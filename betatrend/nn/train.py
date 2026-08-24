"""Walk-forward GRPO：观测是最近 7 根 bar 的 [7, n_feat] 窗口，Actor 输出仓位。

关键约束：
  - 逐步奖励：desk 执行后的窗口 PnL（8h 冻结 + 实盘费率）→ 波动标准化 → r − λ·min(r,0)²
  - 同一再平衡状态下采 G 个动作，组内 (r-mean)/std 当优势，没有 Critic / GAE
  - 每折只在训练集上拟合 scaler，测试折绝不回头挑参
  - 损失 = clipped surrogate + KL(π || π_old) − entropy，没有 TSMOM 残差项
  - 全样本权重只在 walk-forward 报告过关后用于推理，OOS 数字才是诚实成绩
  - 折与折之间按 seed 链条热启动：网络权重接着用，scaler / 优化器每折重来
  - 同一折的各 seed 之间没有依赖，nn_jobs>1 时用 spawn 进程池并行；折与折仍严格串行
"""
from __future__ import annotations

import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch import nn

from betatrend.config import ROOT, Settings, StrategyCfg, desk_hold_bars, execution_cost_rate
from betatrend.mathx import annualized_return, calmar_ratio, max_drawdown, sharpe_ratio, sortino_ratio
from betatrend.nn.buffer import ReplayBuffer
from betatrend.nn.dataset import (
    FEATURE_NAMES,
    N_FEAT,
    SEQ_LEN,
    build_feature_frame,
    execution_aligned_returns,
    make_windows,
    shuffle_rebalance_index,
    sizing_vol,
    vol_leverage,
)
from betatrend.nn.model import GRPOActor
from betatrend.nn.reward import group_advantages, hold_group_rewards
from betatrend.signals import smooth_unit

HOURLY_BARS_PER_YEAR = 24 * 365
FEAT_CLIP = 8.0
ROLLOUT_BS = 256


@dataclass
class TrainResult:
    """一次 walk-forward 训练的产物：OOS 指标、权重路径、折数。"""

    oos_sharpe: float
    oos_return: float
    oos_max_dd: float
    path: Path
    n_folds: int
    metrics: dict = field(default_factory=dict)


def _robust_scale(x: np.ndarray, median: np.ndarray, iqr: np.ndarray) -> np.ndarray:
    z = (x - median) / np.clip(iqr, 1e-6, None)
    return np.clip(z, -FEAT_CLIP, FEAT_CLIP).astype(np.float32)


def sizing_series(panel: pd.DataFrame, cfg: StrategyCfg) -> tuple[np.ndarray, np.ndarray]:
    """(年化波动, 杠杆)。窗口跟着 ``strategy.vol_lookback`` 走，与 desk 同一口径。

    特征表里的 ``vol_24`` 只是模型输入，不再拿来定杠杆。
    """
    vol = sizing_vol(panel, cfg.vol_lookback)
    lev = vol_leverage(vol, target=cfg.target_vol_annual, max_leverage=cfg.max_leverage)
    return vol.to_numpy(dtype=np.float64), lev.to_numpy(dtype=np.float64)


def _fit_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(x, axis=0)
    iqr = np.clip(np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0), 1e-6, None)
    return median.astype(np.float32), iqr.astype(np.float32)


@torch.no_grad()
def _grpo_rollout(
    net: GRPOActor,
    obs: torch.Tensor,
    y: np.ndarray,
    lev: np.ndarray,
    vol_ann: np.ndarray,
    cost: float,
    cfg: StrategyCfg,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """在每个再平衡点采 G 个动作，用窗口总奖励做组内相对优势。

    观测不含仓位，所以所有再平衡点可以先一批采样；奖励仍按访问顺序推进
    EMA / 暴露（换手依赖上一窗口结束时的仓位）。``rng`` 非空时打乱再平衡点
    的先后，每个点的 8h 收益窗仍是原来的连续 bar。
    """
    net.eval()
    n = int(obs.shape[0])
    hold = desk_hold_bars(cfg)
    group = max(int(cfg.grpo_group_size), 2)
    if rng is None:
        reb = np.arange(0, n, hold, dtype=np.int64)
    else:
        reb = shuffle_rebalance_index(n, hold, rng)
    action_g, logp_g = net.sample_group(obs.index_select(0, torch.from_numpy(reb)), group)
    action_g = action_g.cpu().numpy()
    logp_g = logp_g.cpu().numpy()
    last_unit = 0.0
    prev_exp = 0.0
    idx_out: list[np.ndarray] = []
    act_out: list[np.ndarray] = []
    lp_out: list[np.ndarray] = []
    adv_out: list[np.ndarray] = []
    exec_r: list[float] = []
    kw = dict(
        cost=float(cost),
        smooth=float(cfg.nn_smooth),
        min_position=float(cfg.min_position),
        long_only=bool(cfg.long_only),
        down_lambda=float(cfg.reward_down_lambda),
        dd_inc=float(cfg.reward_dd_inc),
        dd_level=float(cfg.reward_dd_level),
        clip=float(cfg.reward_clip),
    )
    for i, t in enumerate(reb):
        sl = slice(int(t), min(int(t) + hold, n))
        rewards, held, ema = hold_group_rewards(
            action_g[:, i],
            last_unit=last_unit,
            prev_exp=prev_exp,
            y=y[sl],
            lev=lev[sl],
            vol_ann=vol_ann[sl],
            **kw,
        )
        adv = np.nan_to_num(group_advantages(rewards), nan=0.0, posinf=0.0, neginf=0.0)
        idx_out.append(np.full(group, int(t), dtype=np.int64))
        act_out.append(np.asarray(action_g[:, i], dtype=np.float32))
        lp_out.append(np.asarray(logp_g[:, i], dtype=np.float32))
        adv_out.append(adv)
        last_unit = float(ema[0])
        prev_exp = float(held[0] * lev[sl][-1])
        exec_r.append(float(rewards[0]))
    return (
        np.concatenate(idx_out),
        np.concatenate(act_out),
        np.concatenate(lp_out),
        np.concatenate(adv_out),
        float(np.mean(exec_r)) if exec_r else 0.0,
    )


def _grpo_update(
    net: GRPOActor,
    opt: torch.optim.Optimizer,
    obs: torch.Tensor,
    packed: dict[str, np.ndarray],
    cfg: StrategyCfg,
    inner_epochs: int,
) -> None:
    """obs 是整段观测；packed["index"] 指回再平衡 bar。优势已经是组内相对值，不再全局标准化。"""
    net.train()
    t_len = len(packed["advantages"])
    batch = min(max(int(cfg.ppo_batch), 16), max(t_len, 1))
    clip = float(cfg.ppo_clip)
    kl_coef = float(cfg.grpo_kl_coef)
    rows = torch.from_numpy(packed["index"])
    act_all = torch.from_numpy(packed["actions"])
    logp_all = torch.from_numpy(packed["logp"])
    adv_all = torch.from_numpy(packed["advantages"])
    idx = np.arange(t_len)
    for _ in range(inner_epochs):
        np.random.shuffle(idx)
        for start in range(0, t_len, batch):
            if t_len - start < 2:
                continue
            mb = torch.from_numpy(idx[start : start + batch])
            xt = obs.index_select(0, rows.index_select(0, mb))
            act = act_all.index_select(0, mb)
            oldlp = logp_all.index_select(0, mb)
            adv = adv_all.index_select(0, mb)
            logp, ent = net.evaluate(xt, act)
            log_ratio = (logp - oldlp).clamp(-10.0, 10.0)
            ratio = log_ratio.exp()
            surr1 = ratio * adv
            surr2 = ratio.clamp(1.0 - clip, 1.0 + clip) * adv
            pg = -torch.min(surr1, surr2).mean()
            # DeepSeek GRPO：β (π_old/π_new − log(π_old/π_new) − 1)
            kl = ((-log_ratio).exp() + log_ratio - 1.0).mean()
            loss = pg + kl_coef * kl - float(cfg.ppo_ent_coef) * ent.mean()
            if not torch.isfinite(loss):
                continue
            opt.zero_grad()
            loss.backward()
            for p in net.parameters():
                if p.grad is not None:
                    p.grad.nan_to_num_(0.0)
            nn.utils.clip_grad_norm_(net.parameters(), float(cfg.ppo_max_grad_norm))
            opt.step()


def _dump_ckpt(path: Path, payload: dict) -> Path:
    """Atomic torch.save so a kill mid-write cannot leave a truncated file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)
    logger.info("saved checkpoint {}", path)
    return path


def _ckpt_payload(
    *,
    states: list,
    median: np.ndarray,
    iqr: np.ndarray,
    cfg: StrategyCfg,
    seq_len: int,
    extra: dict | None = None,
) -> dict:
    blob = {
        "kind": "grpo",
        "seq_len": int(seq_len),
        "n_feat": N_FEAT,
        "feature_names": FEATURE_NAMES,
        "median": median,
        "iqr": iqr,
        "dropout": cfg.nn_dropout,
        "states": states,
    }
    if extra:
        blob.update(extra)
    return blob


def _train_one(
    windows: np.ndarray,
    y: np.ndarray,
    lev: np.ndarray,
    vol_ann: np.ndarray,
    *,
    cfg: StrategyCfg,
    cost: float,
    seed: int,
    y_clip: float,
    ckpt_prefix: Path | None = None,
    ckpt_meta: dict | None = None,
    init_state: dict | None = None,
) -> GRPOActor:
    torch.manual_seed(seed)
    np.random.seed(seed)
    seq_len = int(cfg.seq_len) if cfg.seq_len else SEQ_LEN
    net = GRPOActor(n_feat=N_FEAT, seq_len=seq_len, dropout=cfg.nn_dropout)
    if init_state is not None:
        net.load_state_dict(init_state)
    opt = torch.optim.AdamW(net.parameters(), lr=float(cfg.ppo_lr), weight_decay=1e-4)
    yc = np.clip(y, -y_clip, y_clip)
    inner = max(int(cfg.ppo_inner_epochs), 1)
    epochs = max(int(cfg.nn_epochs), 1)
    replay = ReplayBuffer(n_rollouts=int(cfg.ppo_replay_rollouts))
    ckpt_every = int(getattr(cfg, "nn_ckpt_every", 0) or 0)
    shuffle = bool(getattr(cfg, "nn_shuffle", True))
    rng = np.random.default_rng(int(seed) + 17) if shuffle else None
    # 观测在整段训练里不变：转一次张量，之后 rollout 切片和 minibatch 都只做 index_select
    obs = torch.from_numpy(np.ascontiguousarray(windows, dtype=np.float32))
    logger.info(
        "GRPO start seed={} n={} epochs={} group={} inner={} replay={} shuffle={} warm_start={}",
        seed,
        len(windows),
        epochs,
        max(int(cfg.grpo_group_size), 2),
        inner,
        replay.n_rollouts,
        shuffle,
        init_state is not None,
    )
    for ep in range(epochs):
        rows, actions, logps, adv, mean_r = _grpo_rollout(net, obs, yc, lev, vol_ann, cost, cfg, rng=rng)
        replay.add(rows, actions, logps, adv)
        _grpo_update(net, opt, obs, replay.packed(), cfg, inner)
        step = ep + 1
        if ep == 0 or step % 20 == 0 or step == epochs:
            logger.info("GRPO seed={} epoch {}/{} mean_r={:.4f}", seed, step, epochs, mean_r)
        if ckpt_prefix is not None and ckpt_every > 0 and step % ckpt_every == 0:
            meta = dict(ckpt_meta or {})
            median = meta.pop("median", None)
            iqr = meta.pop("iqr", None)
            if median is None or iqr is None:
                continue
            meta.update({"step": step, "seed": int(seed), "partial": step < epochs})
            tags = []
            if meta.get("fold_id") is not None:
                tags.append(f"fold{int(meta['fold_id'])}")
            tags.append(f"s{int(seed)}")
            tags.append(f"e{step}")
            mid = Path(ckpt_prefix).with_name(
                f"{Path(ckpt_prefix).stem}_{'_'.join(tags)}{Path(ckpt_prefix).suffix}"
            )
            _dump_ckpt(
                mid,
                _ckpt_payload(
                    states=[{k: v.cpu() for k, v in net.state_dict().items()}],
                    median=median,
                    iqr=iqr,
                    cfg=cfg,
                    seq_len=seq_len,
                    extra=meta,
                ),
            )
    net.eval()
    return net


@torch.no_grad()
def _predict(net: GRPOActor, windows: np.ndarray) -> np.ndarray:
    net.eval()
    obs = torch.from_numpy(np.ascontiguousarray(windows, dtype=np.float32))
    out = np.zeros(len(windows), dtype=np.float32)
    for i in range(0, len(windows), ROLLOUT_BS):
        sl = slice(i, min(i + ROLLOUT_BS, len(windows)))
        out[sl] = net(obs[sl]).squeeze(-1).cpu().numpy()
    return out


def _finite(x: float, default: float = 0.0) -> float:
    v = float(x)
    return v if np.isfinite(v) else default


# 同一折里的各个 seed 互不依赖（warm-start 是每个 seed 各自成链），可以并行。
# 恒定的特征/收益矩阵通过进程池 initializer 送一次，之后每个任务只 pickle
# 标量、下标和上一折的权重。窗口在子进程里按 scaler 现算，省掉每任务十几 MB。
_SHARED: dict[str, Any] = {}


def _bind_shared(x_all: np.ndarray, y_all: np.ndarray, lev_all: np.ndarray, vol_all: np.ndarray) -> None:
    _SHARED.update(x=x_all, y=y_all, lev=lev_all, vol=vol_all)
    _SHARED.pop("win_key", None)
    _SHARED.pop("win", None)


def _seed_worker_init(
    x_all: np.ndarray,
    y_all: np.ndarray,
    lev_all: np.ndarray,
    vol_all: np.ndarray,
    threads: int,
) -> None:
    torch.set_num_threads(max(int(threads), 1))
    _bind_shared(x_all, y_all, lev_all, vol_all)


def _fold_windows(median: np.ndarray, iqr: np.ndarray, seq_len: int) -> np.ndarray:
    """按本折 scaler 缩放后的整段窗口。同一折的多个 seed 共用，缓存一份。"""
    key = (median.tobytes(), iqr.tobytes(), int(seq_len))
    if _SHARED.get("win_key") != key:
        x_all = _SHARED["x"]
        _SHARED["win"] = make_windows(_robust_scale(x_all, median, iqr), seq_len=seq_len)
        _SHARED["win_key"] = key
    return _SHARED["win"]


def _seed_worker(job: dict) -> dict:
    """训练一个 (fold, seed)，返回权重和测试折预测。可在子进程里跑。"""
    tr = np.asarray(job["train_idx"], dtype=int)
    te = job["test_idx"]
    win = _fold_windows(job["median"], job["iqr"], int(job["seq_len"]))
    net = _train_one(
        win[tr],
        _SHARED["y"][tr],
        _SHARED["lev"][tr],
        _SHARED["vol"][tr],
        cfg=job["cfg"],
        cost=float(job["cost"]),
        seed=int(job["seed"]),
        y_clip=float(job["y_clip"]),
        ckpt_prefix=job["ckpt_prefix"],
        ckpt_meta=job["ckpt_meta"],
        init_state=job["init_state"],
    )
    pred = None
    if te is not None and len(te):
        pred = _predict(net, win[np.asarray(te, dtype=int)])
    return {
        "seed": int(job["seed"]),
        "state": {k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
        "pred": pred,
    }


MIN_THREADS_PER_JOB = 2


def resolve_seed_jobs(cfg: StrategyCfg, n_seeds: int) -> tuple[int, int]:
    """(并行进程数, 每进程 torch 线程数)。nn_jobs<=0 表示按 seed 数铺满。

    线程数不降到 2 以下：单线程 worker 慢得太多，3 个并行反而不如 1 个串行
    （实测 4 核机上 3×1 线程是 0.81×，3×2 线程是 1.38×）。超订交给 OS 调度。
    """
    want = int(getattr(cfg, "nn_jobs", 1) or 0)
    if want <= 0:
        want = n_seeds
    n_jobs = max(1, min(want, n_seeds))
    per = int(getattr(cfg, "nn_threads_per_job", 0) or 0)
    if per <= 0:
        per = max(MIN_THREADS_PER_JOB, torch.get_num_threads() // n_jobs)
    return n_jobs, per


def _run_seed_jobs(jobs: list[dict], pool) -> list[dict]:
    """顺序返回结果。pool 为 None 时在本进程串行跑。"""
    if pool is None:
        return [_seed_worker(job) for job in jobs]
    futures = [pool.submit(_seed_worker, job) for job in jobs]
    return [f.result() for f in futures]


def desk_positions(
    unit: np.ndarray,
    *,
    smooth: float,
    min_position: float,
    hold: int,
    long_only: bool = False,
) -> np.ndarray:
    """把每根 bar 的原始 unit 走一遍 desk 的执行契约，得到实际会持有的仓位。

    desk 只在再平衡 bar 上调 ``DeskCycle.run``（``research.py`` / ``backtest.py``
    都是这样），所以 ``NeuralPolicy.predict_unit`` 每 hold 根才被调用一次，
    EMA 也只在那时前进一步——不是每小时。中间的 bar 冻结在上次成交的仓位上。

    单步后处理走 ``signals.smooth_unit``——和 ``NeuralPolicy.predict_unit`` 同一个
    函数，所以两边不可能再漂。
    """
    n = len(unit)
    out = np.zeros(n, dtype=np.float64)
    hold = max(int(hold), 1)
    last = 0.0
    held = 0.0
    for t in range(n):
        if t % hold == 0:
            last = smooth_unit(
                unit[t], last, smooth=smooth, min_position=min_position, long_only=long_only
            )
            held = float(np.clip(last, -1.0, 1.0))
        out[t] = held
    return out


def _overlay_metrics(y, pos, lev, cost, steps_per_year) -> dict:
    """评估主指标是 Sharpe 与最大回撤；收益率只作次要参考。"""
    exp = pos * lev
    net = exp * y - cost * np.abs(np.diff(exp, prepend=exp[:1]))
    equity = np.cumprod(1.0 + net)
    eq = pd.Series(equity)
    spy = int(round(steps_per_year))
    mdd = float(max_drawdown(eq))
    ann = float(annualized_return(eq, bars_per_year=spy))
    return {
        "sharpe": _finite(sharpe_ratio(pd.Series(net), bars_per_year=spy)),
        "max_drawdown": mdd,
        "calmar": _finite(calmar_ratio(ann, mdd)),
        "sortino": _finite(sortino_ratio(pd.Series(net), bars_per_year=spy)),
        "annualized_return": ann,
        "total_return": float(equity[-1] - 1.0) if len(equity) else 0.0,
        "turnover": float(np.mean(np.abs(np.diff(exp, prepend=exp[:1])))),
        "mean_pos": float(np.mean(pos)),
    }


def fold_ckpt_path(prefix: Path, fold_id: int, seed: int, step: int) -> Path:
    """``eth_decision_fold{id}_s{seed}_e{step}.pt``，和 ``_train_one`` 中途落盘同名。"""
    prefix = Path(prefix)
    return prefix.with_name(f"{prefix.stem}_fold{int(fold_id)}_s{int(seed)}_e{int(step)}{prefix.suffix}")


def fold_seed(fold: int, seed_i: int) -> int:
    return 7 + int(seed_i) + int(fold) * 17


def latest_complete_fold(prefix: Path, n_seeds: int, epochs: int) -> int:
    """从 0 起连续找齐了 n_seeds 个 e{epochs} 文件的最后一折。没有则 -1。"""
    last = -1
    fold = 0
    n_seeds = max(int(n_seeds), 1)
    epochs = max(int(epochs), 1)
    while True:
        paths = [fold_ckpt_path(prefix, fold, fold_seed(fold, s), epochs) for s in range(n_seeds)]
        if all(p.is_file() for p in paths):
            last = fold
            fold += 1
            continue
        break
    return last


def _load_ckpt_state(path: Path) -> dict:
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    states = blob.get("states") or []
    if not states:
        raise ValueError(f"checkpoint has no states: {path}")
    return states[0]


def _actor_from_ckpt(path: Path) -> tuple[GRPOActor, dict]:
    blob = torch.load(Path(path), map_location="cpu", weights_only=False)
    net = GRPOActor(
        n_feat=int(blob.get("n_feat") or N_FEAT),
        seq_len=int(blob.get("seq_len") or SEQ_LEN),
        dropout=float(blob.get("dropout", 0.2)),
    )
    states = blob.get("states") or []
    if not states:
        raise ValueError(f"checkpoint has no states: {path}")
    net.load_state_dict(states[0])
    net.eval()
    return net, blob


def _replay_completed_oos(
    oos_nn: np.ndarray,
    x_all: np.ndarray,
    folds: list[tuple[int, np.ndarray, np.ndarray]],
    *,
    start_fold: int,
    path: Path,
    n_seeds: int,
    epochs: int,
    seq_len: int,
) -> None:
    """用已落盘的折权重回放跳过折的测试预测，好让最终 walk-forward OOS 仍然完整。"""
    for fold, _tr, te in folds:
        if fold >= start_fold:
            break
        preds: list[np.ndarray] = []
        win = None
        for s in range(n_seeds):
            ckpt = fold_ckpt_path(path, fold, fold_seed(fold, s), epochs)
            if not ckpt.is_file():
                raise FileNotFoundError(f"missing fold checkpoint for OOS replay: {ckpt}")
            net, blob = _actor_from_ckpt(ckpt)
            if win is None:
                median = np.asarray(blob["median"], dtype=np.float32)
                iqr = np.asarray(blob["iqr"], dtype=np.float32)
                win = make_windows(_robust_scale(x_all, median, iqr), seq_len=seq_len)
            preds.append(_predict(net, win[np.asarray(te, dtype=int)]))
        oos_nn[np.asarray(te)] = np.mean(np.stack(preds, axis=0), axis=0).reshape(-1)
        logger.info("replayed OOS fold {}/{}", fold + 1, len(folds))


def walk_forward_folds(
    n: int,
    *,
    warmup: int,
    min_train: int,
    test_h: int,
    purge: int,
) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Yield (fold_id, train_idx, test_idx). Scaler must be fit on train_idx only."""
    valid = np.arange(n)
    valid = valid[(valid >= warmup) & (valid < n - 2)]
    folds: list[tuple[int, np.ndarray, np.ndarray]] = []
    fold = 0
    start = warmup + min_train
    while start + test_h < n - 2:
        tr_h = valid[(valid >= warmup) & (valid < start - purge)]
        te = np.arange(start, min(start + test_h, n - 2))
        if len(tr_h) < 48 or len(te) < 4:
            break
        folds.append((fold, tr_h, te))
        fold += 1
        start += test_h
    return folds


def list_fold_jobs(
    n: int,
    cfg: StrategyCfg,
    *,
    min_train: int | None = None,
    test_h: int | None = None,
    purge: int = 24,
    prod_holdout: int | None = None,
) -> list[dict]:
    """One walk-forward fold × seed, plus production seeds.

    Independent only if each job starts from scratch. Sequential warm-start
    needs train_decision_net or train_fold(..., init_path=previous.pt).
    """
    warmup = max(cfg.min_history, 200)
    min_train = int(min_train if min_train is not None else 90 * 24)
    test_h = int(test_h if test_h is not None else 21 * 24)
    prod_holdout = int(prod_holdout if prod_holdout is not None else 14 * 24)
    jobs: list[dict] = []
    for fold, tr, te in walk_forward_folds(n, warmup=warmup, min_train=min_train, test_h=test_h, purge=purge):
        for s in range(max(int(cfg.nn_seeds), 1)):
            jobs.append(
                {
                    "fold_id": fold,
                    "seed": 7 + s + fold * 17,
                    "train_start": int(tr[0]),
                    "train_end": int(tr[-1] + 1),
                    "test_start": int(te[0]),
                    "test_end": int(te[-1] + 1),
                    "role": "oos",
                }
            )
    valid = np.arange(n)
    valid = valid[(valid >= warmup) & (valid < n - 2)]
    tr_h = valid[valid < n - prod_holdout]
    if len(tr_h) < 48:
        tr_h = valid
    for s in range(max(int(cfg.nn_seeds), 1)):
        jobs.append(
            {
                "fold_id": -1,
                "seed": 101 + s,
                "train_start": int(tr_h[0]),
                "train_end": int(tr_h[-1] + 1),
                "test_start": None,
                "test_end": None,
                "role": "prod",
            }
        )
    return jobs


def train_fold(
    panel: pd.DataFrame,
    settings: Settings,
    *,
    train_idx: np.ndarray,
    test_idx: np.ndarray | None = None,
    fold_id: int = 0,
    seed: int = 7,
    path: Path | None = None,
    init_path: Path | None = None,
) -> dict:
    """Train one PPO seed on a single fold. Pass init_path to warm-start from the previous fold."""
    cfg = settings.strategy
    seq_len = int(cfg.seq_len) if cfg.seq_len else SEQ_LEN
    cost = execution_cost_rate(settings)
    y_clip = 0.04
    feats = build_feature_frame(panel)
    y_all = execution_aligned_returns(panel).to_numpy(dtype=np.float64)
    vol_all, lev_all = sizing_series(panel, cfg)
    x_all = feats.to_numpy(dtype=np.float64)
    train_idx = np.asarray(train_idx, dtype=int)
    median, iqr = _fit_scaler(x_all[train_idx])
    win = make_windows(_robust_scale(x_all, median, iqr), seq_len=seq_len)
    init_state = None
    if init_path is not None:
        blob = torch.load(Path(init_path), map_location="cpu", weights_only=False)
        states = blob.get("states") or []
        if not states:
            raise ValueError(f"checkpoint has no states: {init_path}")
        init_state = states[0]
    net = _train_one(
        win[train_idx],
        y_all[train_idx],
        lev_all[train_idx],
        vol_all[train_idx],
        cfg=cfg,
        cost=cost,
        seed=int(seed),
        y_clip=y_clip,
        ckpt_prefix=Path(path) if path is not None else None,
        ckpt_meta={"median": median, "iqr": iqr, "fold_id": int(fold_id)} if path is not None else None,
        init_state=init_state,
    )
    pred_te = None
    if test_idx is not None and len(test_idx):
        test_idx = np.asarray(test_idx, dtype=int)
        pred_te = _predict(net, win[test_idx])
    out = {
        "fold_id": int(fold_id),
        "seed": int(seed),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx) if test_idx is not None else 0),
        "median": median,
        "iqr": iqr,
        "pred_te": pred_te,
    }
    if path is not None:
        _dump_ckpt(
            Path(path),
            _ckpt_payload(
                states=[{k: v.cpu() for k, v in net.state_dict().items()}],
                median=median,
                iqr=iqr,
                cfg=cfg,
                seq_len=seq_len,
                extra={"fold_id": int(fold_id), "seed": int(seed)},
            ),
        )
        out["path"] = str(Path(path))
    return out


def train_decision_net(
    panel: pd.DataFrame,
    settings: Settings,
    path: Path | None = None,
    *,
    min_train: int | None = None,
    test_h: int | None = None,
    purge: int = 24,
    prod_holdout: int | None = None,
    min_valid: int = 800,
    start_fold: int | None = None,
    resume: bool = False,
) -> TrainResult:
    cfg = settings.strategy
    seq_len = int(cfg.seq_len) if cfg.seq_len else SEQ_LEN
    train_cost = execution_cost_rate(settings)
    cost = cfg.nn_cost_bps / 10_000.0
    live_cost = train_cost
    horizon = desk_hold_bars(cfg)
    steps_per_year = float(HOURLY_BARS_PER_YEAR)
    y_clip = 0.04
    feats = build_feature_frame(panel)
    y_all = execution_aligned_returns(panel).to_numpy(dtype=np.float64)
    vol_all, lev_all = sizing_series(panel, cfg)
    x_all = feats.to_numpy(dtype=np.float64)
    warmup = max(cfg.min_history, 200)
    valid = np.arange(len(panel))
    valid = valid[(valid >= warmup) & (valid < len(panel) - 2)]
    if len(valid) < min_valid:
        raise ValueError(f"Not enough bars to train NN: {len(valid)}")

    min_train = int(min_train if min_train is not None else 90 * 24)
    test_h = int(test_h if test_h is not None else 21 * 24)
    prod_holdout = int(prod_holdout if prod_holdout is not None else 14 * 24)
    oos_nn = np.full(len(panel), np.nan)
    path = Path(path or (ROOT / cfg.nn_model_path))

    folds = walk_forward_folds(
        len(panel), warmup=warmup, min_train=min_train, test_h=test_h, purge=purge
    )
    logger.info(
        "walk-forward folds={} warmup={} min_train={} test_h={} seeds={} epochs={}",
        len(folds),
        warmup,
        min_train,
        test_h,
        max(int(cfg.nn_seeds), 1),
        max(int(cfg.nn_epochs), 1),
    )
    n_seeds = max(int(cfg.nn_seeds), 1)
    epochs = max(int(cfg.nn_epochs), 1)
    if resume and start_fold is None:
        start_fold = latest_complete_fold(path, n_seeds, epochs) + 1
    start_fold = 0 if start_fold is None else int(start_fold)
    start_fold = max(0, min(start_fold, len(folds)))
    n_jobs, threads_per_job = resolve_seed_jobs(cfg, n_seeds)
    _bind_shared(x_all, y_all, lev_all, vol_all)
    pool = None
    if n_jobs > 1:
        pool = ProcessPoolExecutor(
            max_workers=n_jobs,
            mp_context=mp.get_context("spawn"),
            initializer=_seed_worker_init,
            initargs=(x_all, y_all, lev_all, vol_all, threads_per_job),
        )
        logger.info("seed parallelism: jobs={} threads/job={} seeds={}", n_jobs, threads_per_job, n_seeds)
    try:
        prev_states: list[dict | None] = [None] * n_seeds
        if start_fold > 0:
            prev_fold = start_fold - 1
            for s in range(n_seeds):
                ckpt = fold_ckpt_path(path, prev_fold, fold_seed(prev_fold, s), epochs)
                if not ckpt.is_file():
                    raise FileNotFoundError(f"resume needs {ckpt}")
                prev_states[s] = _load_ckpt_state(ckpt)
            logger.info(
                "resume from fold {}/{} warm_start=fold{} shuffle={}",
                start_fold + 1,
                len(folds),
                prev_fold,
                bool(getattr(cfg, "nn_shuffle", True)),
            )
            _replay_completed_oos(
                oos_nn,
                x_all,
                folds,
                start_fold=start_fold,
                path=path,
                n_seeds=n_seeds,
                epochs=epochs,
                seq_len=seq_len,
            )
        for fold, tr_h, te in folds:
            if fold < start_fold:
                continue
            logger.info(
                "fold {}/{} train_idx=[{}, {}) n_train={} n_test={} warm_start={}",
                fold + 1,
                len(folds),
                int(tr_h[0]),
                int(tr_h[-1] + 1),
                len(tr_h),
                len(te),
                prev_states[0] is not None,
            )
            median, iqr = _fit_scaler(x_all[tr_h])
            results = _run_seed_jobs(
                [
                    {
                        "cfg": cfg,
                        "seq_len": seq_len,
                        "cost": train_cost,
                        "y_clip": y_clip,
                        "median": median,
                        "iqr": iqr,
                        "train_idx": tr_h,
                        "test_idx": te,
                        "seed": 7 + seed + fold * 17,
                        "init_state": prev_states[seed],
                        "ckpt_prefix": path,
                        "ckpt_meta": {"median": median, "iqr": iqr, "fold_id": int(fold)},
                    }
                    for seed in range(n_seeds)
                ],
                pool,
            )
            for seed, res in enumerate(results):
                prev_states[seed] = res["state"]
            pred_te = np.mean(np.stack([res["pred"] for res in results], axis=0), axis=0).reshape(-1)
            oos_nn[np.asarray(te)] = pred_te
        fold = len(folds)

        mask = np.isfinite(oos_nn)
        if int(mask.sum()) < 20:
            raise RuntimeError(
                f"Walk-forward produced too few OOS bars: folds={fold} n_oos={int(mask.sum())}"
            )
        # 两套口径：raw 是每根 bar 换仓、按训练成本记账的裸信号；desk 是走完
        # EMA + 8h 持有 + 真实费率之后、实盘真正会拿到的东西。主指标看 desk。
        raw_unit = np.where(np.abs(oos_nn) < cfg.min_position, 0.0, oos_nn)
        desk_unit = desk_positions(
            oos_nn[mask],
            smooth=cfg.nn_smooth,
            min_position=cfg.min_position,
            hold=horizon,
            long_only=cfg.long_only,
        )
        raw_m = _overlay_metrics(y_all[mask], raw_unit[mask], lev_all[mask], cost, steps_per_year)
        nn_m = _overlay_metrics(y_all[mask], desk_unit, lev_all[mask], live_cost, steps_per_year)
        oos_nn = raw_unit

        prod_end = len(panel) - prod_holdout
        tr_h = valid[valid < prod_end]
        if len(tr_h) < 48:
            tr_h = valid
        median, iqr = _fit_scaler(x_all[tr_h])
        prod = _run_seed_jobs(
            [
                {
                    "cfg": cfg,
                    "seq_len": seq_len,
                    "cost": train_cost,
                    "y_clip": y_clip,
                    "median": median,
                    "iqr": iqr,
                    "train_idx": tr_h,
                    "test_idx": None,
                    "seed": 101 + seed,
                    "init_state": prev_states[seed],
                    "ckpt_prefix": path,
                    "ckpt_meta": {"median": median, "iqr": iqr, "horizon": horizon, "fold_id": -1},
                }
                for seed in range(n_seeds)
            ],
            pool,
        )
    finally:
        if pool is not None:
            pool.shutdown()

    # 整段集成一次性原子落盘：最终权重路径上不会再出现只含部分 seed 的半成品
    states = [{k: v.cpu() for k, v in res["state"].items()} for res in prod]
    _dump_ckpt(
        path,
        _ckpt_payload(
            states=states,
            median=median,
            iqr=iqr,
            cfg=cfg,
            seq_len=seq_len,
            extra={
                "horizon": horizon,
                "n_seeds_done": len(states),
                "partial": len(states) < n_seeds,
            },
        ),
    )
    desk_full = np.full(len(panel), np.nan)
    desk_full[mask] = desk_unit
    oos_df = pd.DataFrame(
        {"nn_unit": oos_nn, "desk_unit": desk_full, "y": y_all, "lev": lev_all},
        index=panel.index,
    )
    oos_path = path.with_name(path.stem + "_oos.parquet")
    oos_df.to_parquet(oos_path)
    report = {
        "oos_neural": nn_m,
        "oos_raw_signal": raw_m,
        "eval_primary": {
            "sharpe": nn_m["sharpe"],
            "max_drawdown": nn_m["max_drawdown"],
            "calmar": nn_m["calmar"],
        },
        "eval_secondary": {
            "total_return": nn_m["total_return"],
            "annualized_return": nn_m["annualized_return"],
            "sortino": nn_m["sortino"],
            "turnover": nn_m["turnover"],
        },
        "execution_contract": {
            "hold_bars": horizon,
            "nn_smooth": float(cfg.nn_smooth),
            "min_position": float(cfg.min_position),
            "vol_lookback": int(cfg.vol_lookback),
            "eval_cost_bps": round(live_cost * 10_000.0, 4),
            "train_cost_bps": round(train_cost * 10_000.0, 4),
        },
        "horizon_hours": horizon,
        "n_folds": fold,
        "n_oos_bars": int(mask.sum()),
        "path": str(path),
        "oos_path": str(oos_path),
        "label": "hourly_grpo_from_next_open",
        "obs": f"[{seq_len}, {N_FEAT}]",
        "note": (
            "Walk-forward OOS for GRPO actor (greedy), warm-started across folds per seed. "
            "oos_neural replays the desk contract: nn_smooth EMA stepped once per hold_bars, "
            "min_position gate, position frozen between rebalances, live fee+slippage. "
            "Training overlay uses the same desk hold window: GRPO samples a group of "
            "actions at each rebalance, scores each candidate on that frozen window, and "
            "uses group-relative advantages (no critic). "
            "oos_raw_signal is the unsmoothed every-bar signal at nn_cost_bps and is a "
            "reference only -- the desk never trades it. Primary eval is Sharpe and max "
            "drawdown, not raw return. Position is the decision net only; no TSMOM residual."
        ),
    }
    path.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(
        "GRPO walk-forward | obs=[{},{}] desk Sharpe={:.2f} mdd={:.2%} calmar={:.2f} ret={:.2%} "
        "turnover={:.4f} | raw Sharpe={:.2f} turnover={:.4f} | folds={}",
        seq_len,
        N_FEAT,
        nn_m["sharpe"],
        nn_m["max_drawdown"],
        nn_m["calmar"],
        nn_m["total_return"],
        nn_m["turnover"],
        raw_m["sharpe"],
        raw_m["turnover"],
        fold,
    )
    return TrainResult(
        oos_sharpe=nn_m["sharpe"],
        oos_return=nn_m["total_return"],
        oos_max_dd=nn_m["max_drawdown"],
        path=path,
        n_folds=fold,
        metrics=report,
    )
