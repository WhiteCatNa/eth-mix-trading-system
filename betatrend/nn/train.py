"""Walk-forward PPO：观测是最近 7 根 bar 的 [7, 30] 窗口，Actor 输出仓位。

关键约束：
  - 逐步奖励用执行对齐 1h 收益（open[t+1] 成交、open[t+2] 平，与回测器一致）
  - 每折只在训练集上拟合 scaler，测试折绝不回头挑参
  - 损失 = clipped surrogate + value + entropy，没有 TSMOM 残差项
  - 全样本权重只在 walk-forward 报告过关后用于推理，OOS 数字才是诚实成绩
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch import nn

from betatrend.config import ROOT, Settings, StrategyCfg
from betatrend.mathx import annualized_return, calmar_ratio, max_drawdown, sharpe_ratio, sortino_ratio
from betatrend.nn.buffer import ReplayBuffer
from betatrend.nn.dataset import (
    FEATURE_NAMES,
    N_FEAT,
    SEQ_LEN,
    build_feature_frame,
    execution_aligned_returns,
    make_windows,
    vol_leverage,
)
from betatrend.nn.model import PPOActorCritic
from betatrend.nn.reward import bar_pnl, shape_rewards

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


def _fit_scaler(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(x, axis=0)
    iqr = np.clip(np.percentile(x, 75, axis=0) - np.percentile(x, 25, axis=0), 1e-6, None)
    return median.astype(np.float32), iqr.astype(np.float32)


def _gae(rewards: np.ndarray, values: np.ndarray, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    t_len = len(rewards)
    adv = np.zeros(t_len, dtype=np.float64)
    last = 0.0
    next_v = 0.0
    for t in range(t_len - 1, -1, -1):
        done = 1.0 if t == t_len - 1 else 0.0
        delta = rewards[t] + gamma * next_v * (1.0 - done) - values[t]
        last = delta + gamma * lam * (1.0 - done) * last
        adv[t] = last
        next_v = values[t]
    ret = adv + values
    return adv.astype(np.float32), ret.astype(np.float32)


@torch.no_grad()
def _rollout(
    net: PPOActorCritic,
    windows: np.ndarray,
    y: np.ndarray,
    lev: np.ndarray,
    cost: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    net.eval()
    t_len = len(windows)
    actions = np.zeros(t_len, dtype=np.float32)
    logps = np.zeros(t_len, dtype=np.float32)
    values = np.zeros(t_len, dtype=np.float32)
    for i in range(0, t_len, ROLLOUT_BS):
        sl = slice(i, min(i + ROLLOUT_BS, t_len))
        xt = torch.tensor(windows[sl], dtype=torch.float32)
        a, lp, v, _ = net.act(xt, deterministic=False)
        actions[sl] = a.cpu().numpy()
        logps[sl] = lp.cpu().numpy()
        values[sl] = v.cpu().numpy()
    exposure = actions * lev.astype(np.float64)
    dlt = np.empty_like(exposure)
    dlt[0] = np.abs(exposure[0])
    dlt[1:] = np.abs(np.diff(exposure))
    rewards = (REWARD_SCALE * (exposure * y - cost * dlt)).astype(np.float32)
    return actions, logps, values, rewards


def _ppo_update(
    net: PPOActorCritic,
    opt: torch.optim.Optimizer,
    windows: np.ndarray,
    actions: np.ndarray,
    old_logp: np.ndarray,
    advantages: np.ndarray,
    returns: np.ndarray,
    cfg: StrategyCfg,
    inner_epochs: int,
) -> None:
    net.train()
    t_len = len(windows)
    batch = min(max(int(cfg.ppo_batch), 16), max(t_len, 1))
    clip = float(cfg.ppo_clip)
    idx = np.arange(t_len)
    adv_full = (advantages - advantages.mean()) / (float(advantages.std()) + 1e-8)
    for _ in range(inner_epochs):
        np.random.shuffle(idx)
        for start in range(0, t_len, batch):
            mb = idx[start : start + batch]
            if len(mb) < 2:
                continue
            xt = torch.tensor(windows[mb], dtype=torch.float32)
            act = torch.tensor(actions[mb], dtype=torch.float32)
            oldlp = torch.tensor(old_logp[mb], dtype=torch.float32)
            adv = torch.tensor(adv_full[mb], dtype=torch.float32)
            ret = torch.tensor(returns[mb], dtype=torch.float32)
            logp, value, ent = net.evaluate(xt, act)
            ratio = (logp - oldlp).exp()
            surr1 = ratio * adv
            surr2 = ratio.clamp(1.0 - clip, 1.0 + clip) * adv
            pg = -torch.min(surr1, surr2).mean()
            vf = (value - ret).pow(2).mean()
            loss = pg + float(cfg.ppo_vf_coef) * vf - float(cfg.ppo_ent_coef) * ent.mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), float(cfg.ppo_max_grad_norm))
            opt.step()


def _train_one(
    windows: np.ndarray,
    y: np.ndarray,
    lev: np.ndarray,
    *,
    cfg: StrategyCfg,
    cost: float,
    seed: int,
    y_clip: float,
) -> PPOActorCritic:
    torch.manual_seed(seed)
    np.random.seed(seed)
    seq_len = int(cfg.seq_len) if cfg.seq_len else SEQ_LEN
    net = PPOActorCritic(n_feat=N_FEAT, seq_len=seq_len, dropout=cfg.nn_dropout)
    opt = torch.optim.AdamW(net.parameters(), lr=float(cfg.ppo_lr), weight_decay=1e-4)
    yc = np.clip(y, -y_clip, y_clip)
    inner = max(int(cfg.ppo_inner_epochs), 1)
    epochs = max(int(cfg.nn_epochs), 1)
    for _ in range(epochs):
        actions, logps, values, rewards = _rollout(net, windows, yc, lev, cost)
        adv, ret = _gae(rewards, values, float(cfg.ppo_gamma), float(cfg.ppo_gae_lambda))
        _ppo_update(net, opt, windows, actions, logps, adv, ret, cfg, inner)
    net.eval()
    return net


@torch.no_grad()
def _predict(net: PPOActorCritic, windows: np.ndarray) -> np.ndarray:
    net.eval()
    out = np.zeros(len(windows), dtype=np.float32)
    for i in range(0, len(windows), ROLLOUT_BS):
        sl = slice(i, min(i + ROLLOUT_BS, len(windows)))
        xt = torch.tensor(windows[sl], dtype=torch.float32)
        out[sl] = net(xt).squeeze(-1).cpu().numpy()
    return out


def _overlay_metrics(y, pos, lev, cost, steps_per_year) -> dict:
    exp = pos * lev
    net = exp * y - cost * np.abs(np.diff(exp, prepend=exp[:1]))
    equity = np.cumprod(1.0 + net)
    eq = pd.Series(equity)
    peak = eq.cummax()
    dd = float(((eq - peak) / peak).min()) if len(eq) else 0.0
    return {
        "sharpe": float(sharpe_ratio(pd.Series(net), bars_per_year=int(round(steps_per_year)))),
        "total_return": float(equity[-1] - 1.0) if len(equity) else 0.0,
        "max_drawdown": dd,
        "turnover": float(np.mean(np.abs(np.diff(exp, prepend=exp[:1])))),
        "mean_pos": float(np.mean(pos)),
    }


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
) -> TrainResult:
    cfg = settings.strategy
    seq_len = int(cfg.seq_len) if cfg.seq_len else SEQ_LEN
    cost = cfg.nn_cost_bps / 10_000.0
    horizon = max(int(cfg.rebalance_hours), 1)
    steps_per_year = float(HOURLY_BARS_PER_YEAR)
    y_clip = 0.04
    feats = build_feature_frame(panel)
    y_all = execution_aligned_returns(panel).to_numpy(dtype=np.float64)
    lev_all = vol_leverage(feats["vol_24"], target=cfg.target_vol_annual, max_leverage=cfg.max_leverage).to_numpy(
        dtype=np.float64
    )
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

    fold = 0
    start = warmup + min_train
    while start + test_h < len(panel) - 2:
        tr_h = valid[(valid >= warmup) & (valid < start - purge)]
        te = np.arange(start, min(start + test_h, len(panel) - 2))
        if len(tr_h) < 48 or len(te) < 4:
            break
        median, iqr = _fit_scaler(x_all[tr_h])
        win = make_windows(_robust_scale(x_all, median, iqr), seq_len=seq_len)
        preds = []
        for seed in range(max(int(cfg.nn_seeds), 1)):
            net = _train_one(
                win[tr_h],
                y_all[tr_h],
                lev_all[tr_h],
                cfg=cfg,
                cost=cost,
                seed=7 + seed + fold * 17,
                y_clip=y_clip,
            )
            preds.append(_predict(net, win[te]))
        pred_te = np.mean(np.stack(preds, axis=0), axis=0).reshape(-1)
        oos_nn[np.asarray(te)] = pred_te
        fold += 1
        start += test_h

    mask = np.isfinite(oos_nn)
    if int(mask.sum()) < 20:
        raise RuntimeError(
            f"Walk-forward produced too few OOS bars: folds={fold} n_oos={int(mask.sum())}"
        )
    oos_nn = np.where(np.abs(oos_nn) < cfg.min_position, 0.0, oos_nn)
    nn_m = _overlay_metrics(y_all[mask], oos_nn[mask], lev_all[mask], cost, steps_per_year)

    prod_end = len(panel) - prod_holdout
    tr_h = valid[valid < prod_end]
    if len(tr_h) < 48:
        tr_h = valid
    median, iqr = _fit_scaler(x_all[tr_h])
    win = make_windows(_robust_scale(x_all, median, iqr), seq_len=seq_len)
    states = []
    for seed in range(max(int(cfg.nn_seeds), 1)):
        net = _train_one(
            win[tr_h],
            y_all[tr_h],
            lev_all[tr_h],
            cfg=cfg,
            cost=cost,
            seed=101 + seed,
            y_clip=y_clip,
        )
        states.append({k: v.cpu() for k, v in net.state_dict().items()})

    path = path or (ROOT / cfg.nn_model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "ppo",
            "seq_len": seq_len,
            "n_feat": N_FEAT,
            "feature_names": FEATURE_NAMES,
            "median": median,
            "iqr": iqr,
            "dropout": cfg.nn_dropout,
            "horizon": horizon,
            "states": states,
        },
        path,
    )
    oos_df = pd.DataFrame(
        {"nn_unit": oos_nn, "y": y_all, "lev": lev_all},
        index=panel.index,
    )
    oos_path = path.with_name(path.stem + "_oos.parquet")
    oos_df.to_parquet(oos_path)
    report = {
        "oos_neural": nn_m,
        "horizon_hours": horizon,
        "n_folds": fold,
        "n_oos_bars": int(mask.sum()),
        "path": str(path),
        "oos_path": str(oos_path),
        "label": "hourly_ppo_from_next_open",
        "obs": f"[{seq_len}, {N_FEAT}]",
        "note": "Walk-forward OOS for PPO actor (greedy). Position is the decision net only; no TSMOM residual.",
    }
    path.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(
        "PPO walk-forward | obs=[{},{}] Sharpe={:.2f} ret={:.2%} mdd={:.2%} | folds={}",
        seq_len,
        N_FEAT,
        nn_m["sharpe"],
        nn_m["total_return"],
        nn_m["max_drawdown"],
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
