"""Walk-forward 训练器：在 TSMOM 残差上训练，用交易台同一持有期打分。

关键约束：
  - 标签是执行对齐收益（在 open[t+1] 成交、open[t+2] 平，与回测器一致）
  - 每折只在训练集上拟合 scaler / 选 blend，测试折绝不回头挑参
  - 损失 ≈ −Sharpe + 偏离 TSMOM 的 L2 + gate 打开的惩罚，避免无成本乱交易
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

from betatrend.config import ROOT, Settings
from betatrend.mathx import sharpe_ratio
from betatrend.nn.dataset import (
    FEATURE_NAMES,
    build_feature_frame,
    execution_aligned_returns,
    vol_leverage,
)
from betatrend.nn.model import DecisionNet

HOURLY_BARS_PER_YEAR = 24 * 365
FEAT_CLIP = 8.0
BLEND_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
BLEND_EDGE = 0.02


@dataclass
class TrainResult:
    """一次 walk-forward 训练的产物：OOS 指标、权重路径、折数。"""
    oos_sharpe: float
    oos_return: float
    oos_max_dd: float
    tsmom_oos_sharpe: float
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


def _hold_returns(y_1h: np.ndarray, horizon: int) -> np.ndarray:
    n = len(y_1h)
    out = np.zeros(n, dtype=np.float64)
    if horizon <= 1:
        return y_1h.astype(np.float64)
    log = np.log1p(np.clip(y_1h, -0.5, 0.5))
    csum = np.concatenate([[0.0], np.cumsum(log)])
    last = n - horizon
    if last <= 0:
        return out
    out[:last] = np.expm1(csum[horizon:horizon + last] - csum[:last])
    return out


def _pnl_loss(
    pos: torch.Tensor,
    y: torch.Tensor,
    lev: torch.Tensor,
    cost: float,
    steps_per_year: float,
    tsmom_unit: torch.Tensor,
    grid_w: torch.Tensor | None = None,
    gate: torch.Tensor | None = None,
) -> torch.Tensor:
    exposure = pos * lev
    gross = exposure * y
    dlt = torch.empty_like(exposure)
    dlt[0] = exposure[0].abs()
    dlt[1:] = (exposure[1:] - exposure[:-1]).abs()
    net = gross - cost * dlt
    mean = net.mean()
    std = net.std(correction=0) + 1e-8
    scale = torch.sqrt(torch.tensor(float(steps_per_year), device=pos.device))
    sharpe = mean / std * scale
    if grid_w is not None:
        w = grid_w.clamp(min=0.0)
        wsum = w.sum().clamp(min=1e-8)
        mean_g = (net * w).sum() / wsum
        var_g = (w * (net - mean_g).pow(2)).sum() / wsum
        sharpe_g = mean_g / (var_g.sqrt() + 1e-8) * scale
        sharpe = 0.55 * sharpe + 0.45 * sharpe_g
    residual = pos - tsmom_unit
    loss = -sharpe + 0.12 * residual.pow(2).mean()
    if gate is not None:
        loss = loss + 0.02 * gate.mean()
    return loss


def _train_one(
    x: np.ndarray,
    y: np.ndarray,
    lev: np.ndarray,
    tsmom_unit: np.ndarray,
    *,
    hidden: tuple[int, ...],
    dropout: float,
    cost: float,
    epochs: int,
    patience: int,
    seed: int,
    steps_per_year: float,
    y_clip: float,
    delta_gain: float,
    val_frac: float = 0.15,
    grid_w: np.ndarray | None = None,
) -> DecisionNet:
    torch.manual_seed(seed)
    np.random.seed(seed)
    n = len(x)
    split = max(int(n * (1.0 - val_frac)), n // 2)
    if n - split < 8:
        split = max(n - max(n // 10, 8), n // 2)
    net = DecisionNet(x.shape[1], hidden=hidden, dropout=dropout, delta_gain=delta_gain)
    opt = torch.optim.AdamW(net.parameters(), lr=6e-4, weight_decay=1e-4)
    yc = np.clip(y, -y_clip, y_clip)

    def pack(sl: slice):
        gw = None
        if grid_w is not None:
            gw = torch.tensor(grid_w[sl], dtype=torch.float32)
        return (
            torch.tensor(x[sl], dtype=torch.float32),
            torch.tensor(yc[sl], dtype=torch.float32),
            torch.tensor(lev[sl], dtype=torch.float32),
            torch.tensor(tsmom_unit[sl], dtype=torch.float32),
            gw,
        )

    xt, yt, lt, tt, gt = pack(slice(0, split))
    xv, yv, lv, tv, gv = pack(slice(split, None))
    has_val = len(xv) >= 8
    best_state = None
    best_val = float("inf")
    bad = 0
    for _ in range(epochs):
        net.train()
        opt.zero_grad()
        pred = net(xt, tt).squeeze(-1)
        loss = _pnl_loss(
            pred, yt, lt, cost, steps_per_year, tt, grid_w=gt, gate=getattr(net, "_last_gate", None)
        )
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        net.eval()
        with torch.no_grad():
            if has_val:
                pred_v = net(xv, tv).squeeze(-1)
                val = float(
                    _pnl_loss(
                        pred_v, yv, lv, cost, steps_per_year, tv, grid_w=gv, gate=getattr(net, "_last_gate", None)
                    )
                )
            else:
                val = float(loss.detach())
        if val < best_val - 1e-5:
            best_val = val
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()
    return net


@torch.no_grad()
def _predict(net: DecisionNet, x: np.ndarray, tsmom_unit: np.ndarray) -> np.ndarray:
    xt = torch.tensor(x, dtype=torch.float32)
    tt = torch.tensor(tsmom_unit, dtype=torch.float32)
    return net(xt, tt).squeeze(-1).cpu().numpy()


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


def _pick_blend(y, nn, ts, lev, cost, spy) -> float:
    ts_s = _overlay_metrics(y, ts, lev, cost, spy)["sharpe"]
    best_b, best_s = 0.0, ts_s
    for b in BLEND_GRID:
        if b == 0.0:
            continue
        pos = b * nn + (1.0 - b) * ts
        s = _overlay_metrics(y, pos, lev, cost, spy)["sharpe"]
        if s > best_s + BLEND_EDGE:
            best_b, best_s = float(b), s
    return best_b


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
    hidden = tuple(int(h) for h in cfg.nn_hidden)
    cost = cfg.nn_cost_bps / 10_000.0
    horizon = max(int(cfg.rebalance_hours), 1)
    delta_gain = float(getattr(cfg, "nn_delta_gain", 0.5))
    steps_per_year = HOURLY_BARS_PER_YEAR / horizon
    y_clip = 0.04 * np.sqrt(horizon)
    feats = build_feature_frame(panel)
    y_1h = execution_aligned_returns(panel).to_numpy(dtype=np.float64)
    y_all = _hold_returns(y_1h, horizon)
    lev_all = vol_leverage(feats["vol_24"], target=cfg.target_vol_annual, max_leverage=cfg.max_leverage).to_numpy(
        dtype=np.float64
    )
    x_all = feats.to_numpy(dtype=np.float64)
    tsmom_unit = np.tanh(feats["tsmom"].to_numpy() / max(cfg.score_scale, 1e-6))
    tsmom_unit = np.where(np.abs(tsmom_unit) < cfg.min_position, 0.0, tsmom_unit)
    warmup = max(cfg.min_history, 200)
    valid = np.arange(len(panel))
    valid = valid[(valid >= warmup) & (valid < len(panel) - horizon - 2)]
    if len(valid) < min_valid:
        raise ValueError(f"Not enough bars to train NN: {len(valid)}")

    min_train = int(min_train if min_train is not None else 90 * 24)
    test_h = int(test_h if test_h is not None else 21 * 24)
    prod_holdout = int(prod_holdout if prod_holdout is not None else 14 * 24)
    oos_nn = np.full(len(panel), np.nan)
    oos_ts = np.full(len(panel), np.nan)
    oos_blend = np.full(len(panel), np.nan)
    fold_blends: list[float] = []

    fold = 0
    start = warmup + min_train
    while start + test_h < len(panel) - horizon - 2:
        tr_h = valid[(valid >= warmup) & (valid < start - purge)]
        te_h = np.arange(start, min(start + test_h, len(panel) - horizon - 2))
        te = te_h[te_h % horizon == 0]
        if len(tr_h) < 48 or len(te) < 4:
            break
        median, iqr = _fit_scaler(x_all[tr_h])
        xt = _robust_scale(x_all[tr_h], median, iqr)
        preds = []
        last_net = None
        for seed in range(max(int(cfg.nn_seeds), 1)):
            net = _train_one(
                xt,
                y_all[tr_h],
                lev_all[tr_h],
                tsmom_unit[tr_h],
                hidden=hidden,
                dropout=cfg.nn_dropout,
                cost=cost,
                epochs=cfg.nn_epochs,
                patience=cfg.nn_patience,
                seed=7 + seed + fold * 17,
                steps_per_year=steps_per_year,
                y_clip=y_clip,
                delta_gain=delta_gain,
                grid_w=np.where((tr_h % horizon) == 0, 1.0, 0.25).astype(np.float64),
            )
            last_net = net
            preds.append(_predict(net, _robust_scale(x_all[te], median, iqr), tsmom_unit[te]))
        nn_te = np.mean(np.stack(preds, axis=0), axis=0)
        val = tr_h[tr_h % horizon == 0]
        val = val[-max(len(val) // 6, 8) :]
        nn_val = _predict(last_net, _robust_scale(x_all[val], median, iqr), tsmom_unit[val])
        blend_f = _pick_blend(y_all[val], nn_val, tsmom_unit[val], lev_all[val], cost, steps_per_year)
        fold_blends.append(blend_f)
        oos_nn[te] = nn_te
        oos_ts[te] = tsmom_unit[te]
        oos_blend[te] = blend_f * nn_te + (1.0 - blend_f) * tsmom_unit[te]
        fold += 1
        start += test_h

    mask = np.isfinite(oos_blend)
    if int(mask.sum()) < 20:
        raise RuntimeError("Walk-forward produced too few OOS bars")
    oos_nn = np.where(np.abs(oos_nn) < cfg.min_position, 0.0, oos_nn)
    oos_blend = np.where(np.abs(oos_blend) < cfg.min_position, 0.0, oos_blend)
    # Live residual overlay uses blend=1 (net already contains TSMOM). Report that path.
    nn_m = _overlay_metrics(y_all[mask], oos_nn[mask], lev_all[mask], cost, steps_per_year)
    ts_m = _overlay_metrics(y_all[mask], oos_ts[mask], lev_all[mask], cost, steps_per_year)
    gated_m = _overlay_metrics(y_all[mask], oos_blend[mask], lev_all[mask], cost, steps_per_year)
    blend = float(np.median(np.asarray(fold_blends, dtype=float))) if fold_blends else 0.0

    prod_end = len(panel) - prod_holdout
    tr_h = valid[valid < prod_end]
    if len(tr_h) < 48:
        tr_h = valid
    median, iqr = _fit_scaler(x_all[tr_h])
    xt = _robust_scale(x_all[tr_h], median, iqr)
    states = []
    for seed in range(max(int(cfg.nn_seeds), 1)):
        net = _train_one(
            xt,
            y_all[tr_h],
            lev_all[tr_h],
            tsmom_unit[tr_h],
            hidden=hidden,
            dropout=cfg.nn_dropout,
            cost=cost,
            epochs=cfg.nn_epochs,
            patience=cfg.nn_patience,
            seed=101 + seed,
            steps_per_year=steps_per_year,
            y_clip=y_clip,
            delta_gain=delta_gain,
            grid_w=np.where((tr_h % horizon) == 0, 1.0, 0.25).astype(np.float64),
        )
        states.append({k: v.cpu() for k, v in net.state_dict().items()})

    path = path or (ROOT / cfg.nn_model_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "feature_names": FEATURE_NAMES,
            "median": median,
            "iqr": iqr,
            "hidden": list(hidden),
            "dropout": cfg.nn_dropout,
            "blend": blend,
            "delta_gain": delta_gain,
            "horizon": horizon,
            "states": states,
        },
        path,
    )
    oos_df = pd.DataFrame(
        {"nn_unit": oos_nn, "tsmom_unit": oos_ts, "blend_unit": oos_nn, "y": y_all, "lev": lev_all},
        index=panel.index,
    )
    oos_path = path.with_name(path.stem + "_oos.parquet")
    oos_df.to_parquet(oos_path)
    report = {
        "oos_neural_blend": nn_m,
        "oos_neural_gated": gated_m,
        "oos_tsmom": ts_m,
        "chosen_blend": blend,
        "fold_blends": fold_blends,
        "horizon_hours": horizon,
        "n_folds": fold,
        "n_oos_bars": int(mask.sum()),
        "path": str(path),
        "oos_path": str(oos_path),
        "label": f"hold_{horizon}h_from_next_open",
        "note": "Walk-forward OOS at the desk rebalance horizon. Gated residual around TSMOM; fold blend from in-fold val only.",
    }
    path.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info(
        "NN walk-forward | H={}h blend={:.2f} Sharpe={:.2f} ret={:.2%} mdd={:.2%} | TSMOM Sharpe={:.2f} | folds={}",
        horizon, blend, nn_m["sharpe"], nn_m["total_return"], nn_m["max_drawdown"], ts_m["sharpe"], fold,
    )
    return TrainResult(
        oos_sharpe=nn_m["sharpe"],
        oos_return=nn_m["total_return"],
        oos_max_dd=nn_m["max_drawdown"],
        tsmom_oos_sharpe=ts_m["sharpe"],
        path=path,
        n_folds=fold,
        metrics=report,
    )
