"""交易 RL 的逐步奖励。评估看 Sharpe / 最大回撤，奖励必须朝这两个量对齐。

不要直接用收益率：ETH 1h 噪声大、肥尾，高波动时段会主导梯度，智能体学会“赌波动”
而不是提高风险调整后收益。

本模块的路径：

    1. 执行对齐 PnL（含手续费/换手）——和回测器同一套钱
    2. 用当时的小时波动把 PnL 标准化——去掉波动尺度
    3. Differential Sharpe（Moody & Saffell）：奖励 = 对滚动夏普的瞬时贡献
    4. 回撤惩罚：加深回撤时扣分，停留在深回撤里持续扣一点，促使回本
    5. clip，避免单根暴跌把 GAE 打爆

Differential Sharpe 递推（指数滑动矩）：

    A_t = A_{t-1} + η (R_t - A_{t-1})          # E[R]
    B_t = B_{t-1} + η (R_t² - B_{t-1})         # E[R²]
    dS  = (B ΔA - 0.5 A ΔB) / (B - A²)^{1.5}   # ∂Sharpe / ∂R 的一阶增量

η ≈ 1/72：大约三天 1h bar，既跟得上制度切换，又不会被单根噪声带跑。
"""
from __future__ import annotations

import numpy as np

from betatrend.mathx import BARS_PER_YEAR

_PNL_CLIP = 0.4
_R_VOL_CLIP = 8.0


def bar_pnl(
    actions: np.ndarray,
    y: np.ndarray,
    lev: np.ndarray,
    cost: float,
) -> np.ndarray:
    """扣费后的 bar PnL。仓位在 open[t+1] 成交时的暴露 = action * lev。"""
    exposure = np.asarray(actions, dtype=np.float64) * np.asarray(lev, dtype=np.float64)
    dlt = np.empty_like(exposure)
    dlt[0] = np.abs(exposure[0])
    dlt[1:] = np.abs(np.diff(exposure))
    y = np.asarray(y, dtype=np.float64)
    return exposure * y - float(cost) * dlt


def shape_rewards(
    pnl: np.ndarray,
    vol_annual: np.ndarray,
    *,
    eta: float = 1.0 / 72.0,
    dd_inc: float = 1.0,
    dd_level: float = 0.05,
    clip: float = 5.0,
) -> np.ndarray:
    """把 PnL 序列变成 PPO 用的逐步奖励（对齐 Sharpe + 最大回撤）。"""
    pnl = np.asarray(pnl, dtype=np.float64)
    vol_annual = np.asarray(vol_annual, dtype=np.float64)
    if pnl.shape != vol_annual.shape:
        raise ValueError(f"pnl/vol length mismatch: {pnl.shape} vs {vol_annual.shape}")
    n = len(pnl)
    if n == 0:
        return pnl.astype(np.float32)

    hourly_vol = np.clip(vol_annual / np.sqrt(float(BARS_PER_YEAR)), 1e-6, None)
    r = np.clip(pnl / hourly_vol, -_R_VOL_CLIP, _R_VOL_CLIP)

    eta = float(np.clip(eta, 1e-4, 0.5))
    a = 0.0
    b = 1.0
    d_sharpe = np.empty(n, dtype=np.float64)
    for t in range(n):
        rt = float(r[t])
        d_a = rt - a
        d_b = rt * rt - b
        var = max(b - a * a, 1e-8)
        d_sharpe[t] = (b * d_a - 0.5 * a * d_b) / (var**1.5)
        a += eta * d_a
        b += eta * d_b

    equity = np.cumprod(1.0 + np.clip(pnl, -_PNL_CLIP, _PNL_CLIP))
    peak = np.maximum.accumulate(equity)
    depth = (peak - equity) / np.clip(peak, 1e-12, None)
    deepen = np.maximum(np.diff(depth, prepend=0.0), 0.0)
    dd_pen = float(dd_inc) * deepen + float(dd_level) * depth

    shaped = np.clip(d_sharpe - dd_pen, -float(clip), float(clip))
    return shaped.astype(np.float32)
