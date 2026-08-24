"""交易 RL 的逐步奖励。评估看 Sharpe / 最大回撤，奖励必须朝这两个量对齐。

不要直接用收益率：ETH 1h 噪声大、肥尾，高波动时段会主导梯度，智能体学会“赌波动”
而不是提高风险调整后收益。

本模块的路径：

    1. 执行对齐 PnL（含手续费/换手）——和回测器同一套钱
    2. 用当时的小时波动把 PnL 标准化，得到 r——去掉波动尺度
    3. 下行方差惩罚 r - λ·min(r,0)²——逐 bar 形式的“均值 - 下行方差”
    4. 路径回撤惩罚（默认关闭，见下）
    5. clip，纯护栏；正常量级下不触发

为什么不再用 differential Sharpe
--------------------------------
DSR（Moody & Saffell）保证的是“增量 ≈ 滚动夏普的一阶更新”，是给在线学习设计的。
PPO 最大化折现回报，对平稳策略等价于奖励的路径均值，而 DSR 增量的路径均值由 O(η²)
的异方差偏差主导，并不随夏普单调——ETH 的波动聚集恰好把这个偏差放大：大波动之后
E[R²] 抬升，随后的收益被更大的分母除，信号越强、暴露波动越大、被压得越狠。

在 ETH 1h 上实测过：构造 20 个候选信号（噪声、动量、常仓、不同强度的 oracle、
反向 oracle），按 desk 执行契约算出各自真实的夏普与最大回撤，再看奖励的路径均值
能不能把它们排对。两个互不重叠的窗口上：

                        秩相关 vs 夏普   vs 最大回撤   奖励最优信号
    DSR（旧）              0.40 ~ 0.44    0.47 ~ 0.58   全程空仓  ← 最优解是不交易
    r - 0.5·min(r,0)²      0.81 ~ 0.88    0.87 ~ 0.90   高夏普 oracle

旧奖励下，夏普 +12.6 的 oracle 得分低于全程空仓，亏掉 99% 本金的反向 oracle 还排在
小赚的动量之上。这不是 clip 或回撤权重造成的：把 clip 放开到无穷、回撤罚整个去掉，
差分项自己的均值依旧排反。

λ 的选择
--------
0.25 ~ 1.0 之间是平坦最优，取 0.5。λ=0 也能排对夏普，但奖励对暴露是线性的，最优仓位
会恒等于满仓；加上下行项后最优暴露 ∝ 边缘/λ，定仓才是个良定义的问题。

回撤项默认关掉
--------------
``dd_inc`` / ``dd_level`` 默认 0：实测下行方差项已经把回撤管住（秩相关 0.90），
再叠加路径回撤项对排序没有增益（0.899 vs 0.903，略降）。留作可调旋钮。

训练时先走 desk 再计 PnL
------------------------
``shape_rewards`` 本身不管换仓频率。PPO 的 rollout 会先把采样动作送进
``desk_positions``（EMA + ``min_position`` 死区，每 ``rebalance_hours`` 根才成交），
再用 desk 真正持有的仓位调用 ``bar_pnl``，费率与实盘相同（手续费+滑点）。

否则奖励按每小时 |Δexposure| 扣费，评估却是 8h 冻结：高斯探索的翻仓是确定性
策略的十几倍，训练信号被自伤换手主导，而那笔费用上线根本不会发生。
"""
from __future__ import annotations

import numpy as np

from betatrend.mathx import BARS_PER_YEAR

# 两个纯护栏。r 的标准差实测 ~0.32、单 bar PnL ~0.001，正常量级下都不触发；
# 留着只为挡住数据异常（除零、脏 bar）造成的爆炸值。
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
    down_lambda: float = 0.5,
    dd_inc: float = 0.0,
    dd_level: float = 0.0,
    clip: float = 5.0,
) -> np.ndarray:
    """把 PnL 序列变成逐步奖励（均值 - 下行方差，可选路径回撤）。

    与 ``env.RewardMachine`` 逐 bar 等价，两者由 ``test_env`` 锁在 1e-6 内。
    GRPO 在再平衡窗口上对这条序列求和，再做组内标准化。
    """
    pnl = np.asarray(pnl, dtype=np.float64)
    vol_annual = np.asarray(vol_annual, dtype=np.float64)
    if pnl.shape != vol_annual.shape:
        raise ValueError(f"pnl/vol length mismatch: {pnl.shape} vs {vol_annual.shape}")
    if len(pnl) == 0:
        return pnl.astype(np.float32)

    hourly_vol = np.clip(vol_annual / np.sqrt(float(BARS_PER_YEAR)), 1e-6, None)
    r = np.clip(pnl / hourly_vol, -_R_VOL_CLIP, _R_VOL_CLIP)
    down = np.minimum(r, 0.0)
    shaped = r - float(down_lambda) * down * down

    equity = np.cumprod(1.0 + np.clip(pnl, -_PNL_CLIP, _PNL_CLIP))
    peak = np.maximum.accumulate(equity)
    depth = (peak - equity) / np.clip(peak, 1e-12, None)
    deepen = np.maximum(np.diff(depth, prepend=0.0), 0.0)
    shaped -= float(dd_inc) * deepen + float(dd_level) * depth

    return np.clip(shaped, -float(clip), float(clip)).astype(np.float32)


def hold_group_rewards(
    actions: np.ndarray,
    *,
    last_unit: float,
    prev_exp: float,
    y: np.ndarray,
    lev: np.ndarray,
    vol_ann: np.ndarray,
    cost: float,
    smooth: float,
    min_position: float,
    long_only: bool = False,
    down_lambda: float = 0.5,
    dd_inc: float = 0.0,
    dd_level: float = 0.0,
    clip: float = 5.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """同一再平衡点上 G 个候选动作的窗口总奖励。

    每个候选先走 ``smooth_unit``（与 desk 相同），然后在窗口内冻结仓位，
    只在第一根支付换手。返回 ``(reward[G], held[G], ema[G])``。
    """
    raw = np.asarray(actions, dtype=np.float64).reshape(-1)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    lev = np.asarray(lev, dtype=np.float64).reshape(-1)
    vol_ann = np.asarray(vol_ann, dtype=np.float64).reshape(-1)
    if y.shape != lev.shape or y.shape != vol_ann.shape:
        raise ValueError("y/lev/vol length mismatch")
    if long_only:
        raw = np.maximum(raw, 0.0)
    alpha = min(max(float(smooth), 0.0), 0.95)
    ema = (1.0 - alpha) * raw + alpha * float(last_unit)
    floor = max(float(min_position), 0.0)
    ema = np.where(np.abs(ema) < floor, 0.0, ema)
    held = np.clip(ema, -1.0, 1.0)
    if y.size == 0:
        z = np.zeros(len(raw), dtype=np.float64)
        return z, held, ema
    exp = held[:, None] * lev[None, :]
    dlt = np.zeros_like(exp)
    dlt[:, 0] = np.abs(exp[:, 0] - float(prev_exp))
    pnl = exp * y[None, :] - float(cost) * dlt
    hourly = np.clip(vol_ann / np.sqrt(float(BARS_PER_YEAR)), 1e-6, None)
    r = np.clip(pnl / hourly[None, :], -_R_VOL_CLIP, _R_VOL_CLIP)
    down = np.minimum(r, 0.0)
    shaped = r - float(down_lambda) * down * down
    if float(dd_inc) or float(dd_level):
        growth = 1.0 + np.clip(pnl, -_PNL_CLIP, _PNL_CLIP)
        equity = np.cumprod(growth, axis=1)
        peak = np.maximum.accumulate(equity, axis=1)
        depth = (peak - equity) / np.clip(peak, 1e-12, None)
        deepen = np.maximum(np.diff(depth, axis=1, prepend=0.0), 0.0)
        shaped = shaped - float(dd_inc) * deepen - float(dd_level) * depth
    shaped = np.clip(shaped, -float(clip), float(clip))
    return shaped.sum(axis=1), held, ema


def group_advantages(rewards: np.ndarray) -> np.ndarray:
    """GRPO：组内 (r - mean) / std。全员分数一样时优势为 0。"""
    r = np.asarray(rewards, dtype=np.float64).reshape(-1)
    if r.size < 2:
        return np.zeros_like(r, dtype=np.float32)
    sd = float(r.std())
    if sd < 1e-8:
        return np.zeros_like(r, dtype=np.float32)
    return ((r - r.mean()) / sd).astype(np.float32)
