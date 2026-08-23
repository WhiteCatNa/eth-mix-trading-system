"""表现与仓位数学。默认按 1h K 线、全年 24×365 根年化（永续 7×24 交易）。"""
from __future__ import annotations

import numpy as np
import pandas as pd

BARS_PER_YEAR = 24 * 365


def sharpe_ratio(returns: pd.Series | np.ndarray, bars_per_year: int = BARS_PER_YEAR) -> float:
    """年化夏普：mean / std × √N。样本不足或波动为 0 时返回 0，避免除零。"""
    r = pd.Series(returns, dtype=float).dropna()
    if len(r) < 2:
        return 0.0
    std = float(r.std(ddof=1))
    if std == 0 or np.isnan(std):
        return 0.0
    return float(r.mean() / std * np.sqrt(bars_per_year))


def sortino_ratio(returns: pd.Series | np.ndarray, bars_per_year: int = BARS_PER_YEAR) -> float:
    """索提诺：只用下行标准差做分母。全是正收益时视为 +inf。"""
    r = pd.Series(returns, dtype=float).dropna()
    if len(r) < 2:
        return 0.0
    down = r[r < 0]
    if len(down) < 1:
        return float("inf") if r.mean() > 0 else 0.0
    dstd = float(down.std(ddof=1))
    if dstd == 0 or np.isnan(dstd):
        return 0.0
    return float(r.mean() / dstd * np.sqrt(bars_per_year))


def max_drawdown(equity: pd.Series | np.ndarray) -> float:
    """最大回撤，返回负值（例如 -0.12 表示从峰值跌了 12%）。"""
    eq = pd.Series(equity, dtype=float)
    if eq.empty:
        return 0.0
    peak = eq.cummax()
    dd = (eq - peak) / peak.replace(0, np.nan)
    return float(dd.min()) if len(dd) else 0.0


def calmar_ratio(ann_return: float, mdd: float) -> float:
    """卡玛：年化收益 / |最大回撤|。没有回撤时无定义，返回 0。"""
    if mdd >= 0 or abs(mdd) < 1e-12:
        return 0.0
    return float(ann_return / abs(mdd))


def annualized_return(equity: pd.Series, bars_per_year: int = BARS_PER_YEAR) -> float:
    """按几何增长把整段权益曲线折成年化收益率。"""
    eq = pd.Series(equity, dtype=float).dropna()
    if len(eq) < 2 or eq.iloc[0] <= 0:
        return 0.0
    n = len(eq) - 1
    return float((eq.iloc[-1] / eq.iloc[0]) ** (bars_per_year / n) - 1.0)


def ols_beta(y: np.ndarray, x: np.ndarray, clip: tuple[float, float] = (0.15, 3.0)) -> float:
    """y 对 x 的无截距 OLS β（先去均值，等价于有截距回归的斜率）。

    样本太短、x 方差为 0 时退回 1.0。结果夹在 [0.15, 3] 防止异常 β 撑爆组合帽。
    """
    n = min(len(y), len(x))
    if n < 20:
        return 1.0
    yy = np.asarray(y[-n:], dtype=float)
    xx = np.asarray(x[-n:], dtype=float)
    xx = xx - xx.mean()
    yy = yy - yy.mean()
    den = float(np.dot(xx, xx))
    if den < 1e-18:
        return 1.0
    b = float(np.dot(xx, yy) / den)
    return float(np.clip(b, clip[0], clip[1]))


def horizon_return(close: np.ndarray, lookback: int, skip: int = 0) -> float:
    """过去 lookback 根的简单收益率，可选再往前 skip 根（跳过最近一段噪声）。

    窗口是 [n-1-skip-lookback, n-1-skip]，只用已经发生的收盘价，没有未来函数。
    """
    n = len(close)
    end = n - 1 - skip
    start = end - lookback
    if start < 0 or end <= start:
        return 0.0
    c0, c1 = float(close[start]), float(close[end])
    if c0 <= 0:
        return 0.0
    return c1 / c0 - 1.0


def realized_vol(returns: np.ndarray, lookback: int, bars_per_year: int = BARS_PER_YEAR) -> float:
    """最近 lookback 根收益的样本标准差，再年化。数据不够时给 20% 占位，避免除零杠杆爆炸。"""
    if len(returns) < max(lookback, 5):
        return 0.20
    s = float(np.std(returns[-lookback:], ddof=1))
    return max(s * np.sqrt(bars_per_year), 1e-6)


def tsmom_score(
    close: np.ndarray,
    returns: np.ndarray,
    lookbacks: list[int],
    weights: list[float],
    skip: int = 0,
) -> float:
    """多周期、波动率缩放的时间序列动量分数。

    每个 horizon：r / (σ_bar × √lookback)。除以 √L 是把“L 根累计收益”变成
    近似单位根波动下的 t 统计量，短周期不会因为噪声绝对值小而被长周期淹没。
    再按权重加权。正分=上涨趋势，负分=下跌趋势。
    """
    wsum = sum(weights) or 1.0
    acc = 0.0
    for lb, w in zip(lookbacks, weights):
        r = horizon_return(close, lb, skip)
        sig = float(np.std(returns[-max(lb, 5) :], ddof=1)) if len(returns) >= 5 else 0.01
        sig = max(sig, 1e-6)
        acc += (w / wsum) * (r / (sig * np.sqrt(max(lb, 1))))
    return float(acc)


def score_to_unit(
    score: float,
    scale: float = 1.0,
    min_position: float = 0.0,
    long_only: bool = False,
) -> float:
    """把有符号 TSMOM 分数压成连续仓位 unit ∈ [-1, 1]。

    tanh 在零点附近近似线性，两端饱和：分数极强时也不会超过满仓。
    long_only 把负 unit 裁成 0（跌势空仓而不是做空）。
    |unit| 小于 min_position 归零，避免噪声仓位来回付费。
    """
    scale = max(float(scale), 1e-6)
    unit = float(np.tanh(score / scale))
    if long_only:
        unit = max(unit, 0.0)
    if abs(unit) < min_position:
        return 0.0
    return unit


def round_step(qty: float, step: float) -> float:
    """按交易所数量步进向下取整（远离零的方向截断），避免因精度被拒单。"""
    if step <= 0:
        return qty
    return float(np.floor(abs(qty) / step) * step) * (1.0 if qty >= 0 else -1.0)
