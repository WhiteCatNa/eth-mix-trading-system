"""核心交易信号：连续仓位 → 多/空/平。

核心闭环（研究、paper、实盘同一条）：

    拉 ETH K 线 → 策略网络输出 unit ∈ [-1, 1]
    → 本模块做成买/卖信号（仓位 + 方向）
    → OMS 下单

规则：
- unit > 0 做多，unit < 0 做空，unit = 0 空仓。
- |unit| 低于全仓的 ``min_position``（默认 5%）视为空仓，**不开新仓**。
- 已有仓位且新 unit 落到 5% 以下时，信号是平仓，不是继续拿着灰尘仓。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class PositionSide(str, Enum):
    """持仓方向。和订单 Side（BUY/SELL）分开：这是策略仓位，不是单笔委托。"""

    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"


@dataclass(frozen=True)
class TradeSignal:
    """网络决策落地后的可执行信号。"""

    side: PositionSide
    unit: float  # 相对全仓比例，[-1, 1]；0 = 不开仓/平仓
    target_notional: float  # 带符号名义本金，正多负空

    @property
    def is_flat(self) -> bool:
        return self.side is PositionSide.FLAT or abs(self.unit) < 1e-12


def apply_open_threshold(unit: float, min_position: float = 0.05) -> float:
    """|unit| 低于全仓比例则归零：不开发新仓。"""
    u = float(unit)
    floor = max(float(min_position), 0.0)
    if abs(u) < floor:
        return 0.0
    return float(max(-1.0, min(1.0, u)))


def smooth_unit(
    raw_unit: float,
    last_unit: float,
    *,
    smooth: float,
    min_position: float = 0.05,
    long_only: bool = False,
) -> float:
    """一步 unit 后处理：long_only 截断 → EMA → min_position 归零。

    实盘推理（``NeuralPolicy.predict_unit``）和训练评估（``nn.train.desk_positions``）
    共用这一份，两边的口径就不会再各走各的。

    返回的是新的 EMA 状态，**没有** clip 到 [-1, 1]：被 min_position 归零时状态
    一起归零，这是原实现的行为。调用方拿去当仓位用时自己 clip。
    """
    u = float(raw_unit)
    if long_only:
        u = max(u, 0.0)
    a = min(max(float(smooth), 0.0), 0.95)
    u = (1.0 - a) * u + a * float(last_unit)
    if abs(u) < max(float(min_position), 0.0):
        return 0.0
    return u


def side_from_unit(unit: float) -> PositionSide:
    if unit > 0.0:
        return PositionSide.LONG
    if unit < 0.0:
        return PositionSide.SHORT
    return PositionSide.FLAT


def make_signal(
    unit: float,
    target_notional: float,
    min_position: float = 0.05,
) -> TradeSignal:
    """把连续仓位收成多/空/平。低于全仓 5% 一律 FLAT、名义为 0。"""
    unit = apply_open_threshold(unit, min_position)
    if unit == 0.0:
        return TradeSignal(side=PositionSide.FLAT, unit=0.0, target_notional=0.0)
    return TradeSignal(
        side=side_from_unit(unit),
        unit=unit,
        target_notional=float(target_notional),
    )


def full_notional(equity: float, risk_budget: float, max_leverage: float) -> float:
    """unit=1 时的全仓名义（权益 × 风险预算 × 杠杆帽）。"""
    return abs(float(equity) * float(risk_budget) * float(max_leverage))


def min_open_notional(
    equity: float,
    risk_budget: float,
    max_leverage: float,
    min_position: float = 0.05,
) -> float:
    """低于该名义视为“不到全仓 5%”，空仓时不开新单。"""
    return full_notional(equity, risk_budget, max_leverage) * max(float(min_position), 0.0)


min_open_notional = min_open_notional
