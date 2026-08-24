"""核心闭环：ETH 行情 → 决策网 → 多空信号 → OMS。

    1. 特征（实现波动，供反波动杠杆）
    2. 决策网输出 unit ∈ [-1, 1]（正多负空；|unit|<5% 不开仓）
    3. OMS 把目标名义拆成 BUY/SELL
    4. 执行由调用方完成（回测撮合 / paper）
"""
from __future__ import annotations

from dataclasses import dataclass

from betatrend.config import Settings
from betatrend.domain import MarketSnapshot, OrderIntent, TargetPosition
from betatrend.features import FeatureSet, compute_features
from betatrend.oms import OrderManager, ParentOrder
from betatrend.strategy import TimingStrategy


@dataclass
class CycleResult:
    features: FeatureSet
    targets: list[TargetPosition]
    notionals: dict[str, float]
    parent: ParentOrder
    intents: list[OrderIntent]


class DeskCycle:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.strategy = TimingStrategy(settings)
        self.oms = OrderManager(settings)

    def reset(self, capital: float | None = None) -> None:
        _ = capital
        self.strategy.reset()

    def restore_smooth(self, unit: float) -> None:
        self.strategy.restore_smooth(unit)

    def last_smooth_unit(self) -> float:
        return self.strategy.last_smooth_unit()

    def run(
        self,
        snap: MarketSnapshot,
        current_notional: dict[str, float],
        reason: str = "rebalance",
    ) -> CycleResult:
        feat = compute_features(
            snap.panels,
            snap.market_symbol,
            vol_lookback=self.settings.strategy.vol_lookback,
        )
        targets = self.strategy.generate(snap, feat)
        notionals = {t.symbol: t.target_notional for t in targets}
        for s in set(current_notional) | set(snap.panels):
            notionals.setdefault(s, 0.0)
        parent = self.oms.rebalance_intents(
            current_notional, notionals, snap.prices, snap.equity, reason=reason
        )
        return CycleResult(
            features=feat,
            targets=targets,
            notionals=notionals,
            parent=parent,
            intents=parent.children,
        )


DeskCycle = DeskCycle
CycleResult = CycleResult
