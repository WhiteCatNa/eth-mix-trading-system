"""OMS：一次再平衡父单 → 若干市价子意图。

不做智能拆单/冰山，只做：
  - 周转带（turnover band）：名义变化太小不下单，省手续费
  - 最小名义过滤
  - 数量按 step_size 向下取整
  - 同向减仓标 reduce_only，防止减仓单在仓位已变时反手开仓
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from betatrend.config import Settings
from betatrend.domain import OrderIntent, OrderStatus, Side
from betatrend.mathx import round_step
from betatrend.signals import min_open_notional


@dataclass
class ParentOrder:
    """一次再平衡的父单。无子单时直接标 FILLED（视为“已经在目标上”）。"""

    parent_id: str
    timestamp: object
    status: OrderStatus
    children: list[OrderIntent] = field(default_factory=list)
    reason: str = ""


class OrderManager:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.history: list[ParentOrder] = []

    def _cid(self) -> str:
        """币安 clientOrderId 最长 36；前缀 + uuid 截断后仍可在账本/对账里追踪。"""
        prefix = self.settings.oms.client_id_prefix
        return f"{prefix}{uuid.uuid4().hex[:16]}"[:36]

    def rebalance_intents(
        self,
        current_notional: dict[str, float],
        target_notional: dict[str, float],
        prices: dict[str, float],
        equity: float,
        reason: str = "rebalance",
        step_size: float = 0.001,
    ) -> ParentOrder:
        """把「当前名义 → 目标名义」差拆成 BUY/SELL 子单。价格≤0 的品种跳过。"""
        parent_id = self._cid()
        children: list[OrderIntent] = []
        # 绝对下限 75 USDT，避免权益很小时 band 小到产生无意义碎单。
        band = max(equity * self.settings.backtest.turnover_band_equity, 75.0)
        symbols = set(current_notional) | set(target_notional)
        for s in sorted(symbols):
            px = float(prices.get(s, 0.0))
            if px <= 0:
                continue
            cur = float(current_notional.get(s, 0.0))
            tgt = float(target_notional.get(s, 0.0))
            # 空仓且目标低于全仓 5%：不开新仓。已有仓位落到 5% 以下时 tgt 已被策略打成 0，这里会平仓。
            min_open = min_open_notional(
                equity,
                self.settings.strategy.risk_budget,
                self.settings.strategy.max_leverage,
                self.settings.strategy.min_position,
            )
            if abs(cur) < min_open and abs(tgt) < min_open:
                continue
            delta = tgt - cur
            if abs(delta) < band:
                continue
            if abs(delta) < self.settings.oms.min_notional:
                continue
            qty = round_step(delta / px, step_size)
            if abs(qty) * px < self.settings.oms.min_notional:
                continue
            # 同号且 |目标| < |当前| → 纯减仓；穿越零或加仓则不能 reduce_only。
            reducing = abs(tgt) < abs(cur) - 1e-12 and tgt * cur >= -1e-12
            children.append(
                OrderIntent(
                    client_order_id=self._cid(),
                    symbol=s,
                    side=Side.BUY if qty > 0 else Side.SELL,
                    qty=abs(qty),
                    notional=abs(qty) * px,
                    parent_id=parent_id,
                    reason=reason,
                    reduce_only=reducing,
                )
            )
        parent = ParentOrder(
            parent_id=parent_id,
            timestamp=None,
            status=OrderStatus.PARENT if children else OrderStatus.FILLED,
            children=children,
            reason=reason,
        )
        self.history.append(parent)
        return parent
