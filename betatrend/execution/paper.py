"""进程内纸交易经纪：按最新价立刻成交 OMS 意图，不发真实订单。

用于 paper / 回测对齐。手续费按 maker 或 taker 配置，滑点按 market_bps。
成交会立刻写进传入的 Ledger，与实盘桥共用同一套 Fill 结构。
"""
from __future__ import annotations

from dataclasses import dataclass, field

from betatrend.config import Settings
from betatrend.domain import Fill, OrderIntent
from betatrend.ledger import Ledger


@dataclass
class PaperBroker:
    settings: Settings
    ledger: Ledger
    fills: list[Fill] = field(default_factory=list)

    def execute(self, intents: list[OrderIntent], prices: dict[str, float], ts) -> list[Fill]:
        """按 last price 全成。qty≤0 或价格无效的意图跳过。BUY 数量为正，SELL 为负。"""
        out: list[Fill] = []
        fee_rate = (
            self.settings.fees.maker if self.settings.fees.use_maker_for_entries else self.settings.fees.taker
        )
        slip_bps = self.settings.slippage.market_bps / 10_000.0
        for it in intents:
            px = float(prices.get(it.symbol, 0.0))
            if px <= 0 or it.qty <= 0:
                continue
            signed_qty = it.qty if it.side.value == "BUY" else -it.qty
            notional = signed_qty * px
            fill = Fill(
                timestamp=ts,
                symbol=it.symbol,
                notional_delta=notional,
                qty_delta=signed_qty,
                price=px,
                fee=abs(notional) * fee_rate,
                slippage=abs(notional) * slip_bps,
                reason=it.reason,
                client_order_id=it.client_order_id,
            )
            self.ledger.apply_fill(fill)
            self.fills.append(fill)
            out.append(fill)
        return out
