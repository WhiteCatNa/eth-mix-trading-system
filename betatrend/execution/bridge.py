"""OMS 桥：paper 本地成交；签名路径仅 testnet；成交后对账，对不上就 kill。

研究/paper 模式走 PaperBroker。testnet/live 走签名 REST，但 oms.testnet_only
为 True 时拒绝主网。每笔成交写审计日志，最后把本地 Ledger 与交易所持仓对账。
"""
from __future__ import annotations

import pandas as pd

from betatrend.config import Settings
from betatrend.control import ControlPlane
from betatrend.domain import Fill, OrderIntent
from betatrend.execution.paper import PaperBroker
from betatrend.execution.signed import BinanceSignedClient
from betatrend.ledger import Ledger
from betatrend.logutil import audit


def _qty_from_position(row: dict) -> float:
    for key in ("positionAmt", "positionAmt", "position_amt"):
        if key in row and row[key] not in (None, ""):
            try:
                return float(row[key])
            except (TypeError, ValueError):
                continue
    return 0.0


def exchange_qty_map(positions: list[dict]) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in positions:
        symbol = str(row.get("symbol") or "")
        if not symbol:
            continue
        out[symbol] = _qty_from_position(row)
    return out


def fill_from_order(
    intent: OrderIntent,
    resp: dict,
    ts,
    prices: dict[str, float],
    settings: Settings,
) -> Fill:
    qty = float(resp.get("executedQty") or resp.get("origQty") or intent.qty or 0.0)
    avg = float(resp.get("avgPrice") or 0.0) or float(prices.get(intent.symbol, 0.0))
    signed = qty if intent.side.value == "BUY" else -qty
    notional = signed * avg
    fee = abs(notional) * float(settings.fees.taker)
    slip = abs(notional) * float(settings.slippage.market_bps) / 10_000.0
    return Fill(
        timestamp=pd.Timestamp(ts),
        symbol=intent.symbol,
        notional_delta=notional,
        qty_delta=signed,
        price=avg,
        fee=fee,
        slippage=slip,
        reason=intent.reason,
        client_order_id=intent.client_order_id,
    )


def reconcile_or_kill(
    settings: Settings,
    ledger: Ledger,
    exchange_qty: dict[str, float],
    *,
    control: ControlPlane | None = None,
) -> None:
    tol = float(settings.oms.reconcile_qty_tol)
    mismatches: list[str] = []
    symbols = set(ledger.qty) | set(exchange_qty)
    for symbol in sorted(symbols):
        local = float(ledger.qty.get(symbol, 0.0))
        remote = float(exchange_qty.get(symbol, 0.0))
        if abs(local - remote) > max(tol, 1e-8):
            mismatches.append(f"{symbol}: local={local} exchange={remote}")
    if not mismatches:
        return
    ctrl = control or ControlPlane(settings)
    detail = "; ".join(mismatches)
    ctrl.trip_kill(reason=f"reconcile mismatch: {detail}")
    audit(settings, "reconcile_kill", mismatches=mismatches)
    raise RuntimeError(f"Position reconcile failed: {detail}")


def submit_intents(
    settings: Settings,
    intents: list[OrderIntent],
    prices: dict[str, float],
    ts,
    *,
    ledger: Ledger,
    confirm: str = "",
    client: BinanceSignedClient | None = None,
) -> list[Fill]:
    mode = settings.account.mode
    if mode in ("research", "paper"):
        broker = PaperBroker(settings, ledger)
        return broker.execute(intents, prices, ts)

    if mode == "live" or (client is not None and not getattr(client, "testnet", True)):
        if settings.oms.testnet_only:
            raise RuntimeError("OMS signed path is testnet-only")

    ctrl = ControlPlane(settings)
    ctrl.assert_can_send_orders(confirm=confirm)
    if client is None:
        client = BinanceSignedClient(settings)
    if not getattr(client, "testnet", True) and settings.oms.testnet_only:
        raise RuntimeError("OMS signed path is testnet-only")

    fills: list[Fill] = []
    for intent in intents:
        if intent.qty <= 0:
            continue
        resp = client.new_order(
            intent.symbol,
            intent.side.value,
            intent.qty,
            confirm=confirm,
            reduce_only=bool(intent.reduce_only),
        )
        fill = fill_from_order(intent, resp, ts, prices, settings)
        ledger.apply_fill(fill)
        fills.append(fill)
        audit(
            settings,
            "oms_fill",
            symbol=intent.symbol,
            side=intent.side.value,
            qty=intent.qty,
            client_order_id=intent.client_order_id,
        )
    try:
        remote = exchange_qty_map(client.positions())
    except Exception as exc:
        ctrl.trip_kill(reason=f"reconcile positions fetch failed: {exc}")
        raise
    reconcile_or_kill(settings, ledger, remote, control=ctrl)
    return fills


# 测试夹具使用的旧名
exchange_qty_map = exchange_qty_map
reconcile_or_kill = reconcile_or_kill
submit_intents = submit_intents
