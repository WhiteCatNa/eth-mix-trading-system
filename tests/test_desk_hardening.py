from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from betatrend.backtest import Backtester
from betatrend.domain import OrderIntent, Side
from betatrend.execution.bridge import exchange_qty_map, reconcile_or_kill, submit_intents
from betatrend.ledger import Ledger
from betatrend.marketdata.synthetic import make_trending_panels
from betatrend.research import paper_run


def test_paper_run_persists_state(settings, tmp_path, stub_rl):
    settings.strategy.decision = "rl"
    settings.backtest.warmup_bars = 160
    settings.paper.dry_run = False
    settings.paper.state_file = str(tmp_path / "paper.json")
    panels = make_trending_panels(n=220, seed=5, symbols=["ETHUSDT"])
    out = paper_run(settings, panels=panels, execute=True, max_bars=8, reset_state=True)
    path = Path(out["state_file"])
    assert path.exists()
    payload = json.loads(path.read_text())
    assert payload["bars"] == 8
    assert out["bars_processed"] == 8
    out2 = paper_run(settings, panels=panels, execute=True, max_bars=4, reset_state=False)
    assert out2["bars_processed"] == 4
    payload2 = json.loads(path.read_text())
    assert payload2["bars"] == 12


def test_paper_run_matches_backtest_sign(settings, tmp_path, stub_rl):
    settings.strategy.decision = "rl"
    settings.strategy.min_history = 150
    settings.backtest.warmup_bars = 160
    settings.paper.dry_run = False
    settings.paper.state_file = str(tmp_path / "paper.json")
    settings.backtest.turnover_band_equity = 0.001
    settings.oms.min_notional = 1.0
    panels = make_trending_panels(n=420, seed=6, symbols=["ETHUSDT"])
    bt = Backtester(settings).run(panels)
    paper_run(settings, panels=panels, execute=True, max_bars=48, reset_state=True)
    paper_state = json.loads((tmp_path / "paper.json").read_text())
    ts = pd.Timestamp(paper_state["last_ts"])
    if ts not in bt.positions.index:
        pytest.skip("paper last bar not in backtest index")
    bt_n = float(bt.positions.loc[ts, "ETHUSDT"])
    tgt = float(paper_state.get("notionals", {}).get("ETHUSDT", 0.0))
    if abs(bt_n) < 1.0 and abs(tgt) < 1.0:
        pytest.skip("both overlays flat at overlap")
    if abs(bt_n) >= 1.0 and abs(tgt) >= 1.0:
        assert np.sign(tgt) == np.sign(bt_n)


class _FakeSigned:
    testnet = True

    def __init__(self, fills_px: float = 2000.0):
        self.fills_px = fills_px
        self.qty: dict[str, float] = {}
        self.orders: list[dict] = []

    def new_order(self, symbol, side, quantity, confirm="", reduce_only=False):
        signed = float(quantity) if side == "BUY" else -float(quantity)
        self.qty[symbol] = self.qty.get(symbol, 0.0) + signed
        self.orders.append(
            {"symbol": symbol, "side": side, "quantity": quantity, "reduce_only": reduce_only}
        )
        return {
            "status": "FILLED",
            "executedQty": str(quantity),
            "avgPrice": str(self.fills_px),
            "symbol": symbol,
        }

    def positions(self):
        return [{"symbol": k, "positionAmt": str(v)} for k, v in self.qty.items() if abs(v) > 0]


def test_oms_testnet_fill_and_reconcile(settings, monkeypatch):
    monkeypatch.setenv("BINANCE_TESTNET", "1")
    settings.account.mode = "testnet"
    settings.oms.min_notional = 1.0
    ledger = Ledger(cash=100_000.0, qty={"ETHUSDT": 0.0})
    client = _FakeSigned()
    intent = OrderIntent(
        client_order_id="bt1",
        symbol="ETHUSDT",
        side=Side.BUY,
        qty=0.5,
        notional=1000.0,
        reason="test",
    )
    fills = submit_intents(
        settings,
        [intent],
        {"ETHUSDT": 2000.0},
        pd.Timestamp("2024-01-01", tz="UTC"),
        ledger=ledger,
        confirm="",
        client=client,
    )
    assert len(fills) == 1
    assert ledger.qty["ETHUSDT"] == pytest.approx(0.5)
    assert client.qty["ETHUSDT"] == pytest.approx(0.5)


def test_oms_rejects_live_signed_path(settings):
    settings.account.mode = "live"
    ledger = Ledger(cash=100_000.0, qty={"ETHUSDT": 0.0})
    client = _FakeSigned()
    client.testnet = False
    intent = OrderIntent(
        client_order_id="bt2",
        symbol="ETHUSDT",
        side=Side.BUY,
        qty=0.5,
        notional=1000.0,
    )
    with pytest.raises(RuntimeError, match="testnet-only"):
        submit_intents(
            settings,
            [intent],
            {"ETHUSDT": 2000.0},
            pd.Timestamp("2024-01-01", tz="UTC"),
            ledger=ledger,
            client=client,
        )


def test_reconcile_mismatch_raises(settings):
    ledger = Ledger(cash=100_000.0, qty={"ETHUSDT": 1.0})
    with pytest.raises(RuntimeError, match="reconcile"):
        reconcile_or_kill(settings, ledger, {"ETHUSDT": 0.0})


def test_exchange_qty_map_reads_binance_amt():
    mapped = exchange_qty_map([{"symbol": "ETHUSDT", "positionAmt": "-2.5"}])
    assert mapped["ETHUSDT"] == pytest.approx(-2.5)
