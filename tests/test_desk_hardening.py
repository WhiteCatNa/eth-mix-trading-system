from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from betatrend.backtest import Backtester
from betatrend.domain import OrderIntent, Side
from betatrend.execution.bridge import exchange_qty_map, reconcile_or_kill, submit_intents
from betatrend.execution.signed import BinanceSignedClient
from betatrend.ledger import Ledger
from betatrend.marketdata.synthetic import make_trending_panels
from betatrend.research import paper_once, paper_run


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


def test_paper_run_persists_rebalance_clock(settings, tmp_path, stub_rl):
    settings.strategy.decision = "rl"
    settings.backtest.warmup_bars = 160
    settings.strategy.rebalance_hours = 8
    settings.paper.dry_run = False
    settings.paper.state_file = str(tmp_path / "paper.json")
    panels = make_trending_panels(n=220, seed=5, symbols=["ETHUSDT"])
    paper_run(settings, panels=panels, execute=True, max_bars=2, reset_state=True)
    payload = json.loads((tmp_path / "paper.json").read_text())
    reb1 = payload.get("last_reb_ts")
    assert reb1
    paper_run(settings, panels=panels, execute=True, max_bars=2, reset_state=False)
    payload2 = json.loads((tmp_path / "paper.json").read_text())
    assert pd.Timestamp(payload2["last_reb_ts"]) == pd.Timestamp(reb1)


def test_paper_once_fills_at_next_open(settings, stub_rl):
    settings.strategy.min_history = 150
    settings.paper.dry_run = False
    settings.oms.min_notional = 1.0
    settings.backtest.turnover_band_equity = 0.0
    panels = make_trending_panels(n=220, seed=7, symbols=["ETHUSDT"])
    df = panels["ETHUSDT"]
    last_close = float(df["close"].iloc[-1])
    last_open = last_close * 0.93
    df.iloc[-1, df.columns.get_loc("open")] = last_open
    df.iloc[-1, df.columns.get_loc("high")] = max(last_open, last_close)
    df.iloc[-1, df.columns.get_loc("low")] = min(last_open, last_close)
    out = paper_once(settings, panels=panels, execute=True)
    assert out["fill_at"] == "next_open"
    assert out["fill_prices"]["ETHUSDT"] == pytest.approx(last_open)
    assert abs(last_open - last_close) > 1.0
    assert pd.Timestamp(out["timestamp"]) == df.index[-2]
    assert pd.Timestamp(out["fill_timestamp"]) == df.index[-1]
    assert out["executed_fills"] >= 1
    assert out["executed_fill_prices"]["ETHUSDT"] == pytest.approx(last_open)


def test_paper_once_execute_overrides_dry_run(settings, stub_rl):
    settings.strategy.min_history = 150
    settings.paper.dry_run = True
    settings.oms.min_notional = 1.0
    settings.backtest.turnover_band_equity = 0.0
    panels = make_trending_panels(n=220, seed=7, symbols=["ETHUSDT"])
    out = paper_once(settings, panels=panels, execute=True)
    assert out["dry_run"] is False
    assert out["executed_fills"] >= 1


def test_paper_run_applies_funding(settings, tmp_path, stub_rl):
    settings.strategy.decision = "rl"
    settings.strategy.min_history = 150
    settings.backtest.warmup_bars = 160
    settings.paper.dry_run = True
    settings.paper.state_file = str(tmp_path / "paper.json")
    settings.data.funding_interval_hours = 1
    settings.oms.min_notional = 1.0
    settings.backtest.turnover_band_equity = 0.001
    base = make_trending_panels(n=200, seed=3, symbols=["ETHUSDT"])["ETHUSDT"]
    zero = base.copy()
    zero["funding_rate"] = 0.0
    fat = base.copy()
    fat["funding_rate"] = 0.01
    out0 = paper_run(settings, panels={"ETHUSDT": zero}, execute=True, max_bars=16, reset_state=True)
    out1 = paper_run(settings, panels={"ETHUSDT": fat}, execute=True, max_bars=16, reset_state=True)
    assert abs(out1["qty"]["ETHUSDT"]) >= 1.0
    assert out1["n_fills"] >= 1
    assert out1["funding_pnl"] != pytest.approx(0.0)
    assert out0["cash"] != pytest.approx(out1["cash"])


class _SmoothPolicy:
    """Uses restored EMA so resume tests can see a non-zero last_unit."""

    ready = True

    def __init__(self, settings=None, *, raw: float = 1.0, smooth: float = 0.5):
        self.settings = settings
        self.raw = float(raw)
        self.smooth = float(smooth)
        self._last_unit = 0.0

    def reset(self) -> None:
        self._last_unit = 0.0

    def last_unit(self) -> float:
        return float(self._last_unit)

    def restore_last_unit(self, unit: float) -> None:
        self._last_unit = float(unit)

    def predict_unit(self, panel, *args, **kwargs) -> float:
        unit = (1.0 - self.smooth) * self.raw + self.smooth * self._last_unit
        self._last_unit = unit
        return unit


def test_paper_run_restores_ema_on_resume(settings, tmp_path, monkeypatch):
    settings.strategy.min_history = 150
    settings.backtest.warmup_bars = 160
    settings.strategy.rebalance_hours = 8
    settings.paper.state_file = str(tmp_path / "paper.json")
    settings.oms.min_notional = 1.0
    monkeypatch.setattr("betatrend.nn.policy.NeuralPolicy", lambda settings: _SmoothPolicy(settings))
    panels = make_trending_panels(n=220, seed=4, symbols=["ETHUSDT"])
    paper_run(settings, panels=panels, execute=True, max_bars=1, reset_state=True)
    payload = json.loads((tmp_path / "paper.json").read_text())
    saved = float(payload["nn_last_unit"])
    assert saved == pytest.approx(0.5)
    paper_run(settings, panels=panels, execute=True, max_bars=8, reset_state=False)
    payload2 = json.loads((tmp_path / "paper.json").read_text())
    assert float(payload2["nn_last_unit"]) == pytest.approx(0.75)


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


def _intent() -> OrderIntent:
    return OrderIntent(
        client_order_id="bt3",
        symbol="ETHUSDT",
        side=Side.BUY,
        qty=0.5,
        notional=1000.0,
        reason="test",
    )


def test_testnet_mode_mainnet_env_requires_live_gates(settings, monkeypatch):
    monkeypatch.setenv("BINANCE_TESTNET", "0")
    monkeypatch.delenv("BETATREND_ALLOW_LIVE", raising=False)
    settings.account.mode = "testnet"
    settings.oms.testnet_only = False
    constructed = {}

    class FakeMainnet(_FakeSigned):
        def __init__(self, _settings, testnet=None):
            super().__init__()
            if testnet is None:
                testnet = os.environ.get("BINANCE_TESTNET", "1") != "0"
            self.testnet = bool(testnet)
            constructed["testnet"] = self.testnet

        def new_order(self, *args, **kwargs):
            raise AssertionError("mainnet order must not be sent without live gates")

    monkeypatch.setattr("betatrend.execution.bridge.BinanceSignedClient", FakeMainnet)
    ledger = Ledger(cash=100_000.0, qty={"ETHUSDT": 0.0})
    with pytest.raises(RuntimeError, match="BETATREND_ALLOW_LIVE|confirm=YES"):
        submit_intents(
            settings,
            [_intent()],
            {"ETHUSDT": 2000.0},
            pd.Timestamp("2024-01-01", tz="UTC"),
            ledger=ledger,
            confirm="",
            client=None,
        )
    assert constructed.get("testnet") is False


def test_testnet_mode_mainnet_env_orders_when_gated(settings, monkeypatch):
    monkeypatch.setenv("BINANCE_TESTNET", "0")
    monkeypatch.setenv("BETATREND_ALLOW_LIVE", "1")
    settings.account.mode = "testnet"
    settings.oms.testnet_only = False

    class FakeMainnet(_FakeSigned):
        def __init__(self, _settings, testnet=None):
            super().__init__()
            self.testnet = False

    monkeypatch.setattr("betatrend.execution.bridge.BinanceSignedClient", FakeMainnet)
    ledger = Ledger(cash=100_000.0, qty={"ETHUSDT": 0.0})
    fills = submit_intents(
        settings,
        [_intent()],
        {"ETHUSDT": 2000.0},
        pd.Timestamp("2024-01-01", tz="UTC"),
        ledger=ledger,
        confirm="YES",
        client=None,
    )
    assert len(fills) == 1
    assert ledger.qty["ETHUSDT"] == pytest.approx(0.5)


def test_new_order_mainnet_requires_confirm_even_with_allow_live(settings, monkeypatch):
    monkeypatch.setenv("BETATREND_ALLOW_LIVE", "1")
    settings.account.mode = "testnet"
    settings.oms.testnet_only = False
    client = BinanceSignedClient(settings, testnet=False)

    def _boom(*_args, **_kwargs):
        raise AssertionError("no http")

    client._signed = _boom
    try:
        with pytest.raises(RuntimeError, match="confirm=YES"):
            client.new_order("ETHUSDT", "BUY", 0.1, confirm="")
    finally:
        client.close()


def test_new_order_mainnet_requires_live_gates_in_testnet_mode(settings, monkeypatch):
    monkeypatch.delenv("BETATREND_ALLOW_LIVE", raising=False)
    settings.account.mode = "testnet"
    settings.oms.testnet_only = False
    client = BinanceSignedClient(settings, testnet=False)

    def _boom(*_args, **_kwargs):
        raise AssertionError("no http")

    client._signed = _boom
    try:
        with pytest.raises(RuntimeError, match="BETATREND_ALLOW_LIVE"):
            client.new_order("ETHUSDT", "BUY", 0.1, confirm="YES")
    finally:
        client.close()
