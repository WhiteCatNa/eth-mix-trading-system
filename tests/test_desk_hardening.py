from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from betatrend.backtest import Backtester
from betatrend.control import ControlPlane
from betatrend.domain import MarketSnapshot, OrderIntent, Side
from betatrend.execution.bridge import exchange_qty_map, reconcile_or_kill, submit_intents
from betatrend.gate import evaluate_deploy_gate
from betatrend.ledger import Ledger
from betatrend.marketdata.synthetic import make_trending_panels
from betatrend.pipeline import DeskCycle
from betatrend.qc import inspect_panels
from betatrend.research import paper_run


def _passing_report() -> dict:
    return {
        "oos_neural_blend": {
            "sharpe": 0.42,
            "total_return": 0.08,
            "max_drawdown": -0.05,
            "turnover": 0.1,
            "mean_pos": 0.2,
        },
        "chosen_blend": 0.5,
        "fold_blends": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        "n_folds": 8,
        "n_oos_bars": 400,
    }


def test_eth_report_fails_deploy_gate(settings):
    result = evaluate_deploy_gate(settings)
    assert result.passed is False
    assert result.oos_sharpe is not None and result.oos_sharpe <= 0
    blob = " ".join(result.reasons)
    assert "Sharpe" in blob


def test_passing_fold_blend_report_clears_gate(settings, tmp_path):
    path = tmp_path / "ok.json"
    path.write_text(json.dumps(_passing_report()), encoding="utf-8")
    settings.deploy.report_path = str(path)
    result = evaluate_deploy_gate(settings)
    assert result.passed is True


def test_live_blocked_by_deploy_gate_even_with_env(settings, monkeypatch):
    monkeypatch.setenv("BETATREND_ALLOW_LIVE", "1")
    monkeypatch.setenv("BINANCE_TESTNET", "0")
    settings.account.mode = "live"
    with pytest.raises(RuntimeError, match="deploy gate"):
        ControlPlane(settings).assert_can_send_orders(confirm="YES")


def test_live_allowed_when_gate_passes(settings, tmp_path, monkeypatch):
    path = tmp_path / "ok.json"
    path.write_text(json.dumps(_passing_report()), encoding="utf-8")
    settings.deploy.report_path = str(path)
    monkeypatch.setenv("BETATREND_ALLOW_LIVE", "1")
    monkeypatch.setenv("BINANCE_TESTNET", "0")
    settings.account.mode = "live"
    ControlPlane(settings).assert_can_send_orders(confirm="YES")


def test_qc_flags_gap_ohlc_jump_and_stale_funding(settings):
    panels = make_trending_panels(n=80, seed=3, symbols=["ETHUSDT"])
    assert inspect_panels(panels, settings).ok
    broken = {k: v.copy() for k, v in panels.items()}
    df = broken["ETHUSDT"]
    df.iloc[-2, df.columns.get_loc("high")] = float(df["close"].iloc[-2]) * 0.5
    assert inspect_panels(broken, settings).ok is False

    gapped = {k: v.copy() for k, v in panels.items()}
    gapped["ETHUSDT"] = gapped["ETHUSDT"].drop(gapped["ETHUSDT"].index[-5])
    assert inspect_panels(gapped, settings).ok is False

    jumped = {k: v.copy() for k, v in panels.items()}
    jumped["ETHUSDT"].iloc[-1, jumped["ETHUSDT"].columns.get_loc("close")] *= 2.0
    jumped["ETHUSDT"].iloc[-1, jumped["ETHUSDT"].columns.get_loc("high")] = max(
        float(jumped["ETHUSDT"]["high"].iloc[-1]),
        float(jumped["ETHUSDT"]["close"].iloc[-1]),
    )
    assert inspect_panels(jumped, settings).ok is False

    stale = {k: v.copy() for k, v in panels.items()}
    if "funding_rate" in stale["ETHUSDT"].columns:
        stale["ETHUSDT"].loc[stale["ETHUSDT"].index[-20:], "funding_rate"] = np.nan
        assert inspect_panels(stale, settings).ok is False


def test_desk_cycle_flattens_on_qc_fail(settings):
    panels = make_trending_panels(n=400, seed=4, symbols=["ETHUSDT"])
    df = panels["ETHUSDT"].copy()
    df.iloc[-1, df.columns.get_loc("low")] = float(df["close"].iloc[-1]) * 2.0
    panels["ETHUSDT"] = df
    prices = {s: float(p["close"].iloc[-1]) for s, p in panels.items()}
    snap = MarketSnapshot(
        timestamp=panels["ETHUSDT"].index[-1],
        panels=panels,
        prices=prices,
        equity=100_000,
        bar_index=len(df) - 1,
        market_symbol="ETHUSDT",
    )
    cycle = DeskCycle(settings).run(snap, {"ETHUSDT": 12_000.0})
    assert cycle.flatten
    assert cycle.clipped["ETHUSDT"] == 0.0
    assert any("qc:" in m for m in cycle.messages)


def test_paper_run_persists_state(settings, tmp_path):
    settings.strategy.decision = "tsmom"
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


def test_paper_run_matches_backtest_sign(settings, tmp_path):
    settings.strategy.decision = "tsmom"
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
    clip = float(paper_state["clipped"].get("ETHUSDT", 0.0))
    if abs(bt_n) < 1.0 and abs(clip) < 1.0:
        pytest.skip("both overlays flat at overlap")
    if abs(bt_n) >= 1.0 and abs(clip) >= 1.0:
        assert np.sign(clip) == np.sign(bt_n)


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


def test_reconcile_mismatch_trips_kill(settings, tmp_path):
    settings.control.kill_file = str(tmp_path / "KILL")
    ledger = Ledger(cash=100_000.0, qty={"ETHUSDT": 1.0})
    with pytest.raises(RuntimeError, match="reconcile"):
        reconcile_or_kill(settings, ledger, {"ETHUSDT": 0.0})
    assert Path(settings.control.kill_file).exists()


def test_exchange_qty_map_reads_binance_amt():
    mapped = exchange_qty_map([{"symbol": "ETHUSDT", "positionAmt": "-2.5"}])
    assert mapped["ETHUSDT"] == pytest.approx(-2.5)
