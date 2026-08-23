from __future__ import annotations

from betatrend.backtest import Backtester
from betatrend.domain import MarketSnapshot
from betatrend.marketdata.synthetic import make_trending_panels
from betatrend.pipeline import DeskCycle


def test_desk_cycle_emits_eth_target(settings):
    panels = make_trending_panels(n=700, seed=5, symbols=["ETHUSDT"])
    prices = {s: float(df["close"].iloc[-1]) for s, df in panels.items()}
    snap = MarketSnapshot(
        timestamp=panels["ETHUSDT"].index[-1],
        panels=panels,
        prices=prices,
        equity=100_000,
        bar_index=len(panels["ETHUSDT"]) - 1,
        market_symbol="ETHUSDT",
    )
    cycle = DeskCycle(settings).run(snap, {s: 0.0 for s in panels})
    assert "ETHUSDT" in cycle.clipped
    assert all(abs(cycle.clipped.get(s, 0.0)) < 1e-9 for s in cycle.clipped if s != "ETHUSDT")


def test_no_lookahead_fill_at_next_open(settings):
    """Signal uses close[t]; the queued fill is executed at open[t+1]."""
    panels = make_trending_panels(n=500, seed=9, symbols=["ETHUSDT"])
    for df in panels.values():
        df["open"] = df["close"].shift(1).fillna(df["close"].iloc[0]) * 0.99
        df["high"] = df[["open", "high", "close"]].max(axis=1)
        df["low"] = df[["open", "low", "close"]].min(axis=1)
    result = Backtester(settings).run(panels)
    assert result.fills, "synthetic trend path should produce fills"
    for f in result.fills[:20]:
        expected_open = float(panels[f.symbol].loc[f.timestamp, "open"])
        assert abs(f.price - expected_open) < 1e-8
        close = float(panels[f.symbol].loc[f.timestamp, "close"])
        assert abs(close - expected_open) > 1e-9
        assert f.symbol == "ETHUSDT"


def test_synthetic_trend_is_tradeable(settings):
    """Designed two-regime ETH path: continuous timing should capture the trend."""
    panels = make_trending_panels(n=1800, seed=2, symbols=["ETHUSDT"])
    result = Backtester(settings).run(panels)
    assert result.metrics["n_fills"] > 0
    assert result.metrics["final_equity"] > 0
    assert result.metrics["total_return"] > 0
    assert result.metrics["sharpe"] > 0
    assert all(f.symbol == "ETHUSDT" for f in result.fills)


def test_kill_file_flattens_cycle(settings, tmp_path, monkeypatch):
    monkeypatch.setenv("BETATREND_KILL", "1")
    panels = make_trending_panels(n=400, seed=1, symbols=["ETHUSDT"])
    prices = {s: float(df["close"].iloc[-1]) for s, df in panels.items()}
    snap = MarketSnapshot(
        timestamp=panels["ETHUSDT"].index[-1],
        panels=panels,
        prices=prices,
        equity=100_000,
        bar_index=399,
        market_symbol="ETHUSDT",
    )
    cycle = DeskCycle(settings).run(snap, {"ETHUSDT": 10_000})
    assert cycle.flatten
    assert all(v == 0.0 for v in cycle.clipped.values())
