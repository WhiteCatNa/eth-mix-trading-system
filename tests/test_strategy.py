from __future__ import annotations

import numpy as np
import pytest

from betatrend.control import ControlPlane
from betatrend.domain import MarketSnapshot
from betatrend.features import compute_features
from betatrend.marketdata.synthetic import make_trending_panels
from betatrend.mathx import score_to_unit
from betatrend.strategy import TimingStrategy


def _snap(settings, n=800):
    sym = settings.universe.trade_symbol
    panels = make_trending_panels(n=n, seed=11, symbols=[sym])
    prices = {s: float(df["close"].iloc[-1]) for s, df in panels.items()}
    return MarketSnapshot(
        timestamp=panels[sym].index[-1],
        panels=panels,
        prices=prices,
        equity=settings.account.initial_capital,
        bar_index=n - 1,
        market_symbol=settings.universe.market_symbol,
    )


def _feat(settings, snap):
    return compute_features(
        snap.panels,
        snap.market_symbol,
        beta_lookback=settings.strategy.beta_lookback,
        vol_lookback=settings.strategy.vol_lookback,
        lookbacks=settings.strategy.lookbacks_hours,
        weights=settings.strategy.lookback_weights,
        skip_hours=0,
    )


def test_features_own_score_is_finite(settings):
    snap = _snap(settings)
    feat = _feat(settings, snap)
    assert np.isfinite(feat.own_scores["ETHUSDT"])
    assert feat.betas["ETHUSDT"] == 1.0


def test_score_to_unit_is_continuous_and_odd():
    a = score_to_unit(0.4, scale=1.0)
    b = score_to_unit(1.2, scale=1.0)
    assert 0 < a < b < 1
    assert abs(score_to_unit(-0.4) + a) < 1e-12
    assert score_to_unit(0.01, min_position=0.05) == 0.0


def test_stronger_score_larger_position(settings):
    strat = TimingStrategy(settings)
    snap = _snap(settings)
    feat = _feat(settings, snap)
    feat.own_scores["ETHUSDT"] = 0.35
    small = strat.generate(snap, feat)[0].target_notional
    feat.own_scores["ETHUSDT"] = 1.40
    large = strat.generate(snap, feat)[0].target_notional
    assert large > small > 0
    feat.own_scores["ETHUSDT"] = -1.40
    short = strat.generate(snap, feat)[0].target_notional
    assert short < 0
    assert abs(abs(short) - large) < 1.0  # symmetric overlay


def test_long_only_blocks_shorts(settings):
    settings.strategy.long_only = True
    strat = TimingStrategy(settings)
    snap = _snap(settings)
    feat = _feat(settings, snap)
    feat.own_scores["ETHUSDT"] = -2.0
    feat.market_score = -2.0
    targets = strat.generate(snap, feat)
    assert all(t.target_notional >= -1e-9 for t in targets)


def test_live_orders_blocked(settings, monkeypatch):
    monkeypatch.delenv("BETATREND_ALLOW_LIVE", raising=False)
    settings.account.mode = "live"
    with pytest.raises(RuntimeError, match="Live blocked"):
        ControlPlane(settings).assert_can_send_orders("YES")


def test_live_orders_blocked_without_yes(settings, monkeypatch):
    monkeypatch.setenv("BETATREND_ALLOW_LIVE", "1")
    monkeypatch.setenv("BINANCE_TESTNET", "0")
    settings.account.mode = "live"
    with pytest.raises(RuntimeError, match="YES"):
        ControlPlane(settings).assert_can_send_orders("")


def test_unit_below_five_percent_is_flat(settings):
    from betatrend.mathx import score_to_unit
    from betatrend.signals import make_signal

    settings.strategy.min_position = 0.05
    settings.strategy.long_only = False
    unit = score_to_unit(0.02, scale=1.0, min_position=0.05, long_only=False)
    assert unit == 0.0
    sig = make_signal(0.04, target_notional=8_000, min_position=0.05)
    assert sig.side.value == "FLAT"
    assert sig.target_notional == 0.0


def test_negative_unit_is_short(settings):
    settings.strategy.long_only = False
    settings.strategy.min_position = 0.05
    strat = TimingStrategy(settings)
    snap = _snap(settings)
    feat = _feat(settings, snap)
    feat.own_scores["ETHUSDT"] = -1.40
    feat.market_score = -1.40
    tgt = strat.generate(snap, feat)[0]
    assert tgt.target_notional < 0
    assert tgt.extras["side"] == "SHORT"
    assert tgt.extras["unit"] < 0
