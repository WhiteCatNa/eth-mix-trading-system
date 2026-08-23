from __future__ import annotations

import numpy as np
import pytest

from betatrend.marketdata.synthetic import make_trending_panels
from betatrend.mathx import score_to_unit
from betatrend.nn.dataset import FEATURE_NAMES, build_feature_frame
from betatrend.strategy import TimingStrategy

torch = pytest.importorskip("torch")


def test_features_do_not_look_ahead():
    panel = make_trending_panels(n=500, seed=3, symbols=["ETHUSDT"])["ETHUSDT"]
    base = build_feature_frame(panel)
    assert list(base.columns) == list(FEATURE_NAMES)
    mutated = panel.copy()
    mutated.iloc[250, mutated.columns.get_loc("close")] *= 1.08
    after = build_feature_frame(mutated)
    np.testing.assert_allclose(base.iloc[:250].to_numpy(), after.iloc[:250].to_numpy(), atol=1e-12)
    assert not np.allclose(base.iloc[250:].to_numpy(), after.iloc[250:].to_numpy())


def test_neural_falls_back_to_tsmom_without_weights(settings):
    settings.strategy.decision = "neural"
    settings.strategy.nn_model_path = "models/_missing_eth_decision.pt"
    snap_panels = make_trending_panels(n=400, seed=4, symbols=["ETHUSDT"])
    from betatrend.domain import MarketSnapshot
    from betatrend.features import compute_features

    panels = snap_panels
    prices = {s: float(df["close"].iloc[-1]) for s, df in panels.items()}
    snap = MarketSnapshot(
        timestamp=panels["ETHUSDT"].index[-1],
        panels=panels,
        prices=prices,
        equity=settings.account.initial_capital,
        bar_index=len(panels["ETHUSDT"]) - 1,
        market_symbol=settings.universe.market_symbol,
    )
    feat = compute_features(
        snap.panels,
        snap.market_symbol,
        beta_lookback=settings.strategy.beta_lookback,
        vol_lookback=settings.strategy.vol_lookback,
        lookbacks=settings.strategy.lookbacks_hours,
        weights=settings.strategy.lookback_weights,
        skip_hours=0,
    )
    feat.own_scores["ETHUSDT"] = 0.9
    neural = TimingStrategy(settings).generate(snap, feat)[0]
    settings.strategy.decision = "tsmom"
    tsmom = TimingStrategy(settings).generate(snap, feat)[0]
    assert neural.extras["decision"] == "tsmom"
    assert abs(neural.target_notional - tsmom.target_notional) < 1e-6


def test_tiny_walk_forward_writes_weights(settings, tmp_path):
    from betatrend.nn.policy import NeuralPolicy
    from betatrend.nn.train import train_decision_net

    panel = make_trending_panels(n=1400, seed=8, symbols=["ETHUSDT"])["ETHUSDT"]
    settings.strategy.nn_model_path = str(tmp_path / "tiny.pt")
    settings.strategy.nn_epochs = 8
    settings.strategy.nn_patience = 3
    settings.strategy.nn_seeds = 1
    settings.strategy.min_history = 80
    result = train_decision_net(
        panel,
        settings,
        path=tmp_path / "tiny.pt",
        min_train=400,
        test_h=96,
        purge=12,
        prod_holdout=48,
        min_valid=400,
    )
    assert result.path.exists()
    assert result.n_folds >= 2
    assert np.isfinite(result.oos_sharpe)
    policy = NeuralPolicy(settings)
    assert policy.ready
    unit = policy.predict_unit(panel, tsmom_score=0.4)
    assert -1.0 <= unit <= 1.0
    prior = score_to_unit(0.4, scale=1.0, min_position=settings.strategy.min_position)
    assert abs(unit) <= 1.0
    _ = prior


def test_untrained_residual_equals_tsmom():
    from betatrend.nn.model import DecisionNet
    import torch

    net = DecisionNet(n_in=8, hidden=(4, 4), dropout=0.0, delta_gain=0.5)
    net.eval()
    x = torch.randn(5, 8)
    prior = torch.tensor([-0.8, -0.2, 0.0, 0.3, 0.9])
    out = net(x, prior).squeeze(-1)
    torch.testing.assert_close(out, prior, atol=1e-5, rtol=0.0)


def test_new_regime_features_are_present():
    from betatrend.nn.dataset import FEATURE_NAMES

    for name in ("funding_z", "funding_d8", "range_pos", "ret_streak", "trend_persist"):
        assert name in FEATURE_NAMES
    assert FEATURE_NAMES[-1] == "tsmom"


def test_gate_starts_closed():
    from betatrend.nn.model import DecisionNet
    import torch

    net = DecisionNet(n_in=6, hidden=(4, 4), dropout=0.0, delta_gain=0.5)
    x = torch.randn(8, 6)
    prior = torch.linspace(-0.9, 0.9, 8)
    out = net(x, prior).squeeze(-1)
    torch.testing.assert_close(out, prior, atol=1e-5, rtol=0.0)
    assert float(net._last_gate.detach().mean()) < 0.20
