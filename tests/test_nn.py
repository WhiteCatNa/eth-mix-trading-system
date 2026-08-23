from __future__ import annotations

import numpy as np
import pytest

from betatrend.domain import MarketSnapshot
from betatrend.features import compute_features
from betatrend.marketdata.synthetic import make_trending_panels
from betatrend.nn.dataset import FEATURE_NAMES, N_FEAT, SEQ_LEN, build_feature_frame, last_feature_window, make_windows
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


def test_missing_weights_stay_flat(settings):
    settings.strategy.decision = "rl"
    settings.strategy.nn_model_path = "models/_missing_eth_decision.pt"
    panels = make_trending_panels(n=400, seed=4, symbols=["ETHUSDT"])
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
        vol_lookback=settings.strategy.vol_lookback,
    )
    tgt = TimingStrategy(settings).generate(snap, feat)[0]
    assert tgt.extras["decision"] == "flat"
    assert abs(tgt.target_notional) < 1e-9
    assert tgt.extras["unit"] == 0.0


def test_tiny_walk_forward_writes_weights(settings, tmp_path):
    from betatrend.nn.policy import NeuralPolicy
    from betatrend.nn.train import train_decision_net

    panel = make_trending_panels(n=1400, seed=8, symbols=["ETHUSDT"])["ETHUSDT"]
    settings.strategy.nn_model_path = str(tmp_path / "tiny.pt")
    settings.strategy.nn_epochs = 2
    settings.strategy.nn_patience = 3
    settings.strategy.nn_seeds = 1
    settings.strategy.min_history = 80
    settings.strategy.ppo_inner_epochs = 1
    result = train_decision_net(
        panel,
        settings,
        path=tmp_path / "tiny.pt",
        min_train=400,
        test_h=240,
        purge=12,
        prod_holdout=48,
        min_valid=400,
    )
    assert result.path.exists()
    assert result.n_folds >= 2
    assert np.isfinite(result.oos_sharpe)
    policy = NeuralPolicy(settings)
    assert policy.ready
    unit = policy.predict_unit(panel)
    assert -1.0 <= unit <= 1.0


def test_untrained_net_outputs_flat():
    from betatrend.nn.model import DecisionNet
    import torch

    net = DecisionNet(n_feat=30, seq_len=7, dropout=0.0)
    net.eval()
    x = torch.randn(4, 7, 30)
    out = net(x).squeeze(-1)
    torch.testing.assert_close(out, torch.zeros_like(out), atol=1e-5, rtol=0.0)


def test_actor_has_positive_std_head_and_shared_trunk():
    from betatrend.nn.model import PPOActorCritic
    import torch

    net = PPOActorCritic(n_feat=30, seq_len=7, dropout=0.2)
    net.eval()
    x = torch.randn(3, 7, 30)
    shared = net._encode(x)
    assert shared.shape == (3, 64)
    mu_raw, std, value = net._heads(shared)
    assert mu_raw.shape == std.shape == value.shape == (3,)
    assert torch.all(std > 0)
    action, logp, v, ent = net.act(x, deterministic=True)
    torch.testing.assert_close(action, torch.tanh(mu_raw), atol=1e-5, rtol=0.0)
    assert logp.shape == v.shape == ent.shape == (3,)
    assert any(isinstance(m, torch.nn.LayerNorm) for m in net.modules())
    assert any(isinstance(m, torch.nn.ReLU) for m in net.modules())


def test_feature_window_is_seven_by_thirty():
    panel = make_trending_panels(n=80, seed=1, symbols=["ETHUSDT"])["ETHUSDT"]
    feats = build_feature_frame(panel).to_numpy(dtype=np.float32)
    windows = make_windows(feats, seq_len=SEQ_LEN)
    assert feats.shape[1] == N_FEAT == 30
    assert windows.shape == (len(panel), SEQ_LEN, N_FEAT)
    last = last_feature_window(panel)
    assert last.shape == (SEQ_LEN, N_FEAT)
    np.testing.assert_allclose(last, windows[-1], atol=1e-6)


def test_new_regime_features_are_present():
    for name in ("funding_z", "funding_d8", "range_pos", "ret_streak", "trend_persist", "close_z"):
        assert name in FEATURE_NAMES
    assert "tsmom" not in FEATURE_NAMES
    assert len(FEATURE_NAMES) == 30
