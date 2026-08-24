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
    settings.strategy.ppo_replay_rollouts = 2
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
    assert np.isfinite(result.oos_max_dd)
    assert result.oos_max_dd <= 1e-12
    assert "eval_primary" in result.metrics
    assert "sharpe" in result.metrics["eval_primary"]
    assert "max_drawdown" in result.metrics["eval_primary"]
    policy = NeuralPolicy(settings)
    assert policy.ready
    unit = policy.predict_unit(panel)
    assert -1.0 <= unit <= 1.0


def test_untrained_net_outputs_flat():
    from betatrend.nn.model import DecisionNet
    import torch

    net = DecisionNet(n_feat=N_FEAT, seq_len=7, dropout=0.0)
    net.eval()
    x = torch.randn(4, 7, N_FEAT)
    out = net(x).squeeze(-1)
    torch.testing.assert_close(out, torch.zeros_like(out), atol=1e-5, rtol=0.0)


def test_actor_has_positive_std_head_and_shared_trunk():
    from betatrend.nn.model import PPOActorCritic
    import torch

    net = PPOActorCritic(n_feat=N_FEAT, seq_len=7, dropout=0.2)
    net.eval()
    x = torch.randn(3, 7, N_FEAT)
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


def test_feature_window_is_seven_by_n_feat():
    panel = make_trending_panels(n=80, seed=1, symbols=["ETHUSDT"])["ETHUSDT"]
    feats = build_feature_frame(panel).to_numpy(dtype=np.float32)
    windows = make_windows(feats, seq_len=SEQ_LEN)
    assert feats.shape[1] == N_FEAT
    assert N_FEAT > 30
    assert windows.shape == (len(panel), SEQ_LEN, N_FEAT)
    last = last_feature_window(panel)
    assert last.shape == (SEQ_LEN, N_FEAT)
    np.testing.assert_allclose(last, windows[-1], atol=1e-6)


def test_new_regime_features_are_present():
    for name in ("funding_z", "funding_d8", "range_pos", "ret_streak", "trend_persist", "close_z"):
        assert name in FEATURE_NAMES
    assert "tsmom" not in FEATURE_NAMES
    for name in ("taker_imb", "body", "wick_imb", "gap", "atr_n", "trades_z", "basis", "oi_chg", "lsr_dev"):
        assert name in FEATURE_NAMES
    assert len(FEATURE_NAMES) == N_FEAT == 42


def test_missing_futures_columns_fill_neutral():
    panel = make_trending_panels(n=80, seed=1, symbols=["ETHUSDT"])["ETHUSDT"]
    dropped = panel.drop(columns=[c for c in ("taker_buy_base", "trades", "mark_close", "index_close", "open_interest", "long_short_ratio") if c in panel.columns])
    feats = build_feature_frame(dropped)
    assert (feats["taker_imb"] == 0.0).all()
    assert (feats["trades_z"] == 0.0).all()
    assert (feats["basis"] == 0.0).all()
    assert (feats["oi_chg"] == 0.0).all()
    assert (feats["lsr_dev"] == 0.0).all()


def test_extra_features_do_not_look_ahead():
    panel = make_trending_panels(n=200, seed=2, symbols=["ETHUSDT"])["ETHUSDT"]
    base = build_feature_frame(panel)
    mutated = panel.copy()
    mutated.iloc[120, mutated.columns.get_loc("taker_buy_base")] *= 3.0
    mutated.iloc[120, mutated.columns.get_loc("open_interest")] *= 1.5
    after = build_feature_frame(mutated)
    np.testing.assert_allclose(base.iloc[:120].to_numpy(), after.iloc[:120].to_numpy(), atol=1e-12)
    assert not np.allclose(base.iloc[120:].to_numpy(), after.iloc[120:].to_numpy())


def test_reward_shaping_penalizes_drawdown_more_than_flat():
    from betatrend.nn.reward import bar_pnl, shape_rewards

    n = 64
    y = np.zeros(n)
    y[20:28] = -0.02
    lev = np.ones(n)
    vol = np.full(n, 0.20)
    crash = bar_pnl(np.ones(n), y, lev, cost=0.0)
    flat = bar_pnl(np.zeros(n), y, lev, cost=0.0)
    r_crash = shape_rewards(crash, vol)
    r_flat = shape_rewards(flat, vol)
    assert r_crash.mean() < r_flat.mean()
    assert np.all(np.isfinite(r_crash))
    assert r_crash.min() >= -5.0 - 1e-6


def test_replay_buffer_keeps_recent_rollouts():
    from betatrend.nn.buffer import ReplayBuffer

    buf = ReplayBuffer(n_rollouts=2)
    for i in range(3):
        buf.add(
            obs=np.ones((4, 7, N_FEAT), dtype=np.float32) * i,
            actions=np.full(4, float(i), dtype=np.float32),
            logp=np.zeros(4, dtype=np.float32),
            advantages=np.ones(4, dtype=np.float32),
            returns=np.ones(4, dtype=np.float32),
        )
    assert buf.n_stored == 2
    packed = buf.packed()
    assert packed["obs"].shape == (8, 7, N_FEAT)
    assert packed["actions"].shape == (8,)
    np.testing.assert_allclose(packed["actions"][:4], 1.0)
    np.testing.assert_allclose(packed["actions"][4:], 2.0)
