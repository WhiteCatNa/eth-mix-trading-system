from __future__ import annotations

import numpy as np
import pytest

from betatrend.config import load_settings
from betatrend.marketdata.synthetic import make_trending_panels
from betatrend.nn.dataset import N_FEAT, SEQ_LEN, last_feature_window
from betatrend.nn.env import BacktestEnv, OverlayEnv, ResetCfg, last_window, overlay_rewards_py, series_flags
from betatrend.nn.reward import bar_pnl, shape_rewards
from betatrend.nn.train import list_fold_jobs, walk_forward_folds


def _panel(n: int = 400, seed: int = 7):
    return make_trending_panels(n=n, seed=seed, symbols=["ETHUSDT"])["ETHUSDT"]


def test_stepwise_reward_matches_batch_shape_rewards():
    panel = _panel()
    cfg = ResetCfg(start=180, end=len(panel) - 2, cost=0.0008, seed=7, train_end=300)
    env = OverlayEnv(panel, cfg)
    env.reset(cfg)
    n_steps = env.end - cfg.start
    actions = np.tanh(np.sin(np.arange(n_steps, dtype=np.float64) / 11.0))
    sl = slice(cfg.start, cfg.start + n_steps)
    pnl, rew = overlay_rewards_py(
        actions,
        env.y[sl],
        env.lev[sl],
        env.vol[sl],
        cfg.cost,
        eta=cfg.eta,
        dd_inc=cfg.dd_inc,
        dd_level=cfg.dd_level,
        clip=cfg.clip,
        so_w=cfg.so_w,
    )
    ref_pnl = bar_pnl(actions, env.y[sl], env.lev[sl], cfg.cost)
    ref_rew = shape_rewards(
        ref_pnl,
        env.vol[sl],
        eta=cfg.eta,
        dd_inc=cfg.dd_inc,
        dd_level=cfg.dd_level,
        clip=cfg.clip,
        so_w=cfg.so_w,
    )
    np.testing.assert_allclose(pnl, ref_pnl, atol=1e-12)
    np.testing.assert_allclose(rew, ref_rew, atol=1e-6)


def test_overlay_env_stepwise_matches_batch():
    panel = _panel()
    cfg = ResetCfg(start=180, end=len(panel) - 2, cost=0.0008, seed=7, train_end=300)
    env = OverlayEnv(panel, cfg)
    env.reset(cfg)
    n_steps = cfg.end - cfg.start
    actions = np.tanh(np.sin(np.arange(n_steps, dtype=np.float64) / 11.0))
    rews = []
    for a in actions:
        step = env.step(float(a))
        rews.append(step.reward)
        if step.done:
            break
    sl = slice(cfg.start, cfg.start + len(rews))
    _, batch = overlay_rewards_py(
        np.asarray(actions[: len(rews)]),
        env.y[sl],
        env.lev[sl],
        env.vol[sl],
        cfg.cost,
        eta=cfg.eta,
        dd_inc=cfg.dd_inc,
        dd_level=cfg.dd_level,
        clip=cfg.clip,
        so_w=cfg.so_w,
    )
    np.testing.assert_allclose(rews, batch, atol=1e-6)


def test_last_window_shape_and_flags():
    panel = _panel(n=80, seed=1)
    win = last_window(panel, seq_len=SEQ_LEN)
    assert win.shape == (SEQ_LEN, N_FEAT)
    py = last_feature_window(panel, seq_len=SEQ_LEN)
    np.testing.assert_allclose(win, py, atol=1e-6)
    assert series_flags(panel) == [True, True, True, True, True]
    dropped = panel.drop(columns=[c for c in ("taker_buy_base", "open_interest") if c in panel.columns])
    assert series_flags(dropped)[0] is False
    assert series_flags(dropped)[3] is False


def test_list_fold_jobs_are_independent():
    settings = load_settings()
    jobs = list_fold_jobs(4000, settings.strategy, min_train=90 * 24, test_h=21 * 24, purge=24)
    assert jobs
    oos = [j for j in jobs if j["role"] == "oos"]
    prod = [j for j in jobs if j["role"] == "prod"]
    assert oos and prod
    assert len({(j["fold_id"], j["seed"]) for j in jobs}) == len(jobs)
    folds = walk_forward_folds(
        4000,
        warmup=max(settings.strategy.min_history, 200),
        min_train=90 * 24,
        test_h=21 * 24,
        purge=24,
    )
    assert {j["fold_id"] for j in oos} == {f[0] for f in folds}


def test_backtest_env_order_gap_funding_fill_mark():
    panel = _panel(n=48, seed=3)
    cfg = ResetCfg(
        start=0,
        end=3,
        exec_mode="backtest",
        fee_rate=0.001,
        slip_bps=10.0,
        funding_interval_hours=8,
        initial_equity=10_000.0,
        target_vol=0.20,
        max_leverage=2.0,
        risk_budget=1.0,
        turnover_band_equity=0.0,
        train_end=3,
    )
    env = BacktestEnv(panel, cfg)
    env.reset(cfg)
    s0 = env.step(1.0)
    assert env.qty == pytest.approx(0.0)
    assert env.pending is not None
    pending0 = float(env.pending)
    s1 = env.step(1.0)
    assert abs(env.qty) > 0.0
    assert s1.pnl != 0.0 or pending0 != 0.0
    assert s0.info.get("exec") == "backtest"


def test_funding_column_fallback():
    panel = _panel(n=40, seed=4)
    alt = panel.rename(columns={"funding_rate": "funding"}) if "funding_rate" in panel.columns else panel
    env = BacktestEnv(alt, ResetCfg(start=0, end=5, exec_mode="backtest"))
    env.reset()
    assert env.funding.shape[0] == len(alt)
