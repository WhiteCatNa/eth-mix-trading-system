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


def test_nan_mean_head_does_not_crash_normal():
    from betatrend.nn.model import GRPOActor
    import torch

    net = GRPOActor(n_feat=N_FEAT, seq_len=7, dropout=0.0)
    net.eval()
    with torch.no_grad():
        net.mean_head.weight.fill_(float("nan"))
        dist = net._dist(torch.zeros(4, 7, N_FEAT))
    assert torch.isfinite(dist.mean).all()
    assert torch.isfinite(dist.stddev).all()


def test_actor_has_positive_std_head_and_shared_trunk():
    from betatrend.nn.model import GRPOActor
    import torch

    net = GRPOActor(n_feat=N_FEAT, seq_len=7, dropout=0.2)
    net.eval()
    x = torch.randn(3, 7, N_FEAT)
    shared = net._encode(x)
    assert shared.shape == (3, 64)
    mu_raw, std = net._heads(shared)
    assert mu_raw.shape == std.shape == (3,)
    assert torch.all(std > 0)
    action, logp, ent = net.act(x, deterministic=True)
    torch.testing.assert_close(action, torch.tanh(mu_raw), atol=1e-5, rtol=0.0)
    assert logp.shape == ent.shape == (3,)
    grp_a, grp_lp = net.sample_group(x[:1], 4)
    assert grp_a.shape == grp_lp.shape == (4, 1)
    assert not any(n.startswith("value") or n.startswith("critic") for n, _ in net.named_parameters())
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


def test_shuffle_rebalance_index_permutes_hold_grid():
    from betatrend.nn.dataset import shuffle_rebalance_index

    hold = 8
    n = 100
    base = np.arange(0, n, hold)
    a = shuffle_rebalance_index(n, hold, np.random.default_rng(0))
    b = shuffle_rebalance_index(n, hold, np.random.default_rng(1))
    assert sorted(a.tolist()) == base.tolist()
    assert sorted(b.tolist()) == base.tolist()
    assert not np.array_equal(a, b)
    np.testing.assert_array_equal(shuffle_rebalance_index(5, hold, np.random.default_rng(0)), [0])


def test_latest_complete_fold_stops_at_first_gap(tmp_path):
    from betatrend.nn.train import fold_ckpt_path, fold_seed, latest_complete_fold

    prefix = tmp_path / "eth_decision.pt"
    assert latest_complete_fold(prefix, n_seeds=3, epochs=80) == -1
    for fold in range(2):
        for s in range(3):
            p = fold_ckpt_path(prefix, fold, fold_seed(fold, s), 80)
            p.write_bytes(b"x")
    assert latest_complete_fold(prefix, n_seeds=3, epochs=80) == 1
    fold_ckpt_path(prefix, 2, fold_seed(2, 0), 80).write_bytes(b"x")
    assert latest_complete_fold(prefix, n_seeds=3, epochs=80) == 1


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
    y[20:28] = -0.002
    lev = np.ones(n)
    vol = np.full(n, 0.20)
    crash = bar_pnl(np.ones(n), y, lev, cost=0.0)
    flat = bar_pnl(np.zeros(n), y, lev, cost=0.0)
    assert shape_rewards(crash, vol).mean() < shape_rewards(flat, vol).mean()
    # λ 越大，同一段下跌扣得越狠
    hard = shape_rewards(crash, vol, down_lambda=1.0).mean()
    soft = shape_rewards(crash, vol, down_lambda=0.0).mean()
    assert hard < soft


def test_reward_clip_bounds_extreme_bars():
    from betatrend.nn.reward import shape_rewards

    vol = np.full(16, 0.20)
    pnl = np.zeros(16)
    pnl[5] = -0.5
    r = shape_rewards(pnl, vol, clip=5.0)
    assert np.all(np.isfinite(r))
    assert r.min() >= -5.0 - 1e-6 and r.max() <= 5.0 + 1e-6


def test_downside_term_penalizes_left_tail_more_than_right():
    from betatrend.nn.reward import shape_rewards

    vol = np.full(48, 0.20)
    left = np.zeros(48)
    left[10:14] = -0.003
    right = -left
    gap = shape_rewards(right, vol).mean() - shape_rewards(left, vol).mean()
    # λ=0 时奖励对 PnL 是线性的，左右尾只差符号；不对称性全部来自 min(r,0)²
    sym_l = shape_rewards(left, vol, down_lambda=0.0).mean()
    sym_r = shape_rewards(right, vol, down_lambda=0.0).mean()
    assert sym_l == pytest.approx(-sym_r, abs=1e-6)
    assert gap > sym_r - sym_l


def test_reward_matches_closed_form():
    """锁死奖励的函数形式：r - λ·min(r,0)²，r = PnL / 小时波动。

    这条是防回退的主力。历史上这里是 differential Sharpe，它的路径均值不随夏普
    单调（真实 ETH 上会把「全程空仓」排成最优解），换回去会立刻在这里挂掉。
    """
    from betatrend.mathx import BARS_PER_YEAR
    from betatrend.nn.reward import shape_rewards

    pnl = np.array([0.01, -0.01, 0.0, -0.03, 0.02])
    vol_ann = np.full(5, 0.50)
    hourly = 0.50 / np.sqrt(BARS_PER_YEAR)
    r = pnl / hourly
    for lam in (0.0, 0.5, 2.0):
        expect = r - lam * np.minimum(r, 0.0) ** 2
        got = shape_rewards(pnl, vol_ann, down_lambda=lam, dd_inc=0.0, dd_level=0.0, clip=1e9)
        np.testing.assert_allclose(got, expect, rtol=1e-6)


def test_reward_ranks_signals_by_realised_sharpe():
    """奖励的路径均值必须随真实夏普单调，且「全程空仓」不能是最优解。

    PPO 最大化折现回报，对平稳策略等价于奖励的路径均值——所以路径均值排错，
    学出来的策略就是错的。注意合成面板比真实 ETH 温和得多（峰度约 12 vs 127），
    旧奖励在这条数据上并不会挂；它排错是在真实 ETH 上量到的。
    """
    import pandas as pd

    from betatrend.mathx import BARS_PER_YEAR, sharpe_ratio
    from betatrend.nn.dataset import execution_aligned_returns, sizing_vol, vol_leverage
    from betatrend.nn.reward import bar_pnl, shape_rewards

    panel = make_trending_panels(n=3000, seed=11, symbols=["ETHUSDT"])["ETHUSDT"]
    y = execution_aligned_returns(panel).to_numpy(dtype=float)
    vol = sizing_vol(panel, 72)
    lev = vol_leverage(vol, target=0.20, max_leverage=2.0).to_numpy(dtype=float)
    vol_ann = vol.to_numpy(dtype=float)

    fwd = pd.Series(y).rolling(24).mean().shift(-23).fillna(0.0).to_numpy()
    oracle = np.tanh(fwd / (float(np.std(fwd)) + 1e-12))
    units = {"oracle": oracle, "flat": np.zeros_like(y), "anti": -oracle}

    reward, sharpe = {}, {}
    for name, unit in units.items():
        pnl = bar_pnl(unit, y, lev, cost=0.0008)
        reward[name] = float(shape_rewards(pnl, vol_ann).mean())
        sharpe[name] = sharpe_ratio(pd.Series(pnl), bars_per_year=BARS_PER_YEAR)

    assert sharpe["oracle"] > 1.0 > sharpe["anti"], "前提不成立：oracle 应当远比反向好"
    assert reward["oracle"] > reward["flat"] > reward["anti"]


def test_desk_hold_stops_intra_hold_turnover_from_entering_pnl():
    """训练奖励必须按 8h 冻结记账，否则逐小时翻仓的探索噪声会主导梯度。"""
    from betatrend.config import StrategyCfg, desk_hold_bars
    from betatrend.nn.reward import bar_pnl
    from betatrend.nn.train import desk_positions

    n = 64
    flip = np.where(np.arange(n) % 2 == 0, 0.8, -0.8)
    hold = desk_hold_bars(StrategyCfg())
    held = desk_positions(flip, smooth=0.0, min_position=0.0, hold=hold)
    for start in range(0, n, hold):
        sl = held[start : start + hold]
        assert np.allclose(sl, sl[0])
    y = np.zeros(n)
    lev = np.ones(n)
    raw_cost = float(np.abs(bar_pnl(flip, y, lev, 0.001)).sum())
    held_cost = float(np.abs(bar_pnl(held, y, lev, 0.001)).sum())
    assert raw_cost > held_cost * 4


def test_hold_group_rewards_and_advantages():
    from betatrend.nn.reward import group_advantages, hold_group_rewards

    y = np.array([0.01, 0.01, 0.01, 0.01])
    lev = np.ones(4)
    vol = np.full(4, 0.20)
    r, held, ema = hold_group_rewards(
        np.array([1.0, 0.0, -1.0]),
        last_unit=0.0,
        prev_exp=0.0,
        y=y,
        lev=lev,
        vol_ann=vol,
        cost=0.0,
        smooth=0.0,
        min_position=0.0,
    )
    assert held.tolist() == [1.0, 0.0, -1.0]
    assert r[0] > r[1] > r[2]
    adv = group_advantages(r)
    assert adv.shape == (3,)
    assert abs(float(adv.mean())) < 1e-6
    np.testing.assert_allclose(group_advantages(np.ones(5)), 0.0)


def test_replay_buffer_keeps_recent_rollouts():
    from betatrend.nn.buffer import ReplayBuffer

    buf = ReplayBuffer(n_rollouts=2)
    for i in range(3):
        buf.add(
            index=np.arange(4, dtype=np.int64),
            actions=np.full(4, float(i), dtype=np.float32),
            logp=np.zeros(4, dtype=np.float32),
            advantages=np.ones(4, dtype=np.float32),
        )
    assert buf.n_stored == 2
    assert len(buf) == 8
    packed = buf.packed()
    assert packed["index"].shape == (8,)
    assert packed["actions"].shape == (8,)
    np.testing.assert_array_equal(packed["index"], np.tile(np.arange(4), 2))
    np.testing.assert_allclose(packed["actions"][:4], 1.0)
    np.testing.assert_allclose(packed["actions"][4:], 2.0)


def test_replay_buffer_rejects_ragged_rollout():
    from betatrend.nn.buffer import ReplayBuffer

    buf = ReplayBuffer(n_rollouts=2)
    with pytest.raises(ValueError, match="same length|share|length"):
        buf.add(
            index=np.arange(4, dtype=np.int64),
            actions=np.zeros(3, dtype=np.float32),
            logp=np.zeros(4, dtype=np.float32),
            advantages=np.zeros(4, dtype=np.float32),
        )


def test_resolve_seed_jobs_never_exceeds_seeds(settings):
    from betatrend.nn.train import MIN_THREADS_PER_JOB, resolve_seed_jobs

    settings.strategy.nn_jobs = 8
    settings.strategy.nn_threads_per_job = 0
    jobs, threads = resolve_seed_jobs(settings.strategy, 3)
    assert jobs == 3
    # 单线程 worker 比串行还慢，自动档必须留住下限
    assert threads == max(MIN_THREADS_PER_JOB, torch.get_num_threads() // 3)
    assert threads >= MIN_THREADS_PER_JOB

    settings.strategy.nn_jobs = 0
    assert resolve_seed_jobs(settings.strategy, 2)[0] == 2

    settings.strategy.nn_jobs = 1
    settings.strategy.nn_threads_per_job = 2
    assert resolve_seed_jobs(settings.strategy, 3) == (1, 2)


def _tiny_wf(settings, tmp_path, *, jobs: int, tag: str):
    from betatrend.nn.train import train_decision_net

    panel = make_trending_panels(n=1400, seed=8, symbols=["ETHUSDT"])["ETHUSDT"]
    settings.strategy.nn_epochs = 1
    settings.strategy.nn_seeds = 2
    settings.strategy.nn_ckpt_every = 0
    settings.strategy.min_history = 80
    settings.strategy.ppo_inner_epochs = 1
    settings.strategy.ppo_replay_rollouts = 2
    settings.strategy.nn_jobs = jobs
    settings.strategy.nn_threads_per_job = torch.get_num_threads()
    return train_decision_net(
        panel,
        settings,
        path=tmp_path / f"{tag}.pt",
        min_train=400,
        test_h=240,
        purge=12,
        prod_holdout=48,
        min_valid=400,
    )


def test_parallel_seeds_match_sequential(settings, tmp_path):
    """折内 seed 并行只改调度，不改结果：OOS 与最终权重都要和串行一致。"""
    seq = _tiny_wf(settings, tmp_path, jobs=1, tag="seq")
    par = _tiny_wf(settings, tmp_path, jobs=2, tag="par")

    assert seq.n_folds == par.n_folds
    assert seq.oos_sharpe == pytest.approx(par.oos_sharpe, rel=1e-6, abs=1e-9)
    assert seq.oos_max_dd == pytest.approx(par.oos_max_dd, rel=1e-6, abs=1e-9)

    blob_seq = torch.load(seq.path, map_location="cpu", weights_only=False)
    blob_par = torch.load(par.path, map_location="cpu", weights_only=False)
    assert len(blob_seq["states"]) == len(blob_par["states"]) == 2
    assert not blob_par["partial"]
    for a, b in zip(blob_seq["states"], blob_par["states"]):
        assert a.keys() == b.keys()
        for k in a:
            torch.testing.assert_close(a[k], b[k], rtol=1e-5, atol=1e-6)


def test_sizing_vol_matches_desk_realized_vol(settings):
    """训练/评估的杠杆 σ 必须和 desk 的 realized_vol 逐值相等，不能再是 vol_24。"""
    from betatrend.features import compute_features
    from betatrend.mathx import realized_vol
    from betatrend.nn.dataset import build_feature_frame, sizing_vol

    panel = make_trending_panels(n=600, seed=5, symbols=["ETHUSDT"])["ETHUSDT"]
    lookback = settings.strategy.vol_lookback
    assert lookback != 24, "fixture must not coincide with the vol_24 feature window"

    ret = panel["close"].astype(float).pct_change().fillna(0.0).to_numpy()
    mine = sizing_vol(panel, lookback)
    for t in (200, 401, 599):
        assert float(mine.iloc[t]) == pytest.approx(realized_vol(ret[: t + 1], lookback), rel=1e-12)

    # desk 走 compute_features 拿到的也应是同一个数
    feat = compute_features({"ETHUSDT": panel}, "ETHUSDT", vol_lookback=lookback)
    assert feat.vols["ETHUSDT"] == pytest.approx(float(mine.iloc[-1]), rel=1e-12)

    # 而 vol_24 是另一回事，不该被当成杠杆输入
    assert not np.allclose(build_feature_frame(panel)["vol_24"].to_numpy()[-50:], mine.to_numpy()[-50:])


def test_desk_positions_replays_policy_contract(settings):
    """desk_positions 必须逐 bar 复刻 NeuralPolicy + 再平衡循环的实际持仓。"""
    from betatrend.config import desk_hold_bars
    from betatrend.nn.train import desk_positions
    from betatrend.signals import smooth_unit

    cfg = settings.strategy
    hold = desk_hold_bars(cfg)
    assert hold == 8

    rng = np.random.default_rng(0)
    raw = rng.uniform(-1.0, 1.0, 40)
    got = desk_positions(
        raw, smooth=cfg.nn_smooth, min_position=cfg.min_position, hold=hold, long_only=False
    )

    # 参照实现：只在再平衡 bar 上调策略，EMA 也只在那时前进一步
    last = 0.0
    want = np.zeros(len(raw))
    held = 0.0
    for t in range(len(raw)):
        if t % hold == 0:
            last = smooth_unit(raw[t], last, smooth=cfg.nn_smooth, min_position=cfg.min_position)
            held = float(np.clip(last, -1.0, 1.0))
        want[t] = held
    np.testing.assert_allclose(got, want)

    # 换仓只发生在再平衡 bar 上
    changed = np.flatnonzero(np.abs(np.diff(got)) > 0)
    assert all((i + 1) % hold == 0 for i in changed)
    # 冻结让换手远低于逐 bar 信号
    assert np.abs(np.diff(got)).sum() < np.abs(np.diff(raw)).sum()


def test_smooth_unit_zeroes_ema_state_under_min_position():
    """被 min_position 打掉时 EMA 状态一起归零——照抄原 predict_unit 的行为。"""
    from betatrend.signals import smooth_unit

    assert smooth_unit(0.01, 0.0, smooth=0.2, min_position=0.05) == 0.0
    assert smooth_unit(0.02, 0.10, smooth=0.5, min_position=0.05) == pytest.approx(0.06)
    assert smooth_unit(-0.5, 0.0, smooth=0.2, min_position=0.05, long_only=True) == 0.0
    assert smooth_unit(1.0, 1.0, smooth=0.2, min_position=0.05) == pytest.approx(1.0)


def test_oos_report_carries_both_contracts(settings, tmp_path):
    """报告要同时给出 desk 口径与裸信号口径，且成本/持有参数如实写进去。"""
    res = _tiny_wf(settings, tmp_path, jobs=1, tag="contract")
    m = res.metrics

    assert "oos_neural" in m and "oos_raw_signal" in m
    contract = m["execution_contract"]
    assert contract["hold_bars"] == 8
    assert contract["vol_lookback"] == settings.strategy.vol_lookback
    assert contract["nn_smooth"] == pytest.approx(settings.strategy.nn_smooth)
    # 训练 overlay 与 desk 评估同一套费率（maker 2bps + 滑点 1.5bps）
    assert contract["eval_cost_bps"] == pytest.approx(3.5)
    assert contract["train_cost_bps"] == pytest.approx(3.5)
    # 主指标取 desk 口径
    assert m["eval_primary"]["sharpe"] == pytest.approx(m["oos_neural"]["sharpe"])
    # 冻结 8 根 bar 之后换手必然低于逐 bar 信号
    assert m["oos_neural"]["turnover"] <= m["oos_raw_signal"]["turnover"]


def test_warm_start_does_not_match_scratch_init(settings):
    from betatrend.nn.train import _train_one

    rng = np.random.default_rng(0)
    n = 48
    windows = rng.normal(size=(n, SEQ_LEN, N_FEAT)).astype(np.float32)
    y = rng.normal(size=n).astype(np.float64) * 0.01
    lev = np.ones(n, dtype=np.float64)
    vol = np.full(n, 0.2, dtype=np.float64)
    settings.strategy.nn_epochs = 2
    settings.strategy.ppo_inner_epochs = 1
    settings.strategy.ppo_replay_rollouts = 1
    cfg = settings.strategy
    prior = _train_one(windows, y, lev, vol, cfg=cfg, cost=8e-4, seed=1, y_clip=0.04)
    init_state = {k: v.detach().cpu().clone() for k, v in prior.state_dict().items()}
    warm = _train_one(
        windows, y, lev, vol, cfg=cfg, cost=8e-4, seed=2, y_clip=0.04, init_state=init_state
    )
    scratch = _train_one(windows, y, lev, vol, cfg=cfg, cost=8e-4, seed=2, y_clip=0.04)
    w_warm = next(warm.parameters()).detach()
    w_scratch = next(scratch.parameters()).detach()
    assert not torch.allclose(w_warm, w_scratch)


def test_dump_ckpt_is_atomic_and_loadable(tmp_path):
    from betatrend.config import StrategyCfg
    from betatrend.nn.train import _ckpt_payload, _dump_ckpt

    cfg = StrategyCfg()
    path = tmp_path / "mid.pt"
    states = [{"w": torch.tensor([1.0, 2.0])}]
    _dump_ckpt(
        path,
        _ckpt_payload(
            states=states,
            median=np.zeros(3, dtype=np.float32),
            iqr=np.ones(3, dtype=np.float32),
            cfg=cfg,
            seq_len=7,
            extra={"step": 500, "partial": True},
        ),
    )
    assert path.exists()
    assert not path.with_name("mid.pt.tmp").exists()
    blob = torch.load(path, map_location="cpu", weights_only=False)
    assert blob["kind"] == "grpo"
    assert blob["step"] == 500
    assert blob["partial"] is True
    assert blob["n_feat"] == N_FEAT
