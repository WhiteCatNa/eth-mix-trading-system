from __future__ import annotations

import numpy as np
import pandas as pd

from betatrend.mathx import horizon_return, ols_beta, sharpe_ratio, tsmom_score


def test_ols_beta_recovers_known_slope():
    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 0.01, size=2500)
    y = 1.20 * x + rng.normal(0.0, 0.002, size=2500)
    b = ols_beta(y, x)
    assert abs(b - 1.20) < 0.12


def test_ols_beta_unidentified_is_one():
    x = np.zeros(50)
    y = np.random.default_rng(0).normal(size=50)
    assert ols_beta(y, x) == 1.0


def test_horizon_return_no_lookahead_window():
    close = np.array([1.0, 1.1, 1.21, 1.331])
    # lookback 2, skip 0: 1.331/1.1 - 1
    assert abs(horizon_return(close, 2, 0) - (1.331 / 1.1 - 1)) < 1e-12


def test_tsmom_positive_on_uptrend():
    close = 100 * np.cumprod(np.r_[1.0, np.full(200, 1.002)])
    rets = np.diff(close) / close[:-1]
    rets = np.r_[0.0, rets]
    score = tsmom_score(close, rets, [24, 48], [0.5, 0.5], 0)
    assert score > 0


def test_sharpe_zero_on_flat():
    r = pd.Series(np.zeros(100))
    assert sharpe_ratio(r) == 0.0


def test_score_to_unit_tanh_range():
    from betatrend.mathx import score_to_unit

    assert abs(score_to_unit(0.0)) < 1e-12
    assert 0 < score_to_unit(0.5) < score_to_unit(2.0) < 1
    assert score_to_unit(-10.0) > -1.0

