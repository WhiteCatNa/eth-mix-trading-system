from __future__ import annotations

import numpy as np
import pandas as pd

from betatrend.mathx import horizon_return, sharpe_ratio


def test_horizon_return_no_lookahead_window():
    close = np.array([1.0, 1.1, 1.21, 1.331])
    assert abs(horizon_return(close, 2, 0) - (1.331 / 1.1 - 1)) < 1e-12


def test_sharpe_zero_on_flat():
    r = pd.Series(np.zeros(100))
    assert sharpe_ratio(r) == 0.0


def test_score_to_unit_tanh_range():
    from betatrend.mathx import score_to_unit

    assert abs(score_to_unit(0.0)) < 1e-12
    assert 0 < score_to_unit(0.5) < score_to_unit(2.0) < 1
    assert score_to_unit(-10.0) > -1.0
