from __future__ import annotations

import pytest

from betatrend.config import Settings, load_settings


@pytest.fixture
def settings() -> Settings:
    return load_settings(
        overrides={
            "account": {"initial_capital": 100_000.0, "mode": "paper"},
            "universe": {"symbol": "ETHUSDT"},
            "strategy": {
                "lookbacks_hours": [24, 48, 96],
                "lookback_weights": [0.3, 0.4, 0.3],
                "vol_lookback": 48,
                "min_history": 150,
                "rebalance_hours": 8,
                "score_scale": 1.0,
                "min_position": 0.02,
                "long_only": False,
                "max_leverage": 2.0,
                "target_vol_annual": 0.25,
                "decision": "tsmom",
            },
            "backtest": {"warmup_bars": 160, "turnover_band_equity": 0.01},
            "risk": {
                "vol_target_enabled": False,
                "max_symbol_weight": 1.0,
                "max_gross_leverage": 2.0,
                "max_net_leverage": 2.0,
                "drawdown_soft": 0.08,
                "drawdown_hard": 0.14,
                "drawdown_kill": 0.25,
            },
        }
    )
