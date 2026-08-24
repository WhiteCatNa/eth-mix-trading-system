from __future__ import annotations

import pytest

from betatrend.config import Settings, load_settings


class FakePolicy:
    """Deterministic unit for tests. Not a TSMOM stand-in — just a fixed RL output."""

    ready = True

    def __init__(self, settings=None, *, unit: float = 0.8):
        self.settings = settings
        self.unit = float(unit)

    def reset(self) -> None:
        pass

    def last_unit(self) -> float:
        return float(getattr(self, "_last_unit", self.unit))

    def restore_last_unit(self, unit: float) -> None:
        self._last_unit = float(unit)

    def predict_unit(self, panel, *args, **kwargs) -> float:
        return self.unit


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
                "decision": "rl",
            },
            "backtest": {"warmup_bars": 160, "turnover_band_equity": 0.01},
        }
    )


@pytest.fixture
def stub_rl(monkeypatch):
    monkeypatch.setattr(
        "betatrend.nn.policy.NeuralPolicy",
        lambda settings: FakePolicy(settings, unit=0.8),
    )
