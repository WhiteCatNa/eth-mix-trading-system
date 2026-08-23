from __future__ import annotations

from betatrend.oms import OrderManager
from betatrend.risk import RiskEngine


def test_drawdown_kill_flattens(settings):
    engine = RiskEngine(settings)
    engine.update_equity(100_000)
    engine.update_equity(70_000)  # -30% > kill 25%
    scalar = engine.compute_scalar()
    assert engine.state.kill
    assert scalar == 0.0
    clipped = engine.clip({"ETHUSDT": 50_000}, 70_000)
    assert clipped["ETHUSDT"] == 0.0


def test_soft_drawdown_halves(settings):
    engine = RiskEngine(settings)
    engine.cfg.drawdown_soft = 0.05
    engine.cfg.drawdown_hard = 0.20
    engine.cfg.drawdown_kill = 0.50
    engine.cfg.max_daily_loss = 0.50
    engine.update_equity(100_000)
    engine.set_day_start(94_000)
    engine.update_equity(94_000)  # -6% vs peak, not vs day
    assert abs(engine.compute_scalar() - 0.50) < 1e-9


def test_oms_skips_inside_band(settings):
    oms = OrderManager(settings)
    settings.backtest.turnover_band_equity = 0.05
    parent = oms.rebalance_intents(
        current_notional={"ETHUSDT": 10_000},
        target_notional={"ETHUSDT": 10_200},
        prices={"ETHUSDT": 2000.0},
        equity=100_000,
    )
    assert parent.children == []


def test_oms_creates_sell_to_reduce(settings):
    settings.backtest.turnover_band_equity = 0.001
    settings.oms.min_notional = 1.0
    oms = OrderManager(settings)
    parent = oms.rebalance_intents(
        current_notional={"ETHUSDT": 20_000},
        target_notional={"ETHUSDT": 0.0},
        prices={"ETHUSDT": 2000.0},
        equity=100_000,
    )
    assert len(parent.children) == 1
    assert parent.children[0].side.value == "SELL"


def test_oms_skips_open_below_five_percent_of_full(settings):
    settings.backtest.turnover_band_equity = 0.0
    settings.oms.min_notional = 1.0
    settings.strategy.min_position = 0.05
    settings.strategy.risk_budget = 1.0
    settings.strategy.max_leverage = 2.0
    oms = OrderManager(settings)
    # full = 100k * 1 * 2 = 200k; 5% = 10k. 8k from flat must not open.
    parent = oms.rebalance_intents(
        current_notional={"ETHUSDT": 0.0},
        target_notional={"ETHUSDT": 8_000.0},
        prices={"ETHUSDT": 2000.0},
        equity=100_000,
    )
    assert parent.children == []
    buy = oms.rebalance_intents(
        current_notional={"ETHUSDT": 0.0},
        target_notional={"ETHUSDT": 50_000.0},
        prices={"ETHUSDT": 2000.0},
        equity=100_000,
    )
    assert len(buy.children) == 1
    assert buy.children[0].side.value == "BUY"
    short = oms.rebalance_intents(
        current_notional={"ETHUSDT": 0.0},
        target_notional={"ETHUSDT": -50_000.0},
        prices={"ETHUSDT": 2000.0},
        equity=100_000,
    )
    assert len(short.children) == 1
    assert short.children[0].side.value == "SELL"
