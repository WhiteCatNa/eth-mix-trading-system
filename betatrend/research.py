"""研究与 paper：回测落盘、训练入口、单次/循环纸交易。

CLI 薄封装：拉齐面板 → DeskCycle / Backtester / 训练器 → 写报告或 paper 状态。
不在这里另写信号逻辑。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

from betatrend.backtest import BacktestResult, Backtester
from betatrend.config import ROOT, Settings
from betatrend.domain import MarketSnapshot
from betatrend.execution.paper import PaperBroker
from betatrend.ledger import Ledger
from betatrend.logutil import audit
from betatrend.marketdata.store import MarketDataStore
from betatrend.oms import OrderManager
from betatrend.pipeline import DeskCycle


def save_report(result: BacktestResult, settings: Settings, name: str | None = None) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    name = name or f"backtest_{stamp}"
    folder = settings.report_path()
    metrics_path = folder / f"{name}_metrics.json"
    metrics_path.write_text(json.dumps(result.metrics | result.meta, indent=2, default=str), encoding="utf-8")
    result.equity_curve.to_csv(folder / f"{name}_equity.csv", header=True)
    result.positions.to_csv(folder / f"{name}_positions.csv")
    if not result.targets_log.empty:
        result.targets_log.to_csv(folder / f"{name}_targets.csv", index=False)
    logger.info("Wrote {}", metrics_path)
    return metrics_path


def train_nn(settings: Settings, *, force_demo: bool = False, path: Path | None = None, **kwargs):
    from betatrend.nn.train import train_decision_net

    store = MarketDataStore(settings)
    panels = store.load_universe(
        lookback_days=settings.data.lookback_days,
        force_demo=force_demo,
    )
    symbol = settings.universe.trade_symbol
    if symbol not in panels:
        raise KeyError(f"{symbol} missing from loaded panels")
    return train_decision_net(panels[symbol], settings, path=path, **kwargs)


def run_backtest(settings: Settings, *, force_demo: bool = False, name: str | None = None) -> BacktestResult:
    store = MarketDataStore(settings)
    panels = store.load_universe(
        lookback_days=settings.data.lookback_days,
        force_demo=force_demo,
    )
    result = Backtester(settings).run(panels)
    save_report(result, settings, name=name)
    return result


def _panel_index(panels: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    index = None
    for df in panels.values():
        index = df.index if index is None else index.intersection(df.index)
    if index is None:
        raise ValueError("No panels")
    return index.sort_values()


def _last_reb_index(index: pd.DatetimeIndex, last_reb_ts: pd.Timestamp | None) -> int | None:
    """把上次再平衡时间映射到当前面板下标。找不到则返回 None，改用墙钟小时。"""
    if last_reb_ts is None:
        return None
    try:
        loc = index.get_loc(last_reb_ts)
    except KeyError:
        return None
    if isinstance(loc, slice):
        return int(loc.start)
    if isinstance(loc, int):
        return loc
    return int(list(loc)[0])


def _should_rebalance(
    i: int,
    ts,
    last_reb_i: int | None,
    last_reb_ts: pd.Timestamp | None,
    reb_h: int,
) -> bool:
    if last_reb_i is None and last_reb_ts is None:
        return True
    if last_reb_i is not None:
        return (i - last_reb_i) >= reb_h
    return pd.Timestamp(ts) - pd.Timestamp(last_reb_ts) >= pd.Timedelta(hours=reb_h)


def paper_once(
    settings: Settings,
    *,
    force_demo: bool = False,
    execute: bool = False,
    panels: dict[str, pd.DataFrame] | None = None,
) -> dict:
    """最新一根已完成的 T+1 周期：t 收盘出信号，t+1 开盘成交。

    面板最后一根是成交 bar（open），倒数第二根是信号 bar（close）。
    每次从空仓和 ``initial_capital`` 起算，不读取 ``paper-run`` 账本。
    当前仍未收完、下一根开盘还不存在的那根信号，请用 ``paper_run`` 挂 pending。
    ``execute=True`` 会本地成交，忽略 ``paper.dry_run``。
    """
    if panels is None:
        store = MarketDataStore(settings)
        panels = store.load_universe(
            lookback_days=settings.paper.lookback_days,
            force_demo=force_demo,
        )
    index = _panel_index(panels)
    if len(index) < 2:
        raise ValueError("paper_once needs at least 2 bars (signal at t close, fill at t+1 open)")
    signal_ts = index[-2]
    fill_ts = index[-1]
    symbols = list(panels)
    hist = {s: panels[s].loc[:signal_ts] for s in symbols}
    close_px = {s: float(panels[s].loc[signal_ts, "close"]) for s in symbols}
    open_px = {s: float(panels[s].loc[fill_ts, "open"]) for s in symbols}
    desk = DeskCycle(settings)
    capital = settings.account.initial_capital
    ledger = Ledger(cash=capital, qty={s: 0.0 for s in symbols})
    current = {s: 0.0 for s in symbols}
    snap = MarketSnapshot(
        timestamp=pd.Timestamp(signal_ts),
        panels=hist,
        prices=close_px,
        equity=capital,
        bar_index=max(len(next(iter(hist.values()))) - 1, 0),
        market_symbol=settings.universe.market_symbol,
    )
    cycle = desk.run(snap, current, reason="paper-once")
    fills = []
    intents = cycle.intents
    do_fill = bool(execute)
    if do_fill:
        broker = PaperBroker(settings, ledger)
        oms = OrderManager(settings)
        parent = oms.rebalance_intents(
            current, cycle.notionals, open_px, max(capital, 1.0), reason="paper-once-fill"
        )
        fills = broker.execute(parent.children, open_px, fill_ts)
        intents = parent.children
    extras = cycle.targets[0].extras if cycle.targets else {}
    return {
        "timestamp": str(signal_ts),
        "fill_timestamp": str(fill_ts),
        "fill_at": "next_open",
        "fill_prices": {s: open_px[s] for s in symbols},
        "market_vol": cycle.features.market_vol,
        "notionals": cycle.notionals,
        "n_intents": len(intents),
        "side": extras.get("side"),
        "unit": extras.get("unit"),
        "decision": extras.get("decision"),
        "executed_fills": len(fills),
        "executed_fill_prices": {f.symbol: float(f.price) for f in fills},
        "dry_run": not do_fill,
    }


def _load_paper_state(path: Path, capital: float, symbols: list[str]) -> dict:
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        qty = {s: float((raw.get("qty") or {}).get(s, 0.0)) for s in symbols}
        notionals = raw.get("notionals") or raw.get("clipped") or {}
        pending_raw = raw.get("pending")
        pending = {k: float(v) for k, v in pending_raw.items()} if isinstance(pending_raw, dict) else None
        return {
            "last_ts": raw.get("last_ts"),
            "last_reb_ts": raw.get("last_reb_ts"),
            "cash": float(raw.get("cash", capital)),
            "qty": qty,
            "bars": int(raw.get("bars", 0)),
            "notionals": {k: float(v) for k, v in notionals.items()},
            "pending": pending,
            "nn_last_unit": float(raw.get("nn_last_unit", 0.0)),
        }
    return {
        "last_ts": None,
        "last_reb_ts": None,
        "cash": capital,
        "qty": {s: 0.0 for s in symbols},
        "bars": 0,
        "notionals": {s: 0.0 for s in symbols},
        "pending": None,
        "nn_last_unit": 0.0,
    }


def _save_paper_state(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def paper_run(
    settings: Settings,
    *,
    force_demo: bool = False,
    execute: bool = False,
    max_bars: int | None = 24,
    reset_state: bool = False,
    panels: dict[str, pd.DataFrame] | None = None,
) -> dict:
    if panels is None:
        store = MarketDataStore(settings)
        panels = store.load_universe(
            lookback_days=settings.paper.lookback_days,
            force_demo=force_demo,
        )
    symbols = list(panels)
    index = _panel_index(panels)
    warmup = int(settings.backtest.warmup_bars)
    reb_h = max(int(settings.strategy.rebalance_hours), 8)
    fund_h = int(settings.data.funding_interval_hours)
    state_path = Path(settings.paper.state_file)
    if not state_path.is_absolute():
        state_path = ROOT / settings.paper.state_file
    if reset_state and state_path.exists():
        state_path.unlink()
    capital = float(settings.account.initial_capital)
    state = _load_paper_state(state_path, capital, symbols)
    ledger = Ledger(cash=state["cash"], qty=dict(state["qty"]))
    desk = DeskCycle(settings)
    desk.reset(ledger.cash)
    desk.restore_smooth(float(state.get("nn_last_unit", 0.0)))
    last_ts = pd.Timestamp(state["last_ts"]) if state["last_ts"] else None
    last_reb_ts = pd.Timestamp(state["last_reb_ts"]) if state.get("last_reb_ts") else None
    last_reb_i = _last_reb_index(index, last_reb_ts)
    do_fill = bool(execute)
    broker = PaperBroker(settings, ledger) if do_fill else None
    oms = OrderManager(settings)
    processed = 0
    last_notionals = dict(state["notionals"])
    n_fills = 0
    funding_pnl = 0.0
    pending: dict[str, float] | None = dict(state["pending"]) if state.get("pending") else None

    for i, ts in enumerate(index):
        if i < warmup:
            continue
        if last_ts is not None and pd.Timestamp(ts) <= last_ts:
            continue
        if max_bars is not None and processed >= int(max_bars):
            break

        close_px = {s: float(panels[s].loc[ts, "close"]) for s in symbols}
        open_px = {s: float(panels[s].loc[ts, "open"]) for s in symbols}
        fund_rate = {
            s: float(panels[s].loc[ts, "funding_rate"]) if "funding_rate" in panels[s].columns else 0.0
            for s in symbols
        }

        if i > 0:
            prev = index[i - 1]
            for s in symbols:
                prev_close = float(panels[s].loc[prev, "close"])
                ledger.apply_mark(s, open_px[s] - prev_close)

        if ts.hour % fund_h == 0:
            for s in symbols:
                funding_pnl += ledger.apply_funding(s, open_px[s], fund_rate[s])

        if pending is not None and broker is not None:
            current_open = {s: ledger.notional(s, open_px[s]) for s in symbols}
            parent = oms.rebalance_intents(
                current_open, pending, open_px, max(ledger.cash, 1.0), reason="paper-fill"
            )
            fills = broker.execute(parent.children, open_px, ts)
            n_fills += len(fills)
            pending = None
        elif pending is not None:
            pending = None

        for s in symbols:
            ledger.apply_mark(s, close_px[s] - open_px[s])

        equity = ledger.equity(close_px)
        hist = {s: panels[s].iloc[: i + 1] for s in symbols}
        snap = MarketSnapshot(
            timestamp=pd.Timestamp(ts),
            panels=hist,
            prices=close_px,
            equity=max(equity, 1.0),
            bar_index=i,
            market_symbol=settings.universe.market_symbol,
        )
        current_n = {s: ledger.notional(s, close_px[s]) for s in symbols}
        if _should_rebalance(i, ts, last_reb_i, last_reb_ts, reb_h):
            cycle = desk.run(snap, current_n, reason="paper-run")
            last_notionals = dict(cycle.notionals)
            pending = dict(cycle.notionals)
            last_reb_i = i
            last_reb_ts = pd.Timestamp(ts)
            audit(
                settings,
                "paper_bar",
                timestamp=str(ts),
                notionals=last_notionals,
                equity=equity,
            )
        processed += 1
        state = {
            "last_ts": str(pd.Timestamp(ts)),
            "last_reb_ts": str(last_reb_ts) if last_reb_ts is not None else None,
            "cash": ledger.cash,
            "qty": dict(ledger.qty),
            "bars": int(state["bars"]) + 1,
            "notionals": last_notionals,
            "pending": pending,
            "nn_last_unit": desk.last_smooth_unit(),
            "equity": equity,
            "funding_pnl": funding_pnl,
        }
        _save_paper_state(state_path, state)

    last_px = {s: float(panels[s]["close"].iloc[-1]) for s in symbols}
    return {
        "bars_processed": processed,
        "last_ts": state.get("last_ts"),
        "last_reb_ts": state.get("last_reb_ts"),
        "notionals": last_notionals,
        "qty": dict(ledger.qty),
        "cash": ledger.cash,
        "equity": ledger.equity(last_px) if processed else ledger.cash,
        "n_fills": n_fills,
        "funding_pnl": funding_pnl,
        "state_file": str(state_path),
        "dry_run": not do_fill,
    }
