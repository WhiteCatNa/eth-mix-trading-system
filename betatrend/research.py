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


def paper_once(settings: Settings, *, force_demo: bool = False, execute: bool = False) -> dict:
    store = MarketDataStore(settings)
    panels = store.load_universe(
        lookback_days=settings.paper.lookback_days,
        force_demo=force_demo,
    )
    prices = {s: float(df["close"].iloc[-1]) for s, df in panels.items()}
    ts = next(iter(panels.values())).index[-1]
    desk = DeskCycle(settings)
    capital = settings.account.initial_capital
    ledger = Ledger(cash=capital, qty={s: 0.0 for s in panels})
    current = {s: 0.0 for s in panels}
    snap = MarketSnapshot(
        timestamp=pd.Timestamp(ts),
        panels=panels,
        prices=prices,
        equity=capital,
        bar_index=len(next(iter(panels.values()))) - 1,
        market_symbol=settings.universe.market_symbol,
    )
    cycle = desk.run(snap, current, reason="paper-once")
    fills = []
    if execute and not settings.paper.dry_run:
        broker = PaperBroker(settings, ledger)
        fills = broker.execute(cycle.intents, prices, ts)
    extras = cycle.targets[0].extras if cycle.targets else {}
    return {
        "timestamp": str(ts),
        "market_score": cycle.features.market_score,
        "notionals": cycle.notionals,
        "n_intents": len(cycle.intents),
        "side": extras.get("side"),
        "unit": extras.get("unit"),
        "decision": extras.get("decision"),
        "executed_fills": len(fills),
        "dry_run": settings.paper.dry_run or not execute,
    }


def _load_paper_state(path: Path, capital: float, symbols: list[str]) -> dict:
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        qty = {s: float((raw.get("qty") or {}).get(s, 0.0)) for s in symbols}
        notionals = raw.get("notionals") or raw.get("clipped") or {}
        return {
            "last_ts": raw.get("last_ts"),
            "cash": float(raw.get("cash", capital)),
            "qty": qty,
            "bars": int(raw.get("bars", 0)),
            "notionals": {k: float(v) for k, v in notionals.items()},
        }
    return {
        "last_ts": None,
        "cash": capital,
        "qty": {s: 0.0 for s in symbols},
        "bars": 0,
        "notionals": {s: 0.0 for s in symbols},
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
    index = None
    for df in panels.values():
        index = df.index if index is None else index.intersection(df.index)
    index = index.sort_values()
    warmup = int(settings.backtest.warmup_bars)
    reb_h = max(int(settings.strategy.rebalance_hours), 8)
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
    last_ts = pd.Timestamp(state["last_ts"]) if state["last_ts"] else None
    do_fill = bool(execute) and not bool(settings.paper.dry_run)
    broker = PaperBroker(settings, ledger) if do_fill else None
    oms = OrderManager(settings)
    processed = 0
    last_notionals = dict(state["notionals"])
    n_fills = 0
    last_reb = -10**9
    pending: dict[str, float] | None = None

    for i, ts in enumerate(index):
        if i < warmup:
            continue
        if last_ts is not None and pd.Timestamp(ts) <= last_ts:
            continue
        if max_bars is not None and processed >= int(max_bars):
            break

        close_px = {s: float(panels[s].loc[ts, "close"]) for s in symbols}
        open_px = {s: float(panels[s].loc[ts, "open"]) for s in symbols}

        if i > 0:
            prev = index[i - 1]
            for s in symbols:
                prev_close = float(panels[s].loc[prev, "close"])
                ledger.apply_mark(s, open_px[s] - prev_close)

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
        if (i - last_reb) >= reb_h:
            cycle = desk.run(snap, current_n, reason="paper-run")
            last_notionals = dict(cycle.notionals)
            pending = dict(cycle.notionals)
            last_reb = i
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
            "cash": ledger.cash,
            "qty": dict(ledger.qty),
            "bars": int(state["bars"]) + 1,
            "notionals": last_notionals,
            "equity": equity,
        }
        _save_paper_state(state_path, state)

    last_px = {s: float(panels[s]["close"].iloc[-1]) for s in symbols}
    return {
        "bars_processed": processed,
        "last_ts": state.get("last_ts"),
        "notionals": last_notionals,
        "qty": dict(ledger.qty),
        "cash": ledger.cash,
        "equity": ledger.equity(last_px) if processed else ledger.cash,
        "n_fills": n_fills,
        "state_file": str(state_path),
        "dry_run": settings.paper.dry_run or not execute,
    }
