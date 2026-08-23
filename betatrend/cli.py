"""命令行：拉行情、回测、训练、paper、探活。

``python -m betatrend <子命令>``。研究 / paper / 实盘共用 DeskCycle。
"""
from __future__ import annotations

import argparse
import json
import sys

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from betatrend import __system_name__, __version__
from betatrend.config import ROOT, load_settings
from betatrend.logutil import setup_logging
from betatrend.marketdata import BinancePublicClient
from betatrend.marketdata.store import MarketDataStore
from betatrend.research import paper_once, paper_run, run_backtest, train_nn

console = Console()


def _settings(args: argparse.Namespace):
    overrides = {}
    if getattr(args, "mode", None):
        overrides["account"] = {"mode": args.mode}
    return load_settings(getattr(args, "config", None), overrides or None)


def cmd_version(_args) -> int:
    console.print(f"{__system_name__} {__version__}")
    return 0


def cmd_fetch(args) -> int:
    settings = _settings(args)
    setup_logging(settings)
    store = MarketDataStore(settings)
    panels = store.load_universe(
        lookback_days=args.days,
        force_demo=args.demo,
        refresh=args.refresh,
    )
    for s, df in panels.items():
        last = float(df["close"].iloc[-1])
        console.print(f"{s}: {len(df)} bars  {df.index[0]} → {df.index[-1]}  close={last:.4f}")
    return 0


def cmd_backtest(args) -> int:
    settings = _settings(args)
    setup_logging(settings)
    if args.days:
        settings.data.lookback_days = args.days
    if getattr(args, "decision", None):
        settings.strategy.decision = args.decision
    result = run_backtest(settings, force_demo=args.demo, name=args.name)
    table = Table(title="BETA-TREND backtest")
    table.add_column("metric")
    table.add_column("value", justify="right")
    for k, v in result.metrics.items():
        if isinstance(v, float):
            table.add_row(k, f"{v:.4f}" if abs(v) < 10 else f"{v:.2f}")
        else:
            table.add_row(k, str(v))
    console.print(table)
    console.print("[dim]Not a live performance claim. Demo/costed research only.[/dim]")
    return 0


def cmd_train_nn(args) -> int:
    settings = _settings(args)
    setup_logging(settings)
    if args.days:
        settings.data.lookback_days = args.days
    settings.strategy.decision = "neural"
    from pathlib import Path as P

    result = train_nn(
        settings,
        force_demo=args.demo,
        path=P(args.path) if args.path else None,
    )
    table = Table(title="ETH decision net — walk-forward OOS")
    table.add_column("metric")
    table.add_column("neural", justify="right")
    nn_m = result.metrics.get("oos_neural") or result.metrics["oos_neural"]
    primary = result.metrics.get("eval_primary") or result.metrics.get("eval_primary") or {}
    for k in ("sharpe", "max_drawdown", "calmar"):
        nv = primary.get(k, nn_m.get(k))
        if k == "max_drawdown":
            table.add_row(k, f"{nv:.2%}")
        else:
            table.add_row(k, f"{nv:.3f}")
    for k in ("sortino", "total_return", "turnover"):
        nv = nn_m[k]
        if k == "total_return":
            table.add_row(k + " (secondary)", f"{nv:.2%}")
        else:
            table.add_row(k, f"{nv:.3f}")
    table.add_row("folds", str(result.n_folds))
    table.add_row("weights", str(result.path))
    console.print(table)
    console.print(
        "[dim]Walk-forward OOS is the honest number. Reloading these weights into a full-sample backtest is in-sample for most bars.[/dim]"
    )
    return 0


def cmd_paper_once(args) -> int:
    settings = _settings(args)
    setup_logging(settings)
    out = paper_once(settings, force_demo=args.demo, execute=args.execute)
    console.print_json(json.dumps(out, default=str))
    return 0


def cmd_paper_run(args) -> int:
    settings = _settings(args)
    setup_logging(settings)
    out = paper_run(
        settings,
        force_demo=args.demo,
        execute=args.execute,
        max_bars=args.bars,
        reset_state=args.reset_state,
    )
    console.print_json(json.dumps(out, default=str))
    return 0


def cmd_ping(args) -> int:
    settings = _settings(args)
    setup_logging(settings)
    with BinancePublicClient(settings) as c:
        c.ping()
    console.print("Public REST: ok")
    if args.signed:
        from betatrend.execution import BinanceSignedClient

        client = BinanceSignedClient(settings)
        acct = client.ping_account()
        console.print(f"Signed account: availableBalance={acct.get('availableBalance')}")
        client.close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="betatrend", description="BETA-TREND ETH timing")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--mode", choices=["research", "paper", "testnet", "live"], default=None)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("version").set_defaults(func=cmd_version)

    f = sub.add_parser("fetch")
    f.add_argument("--days", type=int, default=None)
    f.add_argument("--demo", action="store_true")
    f.add_argument("--refresh", action="store_true")
    f.set_defaults(func=cmd_fetch)

    b = sub.add_parser("backtest")
    b.add_argument("--days", type=int, default=None)
    b.add_argument("--demo", action="store_true")
    b.add_argument("--name", type=str, default=None)
    b.add_argument("--decision", choices=["neural", "rl"], default=None)
    b.set_defaults(func=cmd_backtest)

    tn = sub.add_parser("train-nn", help="Walk-forward train the ETH PPO decision net")
    tn.add_argument("--days", type=int, default=None)
    tn.add_argument("--demo", action="store_true")
    tn.add_argument("--path", type=str, default=None)
    tn.set_defaults(func=cmd_train_nn)

    po = sub.add_parser("paper-once")
    po.add_argument("--demo", action="store_true")
    po.add_argument("--execute", action="store_true")
    po.set_defaults(func=cmd_paper_once)

    pr = sub.add_parser("paper-run")
    pr.add_argument("--demo", action="store_true")
    pr.add_argument("--bars", type=int, default=24)
    pr.add_argument("--execute", action="store_true")
    pr.add_argument("--reset-state", action="store_true")
    pr.set_defaults(func=cmd_paper_run)

    pg = sub.add_parser("ping")
    pg.add_argument("--signed", action="store_true")
    pg.set_defaults(func=cmd_ping)
    return p


def main(argv: list[str] | None = None) -> int:
    load_dotenv(ROOT / ".env")
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except Exception as e:
        console.print(f"[red]{type(e).__name__}: {e}[/red]")
        return 1
