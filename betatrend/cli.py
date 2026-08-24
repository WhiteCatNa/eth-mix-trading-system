"""命令行：拉行情、回测、训练、paper、探活。

``python -m betatrend <子命令>``。研究 / paper 共用 DeskCycle。
本 CLI 不发送签名订单；``ping --signed`` 只读账户。
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
from pathlib import Path

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


def cmd_dashboard(args) -> int:
    """回测成交叠加到 TradingView 图上。只读 reports/，不碰训练、不下单。"""
    from betatrend.dashboard.server import serve

    serve(host=args.host, port=args.port, report=args.report)
    return 0


def cmd_explore_nn(args) -> int:
    """对着 train.log + 最新 fold ckpt 起 TorchExplorer 看板。不碰训练进程。"""
    settings = _settings(args)
    setup_logging(settings)
    from betatrend.nn.explore import run_live

    run_live(
        log_path=ROOT / args.log if not Path(args.log).is_absolute() else Path(args.log),
        models_dir=ROOT / args.models_dir if not Path(args.models_dir).is_absolute() else Path(args.models_dir),
        board_dir=ROOT / args.board_dir if not Path(args.board_dir).is_absolute() else Path(args.board_dir),
        explorer_dir=(
            (ROOT / args.board_dir if not Path(args.board_dir).is_absolute() else Path(args.board_dir))
            / "torchexplorer"
        ),
        explorer_port=int(args.port),
        board_port=int(args.board_port),
        poll=float(args.poll),
    )
    return 0


def cmd_train_nn(args) -> int:
    settings = _settings(args)
    setup_logging(settings)
    if args.days:
        settings.data.lookback_days = args.days
    if args.jobs is not None:
        settings.strategy.nn_jobs = args.jobs
    if args.threads_per_job is not None:
        settings.strategy.nn_threads_per_job = args.threads_per_job
    settings.strategy.decision = "neural"
    from pathlib import Path as P

    result = train_nn(
        settings,
        force_demo=args.demo,
        path=P(args.path) if args.path else None,
        resume=bool(getattr(args, "resume", False)),
        start_fold=getattr(args, "start_fold", None),
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


def _load_panel(args, settings):
    import pandas as pd

    if getattr(args, "panel", None):
        p = Path(args.panel)
        if p.suffix.lower() == ".csv":
            return pd.read_csv(p, index_col=0, parse_dates=True)
        return pd.read_parquet(p)
    store = MarketDataStore(settings)
    panels = store.load_universe(
        lookback_days=getattr(args, "days", None) or settings.data.lookback_days,
        force_demo=bool(getattr(args, "demo", False)),
        refresh=False,
    )
    symbol = settings.universe.trade_symbol
    if symbol not in panels:
        raise KeyError(f"{symbol} missing from loaded panels")
    return panels[symbol]


def cmd_train_fold(args) -> int:
    settings = _settings(args)
    setup_logging(settings)
    from betatrend.nn.train import list_fold_jobs, train_fold

    panel = _load_panel(args, settings)
    cfg = settings.strategy
    jobs = list_fold_jobs(len(panel), cfg)
    if args.job_json:
        job = json.loads(Path(args.job_json).read_text(encoding="utf-8"))
    else:
        picked = [j for j in jobs if j["fold_id"] == args.fold and j["seed"] == args.seed]
        if not picked:
            raise SystemExit(f"no job for fold={args.fold} seed={args.seed}; {len(jobs)} jobs listed")
        job = picked[0]
    train_idx = list(range(int(job["train_start"]), int(job["train_end"])))
    test_idx = None
    if job.get("test_start") is not None:
        test_idx = list(range(int(job["test_start"]), int(job["test_end"])))
    out = Path(args.out) if args.out else ROOT / "models" / f"fold{job['fold_id']}_seed{job['seed']}.pt"
    result = train_fold(
        panel,
        settings,
        train_idx=train_idx,
        test_idx=test_idx,
        fold_id=int(job["fold_id"]),
        seed=int(job["seed"]),
        path=out,
        init_path=Path(args.init) if getattr(args, "init", None) else None,
    )
    console.print_json(json.dumps({k: v for k, v in result.items() if k != "pred_te"}, default=str))
    return 0


def cmd_train_fold_jobs(args) -> int:
    settings = _settings(args)
    setup_logging(settings)
    from betatrend.nn.train import list_fold_jobs

    panel = _load_panel(args, settings)
    jobs = list_fold_jobs(len(panel), settings.strategy)
    text = json.dumps(jobs, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        console.print(f"wrote {args.out} ({len(jobs)} jobs)")
    else:
        console.print_json(text)
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
    p = argparse.ArgumentParser(
        prog="betatrend",
        description="BETA-TREND ETH timing (research/paper CLI; does not place signed orders)",
    )
    p.add_argument("--config", type=str, default=None)
    p.add_argument(
        "--mode",
        choices=["research", "paper", "testnet", "live"],
        default=None,
        help="Sets account.mode. paper/backtest still use the local broker.",
    )
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

    tn = sub.add_parser("train-nn", help="Walk-forward train the ETH GRPO decision net")
    tn.add_argument("--days", type=int, default=None)
    tn.add_argument("--demo", action="store_true")
    tn.add_argument("--path", type=str, default=None)
    tn.add_argument(
        "--jobs",
        type=int,
        default=None,
        help="Seeds trained in parallel per fold (1=sequential, <=0=one per seed)",
    )
    tn.add_argument(
        "--threads-per-job",
        type=int,
        default=None,
        help="torch threads inside each training process (0=auto split)",
    )
    tn.add_argument(
        "--resume",
        action="store_true",
        help="Continue from the latest complete fold checkpoint (warm-start + replay skipped OOS)",
    )
    tn.add_argument(
        "--start-fold",
        type=int,
        default=None,
        help="0-indexed fold to start training from; overrides auto-detect when set",
    )
    tn.set_defaults(func=cmd_train_nn)

    dash = sub.add_parser(
        "dashboard",
        help="TradingView analysis dashboard for a backtest report (fills + equity)",
    )
    dash.add_argument("--host", type=str, default="127.0.0.1")
    dash.add_argument("--port", type=int, default=8090)
    dash.add_argument("--report", type=str, default="eval_grpo_fold40")
    dash.set_defaults(func=cmd_dashboard)

    ex = sub.add_parser(
        "explore-nn",
        help="TorchExplorer board for a running or finished GRPO train-nn",
    )
    ex.add_argument(
        "--log",
        type=str,
        default="models/experiments/2026-08-24-grpo-retrain/train.log",
    )
    ex.add_argument("--models-dir", type=str, default="models")
    ex.add_argument(
        "--board-dir",
        type=str,
        default="models/experiments/2026-08-24-grpo-retrain/board",
    )
    ex.add_argument("--port", type=int, default=8080, help="TorchExplorer port")
    ex.add_argument("--board-port", type=int, default=8081, help="metrics board port")
    ex.add_argument("--poll", type=float, default=5.0)
    ex.set_defaults(func=cmd_explore_nn)

    tf = sub.add_parser("train-fold", help="Train one walk-forward fold × seed (distributed worker)")
    tf.add_argument("--days", type=int, default=None)
    tf.add_argument("--demo", action="store_true")
    tf.add_argument("--panel", type=str, default=None, help="Parquet/CSV panel; default loads via MarketDataStore")
    tf.add_argument("--fold", type=int, default=0)
    tf.add_argument("--seed", type=int, default=7)
    tf.add_argument("--job-json", type=str, default=None)
    tf.add_argument("--out", type=str, default=None)
    tf.add_argument("--init", type=str, default=None, help="Previous-fold checkpoint to warm-start")
    tf.set_defaults(func=cmd_train_fold)

    tj = sub.add_parser("train-fold-jobs", help="Print JSON list of independent fold+seed jobs")
    tj.add_argument("--days", type=int, default=None)
    tj.add_argument("--demo", action="store_true")
    tj.add_argument("--panel", type=str, default=None)
    tj.add_argument("--out", type=str, default=None)
    tj.set_defaults(func=cmd_train_fold_jobs)

    po = sub.add_parser("paper-once", help="Latest completed T+1 cycle (signal at t close, fill at t+1 open)")
    po.add_argument("--demo", action="store_true")
    po.add_argument("--execute", action="store_true", help="Fill locally at next open; overrides paper.dry_run")
    po.set_defaults(func=cmd_paper_once)

    pr = sub.add_parser("paper-run", help="Replay bars locally; persists last_reb / pending / EMA")
    pr.add_argument("--demo", action="store_true")
    pr.add_argument("--bars", type=int, default=24)
    pr.add_argument("--execute", action="store_true", help="Fill locally; overrides paper.dry_run")
    pr.add_argument("--reset-state", action="store_true")
    pr.set_defaults(func=cmd_paper_run)

    pg = sub.add_parser("ping", help="Public REST ping; --signed reads account only")
    pg.add_argument("--signed", action="store_true", help="HMAC GET /account; does not place orders")
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
