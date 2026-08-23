"""K 线回测：t 收盘出信号，t+1 开盘成交。禁止未来函数。

一根 bar 内顺序：
  1. 隔夜跳空：旧仓从上一根收盘盯到本根开盘
  2. 整点命中资金费周期则按开盘价结算资金费
  3. 兑现上一根收盘挂出的 pending 目标（开盘价 + 手续费 + 滑点）
  4. 新仓从开盘盯到收盘
  5. 若到再平衡点，用截至本 bar 收盘的历史算下一拍目标
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from loguru import logger

from betatrend.config import Settings
from betatrend.domain import Fill, MarketSnapshot
from betatrend.ledger import Ledger
from betatrend.mathx import annualized_return, calmar_ratio, max_drawdown, sharpe_ratio, sortino_ratio
from betatrend.pipeline import DeskCycle


@dataclass
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    positions: pd.DataFrame
    fills: list[Fill]
    funding_pnl: pd.Series
    price_pnl: pd.Series
    fees_paid: pd.Series
    metrics: dict
    targets_log: pd.DataFrame
    meta: dict = field(default_factory=dict)


class Backtester:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.desk = DeskCycle(settings)

    def run(self, panels: dict[str, pd.DataFrame]) -> BacktestResult:
        if not panels:
            raise ValueError("No panels")
        common = None
        for df in panels.values():
            common = df.index if common is None else common.intersection(df.index)
        common = common.sort_values()
        warmup = self.settings.backtest.warmup_bars
        if len(common) < warmup + 20:
            raise ValueError(f"Not enough bars: {len(common)}")
        panels = {s: df.loc[common].copy() for s, df in panels.items()}
        symbols = list(panels.keys())
        market = self.settings.universe.market_symbol
        if market not in panels:
            raise ValueError(f"Market {market} not in panels")

        equity0 = float(self.settings.account.initial_capital)
        ledger = Ledger(cash=equity0, qty={s: 0.0 for s in symbols})
        self.desk.reset(equity0)

        fills: list[Fill] = []
        equity_rows: list[tuple] = []
        funding_rows: list[tuple] = []
        price_rows: list[tuple] = []
        fee_rows: list[tuple] = []
        pos_rows: list[dict] = []
        target_rows: list[dict] = []

        pending: dict[str, float] | None = None
        last_reb = -10**9
        reb_h = max(int(self.settings.strategy.rebalance_hours), 8)
        fee_entry = (
            self.settings.fees.maker if self.settings.fees.use_maker_for_entries else self.settings.fees.taker
        )
        slip_bps = self.settings.slippage.market_bps / 10_000.0
        fund_h = self.settings.data.funding_interval_hours

        for i, ts in enumerate(common):
            close_px = {s: float(panels[s].loc[ts, "close"]) for s in symbols}
            open_px = {s: float(panels[s].loc[ts, "open"]) for s in symbols}
            fund_rate = {s: float(panels[s].loc[ts, "funding_rate"]) for s in symbols}

            price_pnl = 0.0
            if i > 0:
                prev_ts = common[i - 1]
                prev_close = {s: float(panels[s].loc[prev_ts, "close"]) for s in symbols}
                for s in symbols:
                    price_pnl += ledger.apply_mark(s, open_px[s] - prev_close[s])

            funding_pnl = 0.0
            if ts.hour % fund_h == 0:
                for s in symbols:
                    funding_pnl += ledger.apply_funding(s, open_px[s], fund_rate[s])

            fees_bar = 0.0
            if pending is not None and i >= warmup:
                for s in symbols:
                    px = open_px[s]
                    cur_n = ledger.notional(s, px)
                    tgt = pending.get(s, 0.0)
                    d_n = tgt - cur_n
                    band = max(ledger.cash * self.settings.backtest.turnover_band_equity, 75.0)
                    if abs(d_n) < band or px <= 0:
                        continue
                    fee = abs(d_n) * fee_entry
                    slip = abs(d_n) * slip_bps
                    qty_delta = d_n / px
                    fill = Fill(ts, s, d_n, qty_delta, px, fee, slip, "rebalance")
                    ledger.apply_fill(fill)
                    fills.append(fill)
                    fees_bar += fee + slip
                pending = None

            for s in symbols:
                price_pnl += ledger.apply_mark(s, close_px[s] - open_px[s])

            equity = ledger.cash
            current_n = {s: ledger.notional(s, close_px[s]) for s in symbols}

            do_reb = (i - last_reb) >= reb_h
            if i >= warmup and do_reb:
                hist = {s: panels[s].iloc[: i + 1] for s in symbols}
                snap = MarketSnapshot(
                    timestamp=ts,
                    panels=hist,
                    prices=close_px,
                    equity=max(equity, 1.0),
                    bar_index=i,
                    market_symbol=market,
                )
                cycle = self.desk.run(snap, current_n, reason="rebalance")
                pending = dict(cycle.notionals)
                last_reb = i
                for t in cycle.targets:
                    target_rows.append(
                        {
                            "timestamp": ts,
                            "symbol": t.symbol,
                            "target_notional": t.target_notional,
                            "signal": t.signal,
                            "trend_score": t.trend_score,
                            "reason": t.reason,
                            "side": t.extras.get("side"),
                            "unit": t.extras.get("unit"),
                            "decision": t.extras.get("decision"),
                        }
                    )

            equity_rows.append((ts, equity))
            funding_rows.append((ts, funding_pnl))
            price_rows.append((ts, price_pnl))
            fee_rows.append((ts, fees_bar))
            row = {"timestamp": ts, "equity": equity}
            for s in symbols:
                row[s] = ledger.notional(s, close_px[s])
            pos_rows.append(row)

        equity_curve = pd.Series({t: e for t, e in equity_rows}, name="equity")
        equity_curve.index = pd.to_datetime(equity_curve.index, utc=True)
        returns = equity_curve.pct_change().fillna(0.0)
        metrics = {
            "final_equity": float(equity_curve.iloc[-1]),
            "total_return": float(equity_curve.iloc[-1] / equity0 - 1.0),
            "ann_return": annualized_return(equity_curve),
            "sharpe": sharpe_ratio(returns),
            "sortino": sortino_ratio(returns),
            "max_drawdown": max_drawdown(equity_curve),
            "n_fills": len(fills),
            "fees_total": float(sum(f.fee + f.slippage for f in fills)),
            "funding_total": float(sum(v for _, v in funding_rows)),
            "price_pnl_total": float(sum(v for _, v in price_rows)),
        }
        metrics["calmar"] = calmar_ratio(metrics["ann_return"], metrics["max_drawdown"])
        logger.info(
            "Backtest | eq={:.2f} ret={:.2%} sharpe={:.2f} mdd={:.2%} fills={}",
            metrics["final_equity"],
            metrics["total_return"],
            metrics["sharpe"],
            metrics["max_drawdown"],
            metrics["n_fills"],
        )
        return BacktestResult(
            equity_curve=equity_curve,
            returns=returns,
            positions=pd.DataFrame(pos_rows).set_index("timestamp"),
            fills=fills,
            funding_pnl=pd.Series({t: v for t, v in funding_rows}),
            price_pnl=pd.Series({t: v for t, v in price_rows}),
            fees_paid=pd.Series({t: v for t, v in fee_rows}),
            metrics=metrics,
            targets_log=pd.DataFrame(target_rows) if target_rows else pd.DataFrame(),
            meta={"warmup_bars": warmup, "rebalance_hours": reb_h, "market": market},
        )
