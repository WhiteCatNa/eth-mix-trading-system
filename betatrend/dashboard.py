"""Streamlit 交易台：最近一次回测曲线 + 当前特征快照。

只读展示，不发单。点 Run backtest 会走与 CLI 相同的 ``run_backtest``。
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from betatrend.config import ROOT, load_settings
from betatrend.marketdata.store import MarketDataStore
from betatrend.pipeline import DeskCycle
from betatrend.domain import MarketSnapshot
from betatrend.research import run_backtest

st.set_page_config(page_title="BETA-TREND Desk", layout="wide")
settings = load_settings()
st.title("ETH Timing Desk")
st.caption("Single-name TSMOM, continuous overlay. Not investment advice.")

col1, col2, col3 = st.columns(3)
with col1:
    demo = st.checkbox("Use synthetic demo data", value=True)
with col2:
    run_bt = st.button("Run backtest")
with col3:
    st.write(f"Mode: `{settings.account.mode}` · trade: `{settings.universe.trade_symbol}`")

if run_bt:
    with st.spinner("Running desk backtest…"):
        result = run_backtest(settings, force_demo=demo, name="dashboard")
    st.session_state["result"] = result

result = st.session_state.get("result")
reports = sorted((ROOT / "reports").glob("*_metrics.json"), reverse=True)

if result is None and reports:
    st.info("Load a saved run from reports/ or click Run backtest.")

if result is not None:
    m = result.metrics
    a, b, c, d = st.columns(4)
    a.metric("Total return", f"{m['total_return']:.2%}")
    b.metric("Sharpe", f"{m['sharpe']:.2f}")
    c.metric("Max DD", f"{m['max_drawdown']:.2%}")
    d.metric("Beta PnL / Residual", f"{m['beta_pnl_total']:.0f} / {m['residual_pnl_total']:.0f}")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=result.equity_curve.index, y=result.equity_curve.values, name="equity"))
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Attribution")
        att = pd.DataFrame(
            {
                "beta": result.beta_pnl.cumsum(),
                "residual": result.residual_pnl.cumsum(),
                "funding": result.funding_pnl.cumsum(),
            }
        )
        st.line_chart(att)
    with c2:
        st.subheader("Risk")
        st.line_chart(result.risk_log[["drawdown", "risk_scalar"]])
    if not result.targets_log.empty:
        st.subheader("Last targets")
        last_ts = result.targets_log["timestamp"].iloc[-1]
        st.dataframe(result.targets_log[result.targets_log["timestamp"] == last_ts], use_container_width=True)

st.divider()
st.subheader("Current feature snapshot")
if st.button("Compute snapshot"):
    store = MarketDataStore(settings)
    panels = store.load_universe(force_demo=demo, lookback_days=min(settings.data.lookback_days, 400))
    prices = {s: float(df["close"].iloc[-1]) for s, df in panels.items()}
    ts = next(iter(panels.values())).index[-1]
    desk = DeskCycle(settings)
    snap = MarketSnapshot(
        timestamp=pd.Timestamp(ts),
        panels=panels,
        prices=prices,
        equity=settings.account.initial_capital,
        bar_index=len(next(iter(panels.values()))) - 1,
        market_symbol=settings.universe.market_symbol,
    )
    cycle = desk.run(snap, {s: 0.0 for s in panels})
    st.write(
        {
            "market_score": cycle.features.market_score,
            "market_vol": cycle.features.market_vol,
            "betas": cycle.features.betas,
            "clipped": cycle.clipped,
            "messages": cycle.messages,
        }
    )
