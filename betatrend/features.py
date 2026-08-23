"""ETH 面板特征：收益与实现波动率。滚动统计只看截至当前 bar 的历史。

仓位由决策网输出；这里只提供反波动杠杆所需的 σ。
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from betatrend.mathx import realized_vol


def enrich_panel(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index, utc=True)
    out = out.sort_index()
    out["ret"] = out["close"].astype(float).pct_change().fillna(0.0)
    if "funding_rate" not in out.columns:
        out["funding_rate"] = 0.0
    out["funding_apr"] = out["funding_rate"].astype(float) * 3.0 * 365.0
    return out


@dataclass
class FeatureSet:
    vols: dict[str, float]
    market_vol: float


def compute_features(
    panels: dict[str, pd.DataFrame],
    market_symbol: str,
    *,
    vol_lookback: int,
) -> FeatureSet:
    if market_symbol not in panels:
        raise ValueError(f"Market symbol {market_symbol} missing from panels")
    mkt = panels[market_symbol]
    mkt_ret = mkt["ret"].astype(float).values
    market_vol = realized_vol(mkt_ret, vol_lookback)

    vols: dict[str, float] = {}
    for sym, panel in panels.items():
        ret = panel["ret"].astype(float).values
        vols[sym] = realized_vol(ret, vol_lookback)
    return FeatureSet(vols=vols, market_vol=market_vol)
