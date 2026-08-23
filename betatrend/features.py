"""ETH 面板特征：收益、TSMOM、实现波动率。滚动统计只看截至当前 bar 的历史。"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from betatrend.mathx import realized_vol, tsmom_score


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
    own_scores: dict[str, float]
    market_score: float
    market_vol: float


def compute_features(
    panels: dict[str, pd.DataFrame],
    market_symbol: str,
    *,
    vol_lookback: int,
    lookbacks: list[int],
    weights: list[float],
    skip_hours: int,
) -> FeatureSet:
    if market_symbol not in panels:
        raise ValueError(f"Market symbol {market_symbol} missing from panels")
    mkt = panels[market_symbol]
    mkt_close = mkt["close"].astype(float).values
    mkt_ret = mkt["ret"].astype(float).values
    market_score = tsmom_score(mkt_close, mkt_ret, lookbacks, weights, skip_hours)
    market_vol = realized_vol(mkt_ret, vol_lookback)

    vols: dict[str, float] = {}
    own_scores: dict[str, float] = {}
    for sym, panel in panels.items():
        close = panel["close"].astype(float).values
        ret = panel["ret"].astype(float).values
        vols[sym] = realized_vol(ret, vol_lookback)
        own_scores[sym] = tsmom_score(close, ret, lookbacks, weights, skip_hours)
    return FeatureSet(
        vols=vols,
        own_scores=own_scores,
        market_score=market_score,
        market_vol=market_vol,
    )
