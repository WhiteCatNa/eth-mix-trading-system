"""Parquet 缓存 + 面板拼装。禁止在拉不到真实行情时偷偷换成 demo 数据。

``force_demo=True`` 才用合成路径；真实行情为空直接报错，避免研究报告混进假数据。
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from loguru import logger

from betatrend.config import Settings
from betatrend.features import enrich_panel
from betatrend.marketdata import BinancePublicClient
from betatrend.marketdata.synthetic import make_trending_panels


def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


class MarketDataStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cache_dir = settings.cache_path()

    def _path(self, symbol: str, interval: str, demo: bool) -> Path:
        tag = "demo" if demo else "binance"
        return self.cache_dir / f"panel_{tag}_{symbol}_{interval}.parquet"

    def _need_bars(self, lookback_days: int, interval: str) -> int:
        iv = (interval or "1h").lower()
        if iv.endswith("h"):
            return int(lookback_days * 24 / int(iv[:-1] or 1)) + 8
        if iv.endswith("m"):
            return int(lookback_days * 24 * 60 / int(iv[:-1] or 1)) + 8
        return int(lookback_days) + 8

    def load_symbol(
        self,
        symbol: str,
        lookback_days: int | None = None,
        interval: str | None = None,
        use_cache: bool = True,
        force_demo: bool = False,
        refresh: bool = False,
    ) -> pd.DataFrame:
        interval = interval or self.settings.data.kline_interval
        lookback_days = lookback_days or self.settings.data.lookback_days
        need = self._need_bars(lookback_days, interval)
        path = self._path(symbol, interval, demo=force_demo)

        if force_demo:
            n = min(max(need + 50, 800), 2000)
            panels = make_trending_panels(n=n, symbols=[symbol])
            df = panels[symbol]
            df.to_parquet(path)
            return df.iloc[-need:].copy() if len(df) > need else df.copy()

        if use_cache and not refresh and path.exists():
            df = pd.read_parquet(path)
            df.index = pd.to_datetime(df.index, utc=True)
            if len(df) >= need * 0.9:
                return df.iloc[-need:].copy()

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=lookback_days + 5)
        with BinancePublicClient(self.settings) as client:
            klines = client.fetch_klines(symbol, interval, start_ms=_ms(start), end_ms=_ms(end))
            funding = client.fetch_funding(symbol, start_ms=_ms(start), end_ms=_ms(end))
        if klines.empty:
            raise RuntimeError(f"No Binance klines for {symbol} (refusing silent demo)")
        df = klines.copy()
        if funding.empty:
            df["funding_rate"] = 0.0
        else:
            fund = funding.resample(interval).last()
            df = df.join(fund, how="left")
            df["funding_rate"] = df["funding_rate"].ffill().fillna(0.0)
        df = enrich_panel(df)
        df.to_parquet(path)
        logger.info("Cached {} bars for {}", len(df), symbol)
        return df.iloc[-need:].copy()

    def load_universe(
        self,
        lookback_days: int | None = None,
        force_demo: bool = False,
        refresh: bool = False,
    ) -> dict[str, pd.DataFrame]:
        symbol = self.settings.universe.symbol
        if force_demo:
            lookback_days = lookback_days or self.settings.data.lookback_days
            interval = self.settings.data.kline_interval
            need = self._need_bars(lookback_days, interval)
            n = min(max(need, self.settings.backtest.warmup_bars + 400), 2000)
            panels = make_trending_panels(n=n, symbols=[symbol])
            for s, df in panels.items():
                df.to_parquet(self._path(s, interval, demo=True))
            return panels
        df = self.load_symbol(symbol, lookback_days=lookback_days, force_demo=False, refresh=refresh)
        return {symbol: df}
