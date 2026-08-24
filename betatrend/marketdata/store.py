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

# 旧 parquet 缺这些列会强制重拉，避免缓存把新特征永远填成 0。
PANEL_EXTRA_COLS = (
    "quote_volume",
    "trades",
    "taker_buy_base",
    "mark_close",
    "index_close",
    "open_interest",
    "long_short_ratio",
)


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
            if len(df) >= need * 0.9 and _panel_has_extras(df):
                return df.iloc[-need:].copy()

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=lookback_days + 5)
        start_ms, end_ms = _ms(start), _ms(end)
        with BinancePublicClient(self.settings) as client:
            klines = client.fetch_klines(symbol, interval, start_ms=start_ms, end_ms=end_ms)
            funding = client.fetch_funding(symbol, start_ms=start_ms, end_ms=end_ms)
            extras = _fetch_futures_extras(client, symbol, interval, start_ms, end_ms)
        if klines.empty:
            raise RuntimeError(f"No Binance klines for {symbol} (refusing silent demo)")
        df = klines.copy()
        if funding.empty:
            df["funding_rate"] = 0.0
        else:
            fund = funding.resample(interval).last()
            df = df.join(fund, how="left")
            df["funding_rate"] = df["funding_rate"].ffill().fillna(0.0)
        df = _join_extras(df, extras)
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


def _panel_has_extras(df: pd.DataFrame) -> bool:
    return all(col in df.columns for col in PANEL_EXTRA_COLS)


def _join_asof(left: pd.DataFrame, right: pd.DataFrame) -> pd.DataFrame:
    if right is None or right.empty:
        return left
    extra = right.copy()
    extra.index = pd.to_datetime(extra.index, utc=True)
    extra = extra.sort_index()
    extra = extra[~extra.index.duplicated(keep="last")]
    aligned = extra.reindex(left.index, method="ffill")
    return left.join(aligned, how="left")


def _fetch_futures_extras(
    client: BinancePublicClient,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
) -> dict[str, pd.DataFrame]:
    """标记/指数价有完整历史；OI 与多空比官方约 30 天。失败时跳过，不打断 K 线加载。"""
    extras: dict[str, pd.DataFrame] = {}
    fetches = (
        ("mark", lambda: client.fetch_mark_klines(symbol, interval, start_ms, end_ms)),
        ("index", lambda: client.fetch_index_klines(symbol, interval, start_ms, end_ms)),
        ("oi", lambda: client.fetch_open_interest_hist(symbol, interval, start_ms, end_ms)),
        ("lsr", lambda: client.fetch_global_lsr(symbol, interval, start_ms, end_ms)),
    )
    for name, fn in fetches:
        try:
            extras[name] = fn()
        except Exception as exc:
            logger.warning("optional futures series {} unavailable: {}", name, exc)
    return extras


def _join_extras(df: pd.DataFrame, extras: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = df
    for extra in extras.values():
        out = _join_asof(out, extra)
    close = out["close"].astype(float)
    if "mark_close" not in out.columns:
        out["mark_close"] = close
    else:
        out["mark_close"] = out["mark_close"].ffill().fillna(close)
    if "index_close" not in out.columns:
        out["index_close"] = close
    else:
        out["index_close"] = out["index_close"].ffill().fillna(close)
    if "open_interest" not in out.columns:
        out["open_interest"] = 0.0
    else:
        out["open_interest"] = out["open_interest"].ffill().fillna(0.0)
    if "long_short_ratio" not in out.columns:
        out["long_short_ratio"] = 1.0
    else:
        out["long_short_ratio"] = out["long_short_ratio"].ffill().fillna(1.0)
    for col in ("quote_volume", "trades", "taker_buy_base"):
        if col not in out.columns:
            out[col] = 0.0
        else:
            out[col] = out[col].astype(float).fillna(0.0)
    return out
