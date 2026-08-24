"""币安公开 REST（不需要密钥）：USDT 本位 K 线、资金费率、标记/指数价、OI、多空比。

带 429 重试。K 线按 startTime 分页直到耗尽。资金费率按小时 ffill 对齐到 K 线。
``/futures/data/*`` 统计接口官方只保留约 30 天，更早的 bar 只能填中性值。
"""
from __future__ import annotations

import time
from typing import Any

import httpx
import pandas as pd
from loguru import logger

from betatrend.config import Settings

INTERVAL_MS = {
    "1m": 60_000,
    "5m": 300_000,
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}

STATS_PERIODS = {"5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"}

KLINE_COLS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_volume",
    "trades",
    "taker_buy_base",
    "taker_buy_quote",
    "ignore",
]

KLINE_KEEP = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trades",
    "taker_buy_base",
]


class BinancePublicClient:
    def __init__(self, settings: Settings, timeout: float = 30.0, testnet: bool = False):
        self.settings = settings
        base = settings.binance.testnet_futures_base if testnet else settings.binance.futures_base
        self.base = base.rstrip("/")
        self._client = httpx.Client(timeout=timeout, headers={"User-Agent": "BETA-TREND/0.1"})

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "BinancePublicClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def ping(self) -> bool:
        r = self._client.get(f"{self.base}/fapi/v1/ping")
        r.raise_for_status()
        return True

    def _get(self, path: str, params: dict | None = None) -> Any:
        url = f"{self.base}{path}"
        for attempt in range(5):
            try:
                r = self._client.get(url, params=params or {})
                if r.status_code == 429:
                    time.sleep(1.0 * (attempt + 1))
                    continue
                r.raise_for_status()
                return r.json()
            except httpx.HTTPError as e:
                if attempt == 4:
                    raise
                logger.warning("HTTP {} retry {}", e, attempt + 1)
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"GET failed {url}")

    def _fetch_paged_klines(
        self,
        path: str,
        id_key: str,
        id_value: str,
        interval: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int = 1500,
    ) -> pd.DataFrame:
        rows: list[list] = []
        cursor = start_ms
        step = INTERVAL_MS.get(interval, 3_600_000)
        while True:
            params: dict[str, Any] = {
                id_key: id_value.upper(),
                "interval": interval,
                "limit": min(limit, 1500),
            }
            if cursor is not None:
                params["startTime"] = cursor
            if end_ms is not None:
                params["endTime"] = end_ms
            batch = self._get(path, params)
            if not batch:
                break
            rows.extend(batch)
            last_open = int(batch[-1][0])
            nxt = last_open + step
            if end_ms is not None and nxt >= end_ms:
                break
            if cursor is not None and nxt <= cursor:
                break
            if len(batch) < params["limit"]:
                break
            cursor = nxt
            time.sleep(0.05)
        if not rows:
            return pd.DataFrame(columns=KLINE_KEEP)
        df = pd.DataFrame(rows, columns=KLINE_COLS)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        for col in KLINE_KEEP:
            df[col] = df[col].astype(float)
        return df.drop_duplicates("open_time").sort_values("open_time").set_index("open_time")[KLINE_KEEP]

    def fetch_klines(
        self,
        symbol: str,
        interval: str = "1h",
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int = 1500,
    ) -> pd.DataFrame:
        return self._fetch_paged_klines(
            "/fapi/v1/klines", "symbol", symbol, interval, start_ms, end_ms, limit
        )

    def fetch_mark_klines(
        self,
        symbol: str,
        interval: str = "1h",
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> pd.DataFrame:
        raw = self._fetch_paged_klines(
            "/fapi/v1/markPriceKlines", "symbol", symbol, interval, start_ms, end_ms
        )
        if raw.empty:
            return pd.DataFrame(columns=["mark_close"])
        return raw[["close"]].rename(columns={"close": "mark_close"})

    def fetch_index_klines(
        self,
        symbol: str,
        interval: str = "1h",
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> pd.DataFrame:
        raw = self._fetch_paged_klines(
            "/fapi/v1/indexPriceKlines", "pair", symbol, interval, start_ms, end_ms
        )
        if raw.empty:
            return pd.DataFrame(columns=["index_close"])
        return raw[["close"]].rename(columns={"close": "index_close"})

    def fetch_funding(self, symbol: str, start_ms: int | None = None, end_ms: int | None = None) -> pd.DataFrame:
        rows: list[dict] = []
        cursor = start_ms
        while True:
            params: dict[str, Any] = {"symbol": symbol.upper(), "limit": 1000}
            if cursor is not None:
                params["startTime"] = cursor
            if end_ms is not None:
                params["endTime"] = end_ms
            batch = self._get("/fapi/v1/fundingRate", params)
            if not batch:
                break
            rows.extend(batch)
            last = int(batch[-1]["fundingTime"])
            nxt = last + 1
            if end_ms is not None and nxt >= end_ms:
                break
            if len(batch) < 1000:
                break
            if cursor is not None and nxt <= cursor:
                break
            cursor = nxt
            time.sleep(0.05)
        if not rows:
            return pd.DataFrame(columns=["funding_rate"])
        df = pd.DataFrame(rows)
        df["fundingTime"] = pd.to_datetime(df["fundingTime"].astype(int), unit="ms", utc=True)
        df["funding_rate"] = df["fundingRate"].astype(float)
        df = df.drop_duplicates("fundingTime").sort_values("fundingTime").set_index("fundingTime")
        return df[["funding_rate"]]

    def _fetch_paged_stats(
        self,
        path: str,
        symbol: str,
        period: str,
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> list[dict]:
        # 官方只保留约 30 天；窗口过大直接 400。
        max_span_ms = 30 * 24 * 3600 * 1000
        if start_ms is not None and end_ms is not None and end_ms - start_ms > max_span_ms:
            start_ms = end_ms - max_span_ms
        rows: list[dict] = []
        cursor = start_ms
        while True:
            params: dict[str, Any] = {
                "symbol": symbol.upper(),
                "period": period,
                "limit": 500,
            }
            if cursor is not None:
                params["startTime"] = cursor
            if end_ms is not None:
                params["endTime"] = end_ms
            batch = self._get(path, params)
            if not batch or not isinstance(batch, list):
                break
            rows.extend(batch)
            stamps = [int(item["timestamp"]) for item in batch if "timestamp" in item]
            if not stamps:
                break
            nxt = max(stamps) + 1
            if end_ms is not None and nxt >= end_ms:
                break
            if len(batch) < 500:
                break
            if cursor is not None and nxt <= cursor:
                break
            cursor = nxt
            time.sleep(0.15)
        return rows

    def fetch_open_interest_hist(
        self,
        symbol: str,
        period: str = "1h",
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> pd.DataFrame:
        """``/futures/data/openInterestHist``：合约持仓量。官方约 30 天。"""
        period = period if period in STATS_PERIODS else "1h"
        rows = self._fetch_paged_stats(
            "/futures/data/openInterestHist", symbol, period, start_ms, end_ms
        )
        if not rows:
            return pd.DataFrame(columns=["open_interest"])
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
        oi_col = "sumOpenInterest" if "sumOpenInterest" in df.columns else next(
            c for c in df.columns if c.lower() == "sumopeninterest"
        )
        df["open_interest"] = df[oi_col].astype(float)
        df = df.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
        return df[["open_interest"]]

    def fetch_global_lsr(
        self,
        symbol: str,
        period: str = "1h",
        start_ms: int | None = None,
        end_ms: int | None = None,
    ) -> pd.DataFrame:
        """``/futures/data/globalLongShortAccountRatio``：全市场多空人数比。官方约 30 天。"""
        period = period if period in STATS_PERIODS else "1h"
        rows = self._fetch_paged_stats(
            "/futures/data/globalLongShortAccountRatio", symbol, period, start_ms, end_ms
        )
        if not rows:
            return pd.DataFrame(columns=["long_short_ratio"])
        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
        df["long_short_ratio"] = df["longShortRatio"].astype(float)
        df = df.drop_duplicates("timestamp").sort_values("timestamp").set_index("timestamp")
        return df[["long_short_ratio"]]
