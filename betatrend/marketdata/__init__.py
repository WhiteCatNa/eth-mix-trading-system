"""币安公开 REST（不需要密钥）：拉 USDT 本位 K 线与资金费率。

带 429 重试。K 线按 startTime 分页直到耗尽。资金费率按小时 ffill 对齐到 K 线。
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

    def fetch_klines(
        self,
        symbol: str,
        interval: str = "1h",
        start_ms: int | None = None,
        end_ms: int | None = None,
        limit: int = 1500,
    ) -> pd.DataFrame:
        rows: list[list] = []
        cursor = start_ms
        step = INTERVAL_MS.get(interval, 3_600_000)
        while True:
            params: dict[str, Any] = {
                "symbol": symbol.upper(),
                "interval": interval,
                "limit": min(limit, 1500),
            }
            if cursor is not None:
                params["startTime"] = cursor
            if end_ms is not None:
                params["endTime"] = end_ms
            batch = self._get("/fapi/v1/klines", params)
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
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df = pd.DataFrame(rows, columns=KLINE_COLS)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        df = df.drop_duplicates("open_time").sort_values("open_time").set_index("open_time")
        return df[["open", "high", "low", "close", "volume"]]

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
