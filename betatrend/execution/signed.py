"""币安 USDT 本位永续的 HMAC 签名客户端。默认打 testnet。

密钥只从环境变量读（BINANCE_API_KEY / SECRET），不写进 YAML。
paper/research 模式拒绝签名下单；oms.testnet_only 时禁止主网。
"""
from __future__ import annotations

import hashlib
import hmac
import os
import time
from typing import Any
from urllib.parse import urlencode

import httpx
from loguru import logger

from betatrend.config import Settings


class BinanceSignedClient:
    def __init__(self, settings: Settings, testnet: bool | None = None):
        self.settings = settings
        if testnet is None:
            testnet = os.environ.get("BINANCE_TESTNET", "1") != "0"
        self.testnet = testnet
        base = settings.binance.testnet_futures_base if testnet else settings.binance.futures_base
        self.base = base.rstrip("/")
        self.key = os.environ.get("BINANCE_API_KEY", "")
        self.secret = os.environ.get("BINANCE_API_SECRET", "")
        self._http = httpx.Client(timeout=20.0, headers={"User-Agent": "BETA-TREND/0.1"})

    def close(self) -> None:
        self._http.close()

    def _sign(self, params: dict[str, Any]) -> str:
        """对 querystring 做 HMAC-SHA256，对应币安 SIGNED 接口。"""
        query = urlencode(params, doseq=True)
        return hmac.new(self.secret.encode(), query.encode(), hashlib.sha256).hexdigest()

    def _signed(self, method: str, path: str, params: dict[str, Any] | None = None) -> Any:
        """附加 timestamp、recvWindow、signature 后发请求。缺密钥立即失败。"""
        if not self.key or not self.secret:
            raise RuntimeError("BINANCE_API_KEY / BINANCE_API_SECRET missing")
        payload = dict(params or {})
        payload["timestamp"] = int(time.time() * 1000)
        payload["recvWindow"] = self.settings.binance.recv_window
        payload["signature"] = self._sign(payload)
        headers = {"X-MBX-APIKEY": self.key}
        url = f"{self.base}{path}"
        r = self._http.request(method, url, params=payload, headers=headers)
        r.raise_for_status()
        return r.json()

    def ping_account(self) -> dict:
        return self._signed("GET", "/fapi/v2/account")

    def positions(self) -> list[dict]:
        data = self._signed("GET", "/fapi/v2/positionRisk")
        return [p for p in data if abs(float(p.get("positionAmt") or 0)) > 0]

    def new_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        confirm: str = "",
        reduce_only: bool = False,
    ) -> dict:
        if self.settings.account.mode in ("research", "paper"):
            raise RuntimeError("Signed client refused: account.mode is paper")
        if not self.testnet and self.settings.oms.testnet_only:
            raise RuntimeError("OMS signed path is testnet-only")
        if self.settings.account.mode == "live":
            if os.environ.get("BETATREND_ALLOW_LIVE") != "1":
                raise RuntimeError("live trading requires BETATREND_ALLOW_LIVE=1")
            if confirm != "YES":
                raise RuntimeError("live trading requires confirm=YES")
        params = {
            "symbol": symbol.upper(),
            "side": side,
            "type": "MARKET",
            "quantity": f"{quantity:.8f}".rstrip("0").rstrip("."),
        }
        if reduce_only:
            params["reduceOnly"] = "true"
        logger.warning("Submitting {} {} {}", side, params["quantity"], symbol)
        return self._signed("POST", "/fapi/v1/order", params)
