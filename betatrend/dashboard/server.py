"""本地分析看板：K 线 / 成交 / 净值走 TradingView Lightweight Charts。

只读 ``reports/`` 与行情缓存，不碰训练进程、不下单。
"""
from __future__ import annotations

import gzip
import json
import re
import threading
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pandas as pd
from loguru import logger

from betatrend.config import ROOT

STATIC = Path(__file__).resolve().parent / "static"
REPORT_OK = re.compile(r"^[A-Za-z0-9_-]+$")
PANEL = ROOT / "data" / "cache" / "panel_binance_ETHUSDT_1h.parquet"
MAX_MARKERS = 800
WIDE_DAYS = 60


def _unix(ts) -> int:
    t = pd.Timestamp(ts)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")
    return int(t.timestamp())


def _parse_ts(raw: str | None) -> pd.Timestamp | None:
    if not raw:
        return None
    t = pd.Timestamp(raw)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    return t.tz_convert("UTC")


def list_reports() -> list[dict[str, Any]]:
    out = []
    folder = ROOT / "reports"
    if not folder.exists():
        return out
    for metrics in sorted(folder.glob("*_metrics.json")):
        name = metrics.name[: -len("_metrics.json")]
        row = {
            "name": name,
            "metrics": str(metrics.relative_to(ROOT)),
            "equity": (folder / f"{name}_equity.csv").exists(),
            "fills": (folder / f"{name}_fills.csv").exists(),
            "targets": (folder / f"{name}_targets.csv").exists(),
        }
        out.append(row)
    return out


@lru_cache(maxsize=4)
def _panel() -> pd.DataFrame:
    df = pd.read_parquet(PANEL)
    df.index = pd.to_datetime(df.index, utc=True)
    return df.sort_index()


@lru_cache(maxsize=8)
def _equity(name: str) -> pd.Series:
    path = ROOT / "reports" / f"{name}_equity.csv"
    s = pd.read_csv(path, index_col=0, parse_dates=True).squeeze("columns")
    s.index = pd.to_datetime(s.index, utc=True)
    return s.sort_index().astype(float)


@lru_cache(maxsize=8)
def _fills(name: str) -> pd.DataFrame:
    path = ROOT / "reports" / f"{name}_fills.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, parse_dates=["fill_ts"])
    df["fill_ts"] = pd.to_datetime(df["fill_ts"], utc=True)
    return df.sort_values("fill_ts")


@lru_cache(maxsize=8)
def _metrics(name: str) -> dict[str, Any]:
    path = ROOT / "reports" / f"{name}_metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=8)
def _oos(name: str) -> dict[str, Any] | None:
    path = ROOT / "reports" / f"{name}_oos.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _slice(index: pd.DatetimeIndex, start: pd.Timestamp | None, end: pd.Timestamp | None):
    mask = pd.Series(True, index=index)
    if start is not None:
        mask &= index >= start
    if end is not None:
        mask &= index <= end
    return mask


def _ohlcv_payload(start: pd.Timestamp | None, end: pd.Timestamp | None) -> dict[str, Any]:
    df = _panel()
    mask = _slice(df.index, start, end)
    part = df.loc[mask]
    return {
        "t": [_unix(ts) for ts in part.index],
        "o": part["open"].astype(float).tolist(),
        "h": part["high"].astype(float).tolist(),
        "l": part["low"].astype(float).tolist(),
        "c": part["close"].astype(float).tolist(),
        "v": part["volume"].astype(float).tolist(),
    }


def _equity_payload(name: str, start: pd.Timestamp | None, end: pd.Timestamp | None) -> dict[str, Any]:
    s = _equity(name)
    mask = _slice(s.index, start, end)
    part = s.loc[mask]
    return {"t": [_unix(ts) for ts in part.index], "eq": part.astype(float).tolist()}


def _fill_row(r: pd.Series) -> dict[str, Any]:
    return {
        "n": int(r["n"]) if "n" in r and pd.notna(r["n"]) else 0,
        "t": _unix(r["fill_ts"]),
        "side": str(r.get("buy_sell", "")),
        "action": str(r.get("action", "")),
        "px": float(r["price"]),
        "qty": float(r["qty"]),
        "net": float(r.get("net", 0.0) or 0.0),
        "unit": float(r["unit"]) if "unit" in r and pd.notna(r["unit"]) else None,
        "equity": float(r["equity"]) if "equity" in r and pd.notna(r["equity"]) else None,
    }


def _fills_payload(
    name: str,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
    *,
    want_all: bool,
) -> dict[str, Any]:
    df = _fills(name)
    if df.empty:
        return {"rows": [], "total": 0, "shown": 0, "truncated": False}
    part = df
    if start is not None:
        part = part[part["fill_ts"] >= start]
    if end is not None:
        part = part[part["fill_ts"] <= end]
    total = int(len(part))
    truncated = False
    if not want_all and total > MAX_MARKERS:
        span_days = 0.0
        if start is not None and end is not None:
            span_days = (end - start).total_seconds() / 86400.0
        if span_days > WIDE_DAYS or (start is None and end is None):
            ranked = part.reindex(part["net"].abs().sort_values(ascending=False).index)
            keep = ranked.head(MAX_MARKERS).sort_values("fill_ts")
            part = keep
            truncated = True
    rows = [_fill_row(r) for _, r in part.iterrows()]
    return {"rows": rows, "total": total, "shown": len(rows), "truncated": truncated}


def _yearly(name: str) -> list[dict[str, Any]]:
    df = _fills(name)
    if df.empty:
        return []
    rows = []
    for year, g in df.groupby(df["fill_ts"].dt.year):
        rows.append(
            {
                "year": int(year),
                "n": int(len(g)),
                "buy": int((g["buy_sell"] == "BUY").sum()),
                "sell": int((g["buy_sell"] == "SELL").sum()),
                "net": float(g["net"].sum()) if "net" in g.columns else 0.0,
            }
        )
    return rows


def report_meta(name: str) -> dict[str, Any]:
    if not REPORT_OK.match(name):
        raise ValueError("bad report name")
    metrics = _metrics(name)
    oos = _oos(name)
    fills = _fills(name)
    panel = _panel()
    eq = _equity(name) if (ROOT / "reports" / f"{name}_equity.csv").exists() else None
    presets = [
        {"id": "full", "label": "全样本", "start": _unix(panel.index[0]), "end": _unix(panel.index[-1])},
    ]
    if oos and oos.get("test_start") and oos.get("test_end"):
        presets.insert(
            0,
            {
                "id": "oos",
                "label": "21d OOS",
                "start": _unix(oos["test_start"]),
                "end": _unix(oos["test_end"]) + 8 * 3600,
            },
        )
    if eq is not None and len(eq):
        years = sorted(set(eq.index.year))
        for y in years:
            sl = eq[eq.index.year == y]
            if len(sl) < 24:
                continue
            presets.append(
                {
                    "id": f"y{y}",
                    "label": str(y),
                    "start": _unix(sl.index[0]),
                    "end": _unix(sl.index[-1]),
                }
            )
    return {
        "report": name,
        "symbol": "ETHUSDT",
        "tv_symbol": "BINANCE:ETHUSDT.P",
        "interval": "60",
        "initial_capital": 100_000.0,
        "metrics": metrics,
        "oos": oos,
        "n_fills": int(len(fills)),
        "panel_start": str(panel.index[0]),
        "panel_end": str(panel.index[-1]),
        "equity_start": str(eq.index[0]) if eq is not None and len(eq) else None,
        "equity_end": str(eq.index[-1]) if eq is not None and len(eq) else None,
        "presets": presets,
        "yearly": _yearly(name),
        "reports": list_reports(),
        "note": "回测 ≠ 实盘。全样本净值含训练窗，21d OOS 才是该折的诚实窗口。",
    }


def _json_bytes(payload: Any) -> bytes:
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")


class DashboardHandler(BaseHTTPRequestHandler):
    report: str = "eval_grpo_fold40"

    def log_message(self, fmt: str, *args) -> None:
        logger.debug("dashboard " + fmt, *args)

    def _send(self, code: int, body: bytes, content_type: str, gzip_ok: bool) -> None:
        headers = {"Content-Type": content_type, "Cache-Control": "no-store"}
        if gzip_ok and len(body) > 1024:
            body = gzip.compress(body, compresslevel=5)
            headers["Content-Encoding"] = "gzip"
        self.send_response(code)
        for k, v in headers.items():
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)
        gzip_ok = "gzip" in (self.headers.get("Accept-Encoding") or "")
        report = qs.get("report", [self.report])[0]
        if not REPORT_OK.match(report):
            self._send(400, b'{"error":"bad report"}', "application/json", False)
            return
        try:
            if path in ("/", "/index.html"):
                html = (STATIC / "index.html").read_bytes()
                self._send(200, html, "text/html; charset=utf-8", gzip_ok)
                return
            if path == "/api/meta":
                self._send(200, _json_bytes(report_meta(report)), "application/json", gzip_ok)
                return
            if path == "/api/reports":
                self._send(200, _json_bytes(list_reports()), "application/json", gzip_ok)
                return
            start = _parse_ts(qs.get("start", [None])[0])
            end = _parse_ts(qs.get("end", [None])[0])
            if path == "/api/ohlcv":
                self._send(200, _json_bytes(_ohlcv_payload(start, end)), "application/json", gzip_ok)
                return
            if path == "/api/equity":
                self._send(200, _json_bytes(_equity_payload(report, start, end)), "application/json", gzip_ok)
                return
            if path == "/api/fills":
                want_all = qs.get("all", ["0"])[0] in {"1", "true", "yes"}
                self._send(200, _json_bytes(_fills_payload(report, start, end, want_all=want_all)), "application/json", gzip_ok)
                return
            self._send(404, b'{"error":"not found"}', "application/json", False)
        except FileNotFoundError as exc:
            self._send(404, _json_bytes({"error": str(exc)}), "application/json", False)
        except Exception as exc:  # noqa: BLE001 — HTTP handler must not crash the thread
            logger.exception("dashboard GET {}", path)
            self._send(500, _json_bytes({"error": type(exc).__name__, "detail": str(exc)}), "application/json", False)


def make_server(host: str, port: int, report: str) -> ThreadingHTTPServer:
    if not REPORT_OK.match(report):
        raise ValueError(f"bad report name: {report}")
    DashboardHandler.report = report
    httpd = ThreadingHTTPServer((host, port), DashboardHandler)
    return httpd


def serve(host: str = "127.0.0.1", port: int = 8090, report: str = "eval_grpo_fold40") -> None:
    httpd = make_server(host, port, report)
    url = f"http://{host}:{port}/?report={report}"
    logger.info("analysis dashboard {}", url)
    print(url, flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("dashboard stopped")
    finally:
        httpd.server_close()


def serve_background(host: str = "127.0.0.1", port: int = 8090, report: str = "eval_grpo_fold40") -> ThreadingHTTPServer:
    httpd = make_server(host, port, report)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd
