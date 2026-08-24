from __future__ import annotations

import threading
from urllib.parse import quote
from urllib.request import urlopen

from betatrend.dashboard.server import make_server


def _serve():
    httpd = make_server("127.0.0.1", 0, "eval_grpo_fold40")
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    return httpd, port


def test_dashboard_home_and_meta():
    httpd, port = _serve()
    try:
        html = urlopen(f"http://127.0.0.1:{port}/", timeout=10).read().decode()
        assert "lightweight-charts" in html
        assert "TradingView" in html
        meta = __import__("json").loads(urlopen(f"http://127.0.0.1:{port}/api/meta?report=eval_grpo_fold40", timeout=30).read())
        assert meta["symbol"] == "ETHUSDT"
        assert meta["tv_symbol"] == "BINANCE:ETHUSDT.P"
        assert meta["initial_capital"] == 100_000.0
        assert meta["n_fills"] == 6712
        assert any(p["id"] == "oos" for p in meta["presets"])
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_dashboard_ohlcv_and_fills_oos_window():
    httpd, port = _serve()
    try:
        start = "2022-07-30T21:00:00+00:00"
        end = "2022-08-21T06:00:00+00:00"
        q = f"report=eval_grpo_fold40&start={quote(start)}&end={quote(end)}"
        ohlc = __import__("json").loads(urlopen(f"http://127.0.0.1:{port}/api/ohlcv?{q}", timeout=30).read())
        fills = __import__("json").loads(urlopen(f"http://127.0.0.1:{port}/api/fills?{q}&all=1", timeout=30).read())
        eq = __import__("json").loads(urlopen(f"http://127.0.0.1:{port}/api/equity?{q}", timeout=30).read())
        assert len(ohlc["t"]) > 400
        assert len(ohlc["t"]) == len(ohlc["c"])
        assert fills["total"] >= 50
        assert fills["shown"] == fills["total"]
        assert fills["rows"][0]["side"] in {"BUY", "SELL"}
        assert len(eq["t"]) == len(eq["eq"])
        assert eq["eq"][0] > 0
    finally:
        httpd.shutdown()
        httpd.server_close()
