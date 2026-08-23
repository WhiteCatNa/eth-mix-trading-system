"""离线 ETH 面板：先涨后跌的两段趋势，供测试/demo 在没有网络时跑通链路。

前半段年化漂移 +80%，后半段 −80%，夹着噪声。时间序列动量在这段路径上
应该能做出正收益——这是合成数据的设计边，不是实盘承诺。
``symbols`` 里多出来的名字会被忽略：本系统只交易 ETH。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from betatrend.features import enrich_panel

ETH_PX0 = 2200.0


def make_trending_panels(
    n: int = 2000,
    seed: int = 7,
    start: str = "2023-01-01",
    symbols: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """只生成 ETH 的择时 DGP。``symbols`` 中的其它名字会被丢掉。"""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n, freq="h", tz="UTC")
    half = n // 2
    mu = np.concatenate(
        [np.full(half, 0.80 / (24 * 365)), np.full(n - half, -0.80 / (24 * 365))]
    )
    ret = mu + rng.normal(0.0, 0.004, size=n)
    px = ETH_PX0 * np.cumprod(1.0 + ret)
    fr = rng.normal(0.00003, 0.00005, size=n)
    open_px = np.concatenate([[px[0]], px[:-1]])
    high = np.maximum(open_px, px) * (1.0 + rng.random(n) * 0.001)
    low = np.minimum(open_px, px) * (1.0 - rng.random(n) * 0.001)
    df = pd.DataFrame(
        {
            "open": open_px,
            "high": high,
            "low": low,
            "close": px,
            "volume": rng.uniform(1e3, 1e4, size=n),
            "funding_rate": fr,
        },
        index=idx,
    )
    symbol = (symbols[0] if symbols else None) or "ETHUSDT"
    return {symbol: enrich_panel(df)}
