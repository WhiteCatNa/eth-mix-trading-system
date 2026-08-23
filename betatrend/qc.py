"""行情质检：缺口、OHLC 合法性、跳价、资金费过期、可选时钟漂移。

QC 失败时 DeskCycle 选择 flatten 而不是“带着坏数据继续交易”。
回测历史数据的最后一根相对“现在”必然过期，因此 check_clock 默认关闭；
只有漂移在 2 天内（像实时馈送）或显式打开时钟检查时才报 clock drift。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from betatrend.config import Settings


@dataclass
class QcReport:
    ok: bool  # False → 调用方应 flatten
    messages: list[str] = field(default_factory=list)


def _interval_hours(settings: Settings) -> float:
    """把 ``1h`` / ``15m`` 这类 interval 字符串换成小时数，用于缺口阈值。"""
    raw = str(settings.data.kline_interval).lower().strip()
    if raw.endswith("h"):
        return float(raw[:-1] or 1)
    if raw.endswith("m"):
        return float(raw[:-1] or 60) / 60.0
    return 1.0


def inspect_panels(
    panels: dict[str, pd.DataFrame],
    settings: Settings,
    *,
    now=None,
) -> QcReport:
    """检查每个 symbol 最近 48 根（足够抓到缺口/跳价，又不会被很久以前的脏点误杀）。"""
    cfg = settings.qc
    messages: list[str] = []
    if not cfg.enabled:
        return QcReport(ok=True, messages=["qc disabled"])
    if not panels:
        return QcReport(ok=False, messages=["qc: no panels"])

    step_h = _interval_hours(settings)
    max_gap = pd.Timedelta(hours=float(cfg.max_gap_hours))
    expected = pd.Timedelta(hours=step_h)
    window = 48

    for symbol, df in panels.items():
        if df is None or df.empty:
            messages.append(f"qc: {symbol} empty")
            continue
        need = {"open", "high", "low", "close"}
        missing = need - set(df.columns)
        if missing:
            messages.append(f"qc: {symbol} missing columns {sorted(missing)}")
            continue
        tail = df.iloc[-window:].copy()
        tail.index = pd.to_datetime(tail.index, utc=True)
        if tail.index.has_duplicates:
            messages.append(f"qc: {symbol} duplicate timestamps")
        diffs = tail.index.to_series().diff().dropna()
        if not diffs.empty:
            worst = diffs.max()
            # 允许比理论间隔再宽 30 分钟，消化夏令时/偶发延迟；但仍受 max_gap_hours 硬帽。
            allowed = min(max_gap, expected + pd.Timedelta(minutes=30))
            if worst > allowed:
                messages.append(f"qc: {symbol} gap {worst} > {allowed}")

        o = tail["open"].astype(float)
        h = tail["high"].astype(float)
        low = tail["low"].astype(float)
        c = tail["close"].astype(float)
        # 合法 K 线：high ≥ open/close/low，low ≤ open/close，价格 > 0。
        bad_ohlc = (
            (h + 1e-12 < o).any()
            or (h + 1e-12 < c).any()
            or (low - 1e-12 > o).any()
            or (low - 1e-12 > c).any()
            or (h + 1e-12 < low).any()
            or (c <= 0).any()
            or (o <= 0).any()
        )
        if bad_ohlc:
            messages.append(f"qc: {symbol} OHLC inconsistency")
        ret = c.pct_change().abs()
        if (ret > float(cfg.max_abs_return)).any():
            messages.append(f"qc: {symbol} price jump > {cfg.max_abs_return:.0%}")
        if "volume" in tail.columns and float(cfg.min_volume) > 0:
            if float(tail["volume"].iloc[-1]) < float(cfg.min_volume):
                messages.append(f"qc: {symbol} volume below minimum")
        if "funding_rate" in tail.columns:
            stale_h = float(cfg.funding_stale_hours)
            n_stale = max(int(round(stale_h / max(step_h, 1e-9))), 1)
            fund = tail["funding_rate"].astype(float)
            # 最近 stale_h 小时全是 NaN 才报过期；ffill 后的 0 仍算“有值”。
            if fund.iloc[-n_stale:].isna().all():
                messages.append(f"qc: {symbol} funding stale > {stale_h}h")

        last_ts = tail.index[-1]
        if now is not None:
            asof = pd.Timestamp(now)
            if asof.tzinfo is None:
                asof = asof.tz_localize("UTC")
            else:
                asof = asof.tz_convert("UTC")
            drift = abs(asof - last_ts)
            if bool(cfg.check_clock) or drift <= pd.Timedelta(days=2):
                limit = pd.Timedelta(minutes=float(cfg.clock_drift_minutes))
                if drift > limit:
                    messages.append(f"qc: {symbol} clock drift {drift} > {limit}")

    return QcReport(ok=not messages, messages=messages)


# 测试夹具使用的旧名
inspect_panels = inspect_panels
