"""决策网络的因果特征矩阵。第 t 行只用到 ≤ t 的信息，没有未来函数。

滚动窗口全部 backward-looking（含当前 bar 收盘）。仓位由决策网直接输出，
这些列只是输入特征，不再包含 TSMOM 分数列。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from betatrend.mathx import BARS_PER_YEAR

# 含义简述：
#   ret_*       多周期简单收益（1h / 4h / 12h / 1d / 3d / 7d）
#   vol_*       实现波动（已年化）
#   vol_ratio   短波 / 长波，>1 表示波动抬升
#   ema_gap_*   价相对 EMA 的偏离
#   rsi_14      压到 [-1,1] 的 RSI
#   range_24    日内振幅均值
#   volx_z      成交量 z-score
#   funding*    资金费率水平 / 均线 / z / 8h 差分（拥挤度）
#   tod/dow     小时与星期的傅里叶编码
#   ret_skip    不含最近 24h 的 168h 收益（减轻短期反转噪声）
#   mom_agree   多周期动量符号一致性
#   vov         波动的波动
#   range_pos   价格在 24h 高低点中的位置
#   ret_streak  同向收益持续长度
#   trend_persist 收益一阶自相关
#   close_z     24h 收盘价 z-score
#   taker_imb*  主动买入占比（K 线 taker_buy_base / volume）
#   body/wick   K 线实体与上下影相对振幅
#   gap         开盘相对前收
#   atr_n       ATR(14) / close
#   trades_z    成交笔数 z-score
#   basis*      标记价相对指数价溢价（完整历史）
#   oi_*        持仓量变动 / z（期货统计接口约 30 天，更早为 0）
#   lsr_dev     全市场多空人数比偏离 1（同上，约 30 天）
FEATURE_NAMES = [
    "ret_1",
    "ret_4",
    "ret_12",
    "ret_24",
    "ret_72",
    "ret_168",
    "vol_24",
    "vol_72",
    "vol_168",
    "vol_ratio",
    "ema_gap_24",
    "ema_gap_72",
    "rsi_14",
    "range_24",
    "volx_z",
    "funding",
    "funding_ma",
    "tod_sin",
    "tod_cos",
    "dow_sin",
    "dow_cos",
    "ret_skip",
    "mom_agree",
    "vov",
    "funding_z",
    "funding_d8",
    "range_pos",
    "ret_streak",
    "trend_persist",
    "close_z",
    "taker_imb",
    "taker_imb_ma",
    "body",
    "wick_imb",
    "gap",
    "atr_n",
    "trades_z",
    "basis",
    "basis_z",
    "oi_chg",
    "oi_z",
    "lsr_dev",
]

SEQ_LEN = 7
N_FEAT = len(FEATURE_NAMES)
MAX_LOOKBACK = 168
TAIL_BARS = MAX_LOOKBACK + 48


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0.0).rolling(n, min_periods=n).mean()
    down = (-delta.clip(upper=0.0)).rolling(n, min_periods=n).mean()
    rs = up / down.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return ((rsi - 50.0) / 50.0).fillna(0.0)


def build_feature_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """一根 bar 一行。所有滚动统计都是后向的，并且包含当前 bar t 的收盘。"""
    df = panel.copy()
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    close = df["close"].astype(float)
    open_ = df["open"].astype(float) if "open" in df.columns else close
    high = df["high"].astype(float) if "high" in df.columns else close
    low = df["low"].astype(float) if "low" in df.columns else close
    volume = df["volume"].astype(float) if "volume" in df.columns else pd.Series(0.0, index=df.index)
    ret = close.pct_change().fillna(0.0)
    funding = df["funding_rate"].astype(float) if "funding_rate" in df.columns else pd.Series(0.0, index=df.index)

    out = pd.DataFrame(index=df.index)
    for k in (1, 4, 12, 24, 72, 168):
        out[f"ret_{k}"] = close.pct_change(k).fillna(0.0)
    for k in (24, 72, 168):
        out[f"vol_{k}"] = ret.rolling(k, min_periods=max(k // 2, 8)).std().fillna(0.0) * np.sqrt(24 * 365)
    out["vol_ratio"] = (out["vol_24"] / out["vol_168"].replace(0.0, np.nan)).fillna(1.0).clip(0.2, 5.0)
    ema24 = close.ewm(span=24, adjust=False).mean()
    ema72 = close.ewm(span=72, adjust=False).mean()
    out["ema_gap_24"] = ((close - ema24) / close.replace(0.0, np.nan)).fillna(0.0)
    out["ema_gap_72"] = ((close - ema72) / close.replace(0.0, np.nan)).fillna(0.0)
    out["rsi_14"] = _rsi(close, 14)
    tr = (high - low) / close.replace(0.0, np.nan)
    out["range_24"] = tr.rolling(24, min_periods=8).mean().fillna(0.0)
    vol_ma = volume.rolling(24, min_periods=8).mean()
    vol_sd = volume.rolling(24, min_periods=8).std().replace(0.0, np.nan)
    out["volx_z"] = ((volume - vol_ma) / vol_sd).fillna(0.0).clip(-5, 5)
    out["funding"] = funding.fillna(0.0)
    out["funding_ma"] = funding.rolling(24, min_periods=4).mean().fillna(0.0)
    hours = df.index.hour.astype(float)
    dow = df.index.dayofweek.astype(float)
    out["tod_sin"] = np.sin(2 * np.pi * hours / 24.0)
    out["tod_cos"] = np.cos(2 * np.pi * hours / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    # 经典 TSMOM skip：168h 收益在 24h 前结束，最近一天不进窗口，减轻短期反转噪声。
    out["ret_skip"] = (close.shift(24) / close.shift(24 + 168) - 1.0).replace([np.inf, -np.inf], 0.0).fillna(0.0)
    signs = np.sign(out["ret_24"]) + np.sign(out["ret_72"]) + np.sign(out["ret_168"])
    out["mom_agree"] = (signs / 3.0).fillna(0.0)
    out["vov"] = out["vol_24"].rolling(72, min_periods=24).std().fillna(0.0)
    f_ma72 = funding.rolling(72, min_periods=12).mean()
    f_sd72 = funding.rolling(72, min_periods=12).std().replace(0.0, np.nan)
    out["funding_z"] = ((funding - f_ma72) / f_sd72).fillna(0.0).clip(-5.0, 5.0)
    out["funding_d8"] = (out["funding_ma"] - out["funding_ma"].shift(8)).fillna(0.0)
    hh = high.rolling(24, min_periods=8).max()
    ll = low.rolling(24, min_periods=8).min()
    out["range_pos"] = ((close - ll) / (hh - ll).replace(0.0, np.nan)).fillna(0.5).mul(2.0).sub(1.0)
    grp = (np.sign(ret) != np.sign(ret).shift()).cumsum()
    streak = ret.groupby(grp).cumcount() + 1
    out["ret_streak"] = (np.sign(ret) * np.log1p(streak)).fillna(0.0).clip(-4.0, 4.0)
    out["trend_persist"] = ret.rolling(24, min_periods=12).corr(ret.shift(1)).fillna(0.0).clip(-1.0, 1.0)
    ma = close.rolling(24, min_periods=8).mean()
    sd = close.rolling(24, min_periods=8).std().replace(0.0, np.nan)
    out["close_z"] = ((close - ma) / sd).fillna(0.0).clip(-5.0, 5.0)

    span = (high - low).replace(0.0, np.nan)
    if "taker_buy_base" in df.columns:
        taker = df["taker_buy_base"].astype(float)
        taker_imb = (2.0 * taker / volume.replace(0.0, np.nan) - 1.0).replace([np.inf, -np.inf], 0.0)
        out["taker_imb"] = taker_imb.fillna(0.0).clip(-1.0, 1.0)
    else:
        out["taker_imb"] = 0.0
    out["taker_imb_ma"] = out["taker_imb"].rolling(24, min_periods=8).mean().fillna(0.0)
    out["body"] = ((close - open_) / span).replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-1.0, 1.0)
    upper = high - np.maximum(open_, close)
    lower = np.minimum(open_, close) - low
    out["wick_imb"] = ((upper - lower) / span).replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-1.0, 1.0)
    out["gap"] = (open_ / close.shift(1) - 1.0).replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-0.05, 0.05)
    prev_c = close.shift(1)
    true_range = pd.concat(
        [(high - low), (high - prev_c).abs(), (low - prev_c).abs()],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(14, min_periods=7).mean()
    out["atr_n"] = (atr / close.replace(0.0, np.nan)).replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(0.0, 0.2)
    if "trades" in df.columns:
        t_ma = df["trades"].astype(float).rolling(24, min_periods=8).mean()
        t_sd = df["trades"].astype(float).rolling(24, min_periods=8).std().replace(0.0, np.nan)
        out["trades_z"] = ((df["trades"].astype(float) - t_ma) / t_sd).fillna(0.0).clip(-5.0, 5.0)
    else:
        out["trades_z"] = 0.0

    mark = df["mark_close"].astype(float) if "mark_close" in df.columns else close
    if "index_close" in df.columns:
        index = df["index_close"].astype(float).replace(0.0, np.nan)
        basis = ((mark - index) / index).replace([np.inf, -np.inf], 0.0)
        out["basis"] = basis.fillna(0.0).clip(-0.05, 0.05)
    else:
        out["basis"] = 0.0
    b_ma = out["basis"].rolling(72, min_periods=12).mean()
    b_sd = out["basis"].rolling(72, min_periods=12).std().replace(0.0, np.nan)
    out["basis_z"] = ((out["basis"] - b_ma) / b_sd).fillna(0.0).clip(-5.0, 5.0)

    if "open_interest" in df.columns:
        oi = df["open_interest"].astype(float)
        out["oi_chg"] = oi.pct_change().replace([np.inf, -np.inf], 0.0).fillna(0.0).clip(-0.2, 0.2)
        oi_ma = oi.rolling(72, min_periods=12).mean()
        oi_sd = oi.rolling(72, min_periods=12).std().replace(0.0, np.nan)
        out["oi_z"] = ((oi - oi_ma) / oi_sd).fillna(0.0).clip(-5.0, 5.0)
    else:
        out["oi_chg"] = 0.0
        out["oi_z"] = 0.0
    if "long_short_ratio" in df.columns:
        out["lsr_dev"] = (df["long_short_ratio"].astype(float) - 1.0).clip(-2.0, 2.0).fillna(0.0)
    else:
        out["lsr_dev"] = 0.0
    return out[FEATURE_NAMES].replace([np.inf, -np.inf], 0.0).fillna(0.0)


def make_windows(x: np.ndarray, seq_len: int = SEQ_LEN) -> np.ndarray:
    """因果窗口：第 t 行是 [t-seq_len+1, t]；序列开头用第一行左填充。形状 (n, seq_len, n_feat)。"""
    x = np.asarray(x, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError(f"expected 2D features, got {x.shape}")
    n, f = x.shape
    if n == 0:
        return np.zeros((0, seq_len, f), dtype=np.float32)
    if seq_len <= 1:
        return x[:, None, :]
    pad = np.repeat(x[:1], seq_len - 1, axis=0)
    xp = np.concatenate([pad, x], axis=0)
    windows = np.lib.stride_tricks.sliding_window_view(xp, (seq_len, f))
    return np.ascontiguousarray(windows[:, 0])


def last_feature_row(panel: pd.DataFrame) -> np.ndarray:
    """最后一行因果特征。截一段足够长的尾巴，让 EMA/滚动窗口尽量贴近训练时的状态。"""
    tail = panel.iloc[-max(TAIL_BARS, 256) :] if len(panel) > TAIL_BARS else panel
    return build_feature_frame(tail).iloc[-1].to_numpy(dtype=np.float32)


def last_feature_window(panel: pd.DataFrame, seq_len: int = SEQ_LEN) -> np.ndarray:
    """最近 seq_len 根 bar 的因果特征，形状 (seq_len, n_feat)。"""
    tail = panel.iloc[-max(TAIL_BARS, 256) :] if len(panel) > TAIL_BARS else panel
    x = build_feature_frame(tail).to_numpy(dtype=np.float32)
    return make_windows(x, seq_len=seq_len)[-1]


def next_bar_returns(panel: pd.DataFrame) -> pd.Series:
    """时刻 t 的标签是 close[t+1]/close[t]-1。先 shift 再进特征，避免标签泄漏。"""
    close = panel["close"].astype(float)
    return close.pct_change().shift(-1).fillna(0.0)


def execution_aligned_returns(panel: pd.DataFrame) -> pd.Series:
    """若在 open[t+1] 成交、open[t+2] 离场，实际赚到的收益（与回测器对齐）。"""
    if "open" in panel.columns:
        o = panel["open"].astype(float).replace(0.0, np.nan)
        y = (o.shift(-2) / o.shift(-1) - 1.0)
        return y.replace([np.inf, -np.inf], 0.0).fillna(0.0)
    return next_bar_returns(panel)


def vol_leverage(vol: pd.Series, target: float = 0.20, max_leverage: float = 2.0) -> pd.Series:
    """反波动杠杆：目标波动 / 实现波动，再夹到 [0, max_leverage]。"""
    lev = target / vol.replace(0.0, np.nan)
    return lev.fillna(1.0).clip(0.0, max_leverage)


def sizing_vol(panel: pd.DataFrame, lookback: int) -> pd.Series:
    """给杠杆用的年化波动，口径与 desk 的 ``mathx.realized_vol`` 完全一致。

    特征表里的 ``vol_24`` 是模型输入，窗口写死 24；desk 定仓位用的是
    ``strategy.vol_lookback``（默认 72）。两者不是一回事，训练/评估的杠杆
    必须跟 desk 走，否则 OOS 的仓位曲线实盘根本不会出现。

    对齐点：同为 ddof=1 样本标准差、同样按 √8760 年化、样本不足时同样退到
    0.20 占位（``realized_vol`` 的行为），因此 ``vol_leverage`` 会给出 lev=1.0。
    """
    lookback = max(int(lookback), 2)
    ret = panel["close"].astype(float).pct_change().fillna(0.0)
    vol = ret.rolling(lookback, min_periods=lookback).std() * np.sqrt(BARS_PER_YEAR)
    return vol.fillna(0.20).clip(lower=1e-6)


FEATURE_NAMES = FEATURE_NAMES
N_FEAT = N_FEAT
SEQ_LEN = SEQ_LEN
build_feature_frame = build_feature_frame
make_windows = make_windows
last_feature_window = last_feature_window
execution_aligned_returns = execution_aligned_returns
vol_leverage = vol_leverage
FEATURE_NAMES = FEATURE_NAMES
N_FEAT = N_FEAT
SEQ_LEN = SEQ_LEN
last_feature_window = last_feature_window
build_feature_frame = build_feature_frame
