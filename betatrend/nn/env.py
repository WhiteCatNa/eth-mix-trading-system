"""Frozen trading-env protocol: overlay PnL, stepwise reward, optional backtest exec.

Python is the reference used to dump golden files. The Rust crate must match
these numbers to 1e-6. Train/policy prefer ``betatrend_env`` when installed.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from betatrend.config import ROOT
from betatrend.nn.dataset import (
    FEATURE_NAMES,
    N_FEAT,
    SEQ_LEN,
    build_feature_frame,
    execution_aligned_returns,
    last_feature_window as py_last_window,
    make_windows,
    vol_leverage,
)
from betatrend.nn.reward import bar_pnl, shape_rewards

SPEC_PATH = ROOT / "env_spec.json"
BARS_PER_YEAR = 24 * 365
PNL_CLIP = 0.4
R_VOL_CLIP = 8.0
FEAT_CLIP = 8.0


def load_spec(path: Path | None = None) -> dict[str, Any]:
    return json.loads((path or SPEC_PATH).read_text(encoding="utf-8"))


def try_rust():
    try:
        import betatrend_env  # type: ignore

        return betatrend_env
    except Exception:
        return None


@dataclass
class ResetCfg:
    symbol: str = "ETHUSDT"
    start: int = 0
    end: int | None = None
    cost: float = 0.0008
    down_lambda: float = 0.5
    dd_inc: float = 0.0
    dd_level: float = 0.0
    clip: float = 5.0
    seq_len: int = SEQ_LEN
    fold_id: int = 0
    seed: int = 0
    exec_mode: str = "overlay"
    fee_rate: float = 0.0005
    slip_bps: float = 1.5
    funding_interval_hours: int = 8
    initial_equity: float = 100_000.0
    target_vol: float = 0.20
    max_leverage: float = 2.0
    risk_budget: float = 1.0
    turnover_band_equity: float = 0.015
    train_end: int | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str) -> "ResetCfg":
        return cls(**json.loads(raw))


class RewardMachine:
    """Stepwise copy of ``shape_rewards`` (mean − downside variance − optional drawdown)."""

    def __init__(
        self,
        down_lambda: float = 0.5,
        dd_inc: float = 0.0,
        dd_level: float = 0.0,
        clip: float = 5.0,
        bars_per_year: int = BARS_PER_YEAR,
    ):
        self.down_lambda = float(down_lambda)
        self.dd_inc = float(dd_inc)
        self.dd_level = float(dd_level)
        self.clip = float(clip)
        self.bars_per_year = float(bars_per_year)
        self.reset()

    def reset(self) -> None:
        self.equity = 1.0
        self.peak = 1.0
        self.prev_depth = 0.0

    def step(self, pnl: float, vol_ann: float) -> float:
        hourly_vol = max(vol_ann / math.sqrt(self.bars_per_year), 1e-6)
        rt = float(np.clip(pnl / hourly_vol, -R_VOL_CLIP, R_VOL_CLIP))
        down = min(rt, 0.0)
        self.equity *= 1.0 + float(np.clip(pnl, -PNL_CLIP, PNL_CLIP))
        self.peak = max(self.peak, self.equity)
        depth = (self.peak - self.equity) / max(self.peak, 1e-12)
        deepen = max(depth - self.prev_depth, 0.0)
        self.prev_depth = depth
        dd_pen = self.dd_inc * deepen + self.dd_level * depth
        return float(np.clip(rt - self.down_lambda * down * down - dd_pen, -self.clip, self.clip))


def overlay_rewards_py(
    actions: np.ndarray,
    y: np.ndarray,
    lev: np.ndarray,
    vol_ann: np.ndarray,
    cost: float,
    *,
    down_lambda: float = 0.5,
    dd_inc: float = 0.0,
    dd_level: float = 0.0,
    clip: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    pnl = bar_pnl(actions, y, lev, cost)
    machine = RewardMachine(down_lambda=down_lambda, dd_inc=dd_inc, dd_level=dd_level, clip=clip)
    rewards = np.empty(len(pnl), dtype=np.float32)
    for i in range(len(pnl)):
        rewards[i] = machine.step(float(pnl[i]), float(vol_ann[i]))
    return np.asarray(pnl, dtype=np.float64), rewards


def overlay_rewards(
    actions: np.ndarray,
    y: np.ndarray,
    lev: np.ndarray,
    vol_ann: np.ndarray,
    cost: float,
    *,
    down_lambda: float = 0.5,
    dd_inc: float = 0.0,
    dd_level: float = 0.0,
    clip: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """奖励是纯向量运算，Python 侧比 Rust 快（Rust 要先把 list 编组过去）。

    Rust 分支只在需要验证两侧一致时才有意义，见 ``tests/test_env.py``。
    """
    pnl = bar_pnl(actions, y, lev, cost)
    rewards = shape_rewards(
        pnl,
        np.asarray(vol_ann, dtype=np.float64),
        down_lambda=down_lambda,
        dd_inc=dd_inc,
        dd_level=dd_level,
        clip=clip,
    )
    return pnl, rewards


def overlay_rewards_rust(
    actions: np.ndarray,
    y: np.ndarray,
    lev: np.ndarray,
    vol_ann: np.ndarray,
    cost: float,
    *,
    down_lambda: float = 0.5,
    dd_inc: float = 0.0,
    dd_level: float = 0.0,
    clip: float = 5.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Rust 实现，供一致性测试调用。扩展没装时返回 None。"""
    rust = try_rust()
    if rust is None:
        return None
    pnl, rew = rust.overlay_rewards(
        np.asarray(actions, dtype=np.float64).ravel().tolist(),
        np.asarray(y, dtype=np.float64).ravel().tolist(),
        np.asarray(lev, dtype=np.float64).ravel().tolist(),
        np.asarray(vol_ann, dtype=np.float64).ravel().tolist(),
        float(cost),
        float(down_lambda),
        float(dd_inc),
        float(dd_level),
        float(clip),
    )
    return np.asarray(pnl, dtype=np.float64), np.asarray(rew, dtype=np.float32)


def series_flags(panel: pd.DataFrame) -> list[bool]:
    """Column existence: taker, trades, index, oi, lsr — not per-bar non-zero."""
    return [
        "taker_buy_base" in panel.columns,
        "trades" in panel.columns,
        "index_close" in panel.columns,
        "open_interest" in panel.columns,
        "long_short_ratio" in panel.columns,
    ]


def _funding_array(df: pd.DataFrame) -> np.ndarray:
    if "funding_rate" in df.columns:
        return df["funding_rate"].astype(float).to_numpy()
    if "funding" in df.columns:
        return df["funding"].astype(float).to_numpy()
    return np.zeros(len(df), dtype=np.float64)


def last_window(panel: pd.DataFrame, seq_len: int = SEQ_LEN) -> np.ndarray:
    rust = try_rust()
    if rust is not None:
        flat = rust.last_window(panel_to_bars(panel).tolist(), int(seq_len), series_flags(panel))
        return np.asarray(flat, dtype=np.float32).reshape(int(seq_len), -1)
    return py_last_window(panel, seq_len=seq_len)


def panel_to_bars(panel: pd.DataFrame) -> np.ndarray:
    """ts, open, high, low, close, volume, funding, taker, trades, mark, index, oi, lsr."""
    df = panel.copy()
    df.index = pd.to_datetime(df.index, utc=True)
    n = len(df)
    out = np.zeros((n, 13), dtype=np.float64)
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    out[:, 0] = ((df.index - epoch) / pd.Timedelta(seconds=1)).to_numpy(dtype=np.float64)
    close = df["close"].astype(float).to_numpy()
    out[:, 1] = df["open"].astype(float).to_numpy() if "open" in df.columns else close
    out[:, 2] = df["high"].astype(float).to_numpy() if "high" in df.columns else out[:, 1]
    out[:, 3] = df["low"].astype(float).to_numpy() if "low" in df.columns else out[:, 1]
    out[:, 4] = close
    out[:, 5] = df["volume"].astype(float).to_numpy() if "volume" in df.columns else 0.0
    out[:, 6] = _funding_array(df)
    if "taker_buy_base" in df.columns:
        out[:, 7] = df["taker_buy_base"].astype(float).to_numpy()
    if "trades" in df.columns:
        out[:, 8] = df["trades"].astype(float).to_numpy()
    out[:, 9] = df["mark_close"].astype(float).to_numpy() if "mark_close" in df.columns else close
    out[:, 10] = df["index_close"].astype(float).to_numpy() if "index_close" in df.columns else close
    if "open_interest" in df.columns:
        out[:, 11] = df["open_interest"].astype(float).to_numpy()
    if "long_short_ratio" in df.columns:
        out[:, 12] = df["long_short_ratio"].astype(float).to_numpy()
    return out


@dataclass
class Step:
    obs: np.ndarray
    reward: float
    done: bool
    pnl: float
    info: dict[str, Any] = field(default_factory=dict)


class OverlayEnv:
    """Gym-like wrapper around the current training overlay."""

    def __init__(self, panel: pd.DataFrame, cfg: ResetCfg | None = None):
        self.panel = panel
        self.cfg = cfg or ResetCfg()
        feats = build_feature_frame(panel)
        self.x = feats.to_numpy(dtype=np.float64)
        self.y = execution_aligned_returns(panel).to_numpy(dtype=np.float64)
        self.vol = feats["vol_24"].to_numpy(dtype=np.float64)
        self.lev = vol_leverage(
            pd.Series(self.vol), target=self.cfg.target_vol, max_leverage=self.cfg.max_leverage
        ).to_numpy(dtype=np.float64)
        self.median = np.median(self.x, axis=0)
        self.iqr = np.clip(np.percentile(self.x, 75, axis=0) - np.percentile(self.x, 25, axis=0), 1e-6, None)
        self.rm = RewardMachine(
            down_lambda=self.cfg.down_lambda,
            dd_inc=self.cfg.dd_inc,
            dd_level=self.cfg.dd_level,
            clip=self.cfg.clip,
        )
        self.t = 0
        self.end = len(self.x)
        self.prev_exp = 0.0

    def spec(self) -> dict[str, Any]:
        return load_spec()

    def _scaled_window(self, t: int) -> np.ndarray:
        z = np.clip((self.x - self.median) / self.iqr, -FEAT_CLIP, FEAT_CLIP).astype(np.float32)
        wins = make_windows(z, seq_len=self.cfg.seq_len)
        return wins[min(max(t, 0), len(wins) - 1)]

    def reset(self, cfg: ResetCfg | None = None) -> np.ndarray:
        if cfg is not None:
            self.cfg = cfg
        if self.cfg.train_end is not None:
            sl = slice(0, int(self.cfg.train_end))
            self.median = np.median(self.x[sl], axis=0)
            self.iqr = np.clip(
                np.percentile(self.x[sl], 75, axis=0) - np.percentile(self.x[sl], 25, axis=0), 1e-6, None
            )
        self.t = int(self.cfg.start)
        self.end = int(self.cfg.end) if self.cfg.end is not None else max(len(self.x) - 2, self.t)
        self.rm.down_lambda = float(self.cfg.down_lambda)
        self.rm.dd_inc = float(self.cfg.dd_inc)
        self.rm.dd_level = float(self.cfg.dd_level)
        self.rm.clip = float(self.cfg.clip)
        self.rm.reset()
        self.prev_exp = 0.0
        return self._scaled_window(self.t)

    def rollout_ready(self) -> bool:
        return self.end - self.t >= 1 and self.t >= 0

    def step(self, action: float) -> Step:
        a = float(np.clip(action, -1.0, 1.0))
        exp = a * float(self.lev[self.t])
        pnl = exp * float(self.y[self.t]) - float(self.cfg.cost) * abs(exp - self.prev_exp)
        rew = self.rm.step(pnl, float(self.vol[self.t]))
        self.prev_exp = exp
        self.t += 1
        done = self.t >= self.end
        obs = self._scaled_window(min(self.t, len(self.x) - 1))
        return Step(obs=obs, reward=rew, done=done, pnl=pnl, info={"t": self.t, "exec": "overlay"})


class BacktestEnv:
    """Bar exec matching Backtester: gap → funding → pending fill → mark → next pending."""

    def __init__(self, panel: pd.DataFrame, cfg: ResetCfg | None = None):
        self.panel = panel
        self.cfg = cfg or ResetCfg(exec_mode="backtest")
        df = panel.copy()
        df.index = pd.to_datetime(df.index, utc=True)
        self.open = df["open"].astype(float).to_numpy()
        self.close = df["close"].astype(float).to_numpy()
        self.funding = _funding_array(df)
        self.hours = df.index.hour.to_numpy()
        feats = build_feature_frame(df)
        self.x = feats.to_numpy(dtype=np.float64)
        self.vol = feats["vol_24"].to_numpy(dtype=np.float64)
        self.median = np.median(self.x, axis=0)
        self.iqr = np.clip(np.percentile(self.x, 75, axis=0) - np.percentile(self.x, 25, axis=0), 1e-6, None)
        self.rm = RewardMachine(
            down_lambda=self.cfg.down_lambda,
            dd_inc=self.cfg.dd_inc,
            dd_level=self.cfg.dd_level,
            clip=self.cfg.clip,
        )
        self.cash = self.cfg.initial_equity
        self.qty = 0.0
        self.pending: float | None = None
        self.last_close = float(self.close[0])
        self.t = 0
        self.end = len(self.close)

    def spec(self) -> dict[str, Any]:
        return load_spec()

    def _window(self, t: int) -> np.ndarray:
        z = np.clip((self.x - self.median) / self.iqr, -FEAT_CLIP, FEAT_CLIP).astype(np.float32)
        return make_windows(z, seq_len=self.cfg.seq_len)[min(max(t, 0), len(self.x) - 1)]

    def reset(self, cfg: ResetCfg | None = None) -> np.ndarray:
        if cfg is not None:
            self.cfg = cfg
        self.cash = float(self.cfg.initial_equity)
        self.qty = 0.0
        self.pending = None
        self.t = int(self.cfg.start)
        self.end = int(self.cfg.end) if self.cfg.end is not None else len(self.close)
        self.last_close = float(self.close[max(self.t - 1, 0)]) if self.t > 0 else float(self.close[0])
        self.rm.down_lambda = float(self.cfg.down_lambda)
        self.rm.dd_inc = float(self.cfg.dd_inc)
        self.rm.dd_level = float(self.cfg.dd_level)
        self.rm.clip = float(self.cfg.clip)
        self.rm.reset()
        return self._window(self.t)

    def rollout_ready(self) -> bool:
        return self.t < self.end

    def step(self, action: float) -> Step:
        i = self.t
        o = float(self.open[i])
        c = float(self.close[i])
        pnl = 0.0
        if i > 0:
            mark = self.qty * (o - self.last_close)
            self.cash += mark
            pnl += mark
        if int(self.hours[i]) % int(self.cfg.funding_interval_hours) == 0:
            fp = -(self.qty * o) * float(self.funding[i])
            self.cash += fp
            pnl += fp
        if self.pending is not None:
            d_n = float(self.pending) - self.qty * o
            band = max(self.cash * self.cfg.turnover_band_equity, 75.0)
            if abs(d_n) >= band and o > 0:
                fee = abs(d_n) * self.cfg.fee_rate
                slip = abs(d_n) * (self.cfg.slip_bps / 10_000.0)
                self.cash -= fee + slip
                self.qty += d_n / o
                pnl -= fee + slip
            self.pending = None
        mark2 = self.qty * (c - o)
        self.cash += mark2
        pnl += mark2
        self.last_close = c
        vol = max(float(self.vol[i]), 1e-6)
        lev = min(self.cfg.max_leverage, self.cfg.target_vol / vol)
        unit = float(np.clip(action, -1.0, 1.0))
        self.pending = self.cash * self.cfg.risk_budget * lev * unit
        rew = self.rm.step(pnl / max(self.cfg.initial_equity, 1.0), vol)
        self.t += 1
        done = self.t >= self.end
        return Step(
            obs=self._window(min(self.t, len(self.close) - 1)),
            reward=rew,
            done=done,
            pnl=pnl,
            info={"t": self.t, "exec": "backtest", "equity": self.cash, "qty": self.qty},
        )


def dump_golden(out_dir: Path, n: int = 400, seed: int = 7) -> Path:
    from betatrend.marketdata.synthetic import make_trending_panels

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    panel = make_trending_panels(n=n, seed=seed, symbols=["ETHUSDT"])["ETHUSDT"]
    # 回撤项默认关掉，但黄金文件里开着，好让 Rust 对齐测试覆盖到这条分支。
    cfg = ResetCfg(
        start=180, end=n - 2, cost=0.0008, seed=seed, train_end=300, dd_inc=2.0, dd_level=0.1
    )
    env = OverlayEnv(panel, cfg)
    obs = env.reset(cfg)
    n_steps = int(cfg.end or 0) - cfg.start
    actions = np.tanh(np.sin(np.arange(n_steps, dtype=np.float64) / 11.0))
    rows: list[dict[str, Any]] = []
    for k, a in enumerate(actions):
        t = cfg.start + k
        step = env.step(float(a))
        row: dict[str, Any] = {
            "t": t,
            "action": float(a),
            "y": float(env.y[t]),
            "lev": float(env.lev[t]),
            "vol": float(env.vol[t]),
            "pnl": float(step.pnl),
            "reward": float(step.reward),
            "done": int(step.done),
        }
        for j, name in enumerate(FEATURE_NAMES):
            row[f"x_{name}"] = float(env.x[t, j])
        flat = obs.reshape(-1)
        for j in range(flat.size):
            row[f"obs_{j}"] = float(flat[j])
        rows.append(row)
        obs = step.obs
        if step.done:
            break
    steps = pd.DataFrame(rows)
    panel.to_csv(out_dir / "bars.csv", index=True)
    steps.to_parquet(out_dir / "steps.parquet", index=False)
    steps.to_csv(out_dir / "steps.csv", index=False)
    (out_dir / "reset_cfg.json").write_text(cfg.to_json(), encoding="utf-8")
    (out_dir / "env_spec.json").write_text(json.dumps(load_spec(), indent=2), encoding="utf-8")
    sl = slice(0, len(steps))
    _pnl, _rew = overlay_rewards_py(
        actions[sl],
        env.y[cfg.start : cfg.start + len(steps)],
        env.lev[cfg.start : cfg.start + len(steps)],
        env.vol[cfg.start : cfg.start + len(steps)],
        cfg.cost,
        down_lambda=cfg.down_lambda,
        dd_inc=cfg.dd_inc,
        dd_level=cfg.dd_level,
        clip=cfg.clip,
    )
    ref = shape_rewards(
        _pnl,
        env.vol[cfg.start : cfg.start + len(steps)],
        down_lambda=cfg.down_lambda,
        dd_inc=cfg.dd_inc,
        dd_level=cfg.dd_level,
        clip=cfg.clip,
    )
    err = float(np.max(np.abs(steps["reward"].to_numpy() - ref)))
    (out_dir / "checksum.json").write_text(
        json.dumps({"max_step_vs_batch_reward_abs": err, "n_steps": int(len(steps)), "n_feat": N_FEAT}, indent=2),
        encoding="utf-8",
    )
    if err >= 1e-6:
        raise RuntimeError(f"stepwise reward drifted from batch shape_rewards: {err}")
    return out_dir / "steps.parquet"
