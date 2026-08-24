"""YAML 驱动的类型化配置。

加载：``config/default.yaml`` → 可选覆盖 dict → Pydantic。
只保留 ETH 数据 → 决策网 → 多空信号 → OMS 这条链路上的字段。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config" / "default.yaml"
ROOT = ROOT
DEFAULT_CONFIG = DEFAULT_CONFIG


class AccountCfg(BaseModel):
    model_config = ConfigDict(extra="ignore")
    """账户与运行模式。paper/research 走本地撮合；testnet/live 走签名下单。"""

    initial_capital: float = 100_000.0
    quote_currency: str = "USDT"
    mode: str = "paper"  # research | paper | testnet | live


class UniverseCfg(BaseModel):
    model_config = ConfigDict(extra="ignore")
    """单一交易标的（默认 ETHUSDT）。"""

    symbol: str = "ETHUSDT"

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy(cls, v: Any):
        if not isinstance(v, dict):
            return v
        out = dict(v)
        if not out.get("symbol"):
            if out.get("trade_symbol"):
                out["symbol"] = out["trade_symbol"]
            elif isinstance(out.get("symbols"), list) and out["symbols"]:
                out["symbol"] = out["symbols"][0]
        out.pop("trade_symbol", None)
        out.pop("market_symbol", None)
        out.pop("symbols", None)
        return out

    @property
    def trade_symbol(self) -> str:
        return self.symbol

    @property
    def market_symbol(self) -> str:
        return self.symbol


class BinanceCfg(BaseModel):
    model_config = ConfigDict(extra="ignore")
    futures_base: str = "https://fapi.binance.com"
    spot_base: str = "https://api.binance.com"
    testnet_futures_base: str = "https://testnet.binancefuture.com"
    testnet_spot_base: str = "https://testnet.binance.vision"
    recv_window: int = 5000


class DataCfg(BaseModel):
    model_config = ConfigDict(extra="ignore")
    cache_dir: str = "data/cache"
    kline_interval: str = "1h"
    lookback_days: int = 500
    funding_interval_hours: int = 8


class FeesCfg(BaseModel):
    model_config = ConfigDict(extra="ignore")
    maker: float = 0.0002
    taker: float = 0.0005
    use_maker_for_entries: bool = True


class SlippageCfg(BaseModel):
    model_config = ConfigDict(extra="ignore")
    market_bps: float = 1.5


class StrategyCfg(BaseModel):
    """仓位：notional = equity * risk_budget * min(max_leverage, target_vol / σ) * unit。"""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    lookbacks_hours: list[int] = Field(default_factory=lambda: [24, 72, 168])
    lookback_weights: list[float] = Field(default_factory=lambda: [0.25, 0.40, 0.35])
    skip_hours: int = 0
    vol_lookback: int = 72
    target_vol_annual: float = 0.20
    max_leverage: float = 2.0
    risk_budget: float = 1.0
    score_scale: float = 1.0
    min_position: float = 0.05
    long_only: bool = False
    rebalance_hours: int = 8
    min_history: int = 360
    decision: str = "rl"
    nn_smooth: float = 0.20
    nn_model_path: str = "models/eth_decision.pt"
    nn_hidden: list[int] = Field(default_factory=lambda: [128, 64])
    nn_dropout: float = 0.2
    nn_seeds: int = 3
    nn_cost_bps: float = 8.0
    nn_epochs: int = 80
    nn_patience: int = 12
    seq_len: int = 7
    ppo_gamma: float = 0.99
    ppo_gae_lambda: float = 0.95
    ppo_clip: float = 0.2
    ppo_ent_coef: float = 0.01
    ppo_vf_coef: float = 0.5
    ppo_lr: float = 3e-4
    ppo_batch: int = 256
    ppo_max_grad_norm: float = 0.5
    ppo_inner_epochs: int = 4
    ppo_replay_rollouts: int = 4
    reward_ds_eta: float = 1.0 / 72.0
    reward_dd_inc: float = 1.0
    reward_dd_level: float = 0.05
    reward_clip: float = 5.0


class BacktestCfg(BaseModel):
    model_config = ConfigDict(extra="ignore")
    warmup_bars: int = 400
    report_dir: str = "reports"
    turnover_band_equity: float = 0.015


class PaperCfg(BaseModel):
    model_config = ConfigDict(extra="ignore")
    rebalance_minutes: int = 60
    dry_run: bool = True
    state_file: str = "data/state/paper.json"
    lookback_days: int = 400
    max_notional_per_symbol: float = 25_000.0


class OmsCfg(BaseModel):
    model_config = ConfigDict(extra="ignore")
    min_notional: float = 10.0
    client_id_prefix: str = "bt"
    testnet_only: bool = True
    reconcile_qty_tol: float = 0.001


class LoggingCfg(BaseModel):
    model_config = ConfigDict(extra="ignore")
    level: str = "INFO"
    file: str = "logs/betatrend.log"
    audit_file: str = "logs/audit.jsonl"


class Settings(BaseModel):
    model_config = ConfigDict(extra="ignore")

    account: AccountCfg = Field(default_factory=AccountCfg)
    universe: UniverseCfg = Field(default_factory=UniverseCfg)
    binance: BinanceCfg = Field(default_factory=BinanceCfg)
    data: DataCfg = Field(default_factory=DataCfg)
    fees: FeesCfg = Field(default_factory=FeesCfg)
    slippage: SlippageCfg = Field(default_factory=SlippageCfg)
    strategy: StrategyCfg = Field(default_factory=StrategyCfg)
    backtest: BacktestCfg = Field(default_factory=BacktestCfg)
    paper: PaperCfg = Field(default_factory=PaperCfg)
    oms: OmsCfg = Field(default_factory=OmsCfg)
    logging: LoggingCfg = Field(default_factory=LoggingCfg)

    def cache_path(self) -> Path:
        p = ROOT / self.data.cache_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    def report_path(self) -> Path:
        p = ROOT / self.backtest.report_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    def resolve(self, rel: str) -> Path:
        p = ROOT / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        return p


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_settings(
    path: str | Path | None = None,
    overrides: dict | None = None,
) -> Settings:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    raw: dict[str, Any] = {}
    if cfg_path.exists():
        loaded = yaml.safe_load(cfg_path.read_text()) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"Config must be a mapping: {cfg_path}")
        raw = loaded
    if overrides:
        raw = _deep_merge(raw, overrides)
    return Settings.model_validate(raw)
