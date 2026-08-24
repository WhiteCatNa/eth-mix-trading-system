"""ETH 择时：行情 → 策略网络 → 多空仓位信号。

核心闭环：
    1. 取 ETHUSDT 面板
    2. 决策网输出连续 unit ∈ [-1, 1]（正多负空）
    3. |unit| < 全仓 5% → 空仓，不开新单
    4. 名义本金 = 权益 × 风险预算 × min(杠杆帽, 目标波动 / σ) × unit
    5. OMS 按目标名义买卖

``long_only`` 默认 False，多空都做。权重缺失时保持空仓，不再回退规则基准。
"""
from __future__ import annotations

from loguru import logger

from betatrend.config import Settings
from betatrend.domain import MarketSnapshot, TargetPosition
from betatrend.features import FeatureSet, compute_features
from betatrend.signals import make_signal


class TimingStrategy:
    """根据最新截面生成 ETH 目标名义。无状态，每 bar 可重入。"""

    name = "timing"

    def __init__(self, settings: Settings):
        self.settings = settings
        self.cfg = settings.strategy
        self._policy = None
        self._policy_failed = False
        self._smooth_unit = 0.0

    def reset(self) -> None:
        """清空网络 EMA（新回测 / 新 paper 会话必须调用）。"""
        self._smooth_unit = 0.0
        if self._policy is not None:
            self._policy.reset()

    def restore_smooth(self, unit: float) -> None:
        """从 paper 状态恢复 EMA。策略网尚未加载时先记下，第一次 generate 时套上。"""
        self._smooth_unit = float(unit)
        if self._policy is not None and hasattr(self._policy, "restore_last_unit"):
            self._policy.restore_last_unit(self._smooth_unit)

    def last_smooth_unit(self) -> float:
        pol = self._policy
        if pol is not None and hasattr(pol, "last_unit"):
            return float(pol.last_unit())
        if pol is not None:
            return float(getattr(pol, "_last_unit", self._smooth_unit))
        return float(self._smooth_unit)

    @property
    def trade_symbol(self) -> str:
        return self.settings.universe.trade_symbol

    def _policy_unit(self, panel) -> tuple[float, str]:
        """只走决策网。网未就绪或 torch 不可用时 unit=0（空仓）。"""
        if self._policy is None and not self._policy_failed:
            try:
                from betatrend.nn.policy import NeuralPolicy

                self._policy = NeuralPolicy(self.settings)
                if hasattr(self._policy, "restore_last_unit"):
                    self._policy.restore_last_unit(self._smooth_unit)
            except ImportError:
                logger.warning("torch not installed — staying flat")
                self._policy_failed = True
        if self._policy is not None and self._policy.ready:
            unit = float(self._policy.predict_unit(panel))
            if self.cfg.long_only:
                unit = max(unit, 0.0)
            return unit, "rl"
        return 0.0, "flat"

    def generate(self, snap: MarketSnapshot, features: FeatureSet | None = None) -> list[TargetPosition]:
        """输出 0 或 1 个目标。历史不够或策略关闭时返回空列表。"""
        if not self.cfg.enabled:
            return []
        symbol = self.trade_symbol
        need = max(self.cfg.min_history, max(self.cfg.lookbacks_hours) + 8)
        panel = snap.panels.get(symbol)
        if panel is None or len(panel) < need:
            return []

        feat = features or compute_features(
            snap.panels,
            snap.market_symbol,
            vol_lookback=self.cfg.vol_lookback,
        )
        raw_unit, source = self._policy_unit(panel)
        vol = max(feat.vols.get(symbol, feat.market_vol), 1e-6)
        budget = snap.equity * self.cfg.risk_budget
        lev = min(self.cfg.max_leverage, self.cfg.target_vol_annual / vol)
        raw_notional = budget * lev * raw_unit
        signal = make_signal(raw_unit, raw_notional, min_position=self.cfg.min_position)
        return [
            TargetPosition(
                symbol=symbol,
                strategy=self.name,
                target_notional=float(signal.target_notional),
                signal=signal.unit,
                trend_score=signal.unit,
                reason=(
                    f"{source} {signal.side.value} unit={signal.unit:.3f} "
                    f"lev={lev:.2f} vol={vol:.2%}"
                ),
                extras={
                    "unit": signal.unit,
                    "side": signal.side.value,
                    "vol": vol,
                    "leverage": lev,
                    "decision": source,
                },
            )
        ]


BetaTrendStrategy = TimingStrategy
