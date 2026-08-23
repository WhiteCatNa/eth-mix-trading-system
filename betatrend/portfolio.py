"""组合构建：把策略目标合成名义本金，再套杠杆/β 帽和可选的权益波动目标。

注意：``risk_scalar`` 故意不在这里乘，只在 RiskEngine.clip 乘一次，
避免回撤减速被组合层和风控层各乘一遍。
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from betatrend.config import Settings
from betatrend.domain import TargetPosition
from betatrend.features import FeatureSet


class PortfolioConstructor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.last_scale = 1.0  # 最近一次波动目标乘数，供日志/仪表盘
        self._eq_hist: list[float] = []  # 权益轨迹，用于估计组合实现波动

    def reset(self) -> None:
        self.last_scale = 1.0
        self._eq_hist = []

    def push_equity(self, equity: float) -> None:
        """回测每根 bar 收盘后推入，供下一拍 ``_vol_scale`` 使用。"""
        self._eq_hist.append(float(equity))

    def aggregate(self, targets: list[TargetPosition]) -> dict[str, float]:
        """同标的多个策略目标相加（当前只有 timing 一路，但仍按可加总设计）。"""
        agg: dict[str, float] = defaultdict(float)
        for t in targets:
            agg[t.symbol] += t.target_notional
        return dict(agg)

    def _vol_scale(self) -> float:
        """若组合实现波动偏离策略目标波动，把所有名义按比例缩放并夹在 [min, max]。

        历史不够长时返回 1，避免用噪声波动把仓位打满或打没。
        """
        cfg = self.settings.risk
        if not cfg.vol_target_enabled:
            return 1.0
        lb = cfg.vol_target_lookback
        if len(self._eq_hist) < max(lb, 10):
            return 1.0
        eq = np.asarray(self._eq_hist[-lb:], dtype=float)
        rets = np.diff(eq) / np.clip(eq[:-1], 1e-9, None)
        vol = float(np.std(rets, ddof=1) * np.sqrt(24 * 365))
        if vol < 1e-8:
            return 1.0
        scale = self.settings.strategy.target_vol_annual / vol
        scale = float(np.clip(scale, cfg.vol_target_min_scale, cfg.vol_target_max_scale))
        self.last_scale = scale
        return scale

    def construct(
        self,
        targets: list[TargetPosition],
        equity: float,
        features: FeatureSet,
        risk_scalar: float,
    ) -> dict[str, float]:
        """顺序：聚合 → 波动目标 → 单币帽 → 总杠杆帽 → 净杠杆帽 → 美元 β 帽。"""
        notionals = self.aggregate(targets)
        if equity <= 0:
            return {s: 0.0 for s in notionals}

        risk = self.settings.risk
        # 形参保留是为了 DeskCycle 接口稳定；真正缩放在 RiskEngine.clip。
        _ = risk_scalar
        out = dict(notionals)
        vt = self._vol_scale()
        out = {s: v * vt for s, v in out.items()}

        max_sym = equity * risk.max_symbol_weight * risk.max_gross_leverage
        for s, v in list(out.items()):
            if abs(v) > max_sym:
                out[s] = float(np.sign(v) * max_sym)

        gross = sum(abs(v) for v in out.values())
        max_gross = equity * risk.max_gross_leverage
        if gross > max_gross and gross > 0:
            out = {s: v * (max_gross / gross) for s, v in out.items()}

        net = sum(out.values())
        max_net = equity * risk.max_net_leverage
        if abs(net) > max_net and abs(net) > 1e-9:
            out = {s: v * (max_net / abs(net)) for s, v in out.items()}

        # 美元 β = Σ (名义_i / 权益 × β_i)。超过 max_portfolio_beta 则整体缩。
        dollar_beta = 0.0
        for s, v in out.items():
            dollar_beta += (v / equity) * features.betas.get(s, 1.0)
        cap = risk.max_portfolio_beta
        if abs(dollar_beta) > cap and abs(dollar_beta) > 1e-12:
            out = {s: v * (cap / abs(dollar_beta)) for s, v in out.items()}

        return out
