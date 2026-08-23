"""盘前 / 盘中风控：回撤阶梯、日亏损减速、kill。

``compute_scalar`` 只根据权益路径决定 0~1 的乘数；
``clip`` 再乘到名义上，并套单币权重与总杠杆帽。
kill 在回撤回到 soft 阈值一半以内时自动解除，避免永久锁死回测。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from betatrend.config import Settings


@dataclass
class RiskState:
    equity: float
    peak_equity: float  # 历史峰值，只升不降，用于回撤
    day_start_equity: float  # 每个 UTC 日开盘权益，用于日亏损
    gross_notional: float = 0.0
    net_notional: float = 0.0
    risk_scalar: float = 1.0
    kill: bool = False
    day_halt: bool = False  # 日亏损触发后本交易日把 scalar 压到 0.25
    messages: list[str] = field(default_factory=list)

    @property
    def drawdown(self) -> float:
        """当前回撤，负值。峰值为 0 时视为无回撤。"""
        if self.peak_equity <= 0:
            return 0.0
        return (self.equity - self.peak_equity) / self.peak_equity


class RiskEngine:
    def __init__(self, settings: Settings):
        cap = settings.account.initial_capital
        self.cfg = settings.risk
        self.state = RiskState(equity=cap, peak_equity=cap, day_start_equity=cap)

    def reset(self, capital: float | None = None) -> None:
        cap = float(capital if capital is not None else self.state.equity)
        self.state = RiskState(equity=cap, peak_equity=cap, day_start_equity=cap)

    def update_equity(self, equity: float) -> None:
        """每根 bar 先更新权益和峰值，再算 scalar。"""
        self.state.equity = float(equity)
        self.state.peak_equity = max(self.state.peak_equity, self.state.equity)

    def set_day_start(self, equity: float) -> None:
        """新交易日开始：重置日亏损标记，允许仓位从 0.25 恢复。"""
        self.state.day_start_equity = float(equity)
        self.state.day_halt = False

    def compute_scalar(self) -> float:
        """回撤阶梯（由重到轻覆盖）：

        kill   : DD ≤ -15% → scalar=0，DeskCycle flatten
        hard   : DD ≤ -10% → 0.20
        soft   : DD ≤ -6%  → 0.50
        half   : DD ≤ -3%  → 0.70
        日亏损 : 当日 ≤ -4% → 再与 0.25 取更严（kill 时仍为 0）
        """
        msgs: list[str] = []
        dd = self.state.drawdown
        cfg = self.cfg
        scalar = 1.0

        if dd <= -cfg.drawdown_kill:
            self.state.kill = True
            msgs.append(f"KILL DD {dd:.2%}")
        elif self.state.kill and dd > -cfg.drawdown_soft * 0.5:
            self.state.kill = False
            msgs.append(f"KILL off DD {dd:.2%}")

        if self.state.kill:
            scalar = 0.0
        elif dd <= -cfg.drawdown_hard:
            scalar = 0.20
            msgs.append(f"HARD DD {dd:.2%} → 20%")
        elif dd <= -cfg.drawdown_soft:
            scalar = 0.50
            msgs.append(f"SOFT DD {dd:.2%} → 50%")
        elif dd <= -cfg.drawdown_soft * 0.5:
            scalar = min(scalar, 0.70)

        if self.state.day_start_equity > 0:
            day_pnl = (self.state.equity - self.state.day_start_equity) / self.state.day_start_equity
            if day_pnl <= -cfg.max_daily_loss:
                self.state.day_halt = True
                msgs.append(f"day loss {day_pnl:.2%}")
        if self.state.day_halt and not self.state.kill:
            scalar = min(scalar, 0.25)

        self.state.risk_scalar = float(scalar)
        self.state.messages = msgs
        return self.state.risk_scalar

    def clip(self, notionals: dict[str, float], equity: float) -> dict[str, float]:
        """先乘 risk_scalar，再套单币名义帽与总杠杆帽。权益或 scalar≤0 则全平。"""
        scalar = self.compute_scalar()
        if equity <= 0 or scalar <= 0:
            return {s: 0.0 for s in notionals}
        out = {s: v * scalar for s, v in notionals.items()}
        max_sym = equity * self.cfg.max_symbol_weight * self.cfg.max_gross_leverage
        for s, v in list(out.items()):
            if abs(v) > max_sym:
                out[s] = float(np.sign(v) * max_sym)
        gross = sum(abs(v) for v in out.values())
        max_gross = equity * self.cfg.max_gross_leverage
        if gross > max_gross and gross > 0:
            out = {s: v * (max_gross / gross) for s, v in out.items()}
        self.state.gross_notional = sum(abs(v) for v in out.values())
        self.state.net_notional = sum(out.values())
        return out
