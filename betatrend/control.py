"""控制面：kill 开关、模式门禁、实盘二次确认。

下单路径必须先经过这里。research/paper 直接放行（本地模拟）；
testnet 要求 BINANCE_TESTNET=1；live 还要环境变量、YES 确认和部署门禁。
"""
from __future__ import annotations

import os
from pathlib import Path

from betatrend.config import ROOT, Settings
from betatrend.domain import DeskMode


class ControlPlane:
    def __init__(self, settings: Settings):
        self.settings = settings

    @property
    def mode(self) -> DeskMode:
        """账户 YAML 里的 mode，非法字符串会在枚举转换时直接炸，避免静默跑错环境。"""
        return DeskMode(self.settings.account.mode)

    def kill_tripped(self) -> bool:
        """环境变量 BETATREND_KILL=1 或磁盘上的 kill 文件存在，则视为已熔断。"""
        if os.environ.get("BETATREND_KILL", "0") == "1":
            return True
        return (ROOT / self.settings.control.kill_file).exists()

    def trip_kill(self, reason: str = "manual") -> Path:
        """写入 kill 文件（内容为原因）。DeskCycle 下一拍会 flatten；CLI ``kill`` 也走这里。"""
        p = ROOT / self.settings.control.kill_file
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(reason + "\n", encoding="utf-8")
        return p

    def assert_can_send_orders(self, confirm: str = "") -> None:
        """签名下单前的总闸。任一条件不满足就抛错，不发送 HTTP。

        live 额外要求：
          - BETATREND_ALLOW_LIVE=1（显式打开实盘）
          - BINANCE_TESTNET=0（防止“以为在测试网”却打到主网）
          - confirm == "YES"（CLI/调用方必须口头确认）
          - walk-forward 部署门禁通过
        """
        mode = self.mode
        if mode in (DeskMode.RESEARCH, DeskMode.PAPER):
            return
        if self.kill_tripped():
            raise RuntimeError("Kill switch is on — refusing orders")
        if mode is DeskMode.TESTNET:
            if os.environ.get("BINANCE_TESTNET", "1") != "1":
                raise RuntimeError("Testnet mode requires BINANCE_TESTNET=1")
            return
        if mode is DeskMode.LIVE:
            if os.environ.get("BETATREND_ALLOW_LIVE") != "1":
                raise RuntimeError("Live blocked: set BETATREND_ALLOW_LIVE=1")
            if os.environ.get("BINANCE_TESTNET", "1") != "0":
                raise RuntimeError("Live blocked: set BINANCE_TESTNET=0")
            if confirm != "YES":
                raise RuntimeError("Live blocked: type YES to confirm")
            from betatrend.gate import assert_deploy_gate

            assert_deploy_gate(self.settings)
            return
        raise RuntimeError(f"Unknown mode {mode}")
