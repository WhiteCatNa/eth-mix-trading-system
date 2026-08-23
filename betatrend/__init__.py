"""BETA-TREND：币安 USDT 本位永续上的 ETH 单币择时交易台。

系统做什么
----------
对 ETHUSDT 永续做连续仓位择时，多空都做。研究 / paper / testnet / live 共用 DeskCycle：

    拉 ETH 行情 → 决策网输出 unit ∈ [-1, 1]
    → 买卖信号（仓位 + LONG/SHORT/FLAT）→ OMS → 执行 → 账本

仓位怎么算
----------
1. 决策网（``decision: rl``）输出连续 unit：正=做多，负=做空。无权重时保持空仓。
2. ``|unit| < min_position``（默认全仓 5%）视为空仓，**不开新单**。
3. 名义本金 = 权益 × 风险预算 × min(杠杆帽, 目标波动 / 实现波动) × unit。
4. 默认 ``long_only: false``，多空都做。

免责声明
----------
任何策略都不保证盈利。永续合约带杠杆，本金可能归零。
回测与 paper 成交 ≠ 实盘成绩。本仓库仅供研究与教育，合规与资金风险由使用者自负。
"""

from __future__ import annotations

__version__ = "0.1.0"
__system_name__ = "BETA-TREND"
