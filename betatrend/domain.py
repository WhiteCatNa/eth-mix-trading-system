"""交易台的“通用语言”：从信号到成交共用的枚举与数据结构。

研究、回测、paper、实盘都不允许各自发明一套订单/仓位类型。
这里的对象会在策略、组合、OMS、执行、账本之间原样传递，避免口径漂移。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd


class Venue(str, Enum):
    """成交场所。当前只接币安 USDT 本位永续（fapi）。"""

    BINANCE_USDTM = "binance_usdtm"


class Side(str, Enum):
    """订单方向。与币安 REST 字段一致，用 BUY/SELL 而不是 long/short。

    对永续来说：BUY 增加多头（或减少空头），SELL 增加空头（或减少多头）。
    """

    BUY = "BUY"
    SELL = "SELL"


class OrderType(str, Enum):
    """订单类型。Desk 目前只发市价单，LIMIT 预留给以后做 maker 入场。"""

    MARKET = "MARKET"
    LIMIT = "LIMIT"


class OrderStatus(str, Enum):
    """父单/子单生命周期。

    PARENT：已生成子意图、尚未全部成交。
    FILLED：无子单（无需调仓）或已视为完成。
    """

    NEW = "NEW"
    PARENT = "PARENT"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    REJECTED = "REJECTED"
    CANCELED = "CANCELED"


class DeskMode(str, Enum):
    """账户运行模式。research / paper 走本地模拟成交；testnet/live 走签名 REST。
    """

    RESEARCH = "research"
    PAPER = "paper"
    TESTNET = "testnet"
    LIVE = "live"


@dataclass(frozen=True)
class Instrument:
    """合约规格。frozen：规格是交易所常量，运行时不应被改写。

    tick_size / step_size / min_notional 用于把名义本金换成合法下单数量。
    """

    symbol: str
    venue: Venue = Venue.BINANCE_USDTM
    tick_size: float = 0.01
    step_size: float = 0.001
    min_notional: float = 5.0


@dataclass
class TargetPosition:
    """策略输出的目标仓位（调仓前、尚未交给 OMS）。

    target_notional：带符号的 USDT 名义本金（正=多，负=空）。
    signal：连续仓位 unit ∈ [-1, 1]，不是离散档位。
    trend_score：原始 TSMOM 分数，供归因与日志。
    extras：决策来源（tsmom/neural）、杠杆、波动率等调试字段。
    """

    symbol: str
    strategy: str
    target_notional: float
    signal: float = 0.0
    trend_score: float = 0.0
    reason: str = ""
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class MarketSnapshot:
    """某一根 K 线收盘时的市场截面。

    panels：symbol → 截至当前时刻（含本 bar）的 OHLCV+funding，禁止带未来数据。
    prices：本 bar 用于计价的价格（回测里是收盘价算信号，下一根开盘成交）。
    risk_scalar：风控引擎给出的仓位缩放，1=满仓，0=必须平掉。
    """

    timestamp: pd.Timestamp
    panels: dict[str, pd.DataFrame]
    prices: dict[str, float]
    equity: float
    risk_scalar: float = 1.0
    bar_index: int = 0
    market_symbol: str = "ETHUSDT"


@dataclass
class Fill:
    """一笔已成交记录。账本只认 Fill，不直接认交易所原始 JSON。

    notional_delta / qty_delta 带符号：买入为正、卖出为负。
    fee 与 slippage 都是成本，apply_fill 时从现金里扣。
    """

    timestamp: pd.Timestamp
    symbol: str
    notional_delta: float
    qty_delta: float
    price: float
    fee: float
    slippage: float
    reason: str
    client_order_id: str = ""


@dataclass
class OrderIntent:
    """OMS 拆出来的子单意图，执行层按此下单。

    reduce_only：目标绝对值小于当前且同向时为 True，避免减仓单意外开反向仓。
    parent_id：把一次再平衡的多笔子单绑在同一个父单上，便于审计。
    """

    client_order_id: str
    symbol: str
    side: Side
    qty: float
    notional: float
    order_type: OrderType = OrderType.MARKET
    reduce_only: bool = False
    parent_id: str = ""
    reason: str = ""


# 测试夹具使用的旧名
OrderIntent = OrderIntent
