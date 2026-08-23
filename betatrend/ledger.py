"""仓位账本：永续数量、盯市、资金费、β/残差归因。

线性 USDT 本位合约：每根 bar 把价格变动立刻记进 cash（标记入账），
因此 ``equity()`` 在正常路径下就等于 cash，不再另加未实现盈亏。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from betatrend.domain import Fill


@dataclass
class Ledger:
    cash: float  # 标记入账后的权益代理（含已实现手续费/滑点/资金费）
    qty: dict[str, float] = field(default_factory=dict)  # 带符号张数：正多负空
    avg_px: dict[str, float] = field(default_factory=dict)  # 预留均价字段，当前未用

    def equity(self, prices: dict[str, float]) -> float:
        """返回盯市权益。价格参数保留给将来“不按 bar 标记”的路径。"""
        upnl = 0.0
        for s, q in self.qty.items():
            px = prices.get(s, 0.0)
            if q and px:
                # 线性 USDT-M：PnL 已在 apply_mark 折进 cash。
                upnl += 0.0
        return self.cash + upnl

    def notional(self, symbol: str, price: float) -> float:
        """带符号名义本金 = 数量 × 价格。"""
        return self.qty.get(symbol, 0.0) * price

    def apply_mark(self, symbol: str, d_price: float) -> float:
        """价格变动 d_price 带来的盯市盈亏，写入 cash 并返回该笔 PnL。"""
        pnl = self.qty.get(symbol, 0.0) * d_price
        self.cash += pnl
        return pnl

    def apply_funding(self, symbol: str, price: float, rate: float) -> float:
        """资金费：多头支付正费率（PnL = -名义 × rate），空头方向相反。"""
        n = self.notional(symbol, price)
        pnl = -n * rate
        self.cash += pnl
        return pnl

    def apply_fill(self, fill: Fill) -> None:
        """成交：扣手续费+滑点，按 qty_delta 改持仓。市价全成，不处理部分成交。"""
        self.cash -= fill.fee + fill.slippage
        q = self.qty.get(fill.symbol, 0.0)
        self.qty[fill.symbol] = q + fill.qty_delta


def split_beta_residual_pnl(
    notionals: dict[str, float],
    returns: dict[str, float],
    betas: dict[str, float],
    market_ret: float,
) -> tuple[float, float]:
    """把标的收益拆成 β 部分与残差：r ≈ β·r_m + ε，PnL ≈ 名义 × r。

    单币且 market=trade 时残差应接近 0，β PnL ≈ 全部价格 PnL。
    """
    beta_pnl = 0.0
    resid_pnl = 0.0
    for s, n in notionals.items():
        r = returns.get(s, 0.0)
        b = betas.get(s, 1.0)
        beta_pnl += n * b * market_ret
        resid_pnl += n * (r - b * market_ret)
    return beta_pnl, resid_pnl
