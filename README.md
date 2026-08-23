# BETA-TREND Desk

Binance USDT-M **ETH 单币择时**。核心闭环：

**拉 ETH 数据 → 决策网输出连续仓位 → 多/空/平信号 → 下单。**

1. 行情 QC  
2. 决策网（`decision: rl`）给出 `unit ∈ [-1, 1]`：正做多、负做空  
3. `|unit| < 5%` 全仓 → 空仓，**不开新单**  
4. 组合与杠杆帽  
5. 盘前回撤 / 日亏损 / kill  
6. OMS 买/卖  
7. 执行（paper / testnet）  
8. 账本  

默认 `long_only: false`，多空都做。不是投资建议。回测 ≠ 实盘。

## Signal

交易标的：`universe.symbol`（默认 `ETHUSDT`）。

- 决策网用 ETH 特征输出连续仓位；无权重时退回 TSMOM。  
- `|unit| < min_position`（默认 0.05）视为空仓，避免灰尘仓来回打手续费。  
- 已有仓位而新信号落到 5% 以下时，目标打成 0，**平仓**而不是继续拿着。

成交：**T 收盘算信号，T+1 开盘成交**。

## Quick start

```bash
cd "/Users/jiangzhiwei/eth mix trading system"
source .venv/bin/activate
python -m betatrend backtest --demo
python -m betatrend fetch --days 365
python -m betatrend backtest --days 365
python -m betatrend paper-once --demo
pytest -q
```
