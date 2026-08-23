"""ETH 决策网：用扣费后收益/Sharpe 训练的连续仓位策略（正多负空）。

未训练时输出等于 TSMOM。``NeuralPolicy`` / ``RLPolicy`` 是同一个推理入口。
"""
from __future__ import annotations

from betatrend.nn.policy import NeuralPolicy
from betatrend.nn.train import train_decision_net

RLPolicy = NeuralPolicy

__all__ = ["NeuralPolicy", "RLPolicy", "train_decision_net"]
