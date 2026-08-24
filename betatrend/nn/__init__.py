"""ETH 决策网：PPO Actor-Critic，观测为最近 7 根 K 线 [7, n_feat]，输出连续仓位。

未训练或权重缺失时输出 0（空仓）。``NeuralPolicy`` / ``RLPolicy`` 是同一个推理入口。
"""
from __future__ import annotations

from betatrend.nn.policy import NeuralPolicy
from betatrend.nn.train import list_fold_jobs, train_decision_net, train_fold

RLPolicy = NeuralPolicy

__all__ = ["NeuralPolicy", "RLPolicy", "train_decision_net", "train_fold", "list_fold_jobs"]
