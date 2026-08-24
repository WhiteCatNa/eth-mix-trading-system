"""PPO 用的近期 rollout 回放。

PPO 是 on-policy 算法，不能像 DQN 那样随便抽很旧的转移。这里只保留最近
K 次采样的轨迹，每条带着采集时的 logπ(a|s)。PPO clip 相对“采集策略”
做重要性比，轻度 off-policy 仍有界。

K=1 就是标准 PPO（只用当前 rollout）。K>1 把最近几次轨迹拼起来再打乱
mini-batch,提高样本利用率，又不把半年前的策略混进来。

观测本体不进 buffer，进来的是它在调用方 obs 数组里的行号。一次训练里
obs 是固定的 [n, seq_len, n_feat]，K 份副本会让每个 epoch 多拷 K×n×seq_len×n_feat
个 float，n=1 万量级时是几十 MB 的纯浪费。
"""
from __future__ import annotations

from collections import deque

import numpy as np

_KEYS = ("index", "actions", "logp", "advantages", "returns")


class ReplayBuffer:
    """容量按“整段 rollout”计，不是按单步。"""

    def __init__(self, n_rollouts: int = 4):
        self.n_rollouts = max(int(n_rollouts), 1)
        self._items: deque[dict[str, np.ndarray]] = deque(maxlen=self.n_rollouts)

    def add(
        self,
        index: np.ndarray,
        actions: np.ndarray,
        logp: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> None:
        """index 是观测在调用方 obs 数组里的行号；五个数组必须等长。"""
        item = {
            "index": np.asarray(index, dtype=np.int64),
            "actions": np.asarray(actions, dtype=np.float32),
            "logp": np.asarray(logp, dtype=np.float32),
            "advantages": np.asarray(advantages, dtype=np.float32),
            "returns": np.asarray(returns, dtype=np.float32),
        }
        n = len(item["index"])
        bad = sorted(k for k, v in item.items() if len(v) != n)
        if bad:
            raise ValueError(f"rollout arrays must all have length {n}; mismatched: {bad}")
        self._items.append(item)

    def packed(self) -> dict[str, np.ndarray]:
        if not self._items:
            raise RuntimeError("ReplayBuffer is empty")
        return {k: np.concatenate([it[k] for it in self._items], axis=0) for k in _KEYS}

    def __len__(self) -> int:
        return int(sum(len(it["index"]) for it in self._items))

    @property
    def n_stored(self) -> int:
        return len(self._items)
