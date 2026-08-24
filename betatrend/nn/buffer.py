"""PPO 用的近期 rollout 回放。

PPO 是 on-policy 算法，不能像 DQN 那样随便抽很旧的转移。这里只保留最近
K 次采样的轨迹，每条带着采集时的 logπ(a|s)。PPO clip 相对“采集策略”
做重要性比，轻度 off-policy 仍有界。

K=1 就是标准 PPO（只用当前 rollout）。K>1 把最近几次轨迹拼起来再打乱
mini-batch,提高样本利用率，又不把半年前的策略混进来。
"""
from __future__ import annotations

from collections import deque

import numpy as np


class ReplayBuffer:
    """容量按“整段 rollout”计，不是按单步。"""

    def __init__(self, n_rollouts: int = 4):
        self.n_rollouts = max(int(n_rollouts), 1)
        self._items: deque[dict[str, np.ndarray]] = deque(maxlen=self.n_rollouts)

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        logp: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> None:
        self._items.append(
            {
                "obs": np.asarray(obs, dtype=np.float32),
                "actions": np.asarray(actions, dtype=np.float32),
                "logp": np.asarray(logp, dtype=np.float32),
                "advantages": np.asarray(advantages, dtype=np.float32),
                "returns": np.asarray(returns, dtype=np.float32),
            }
        )

    def packed(self) -> dict[str, np.ndarray]:
        if not self._items:
            raise RuntimeError("ReplayBuffer is empty")
        keys = ("obs", "actions", "logp", "advantages", "returns")
        return {k: np.concatenate([it[k] for it in self._items], axis=0) for k in keys}

    def __len__(self) -> int:
        return int(sum(len(it["actions"]) for it in self._items))

    @property
    def n_stored(self) -> int:
        return len(self._items)


ReplayBuffer = ReplayBuffer
ReplayBuffer = ReplayBuffer
