"""PPO Actor-Critic：最近 7 根 K 线 [B, 7, 30] → 仓位 unit ∈ [-1, 1]。

张量约定（PyTorch LSTM batch_first）：
    [batch, seq=7, feat=30]
取最后时间步后是 [batch, 64]，不是 [batch, 30, 64]。

结构：
    LSTM(128) + Dropout(0.2) + LayerNorm
    → LSTM(64) + Dropout(0.2) + LayerNorm
    → 最后时间步 [B, 64]
    → FC(128) + LayerNorm + ReLU + Dropout(0.3)
    → FC(64) + LayerNorm + ReLU + Dropout(0.3)     ← 共享特征，复制进两个分支
      ├─ Actor: FC(32)+LayerNorm+ReLU+Dropout(0.2)
      │    ├─ 均值头 FC(1)+Tanh        （仓位均值，∈[-1,1]）
      │    └─ 标准差头 FC(1)+Softplus   （探索方差，>0）
      └─ Critic: FC(32)+LayerNorm+ReLU+Dropout(0.2) → FC(1)  （V(s)）

输出层（均值/标准差/价值）不加 Dropout。
均值头零初始化：确定性推理时未训练输出 ≈ 0。
高斯定义在 tanh 之前的空间，动作再 squash 到 [-1, 1]，log_prob 做变量替换。
没有 TSMOM 残差、没有 gate。
"""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

_EPS = 1e-6
_STD_MIN = 1e-4
_STD_MAX = 2.0
_LSTM_DROP = 0.2
_FC_DROP = 0.3
_BRANCH_DROP = 0.2


def _dropouts(dropout: float) -> tuple[float, float, float]:
    """dropout<=0 关闭全部；否则用生产档位，不跟单个标量混用。"""
    if dropout <= 0.0:
        return 0.0, 0.0, 0.0
    return _LSTM_DROP, _FC_DROP, _BRANCH_DROP


class PPOActorCritic(nn.Module):
    """[B, 7, 30] → 仓位。共享 FC(64) 后分 Actor / Critic。"""

    def __init__(
        self,
        n_feat: int = 30,
        seq_len: int = 7,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.n_feat = int(n_feat)
        self.seq_len = int(seq_len)
        lstm_p, fc_p, branch_p = _dropouts(float(dropout))

        self.lstm1 = nn.LSTM(self.n_feat, 128, batch_first=True)
        self.lstm_drop1 = nn.Dropout(lstm_p)
        self.ln1 = nn.LayerNorm(128)
        self.lstm2 = nn.LSTM(128, 64, batch_first=True)
        self.lstm_drop2 = nn.Dropout(lstm_p)
        self.ln2 = nn.LayerNorm(64)

        self.fc = nn.Sequential(
            nn.Linear(64, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(fc_p),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(fc_p),
        )
        self.actor_h = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(branch_p),
        )
        self.mean_head = nn.Linear(32, 1)
        self.std_head = nn.Linear(32, 1)
        self.softplus = nn.Softplus()
        self.critic_h = nn.Sequential(
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(branch_p),
        )
        self.value_head = nn.Linear(32, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.std_head.weight)
        nn.init.zeros_(self.std_head.bias)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """共享特征 [B, 64]。x 必须是 [B, T=7, F=30]。"""
        if x.ndim != 3:
            raise ValueError(f"expected [B, T, F], got {tuple(x.shape)}")
        h, _ = self.lstm1(x)
        h = self.ln1(self.lstm_drop1(h))
        h, _ = self.lstm2(h)
        h = self.ln2(self.lstm_drop2(h))
        shared = self.fc(h[:, -1, :])
        return shared

    def _heads(self, shared: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """同一份 FC(64) 特征复制进 Actor / Critic。返回 mu_raw, std, value，皆 [B]。"""
        actor = self.actor_h(shared)
        critic = self.critic_h(shared)
        mu_raw = self.mean_head(actor).squeeze(-1)
        std = self.softplus(self.std_head(actor)).squeeze(-1).clamp(_STD_MIN, _STD_MAX)
        value = self.value_head(critic).squeeze(-1)
        return mu_raw, std, value

    def _dist(self, x: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        mu_raw, std, value = self._heads(self._encode(x))
        return Normal(mu_raw, std), value

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """确定性仓位 ∈ [-1, 1]，形状 [B, 1]（推理）。"""
        dist, _ = self._dist(x)
        return torch.tanh(dist.mean).unsqueeze(-1)

    def value(self, x: torch.Tensor) -> torch.Tensor:
        _, v = self._dist(x)
        return v

    def act(
        self, x: torch.Tensor, *, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """采样或贪心动作。返回 action, log_prob, value, entropy，皆 [B]。"""
        dist, value = self._dist(x)
        u = dist.mean if deterministic else dist.sample()
        action = torch.tanh(u)
        logp = _tanh_log_prob(dist, u, action)
        return action, logp, value, dist.entropy()

    def evaluate(
        self, x: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """对已存的 tanh 动作 ∈ (-1, 1) 算 log_prob / value / entropy。"""
        dist, value = self._dist(x)
        a = action.clamp(-1.0 + 1e-4, 1.0 - 1e-4)
        u = torch.atanh(a)
        logp = _tanh_log_prob(dist, u, a)
        return logp, value, dist.entropy()


def _tanh_log_prob(dist: Normal, u: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """变量替换：a = tanh(u)。"""
    return dist.log_prob(u) - torch.log(1.0 - action.pow(2) + _EPS)


DecisionNet = PPOActorCritic
