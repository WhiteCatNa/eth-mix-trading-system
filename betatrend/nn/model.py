"""GRPO Actor：最近 7 根 K 线 [B, 7, n_feat] → 仓位 unit ∈ [-1, 1]。

张量约定（PyTorch LSTM batch_first）：
    [batch, seq=7, feat=n_feat]
取最后时间步后是 [batch, 64]，不是 [batch, n_feat, 64]。

结构：
    LSTM(128) + Dropout(0.2) + LayerNorm
    → LSTM(64) + Dropout(0.2) + LayerNorm
    → 最后时间步 [B, 64]
    → FC(128) + LayerNorm + ReLU + Dropout(0.3)
    → FC(64) + LayerNorm + ReLU + Dropout(0.3)
    → Actor: FC(32)+LayerNorm+ReLU+Dropout(0.2)
         ├─ 均值头 FC(1)            （tanh 之前的仓位均值）
         └─ 标准差头 FC(1)+Softplus （探索方差，>0）

没有 Critic。GRPO 用同一状态下 G 个采样动作的奖励做组内标准化当优势，
不需要 V(s)。输出头不加 Dropout。均值头零初始化：未训练时确定性输出 ≈ 0。
高斯定义在 tanh 之前的空间，动作再 squash 到 [-1, 1]，log_prob 做变量替换。
"""
from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Normal

from betatrend.nn.dataset import N_FEAT, SEQ_LEN

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


class GRPOActor(nn.Module):
    """[B, 7, n_feat] → 仓位。只有 Actor，没有价值头。"""

    def __init__(
        self,
        n_feat: int = N_FEAT,
        seq_len: int = SEQ_LEN,
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

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.std_head.weight)
        nn.init.zeros_(self.std_head.bias)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """共享特征 [B, 64]。x 必须是 [B, T=seq_len, F=n_feat]。"""
        if x.ndim != 3:
            raise ValueError(f"expected [B, T, F], got {tuple(x.shape)}")
        h = self.lstm1(x)
        if isinstance(h, tuple):
            h = h[0]
        h = self.ln1(self.lstm_drop1(h))
        h = self.lstm2(h)
        if isinstance(h, tuple):
            h = h[0]
        h = self.ln2(self.lstm_drop2(h))
        return self.fc(h[:, -1, :])

    def _heads(self, shared: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """返回 mu_raw, std，皆 [B]。"""
        actor = self.actor_h(shared)
        mu_raw = self.mean_head(actor).squeeze(-1)
        mu_raw = torch.nan_to_num(mu_raw, nan=0.0, posinf=8.0, neginf=-8.0).clamp(-8.0, 8.0)
        std = self.softplus(self.std_head(actor)).squeeze(-1)
        std = torch.nan_to_num(std, nan=_STD_MIN, posinf=_STD_MAX, neginf=_STD_MIN).clamp(
            _STD_MIN, _STD_MAX
        )
        if getattr(self, "_torchexplore", False):
            from betatrend.nn.explore import attach_tensor

            mu_raw = attach_tensor(mu_raw, self, "mu_raw")
            std = attach_tensor(std, self, "std")
        return mu_raw, std

    def _dist(self, x: torch.Tensor) -> Normal:
        mu_raw, std = self._heads(self._encode(x))
        return Normal(mu_raw, std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """确定性仓位 ∈ [-1, 1]，形状 [B, 1]（推理）。"""
        return torch.tanh(self._dist(x).mean).unsqueeze(-1)

    def act(
        self, x: torch.Tensor, *, deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """采样或贪心动作。返回 action, log_prob, entropy，皆 [B]。"""
        dist = self._dist(x)
        u = dist.mean if deterministic else dist.sample()
        action = torch.tanh(u)
        logp = _tanh_log_prob(dist, u, action)
        return action, logp, dist.entropy()

    def sample_group(self, x: torch.Tensor, n_group: int) -> tuple[torch.Tensor, torch.Tensor]:
        """同一批状态下各采 n_group 个动作。返回 action, log_prob，形状 [G, B]。"""
        g = max(int(n_group), 1)
        dist = self._dist(x)
        u = dist.sample((g,))
        action = torch.tanh(u)
        logp = _tanh_log_prob(dist, u, action)
        return action, logp

    def evaluate(self, x: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """对已存的 tanh 动作 ∈ (-1, 1) 算 log_prob / entropy。"""
        dist = self._dist(x)
        a = action.clamp(-1.0 + 1e-4, 1.0 - 1e-4)
        u = torch.atanh(a)
        logp = _tanh_log_prob(dist, u, a)
        return logp, dist.entropy()


def _tanh_log_prob(dist: Normal, u: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """变量替换：a = tanh(u)。broadcast 对齐 dist 与 sample 的前导维。"""
    return dist.log_prob(u) - torch.log(1.0 - action.pow(2) + _EPS)


DecisionNet = GRPOActor
PPOActorCritic = GRPOActor  # 旧测试 / 导入别名
