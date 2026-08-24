"""PPO Actor-Critic：最近 seq_len 根 K 线 [B, T, n_feat] → 仓位 unit ∈ [-1, 1]。

张量约定（PyTorch batch_first）：
    [batch, seq=T, feat=n_feat]

结构（arch="mlp"，默认）：
    展平 [B, T*n_feat]
    → FC(h1) + LayerNorm + ReLU
    → FC(h2) + LayerNorm + ReLU        ← 共享特征，复制进两个分支
      ├─ Actor: FC(h2//2)+LayerNorm+ReLU
      │    ├─ 均值头 FC(1)        （tanh 之前的仓位均值）
      │    └─ 标准差头 FC(1)+Softplus   （探索方差，>0）
      └─ Critic: FC(h2//2)+LayerNorm+ReLU → FC(1)  （V(s)）

结构（arch="lstm"，仅供对照）：把展平换成 LSTM(h1) → LSTM(h2) 取最后时间步。
特征表里的 ret_1..168 / vol_24..168 / ema_gap 已经是多尺度时序聚合，
所以循环栈能额外榨出的时间信息有限，而它要花掉 ~86% 的参数。

不使用 Dropout。rollout 在 eval 下采样并记录 log_prob，PPO 更新在 train 下重算，
两者的 dropout mask 不同会让重要性比率 π_new/π_old 变成噪声：实测在任何梯度更新之前
就有 27% 的样本落到裁剪带外，信任域裁的是 mask 抖动而不是策略变化。
正则化交给容量控制和验证集早停。

均值头零初始化：确定性推理时未训练输出 ≈ 0。
高斯定义在 tanh 之前的空间，动作再 squash 到 [-1, 1]，log_prob 做变量替换。
没有 TSMOM 残差、没有 gate。
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.distributions import Normal

from betatrend.nn.dataset import N_FEAT, SEQ_LEN

_EPS = 1e-6
_STD_MIN = 1e-4
_STD_MAX = 2.0
DEFAULT_HIDDEN: tuple[int, int] = (64, 64)
DEFAULT_ARCH = "mlp"


def _block(fan_in: int, width: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(fan_in, width), nn.LayerNorm(width), nn.ReLU())


class PPOActorCritic(nn.Module):
    """[B, T, n_feat] → 仓位。共享 trunk 后分 Actor / Critic。"""

    def __init__(
        self,
        n_feat: int = N_FEAT,
        seq_len: int = SEQ_LEN,
        hidden: Sequence[int] = DEFAULT_HIDDEN,
        arch: str = DEFAULT_ARCH,
    ):
        super().__init__()
        self.n_feat = int(n_feat)
        self.seq_len = int(seq_len)
        self.arch = str(arch)
        h = [int(v) for v in hidden]
        if len(h) < 2:
            raise ValueError(f"hidden needs at least two widths, got {hidden}")
        if self.arch not in ("mlp", "lstm"):
            raise ValueError(f"arch must be 'mlp' or 'lstm', got {arch!r}")
        self.hidden = tuple(h)

        if self.arch == "lstm":
            self.lstm1 = nn.LSTM(self.n_feat, h[0], batch_first=True)
            self.ln1 = nn.LayerNorm(h[0])
            self.lstm2 = nn.LSTM(h[0], h[1], batch_first=True)
            self.ln2 = nn.LayerNorm(h[1])
            self.trunk = nn.Sequential(*[_block(h[i], h[i + 1]) for i in range(1, len(h) - 1)])
            trunk_out = h[-1]
        else:
            widths = [self.seq_len * self.n_feat, *h]
            self.trunk = nn.Sequential(*[_block(widths[i], widths[i + 1]) for i in range(len(h))])
            trunk_out = h[-1]

        branch = max(trunk_out // 2, 8)
        self.actor_h = _block(trunk_out, branch)
        self.mean_head = nn.Linear(branch, 1)
        self.std_head = nn.Linear(branch, 1)
        self.softplus = nn.Softplus()
        self.critic_h = _block(trunk_out, branch)
        self.value_head = nn.Linear(branch, 1)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.std_head.weight)
        nn.init.zeros_(self.std_head.bias)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """共享特征 [B, hidden[-1]]。x 必须是 [B, T=seq_len, F=n_feat]。"""
        if x.ndim != 3:
            raise ValueError(f"expected [B, T, F], got {tuple(x.shape)}")
        if self.arch == "lstm":
            h, _ = self.lstm1(x)
            h = self.ln1(h)
            h, _ = self.lstm2(h)
            h = self.ln2(h)
            return self.trunk(h[:, -1, :])
        return self.trunk(x.flatten(1))

    def _heads(self, shared: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """同一份 trunk 特征复制进 Actor / Critic。返回 mu_raw, std, value，皆 [B]。"""
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
        self, x: torch.Tensor, *, deterministic: bool = False, std_scale: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """采样或贪心动作。返回 action, log_prob, value, entropy，皆 [B]。

        ``std_scale`` 缩放探索标准差。采样带来的换手会被奖励里的成本项直接扣掉，
        而观测里没有仓位，agent 无法归因这块自伤成本，所以训练后期需要把它收小。
        """
        dist, value = self._dist(x)
        if std_scale != 1.0:
            dist = Normal(dist.mean, (dist.stddev * float(std_scale)).clamp(_STD_MIN, _STD_MAX))
        u = dist.mean if deterministic else dist.sample()
        action = torch.tanh(u)
        logp = _tanh_log_prob(dist, u, action)
        return action, logp, value, dist.entropy()

    def evaluate(
        self, x: torch.Tensor, action: torch.Tensor, *, std_scale: float = 1.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """对已存的 tanh 动作 ∈ (-1, 1) 算 log_prob / value / entropy。

        ``std_scale`` 必须与产生该动作的 rollout 一致，否则重要性比率会失真。
        """
        dist, value = self._dist(x)
        if std_scale != 1.0:
            dist = Normal(dist.mean, (dist.stddev * float(std_scale)).clamp(_STD_MIN, _STD_MAX))
        a = action.clamp(-1.0 + 1e-4, 1.0 - 1e-4)
        u = torch.atanh(a)
        logp = _tanh_log_prob(dist, u, a)
        return logp, value, dist.entropy()


def _tanh_log_prob(dist: Normal, u: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
    """变量替换：a = tanh(u)。"""
    return dist.log_prob(u) - torch.log(1.0 - action.pow(2) + _EPS)


DecisionNet = PPOActorCritic
