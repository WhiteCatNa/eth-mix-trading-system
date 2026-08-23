"""小型 MLP：在 TSMOM 上加门控残差，未训练网络的输出恒等于 TSMOM。

设计意图：神经网络只能「赚到」偏离基准的权利，不能从随机初始化就胡乱改仓。
delta 头零初始化 → 残差为 0；gate 偏置 -2 → sigmoid≈0.12，再乘 delta_gain，
起步时几乎不偏离。训练必须同时打开门并给出有用的 delta，才会改变仓位。
"""
from __future__ import annotations

import torch
from torch import nn


class DecisionNet(nn.Module):
    """Backbone predicts (delta, gate). Output is clip(tsmom + gain * gate * delta).

    Delta and gate heads start at TSMOM: delta is zero-init, gate bias is strongly
    negative so the net must earn the right to deviate.
    """

    def __init__(
        self,
        n_in: int,
        hidden: tuple[int, ...] = (64, 32),
        dropout: float = 0.25,
        delta_gain: float = 0.5,
    ):
        super().__init__()
        self.delta_gain = float(delta_gain)
        layers: list[nn.Module] = []
        last = n_in
        for h in hidden:
            layers.extend(
                [
                    nn.Linear(last, h),
                    nn.LayerNorm(h),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
            last = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(last, 1)
        self.gate = nn.Linear(last, 1)
        self._last_gate: torch.Tensor | None = None
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.25)
                nn.init.zeros_(m.bias)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, -2.0)

    def forward(self, x: torch.Tensor, tsmom_unit: torch.Tensor) -> torch.Tensor:
        """unit = clip(tsmom + delta_gain * σ(gate) * tanh(delta), -1, 1)。"""
        if tsmom_unit.ndim == 1:
            tsmom_unit = tsmom_unit.unsqueeze(-1)
        h = self.backbone(x)
        delta = torch.tanh(self.head(h))
        gate = torch.sigmoid(self.gate(h))
        self._last_gate = gate
        return torch.clamp(tsmom_unit + self.delta_gain * gate * delta, -1.0, 1.0)
