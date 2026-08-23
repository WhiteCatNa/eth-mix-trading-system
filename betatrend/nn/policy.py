"""加载训练好的集成网络，输出 [-1, 1] 上的连续仓位 unit。

推理路径：特征行 → 用训练折的 median/IQR 稳健标准化 → 各成员网络平均
→ 与 TSMOM prior 按 blend 混合 → EMA 平滑 → min_position 死区。
权重文件缺失或特征 schema 对不上时，退回纯 TSMOM，保证交易台能继续跑。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from loguru import logger

from betatrend.config import ROOT, Settings
from betatrend.mathx import score_to_unit
from betatrend.nn.dataset import FEATURE_NAMES, last_feature_row
from betatrend.nn.model import DecisionNet

FEAT_CLIP = 8.0


class NeuralPolicy:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cfg = settings.strategy
        self._ready = False
        self._nets: list[DecisionNet] = []
        self._median: np.ndarray | None = None
        self._iqr: np.ndarray | None = None
        self._blend = float(self.cfg.nn_blend)
        self._delta_gain = float(getattr(self.cfg, "nn_delta_gain", 0.5))
        self._last_unit = 0.0
        self._load()

    @property
    def ready(self) -> bool:
        return self._ready

    def reset(self) -> None:
        """清空 EMA 状态。新回测 / 新 paper 会话必须调用，否则会把上一段仓位记忆带过来。"""
        self._last_unit = 0.0

    def _load(self) -> None:
        path = ROOT / self.cfg.nn_model_path
        if not path.exists():
            logger.warning("NN weights missing at {} — falling back to TSMOM", path)
            return
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if list(payload.get("feature_names", FEATURE_NAMES)) != list(FEATURE_NAMES):
            logger.warning("NN feature schema mismatch — falling back to TSMOM")
            return
        hidden = tuple(payload.get("hidden", self.cfg.nn_hidden))
        dropout = float(payload.get("dropout", self.cfg.nn_dropout))
        self._delta_gain = float(payload.get("delta_gain", getattr(self.cfg, "nn_delta_gain", 0.5)))
        self._median = np.asarray(payload["median"], dtype=np.float32)
        self._iqr = np.asarray(payload["iqr"], dtype=np.float32)
        self._blend = float(payload.get("blend", self.cfg.nn_blend))
        self._nets = []
        for state in payload["states"]:
            net = DecisionNet(
                len(FEATURE_NAMES), hidden=hidden, dropout=dropout, delta_gain=self._delta_gain
            )
            net.load_state_dict(state)
            net.eval()
            self._nets.append(net)
        self._ready = bool(self._nets)
        logger.info("Loaded {} NN members from {}", len(self._nets), path)

    @torch.no_grad()
    def predict_unit(self, panel: pd.DataFrame, tsmom_score: float) -> float:
        prior = score_to_unit(
            tsmom_score,
            scale=self.cfg.score_scale,
            min_position=self.cfg.min_position,
            long_only=self.cfg.long_only,
        )
        if not self._ready:
            return prior
        row = last_feature_row(panel)
        x = (row - self._median) / np.clip(self._iqr, 1e-6, None)
        x = np.clip(x, -FEAT_CLIP, FEAT_CLIP).astype(np.float32)
        xt = torch.tensor(x.reshape(1, -1), dtype=torch.float32)
        tt = torch.tensor([prior], dtype=torch.float32)
        nn_unit = float(np.mean([float(net(xt, tt).squeeze().cpu()) for net in self._nets]))
        unit = self._blend * nn_unit + (1.0 - self._blend) * prior
        if self.cfg.long_only:
            unit = max(unit, 0.0)
        smooth = float(np.clip(self.cfg.nn_smooth, 0.0, 0.95))
        unit = (1.0 - smooth) * unit + smooth * self._last_unit
        if abs(unit) < self.cfg.min_position:
            unit = 0.0
        self._last_unit = unit
        return float(np.clip(unit, -1.0, 1.0))
