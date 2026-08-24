"""加载训练好的 PPO 集成，输出 [-1, 1] 上的连续仓位 unit。

推理路径：最近 7 根 bar 的 [7, n_feat] 窗口 → median/IQR 稳健标准化 → 各成员
确定性 Actor 平均 → EMA 平滑 → min_position 死区。权重缺失、kind 不是 ppo、
或特征 schema 对不上时不交易（unit=0），不再回退任何规则基准。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from loguru import logger

from betatrend.config import ROOT, Settings
from betatrend.nn.dataset import FEATURE_NAMES, N_FEAT, SEQ_LEN
from betatrend.nn.env import last_window
from betatrend.nn.model import PPOActorCritic
from betatrend.signals import smooth_unit

FEAT_CLIP = 8.0


class NeuralPolicy:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.cfg = settings.strategy
        self._ready = False
        self._nets: list[PPOActorCritic] = []
        self._median: np.ndarray | None = None
        self._iqr: np.ndarray | None = None
        self._seq_len = int(getattr(self.cfg, "seq_len", SEQ_LEN) or SEQ_LEN)
        self._last_unit = 0.0
        self._load()

    @property
    def ready(self) -> bool:
        return self._ready

    def reset(self) -> None:
        """清空 EMA 状态。新回测 / 新 paper 会话必须调用，否则会把上一段仓位记忆带过来。"""
        self._last_unit = 0.0

    def last_unit(self) -> float:
        return float(self._last_unit)

    def restore_last_unit(self, unit: float) -> None:
        """续跑 paper 时恢复 EMA，避免仓位还在、平滑状态却从 0 开始。"""
        self._last_unit = float(np.clip(unit, -1.0, 1.0))

    def _load(self) -> None:
        path = ROOT / self.cfg.nn_model_path
        if not path.exists():
            logger.warning("NN weights missing at {} — staying flat", path)
            return
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("kind") != "ppo":
            logger.warning("NN checkpoint is not PPO (kind={}) — staying flat", payload.get("kind"))
            return
        if list(payload.get("feature_names", FEATURE_NAMES)) != list(FEATURE_NAMES):
            logger.warning("NN feature schema mismatch — staying flat")
            return
        seq_len = int(payload.get("seq_len", self._seq_len))
        n_feat = int(payload.get("n_feat", N_FEAT))
        if seq_len != SEQ_LEN or n_feat != N_FEAT:
            logger.warning("NN observation shape mismatch ({}, {}) — staying flat", seq_len, n_feat)
            return
        hidden = [int(v) for v in payload.get("hidden", self.cfg.nn_hidden)]
        arch = str(payload.get("arch", self.cfg.nn_arch))
        self._seq_len = seq_len
        self._median = np.asarray(payload["median"], dtype=np.float32)
        self._iqr = np.asarray(payload["iqr"], dtype=np.float32)
        if self._median.shape[-1] != N_FEAT or self._iqr.shape[-1] != N_FEAT:
            logger.warning("NN scaler width mismatch — staying flat")
            return
        self._nets = []
        for state in payload["states"]:
            try:
                net = PPOActorCritic(n_feat=n_feat, seq_len=seq_len, hidden=hidden, arch=arch)
                net.load_state_dict(state)
            except (RuntimeError, ValueError) as exc:
                logger.warning("NN weight schema incompatible ({}) — staying flat", exc)
                self._nets = []
                return
            net.eval()
            self._nets.append(net)
        self._ready = bool(self._nets)
        logger.info("Loaded {} PPO members from {}", len(self._nets), path)

    @torch.no_grad()
    def predict_unit(self, panel: pd.DataFrame) -> float:
        if not self._ready:
            return 0.0
        win = last_window(panel, seq_len=self._seq_len)
        x = (win - self._median) / np.clip(self._iqr, 1e-6, None)
        x = np.clip(x, -FEAT_CLIP, FEAT_CLIP).astype(np.float32)
        xt = torch.tensor(x[None, ...], dtype=torch.float32)
        nn_unit = float(np.mean([float(net(xt).squeeze().cpu()) for net in self._nets]))
        unit = smooth_unit(
            nn_unit,
            self._last_unit,
            smooth=self.cfg.nn_smooth,
            min_position=self.cfg.min_position,
            long_only=self.cfg.long_only,
        )
        self._last_unit = unit
        return float(np.clip(unit, -1.0, 1.0))
