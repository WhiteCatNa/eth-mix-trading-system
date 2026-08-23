"""部署门禁：walk-forward 报告不合格时，禁止把神经网络 overlay 送上实盘。

门禁读 ``models/eth_decision.json``（训练结束时写出），检查：
  - OOS 神经网络路径的 Sharpe 必须 **严格大于** min_oos_sharpe
  - 折数足够
  - chosen_blend 必须等于各折验证集选出的 blend 中位数
    （防止用全样本再挑一遍 blend，把 OOS 成绩做高）
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from betatrend.config import ROOT, Settings


@dataclass
class GateResult:
    """一次门禁评估的结构化结果，失败原因全部放进 reasons，便于审计。"""

    passed: bool
    reasons: list[str] = field(default_factory=list)
    oos_sharpe: float | None = None
    n_folds: int | None = None
    chosen_blend: float | None = None
    fold_blend_median: float | None = None
    report_path: str | None = None


def report_path(settings: Settings) -> Path:
    """门禁报告路径：优先 deploy.report_path，否则跟权重文件同 stem 换 .json。"""
    rel = getattr(getattr(settings, "deploy", None), "report_path", "") or ""
    if rel:
        return ROOT / rel
    return ROOT / Path(settings.strategy.nn_model_path).with_suffix(".json")


def evaluate_deploy_gate(settings: Settings, path: Path | None = None) -> GateResult:
    """只评估、不抛错。关闭门禁时视为通过并记下原因，方便测试与人工覆盖。"""
    cfg = settings.deploy
    p = path or report_path(settings)
    result = GateResult(passed=False, report_path=str(p))
    if not cfg.enabled:
        result.passed = True
        result.reasons.append("deploy gate disabled")
        return result
    if not p.exists():
        result.reasons.append(f"missing walk-forward report: {p}")
        return result
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        result.reasons.append(f"invalid report JSON: {exc}")
        return result

    nn = payload.get("oos_neural_blend") or {}
    try:
        sharpe = float(nn.get("sharpe"))
    except (TypeError, ValueError):
        result.reasons.append("report missing oos_neural_blend.sharpe")
        return result
    result.oos_sharpe = sharpe
    try:
        n_folds = int(payload.get("n_folds"))
    except (TypeError, ValueError):
        result.reasons.append("report missing n_folds")
        return result
    result.n_folds = n_folds
    try:
        chosen = float(payload.get("chosen_blend"))
    except (TypeError, ValueError):
        result.reasons.append("report missing chosen_blend")
        return result
    result.chosen_blend = chosen
    folds = payload.get("fold_blends") or []
    if folds:
        result.fold_blend_median = float(statistics.median(float(x) for x in folds))

    if n_folds < int(cfg.min_folds):
        result.reasons.append(f"n_folds {n_folds} < min_folds {cfg.min_folds}")
    # 必须严格大于阈值：Sharpe==0 且阈值==0 时仍拒绝，避免“没边”的模型上实盘。
    if not (sharpe > float(cfg.min_oos_sharpe)):
        result.reasons.append(
            f"OOS net Sharpe {sharpe:.4f} is not > {float(cfg.min_oos_sharpe):.4f}"
        )
    if cfg.require_fold_blend:
        if result.fold_blend_median is None:
            result.reasons.append("fold_blends missing; refusing full-sample blend")
        elif abs(chosen - result.fold_blend_median) > 1e-9:
            result.reasons.append(
                "chosen_blend "
                f"{chosen} != fold-median {result.fold_blend_median} (full-sample blend leak)"
            )
    result.passed = not result.reasons
    return result


def assert_deploy_gate(settings: Settings, path: Path | None = None) -> GateResult:
    """实盘下单前的硬门禁：不通过则抛 RuntimeError，由 ControlPlane 调用。"""
    result = evaluate_deploy_gate(settings, path=path)
    if not result.passed:
        detail = "; ".join(result.reasons) or "unknown"
        raise RuntimeError(f"Live blocked: deploy gate failed: {detail}")
    return result


# 测试夹具使用的旧名
evaluate_deploy_gate = evaluate_deploy_gate
