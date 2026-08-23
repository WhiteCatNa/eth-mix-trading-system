"""日志：控制台 + 滚动文件；审计：JSONL 一行一个事件。"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from loguru import logger

from betatrend.config import ROOT, Settings

_configured = False


def setup_logging(settings: Settings) -> None:
    global _configured
    if _configured:
        return
    logger.remove()
    logger.add(lambda m: print(m, end=""), level=settings.logging.level)
    log_path = ROOT / settings.logging.file
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(str(log_path), rotation="10 MB", level=settings.logging.level)
    _configured = True


def audit(settings: Settings, event: str, **payload: object) -> None:
    path = ROOT / settings.logging.audit_file
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **payload,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")
