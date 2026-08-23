"""日志：控制台 + 滚动文件；审计：JSONL 一行一个事件。

``setup_logging`` 全局只配置一次，避免 pytest / 多次 CLI 把 sink 叠满。
审计日志与普通日志分开，方便事后对账 kill、成交、paper bar。
"""
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
    """追加一条带 UTC 时间戳的 JSON 对象。失败应向上抛，不能默默丢掉审计。"""
    path = ROOT / settings.control.audit_log
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **payload,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")
