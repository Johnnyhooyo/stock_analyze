"""
log_config.py — 统一日志配置
==============================
全项目唯一的日志初始化入口。

用法::

    from log_config import get_logger
    logger = get_logger(__name__)
    logger.info("消息", extra={"ticker": "0700.HK", "action": "BUY"})

控制台输出：
    彩色文本（INFO=蓝, WARNING=黄, ERROR=红），便于开发调试。

文件输出：
    JSON Lines 格式写入 data/logs/app.log，每行一条结构化日志。
    文件自动按 10 MB 滚动，保留最近 5 份。

日志级别：
    通过环境变量 LOG_LEVEL 控制（默认 INFO）。
    示例: LOG_LEVEL=DEBUG python3 daily_run.py

JSON 字段：
    timestamp, level, name, message + extra 字段（ticker, action, elapsed_ms 等）
"""

from __future__ import annotations

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# ──────────────────────────────────────────────────────────────────
#  ANSI 颜色
# ──────────────────────────────────────────────────────────────────

_RESET = "\033[0m"
_COLORS = {
    "DEBUG":    "\033[36m",   # 青色
    "INFO":     "\033[34m",   # 蓝色
    "WARNING":  "\033[33m",   # 黄色
    "ERROR":    "\033[31m",   # 红色
    "CRITICAL": "\033[35m",   # 紫色
}


class _ColorConsoleFormatter(logging.Formatter):
    """控制台彩色 Formatter。"""

    _FMT = "%(asctime)s  %(levelname)-8s  %(name)-28s  %(message)s"
    _DATEFMT = "%H:%M:%S"

    def __init__(self, use_color: bool = True):
        super().__init__(fmt=self._FMT, datefmt=self._DATEFMT)
        self._use_color = use_color and _supports_color()

    def format(self, record: logging.LogRecord) -> str:
        msg = super().format(record)
        if self._use_color:
            color = _COLORS.get(record.levelname, "")
            return f"{color}{msg}{_RESET}"
        return msg


def _supports_color() -> bool:
    """检测终端是否支持 ANSI 颜色。"""
    if not hasattr(sys.stdout, "isatty"):
        return False
    if not sys.stdout.isatty():
        return False
    if os.environ.get("NO_COLOR"):
        return False
    return True


# ──────────────────────────────────────────────────────────────────
#  JSON Lines Formatter（文件）
# ──────────────────────────────────────────────────────────────────

class _JsonLinesFormatter(logging.Formatter):
    """将 LogRecord 序列化为单行 JSON。"""

    # extra 字段白名单（排除 LogRecord 内置字段）
    _SKIP = frozenset(logging.LogRecord(
        "", 0, "", 0, "", (), None
    ).__dict__.keys()) | {"message", "asctime"}

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        entry: dict = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level":   record.levelname,
            "name":    record.name,
            "message": record.message,
        }
        # 附加 extra 字段（ticker, action, elapsed_ms …）
        for k, v in record.__dict__.items():
            if k not in self._SKIP and not k.startswith("_"):
                try:
                    json.dumps(v)   # 确保可序列化
                    entry[k] = v
                except (TypeError, ValueError):
                    entry[k] = str(v)

        if record.exc_info:
            entry["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(entry, ensure_ascii=False)


# ──────────────────────────────────────────────────────────────────
#  初始化函数
# ──────────────────────────────────────────────────────────────────

_initialized = False


def setup_logging(
    log_dir: Path | None = None,
    level: int | None = None,
) -> None:
    """
    初始化全局日志配置（幂等，多次调用无副作用）。

    Args:
        log_dir: 日志目录（默认 <project_root>/data/logs/）
        level:   日志级别（默认读 LOG_LEVEL 环境变量，否则 INFO）
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    # 决定日志级别
    if level is None:
        env_level = os.environ.get("LOG_LEVEL", "INFO").upper()
        level = getattr(logging, env_level, logging.INFO)

    # 根 logger
    root = logging.getLogger()
    root.setLevel(level)

    # ── 控制台 handler ───────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(level)
    console_handler.setFormatter(_ColorConsoleFormatter())
    root.addHandler(console_handler)

    # ── 文件 JSON Lines handler ──────────────────────────────────
    if log_dir is None:
        log_dir = Path(__file__).parent / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "app.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,   # 10 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(_JsonLinesFormatter())
    root.addHandler(file_handler)

    # 静默掉嘈杂的三方库 logger
    for noisy in ("urllib3", "requests", "yfinance", "peewee",
                  "matplotlib", "numba", "lightgbm"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ──────────────────────────────────────────────────────────────────
#  对外接口
# ──────────────────────────────────────────────────────────────────

def get_logger(name: str) -> logging.Logger:
    """
    获取具名 logger，自动确保日志系统已初始化。

    用法::

        logger = get_logger(__name__)
        logger.info("开始分析", extra={"ticker": "0700.HK"})
        logger.warning("信号置信度偏低", extra={"ticker": "0700.HK", "confidence": 0.48})
        logger.error("数据下载失败", exc_info=True)
    """
    setup_logging()
    return logging.getLogger(name)

