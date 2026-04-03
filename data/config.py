"""
数据模块配置
===========
从 config.yaml / keys.yaml 加载数据相关配置，集中管理。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class DataConfig:
    """数据下载模块的统一配置。"""

    # 基础
    ticker: str = ""
    period: str = "1y"
    data_sources: List[str] = field(
        default_factory=lambda: ["tickflow", "yahooquery", "yfinance", "pandas_datareader", "akshare", "alpha_vantage"]
    )

    # API keys
    alpha_vantage_key: Optional[str] = None

    # 路径
    project_root: Path = _PROJECT_ROOT
    historical_dir: Path = field(default=None)  # type: ignore[assignment]

    # 存储
    storage_format: str = "csv"  # "csv" 或 "parquet"

    # 下载
    download_timeout: int = 20
    retry_attempts: int = 3
    retry_backoff: float = 1.0
    batch_delay: float = 0.3
    batch_max_workers: int = 4

    def __post_init__(self) -> None:
        if self.historical_dir is None:
            self.historical_dir = self.project_root / "data" / "historical"
        self.historical_dir.mkdir(parents=True, exist_ok=True)

    # ── 工厂方法 ─────────────────────────────────────────────────

    @classmethod
    def from_yaml(cls, config_path: Optional[Path] = None, keys_path: Optional[Path] = None) -> "DataConfig":
        """从 config.yaml + keys.yaml 构建配置。"""
        if config_path is None:
            config_path = _PROJECT_ROOT / "config.yaml"
        if keys_path is None:
            keys_path = _PROJECT_ROOT / "keys.yaml"

        raw: dict = {}
        if config_path.exists():
            with open(config_path, encoding="utf-8") as f:
                raw = yaml.safe_load(f) or {}

        keys: dict = {}
        if keys_path.exists():
            with open(keys_path, encoding="utf-8") as f:
                keys = yaml.safe_load(f) or {}

        default_sources = ["tickflow", "yahooquery", "yfinance", "pandas_datareader", "akshare", "alpha_vantage"]
        return cls(
            ticker=str(raw.get("ticker", "")),
            period=str(raw.get("period", "1y")),
            data_sources=raw.get("data_sources", default_sources),
            alpha_vantage_key=keys.get("alpha_vantage_key") or raw.get("alpha_vantage_key"),
            storage_format=str(raw.get("storage_format", "csv")),
            download_timeout=int(raw.get("download_timeout", 20)),
            retry_attempts=int(raw.get("retry_attempts", 3)),
            retry_backoff=float(raw.get("retry_backoff", 1.0)),
            batch_delay=float(raw.get("batch_delay", 0.3)),
            batch_max_workers=int(raw.get("batch_max_workers", 4)),
        )


# ── 单例缓存 ────────────────────────────────────────────────────
_cached_config: Optional[DataConfig] = None


def get_config(force_reload: bool = False) -> DataConfig:
    """获取全局数据配置（懒加载，单例）。"""
    global _cached_config
    if _cached_config is None or force_reload:
        _cached_config = DataConfig.from_yaml()
    return _cached_config

