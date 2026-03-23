"""
存储后端
========
抽象存储层，支持 CSV / Parquet 格式。
DataManager 通过 StorageBackend 读写数据，实现存储格式透明切换。

Lazy 迁移策略：
  - 新数据按配置格式写入
  - 读取时自动兼容 CSV 和 Parquet
  - 提供 migrate_csv_to_parquet() 批量迁移工具
"""

from __future__ import annotations

import logging
import os
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """存储后端抽象基类。"""

    @property
    @abstractmethod
    def suffix(self) -> str:
        """文件扩展名 (含点号)。"""
        ...

    @abstractmethod
    def read(self, path: Path) -> pd.DataFrame:
        """读取数据文件为 DataFrame。"""
        ...

    @abstractmethod
    def write(self, df: pd.DataFrame, path: Path) -> None:
        """原子写入 DataFrame 到文件。"""
        ...

    def exists(self, path: Path) -> bool:
        """检查文件是否存在。"""
        return path.exists()

    def glob(self, directory: Path, pattern: str) -> List[Path]:
        """在目录中按模式搜索文件。"""
        return sorted(directory.glob(pattern))


class CsvBackend(StorageBackend):
    """CSV 存储后端（默认）。"""

    @property
    def suffix(self) -> str:
        return ".csv"

    def read(self, path: Path) -> pd.DataFrame:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
        return df

    def write(self, df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(suffix=".csv.tmp", dir=path.parent)
        try:
            os.close(fd)
            df.to_csv(tmp_path, index_label="date")
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


class ParquetBackend(StorageBackend):
    """Parquet 存储后端（需要 pyarrow）。"""

    @property
    def suffix(self) -> str:
        return ".parquet"

    def read(self, path: Path) -> pd.DataFrame:
        df = pd.read_parquet(path)
        if getattr(df.index, "tz", None) is not None:
            df.index = df.index.tz_localize(None)
        return df

    def write(self, df: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(suffix=".parquet.tmp", dir=path.parent)
        try:
            os.close(fd)
            df.to_parquet(tmp_path, index=True, engine="pyarrow")
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise


class AutoBackend(StorageBackend):
    """
    自动后端：写入使用指定格式，读取时自动检测 CSV/Parquet。

    这是 DataManager 的默认后端，实现 lazy 迁移策略。
    """

    def __init__(self, preferred: str = "csv") -> None:
        self._preferred = preferred.lower()
        self._csv = CsvBackend()
        self._parquet = ParquetBackend()
        self._writer = self._parquet if self._preferred == "parquet" else self._csv

    @property
    def suffix(self) -> str:
        return self._writer.suffix

    def read(self, path: Path) -> pd.DataFrame:
        """读取时自动检测格式。"""
        if path.suffix == ".parquet":
            return self._parquet.read(path)
        return self._csv.read(path)

    def write(self, df: pd.DataFrame, path: Path) -> None:
        """按首选格式写入。"""
        self._writer.write(df, path)

    def find_file(self, directory: Path, stem: str) -> Optional[Path]:
        """
        在目录中查找匹配的文件，优先匹配首选格式。

        Args:
            directory: 搜索目录
            stem: 文件名（不含扩展名）

        Returns:
            找到的文件路径，或 None
        """
        # 优先查找首选格式
        preferred = directory / f"{stem}{self.suffix}"
        if preferred.exists():
            return preferred
        # 回退到另一种格式
        fallback_suffix = ".csv" if self._preferred == "parquet" else ".parquet"
        fallback = directory / f"{stem}{fallback_suffix}"
        if fallback.exists():
            return fallback
        return None

    def glob_all(self, directory: Path, pattern_stem: str) -> List[Path]:
        """搜索两种格式的文件。"""
        csv_files = list(directory.glob(f"{pattern_stem}.csv"))
        parquet_files = list(directory.glob(f"{pattern_stem}.parquet"))
        return sorted(csv_files + parquet_files, key=lambda p: p.stat().st_mtime, reverse=True)


def get_backend(format_name: str = "csv") -> AutoBackend:
    """获取存储后端。"""
    return AutoBackend(preferred=format_name)


# ── 迁移工具 ────────────────────────────────────────────────────

def migrate_csv_to_parquet(
    directory: Path,
    *,
    delete_csv: bool = False,
    dry_run: bool = False,
) -> dict:
    """
    将目录下所有 CSV 文件批量转换为 Parquet。

    Args:
        directory: 数据目录
        delete_csv: 转换成功后是否删除原 CSV
        dry_run: 仅打印计划，不实际执行

    Returns:
        {"converted": int, "failed": int, "skipped": int}
    """
    csv_files = list(directory.glob("*.csv"))
    csv_backend = CsvBackend()
    parquet_backend = ParquetBackend()

    converted = 0
    failed = 0
    skipped = 0

    for csv_path in csv_files:
        parquet_path = csv_path.with_suffix(".parquet")
        if parquet_path.exists():
            logger.debug(f"跳过 {csv_path.name} (Parquet 已存在)")
            skipped += 1
            continue

        if dry_run:
            logger.info(f"[DRY RUN] 将转换: {csv_path.name} → {parquet_path.name}")
            converted += 1
            continue

        try:
            df = csv_backend.read(csv_path)
            parquet_backend.write(df, parquet_path)
            logger.info(f"已转换: {csv_path.name} → {parquet_path.name}")
            converted += 1

            if delete_csv:
                csv_path.unlink()
                # 同时删除 meta.json
                meta = csv_path.with_suffix(".meta.json")
                if meta.exists():
                    meta.unlink()
        except Exception as e:
            logger.warning(f"转换失败: {csv_path.name}: {e}")
            # 清理失败的 parquet
            if parquet_path.exists():
                parquet_path.unlink()
            failed += 1

    result = {"converted": converted, "failed": failed, "skipped": skipped}
    logger.info(f"迁移完成: {result}")
    return result

