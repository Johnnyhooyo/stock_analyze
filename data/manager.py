"""
数据管理器 (DataManager)
========================
统一入口，协调 vendor 链 → schema 校验 → 质量检查 → 存储。
替代原 fetch_data.py 中分散的下载/合并/保存逻辑。
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import time as _time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from data.calendar import latest_expected_trading_day
from data.config import DataConfig, get_config
from data.quality import check_quality, save_quality_report
from data.schemas import normalize_columns, validate_ohlcv
from data.storage import AutoBackend, get_backend
from data.vendor_base import DataVendor, parse_period_to_days

logger = logging.getLogger(__name__)


# ── Vendor 注册表 ────────────────────────────────────────────────

def _build_vendor_registry(cfg: DataConfig) -> Dict[str, DataVendor]:
    """延迟导入并构建 vendor 实例。"""
    from data.vendors.yfinance_vendor import YFinanceVendor
    from data.vendors.yahooquery_vendor import YahooQueryVendor
    from data.vendors.pandas_datareader_vendor import PandasDataReaderVendor
    from data.vendors.akshare_vendor import AkShareVendor
    from data.vendors.alpha_vantage_vendor import AlphaVantageVendor

    return {
        "yfinance": YFinanceVendor(),
        "yahooquery": YahooQueryVendor(),
        "pandas_datareader": PandasDataReaderVendor(),
        "akshare": AkShareVendor(),
        "alpha_vantage": AlphaVantageVendor(api_key=cfg.alpha_vantage_key),
    }


# ── 原子写入 ────────────────────────────────────────────────────

def _atomic_write_csv(df: pd.DataFrame, path: Path) -> None:
    """原子写入 CSV：先写临时文件，再 rename，防止中断导致文件损坏。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(suffix=".csv.tmp", dir=path.parent)
    try:
        os.close(fd)
        df.to_csv(tmp_path, index_label="date")
        os.replace(tmp_path, path)  # 原子操作 (POSIX)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


# ── 数据完整性哈希 ──────────────────────────────────────────────

def _compute_hash(path: Path) -> str:
    """计算文件 SHA-256 哈希。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_metadata(path: Path, ticker: str, source: str, rows: int, file_hash: str,
                   last_bar_date: Optional[str] = None) -> None:
    """保存元数据 sidecar 文件。"""
    meta_path = path.with_suffix(".meta.json")
    meta = {
        "ticker": ticker,
        "source": source,
        "download_ts": datetime.now().isoformat(),
        "rows": rows,
        "sha256": file_hash,
        "last_bar_date": last_bar_date,
    }
    # 如果没有传入 last_bar_date，尝试从 CSV 尾行读取
    if last_bar_date is None:
        try:
            with open(path, "rb") as f:
                f.seek(0, 2)
                fsize = f.tell()
                f.seek(max(0, fsize - 2048))
                tail = f.read().decode("utf-8", errors="replace")
            lines = tail.strip().splitlines()
            if len(lines) >= 2:
                date_str = lines[-1].split(",")[0].strip().strip('"')
                meta["last_bar_date"] = str(pd.Timestamp(date_str).date())
        except Exception:
            pass
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")


# ── 缓存过期判断 ────────────────────────────────────────────────

def _is_stale(file_path: Path) -> bool:
    """
    判断本地数据文件是否过期。
    优先读取 .meta.json 中的 last_bar_date (O(1))，
    仅当 meta 不存在时 fallback 到读文件尾行。
    支持 CSV 和 Parquet 格式。
    """
    target = latest_expected_trading_day()

    # ── 快速路径: 读 meta.json ───────────────────────────────────
    meta_path = file_path.with_suffix(".meta.json")
    if meta_path.exists():
        try:
            import json as _json
            meta = _json.loads(meta_path.read_text(encoding="utf-8"))
            last_bar = meta.get("last_bar_date")
            if last_bar:
                from datetime import date as _date
                last_date = _date.fromisoformat(str(last_bar))
                return last_date < target
        except Exception:
            pass  # fallback

    # ── 慢速路径 ─────────────────────────────────────────────────
    try:
        if file_path.suffix == ".parquet":
            # Parquet: 必须读取 index
            df = pd.read_parquet(file_path, columns=[])
            if df.empty:
                return True
            last_date = df.index.max()
            if pd.isna(last_date):
                return True
            if hasattr(last_date, "tzinfo") and last_date.tzinfo is not None:
                last_date = last_date.tz_convert(None)
            return last_date.date() < target
        else:
            # CSV: 只读尾行
            with open(file_path, "rb") as f:
                f.seek(0, 2)
                fsize = f.tell()
                f.seek(max(0, fsize - 2048))
                tail = f.read().decode("utf-8", errors="replace")
            lines = tail.strip().splitlines()
            if len(lines) < 2:
                return True
            last_line = lines[-1]
            date_str = last_line.split(",")[0].strip().strip('"')
            last_date = pd.Timestamp(date_str)
            if hasattr(last_date, "tzinfo") and last_date.tzinfo is not None:
                last_date = last_date.tz_convert(None)
            return last_date.date() < target
    except Exception:
        return True


# ── 核心 DataManager ────────────────────────────────────────────

class DataManager:
    """
    数据管理器：统一数据获取入口。

    Usage::

        mgr = DataManager()
        df, path = mgr.download("0700.HK", period="3y")
        df, path = mgr.download_from_config()  # 读 config.yaml

        # 纯读取（不触发网络请求）
        df = mgr.load("0700.HK", period="3y")
    """

    def __init__(self, config: Optional[DataConfig] = None) -> None:
        self.config = config or get_config()
        self._vendors: Optional[Dict[str, DataVendor]] = None
        self._storage: Optional[AutoBackend] = None

    @property
    def vendors(self) -> Dict[str, DataVendor]:
        if self._vendors is None:
            self._vendors = _build_vendor_registry(self.config)
        return self._vendors

    @property
    def storage(self) -> AutoBackend:
        if self._storage is None:
            self._storage = get_backend(self.config.storage_format)
        return self._storage

    def _ordered_vendors(self, sources_override: Optional[List[str]] = None) -> List[DataVendor]:
        """按配置的数据源优先级返回 vendor 列表。"""
        source_names = sources_override or self.config.data_sources
        result = []
        for name in source_names:
            v = self.vendors.get(name)
            if v and v.is_available():
                result.append(v)
            elif v:
                logger.debug(f"数据源 {name} 不可用（依赖缺失）")
            else:
                logger.debug(f"未知数据源: {name}")
        return result

    # ── 内部: vendor 链下载 ──────────────────────────────────────

    def _fetch_from_vendors(
        self,
        ticker: str,
        period: str,
        sources_override: Optional[List[str]] = None,
    ) -> Tuple[Optional[pd.DataFrame], str]:
        """
        通过 vendor 链尝试下载数据。

        Returns:
            (DataFrame | None, source_name)
        """
        days = parse_period_to_days(period)
        end_date = datetime.today()
        start_date = end_date - timedelta(days=days)

        for vendor in self._ordered_vendors(sources_override):
            logger.info(f"尝试数据源: {vendor.name} for {ticker}")
            try:
                df = vendor.fetch_with_retry(
                    ticker, start_date, end_date,
                    period=period,
                    timeout=self.config.download_timeout,
                    max_attempts=self.config.retry_attempts,
                    backoff=self.config.retry_backoff,
                )
                if df is not None and not df.empty:
                    return df, vendor.name
            except Exception as e:
                logger.info(f"{vendor.name} 失败: {e}")

        return None, "unknown"

    # ── 内部: 标准化 + 合并 + 保存 ──────────────────────────────

    def _normalize_merge_save(
        self,
        new_data: pd.DataFrame,
        ticker: str,
        file_path: Path,
        source: str,
    ) -> Tuple[pd.DataFrame, str]:
        """
        标准化新数据 → 校验 → 质量检查 → 与本地历史合并 → 原子写入 → 元数据。

        Returns:
            (final_df, file_path_str)
        """
        from data.quality import repair_quality

        # ── 标准化 & 校验 ────────────────────────────────────────
        data = normalize_columns(new_data)
        if data is None or data.empty:
            raise ValueError(f"{ticker}: normalize 后为空")

        validation = validate_ohlcv(data)
        if not validation.ok:
            logger.warning(f"数据校验失败 ({ticker}): {validation.errors}")
        if validation.warnings:
            logger.info(f"数据校验警告 ({ticker}): {validation.warnings}")

        # ── 质量检查 + 修复 ──────────────────────────────────────
        quality = check_quality(data, ticker=ticker)
        if quality.has_issues:
            logger.warning(quality.summary())
            data = repair_quality(data, quality)
        else:
            logger.info(quality.summary())

        # ── 合并历史数据 ─────────────────────────────────────────
        # 同时检查 CSV 和 Parquet 的旧文件
        old_files = [file_path]
        alt_suffix = ".parquet" if file_path.suffix == ".csv" else ".csv"
        old_files.append(file_path.with_suffix(alt_suffix))

        for old_path in old_files:
            if old_path.exists():
                try:
                    old_data = self.storage.read(old_path)
                    old_data = normalize_columns(old_data)
                    combined = pd.concat([old_data, data])
                    combined = combined[~combined.index.duplicated(keep="last")]
                    combined = combined.sort_index()
                    rows_added = len(combined) - len(old_data)
                    logger.info(
                        f"合并历史数据: 旧 {len(old_data)} + 新 {len(data)} = {len(combined)} 行 (+{rows_added})"
                    )
                    data = combined
                    break
                except Exception as e:
                    logger.debug(f"读取旧文件 {old_path} 失败: {e}")
        else:
            data = data.sort_index()

        # ── 原子写入（通过存储后端）─────────────────────────────
        # 确保文件扩展名与存储后端一致
        file_path = file_path.with_suffix(self.storage.suffix)
        self.storage.write(data, file_path)
        logger.info(f"数据已保存至: {file_path}")

        # ── 元数据 & 哈希 ────────────────────────────────────────
        last_bar = str(data.index.max().date()) if not data.empty else None
        try:
            file_hash = _compute_hash(file_path)
            _save_metadata(file_path, ticker, source, len(data), file_hash,
                           last_bar_date=last_bar)
        except Exception as e:
            logger.debug(f"元数据保存失败: {e}")

        # ── 保存质量报告 ─────────────────────────────────────────
        try:
            save_quality_report(quality)
        except Exception as e:
            logger.debug(f"质量报告保存失败: {e}")

        return data, str(file_path)

    # ── 纯读取接口 (Phase 3) ────────────────────────────────────

    def load(
        self,
        ticker: str,
        period: str = "1y",
        *,
        out_dir: Optional[Path] = None,
    ) -> pd.DataFrame:
        """
        从本地存储加载数据，不触发网络请求。

        对标 QLib D.features() / Rqalpha history_bars()，
        供策略模块直接调用，与下载解耦。
        自动兼容 CSV 和 Parquet 格式。

        Args:
            ticker: 股票代码
            period: 数据周期
            out_dir: 数据目录（默认使用 config.historical_dir）

        Returns:
            DataFrame (可能为空)

        Raises:
            FileNotFoundError: 没有找到任何本地文件
        """
        search_dir = out_dir or self.config.historical_dir
        safe_name = ticker.replace(".", "_")

        # 精确匹配优先（两种格式）
        for stem in [f"{safe_name}_{period}", f"{ticker}_{period}"]:
            found = self.storage.find_file(search_dir, stem)
            if found is not None:
                return self.storage.read(found)

        # 模糊匹配: 同 ticker 不同 period 的文件
        candidates = self.storage.glob_all(search_dir, f"{safe_name}_*")
        if not candidates:
            candidates = self.storage.glob_all(search_dir, f"{ticker}_*")
        if candidates:
            df = self.storage.read(candidates[0])
            logger.info(f"load: 使用 {candidates[0].name} (精确文件不存在)")
            return df

        raise FileNotFoundError(
            f"本地无 {ticker} 数据文件 (搜索目录: {search_dir})"
        )

    # ── 主下载接口 ───────────────────────────────────────────────

    def download(
        self,
        ticker: str,
        period: str = "1y",
        *,
        sources_override: Optional[List[str]] = None,
        force: bool = False,
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """
        下载单只股票数据。优先使用本地缓存，过期或不存在时从网络下载。

        Args:
            ticker: 股票代码
            period: 数据周期 ('1y', '3y', '5y', '6mo', '30d' …)
            sources_override: 覆盖 config 中的数据源列表
            force: 强制重新下载（忽略本地缓存）

        Returns:
            (DataFrame, file_path) 或 (空DataFrame, None)
        """
        out_dir = self.config.historical_dir
        stem = f"{ticker}_{period}"
        file_path = out_dir / f"{stem}{self.storage.suffix}"

        # ── 查找本地缓存 ─────────────────────────────────────────
        if not force:
            # 搜索两种格式
            candidates = self.storage.glob_all(out_dir, f"{ticker}_*")
            if candidates:
                cached = candidates[0]
                if not _is_stale(cached):
                    data = self.storage.read(cached)
                    if not data.empty:
                        logger.info(f"本地缓存有效，使用: {cached.name}")
                        return data, str(cached)
                else:
                    logger.info(f"本地缓存过期: {cached.name}，需要更新")
                    file_path = cached  # 使用已有文件路径以便合并

        # ── 网络下载 ─────────────────────────────────────────────
        new_data, source = self._fetch_from_vendors(ticker, period, sources_override)

        if new_data is None or new_data.empty:
            logger.warning(f"所有数据源均未获取到 {ticker} 的数据")
            return pd.DataFrame(), None

        # ── 标准化 + 合并 + 保存 ─────────────────────────────────
        file_path = out_dir / f"{stem}{self.storage.suffix}"
        return self._normalize_merge_save(new_data, ticker, file_path, source)

    def download_from_config(
        self, sources_override: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, Optional[str]]:
        """根据 config.yaml 中的 ticker 和 period 下载数据。"""
        if not self.config.ticker:
            raise ValueError("config.yaml 中缺少 ticker 配置")
        return self.download(
            self.config.ticker,
            self.config.period,
            sources_override=sources_override,
        )

    # ── HSI 批量更新 ─────────────────────────────────────────────

    def download_hsi_incremental(
        self,
        period: str = "3y",
        out_dir: Optional[Path] = None,
        delay: float = 0.3,
        stocks: Optional[List[str]] = None,
        max_workers: Optional[int] = None,
    ) -> dict:
        """
        增量更新所有 HSI 成分股数据。

        - 已有文件且是最新的 → 跳过
        - 已有文件但过期 → 下载并合并
        - 无文件 → 全量下载

        Args:
            period: 数据周期
            out_dir: 输出目录
            delay: 串行模式下请求间隔（并发模式由 rate limiter 控制）
            stocks: 股票列表（默认 HSI 成分股）
            max_workers: 并发线程数（None 使用 config.batch_max_workers, 1 = 串行）
        """
        if out_dir is None:
            out_dir = self.config.historical_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        if stocks is None:
            stocks = self._get_hsi_stocks()

        workers = max_workers or self.config.batch_max_workers

        total = len(stocks)
        skipped = 0
        updated = 0
        failed: list[str] = []

        # ── 第一步：过滤出需要更新的股票 ─────────────────────────
        need_update: list[Tuple[str, Path]] = []
        for ticker in stocks:
            safe_name = ticker.replace(".", "_")
            stem = f"{safe_name}_{period}"
            # 查找已有文件（兼容 CSV/Parquet）
            found = self.storage.find_file(out_dir, stem)
            if found is not None and not _is_stale(found):
                skipped += 1
                continue
            file_path = found or (out_dir / f"{stem}{self.storage.suffix}")
            need_update.append((ticker, file_path))

        logger.info(
            f"[HSI增量] 总计 {total} 只, 跳过 {skipped} 只(已是最新), 待更新 {len(need_update)} 只 (workers={workers})"
        )

        if not need_update:
            return {"total": total, "skipped": skipped, "updated": 0, "failed": []}

        # ── 第二步：下载（并发或串行）───────────────────────────
        def _process_one(ticker: str, file_path: Path) -> Optional[str]:
            """处理单只股票，返回 None=成功, str=失败原因"""
            new_data, source = self._fetch_from_vendors(ticker, period)
            if new_data is None or new_data.empty:
                return f"{ticker}: 无数据"
            try:
                self._normalize_merge_save(new_data, ticker, file_path, source)
                return None
            except Exception as e:
                return f"{ticker}: {e}"

        if workers <= 1:
            # 串行模式
            for i, (ticker, file_path) in enumerate(need_update, 1):
                logger.info(f"[{skipped + i}/{total}] ⬇ 更新 {ticker}…")
                err = _process_one(ticker, file_path)
                if err:
                    logger.warning(f"[{skipped + i}/{total}] ❌ {err}")
                    failed.append(ticker)
                else:
                    logger.info(f"[{skipped + i}/{total}] ✅ {ticker}")
                    updated += 1
                _time.sleep(delay)
        else:
            # 并发模式（rate limiter 在 vendor 层自动控制频率）
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_map = {
                    executor.submit(_process_one, ticker, fp): ticker
                    for ticker, fp in need_update
                }
                for i, future in enumerate(as_completed(future_map), 1):
                    ticker = future_map[future]
                    try:
                        err = future.result(timeout=120)
                        if err:
                            logger.warning(f"[{skipped + i}/{total}] ❌ {err}")
                            failed.append(ticker)
                        else:
                            logger.info(f"[{skipped + i}/{total}] ✅ {ticker}")
                            updated += 1
                    except Exception as e:
                        logger.warning(f"[{skipped + i}/{total}] ❌ {ticker}: {e}")
                        failed.append(ticker)


        logger.info(
            f"[HSI增量] 完成: 总计 {total}, 跳过 {skipped}, 更新 {updated}, 失败 {len(failed)}"
            + (f" 失败列表: {failed}" if failed else "")
        )
        return {
            "total": total,
            "skipped": skipped,
            "updated": updated,
            "failed": failed,
        }

    @staticmethod
    def _get_hsi_stocks() -> List[str]:
        """获取 HSI 成分股列表。"""
        try:
            from data.hsi_stocks import HSI_STOCKS
            return list(HSI_STOCKS)
        except ImportError:
            logger.error("无法获取 HSI 成分股列表")
            return []

