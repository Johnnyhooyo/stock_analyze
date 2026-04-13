"""
pipeline/data_prep.py — 数据就绪检查

包含:
  step1_ensure_data()  — 确保目标股票历史日线已就绪
  _ensure_hk_data()    — 全量港股增量更新
  _hist_data_is_stale() — 数据过期判断（考虑港股节假日）
"""

from __future__ import annotations

import sys
import pandas as pd
from datetime import datetime
from pathlib import Path

from log_config import get_logger
from data.manager import DataManager
from data.calendar import prev_trading_day as _prev_hk_trading_day, is_trading_day as _is_hk_trading_day
from config_loader import load_config

logger = get_logger(__name__)


def _hist_data_is_stale(hist_file_path: str) -> bool:
    """
    判断历史数据文件是否过期，正确处理港股节假日。
    - 18点前：当天数据不可用，最新应该是上一个港股交易日
    - 18点后：当天数据可能可用，最新应该是今天（如为交易日）
    """
    from datetime import date as _date, time as _time

    try:
        df = pd.read_csv(hist_file_path, index_col=0, parse_dates=True)
        if df.empty:
            return True
        latest_date = df.index.max()
        if pd.isna(latest_date):
            return True
        if hasattr(latest_date, 'tzinfo') and latest_date.tzinfo is not None:
            latest_date = latest_date.tz_convert(None)

        now = datetime.now()
        today = _date.today()
        cutoff_time = _time(18, 0)

        if now.time() < cutoff_time:
            target_date = _prev_hk_trading_day(today)
        else:
            if _is_hk_trading_day(today):
                target_date = today
            else:
                target_date = _prev_hk_trading_day(today)

        return latest_date.date() < target_date
    except Exception as e:
        logger.debug("数据过期检查失败，视为需要更新", extra={"error": str(e)})
        return True


def step1_ensure_data(
    sources_override=None,
    ticker: str = None,
    skip_download: bool = False,
) -> tuple[pd.DataFrame, str]:
    """
    确保历史日线数据已就绪。

    Returns
    -------
    hist_data : pd.DataFrame   历史日线（Close 等标准列）
    hist_path : str            历史文件路径
    """
    logger.info("步骤1/3: 数据就绪检查开始")

    cfg = load_config()
    effective_ticker = ticker or cfg.get('ticker', '0700.hk')

    hist_dir = Path(__file__).parent.parent / 'data' / 'historical'

    _ticker_lower = effective_ticker.lower()
    _ticker_safe  = effective_ticker.replace('.', '_').lower()

    def _is_ticker_file(p: Path) -> bool:
        stem = p.stem.lower()
        return stem.startswith(_ticker_lower + '_') or stem.startswith(_ticker_safe + '_')

    hist_files = sorted(
        [f for f in hist_dir.glob('*.csv') if _is_ticker_file(f)],
        key=lambda p: p.stat().st_mtime, reverse=True,
    )

    mgr = DataManager()

    if hist_files and not _hist_data_is_stale(str(hist_files[0])):
        hist_path = str(hist_files[0])
        hist_data = pd.read_csv(hist_path, index_col=0, parse_dates=True)
        latest_date = hist_data.index.max()
        _close_nan = int(hist_data['Close'].isna().sum()) if 'Close' in hist_data.columns else -1
        logger.warning(
            "[DEBUG step1_ensure_data] 缓存命中: file=%s, rows=%d, cols=%s, Close_NaN=%d",
            hist_path, len(hist_data), list(hist_data.columns), _close_nan
        )
        logger.info("历史日线数据已是最新", extra={
            "hist_path": hist_path,
            "records": len(hist_data),
            "latest_date": str(latest_date.date())
        })
    else:
        if skip_download:
            if hist_files:
                hist_path = str(hist_files[0])
                hist_data = pd.read_csv(hist_path, index_col=0, parse_dates=True)
                latest_date = hist_data.index.max()
                logger.warning("使用过期本地缓存（%s），--skip-data-download 已设置", latest_date.date())
            else:
                logger.error("本地无 %s 的历史数据缓存，且 --skip-data-download 已设置，流程终止", effective_ticker)
                sys.exit(1)
            return hist_data, hist_path
        if hist_files:
            from datetime import date as _date, time as _time
            try:
                df_tmp = pd.read_csv(hist_files[0], index_col=0, parse_dates=True)
                latest_date = df_tmp.index.max()
            except Exception as e:
                logger.debug("读取 CSV 末行日期失败，视为数据过期", extra={"error": str(e)})
                latest_date = pd.Timestamp('1970-01-01')
            today = _date.today()
            now = datetime.now()
            if now.time() < _time(18, 0):
                target = _prev_hk_trading_day(today)
            else:
                target = today if _is_hk_trading_day(today) else _prev_hk_trading_day(today)
            logger.warning("历史日线数据已过期，正在更新", extra={
                "latest_date": str(latest_date.date()),
                "target_date": str(target)
            })
        else:
            logger.warning("本地无历史日线数据，正在下载")
        hist_data, hist_path = mgr.download(
            effective_ticker,
            period=cfg.get('period', '5y'),
            sources_override=sources_override,
        )
        if hist_data is None or hist_data.empty:
            if hist_files:
                hist_path = str(hist_files[0])
                hist_data = pd.read_csv(hist_path, index_col=0, parse_dates=True)
                logger.warning("数据更新失败，继续使用旧数据", extra={
                    "hist_path": hist_path,
                    "records": len(hist_data)
                })
            else:
                logger.critical("历史数据下载失败，流程终止")
                sys.exit(1)
        else:
            _close_nan2 = int(hist_data['Close'].isna().sum()) if 'Close' in hist_data.columns else -1
            logger.warning(
                "[DEBUG step1_ensure_data] 下载完成: file=%s, rows=%d, cols=%s, Close_NaN=%d",
                hist_path, len(hist_data), list(hist_data.columns), _close_nan2
            )
            logger.info("历史数据已更新", extra={
                "hist_path": hist_path,
                "records": len(hist_data)
            })

    # 校验 Close 列完整性：下载或合并失败时可能产生全 NaN Close，
    # 此时回退到最近的本地缓存文件，避免训练流程用损坏数据运行
    if 'Close' not in hist_data.columns or hist_data['Close'].isna().all():
        logger.error(
            "hist_data Close 列全为 NaN 或缺失，可能为下载/合并阶段数据损坏，尝试回退本地缓存",
            extra={"records": len(hist_data)}
        )
        recovered = False
        for candidate in (hist_files or []):
            try:
                fallback_df = pd.read_csv(str(candidate), index_col=0, parse_dates=True)
                if 'Close' in fallback_df.columns and not fallback_df['Close'].isna().all():
                    hist_data = fallback_df
                    hist_path = str(candidate)
                    logger.warning("已回退到本地缓存", extra={
                        "hist_path": hist_path,
                        "records": len(hist_data)
                    })
                    recovered = True
                    break
            except Exception as e:
                logger.debug("回退候选文件读取失败", extra={"path": str(candidate), "error": str(e)})
        if not recovered:
            logger.critical("hist_data Close 列损坏且无可用本地缓存，流程终止")
            sys.exit(1)

    return hist_data, hist_path


def _ensure_hk_data(cfg: dict = None) -> None:
    """全量港股增量更新 — 单独提取，供 main() 和 train_portfolio_tickers() 调用一次。"""
    if cfg is None:
        cfg = load_config()
    hk_period = cfg.get('hsi_period', '5y')
    logger.info("开始增量更新港股数据", extra={"period": hk_period})
    try:
        mgr = DataManager()
        hk_result = mgr.download_hk_incremental(period=hk_period)
        logger.info("港股数据更新完成", extra={
            "total":          hk_result['total'],
            "skipped":        hk_result['skipped'],
            "updated":        hk_result['updated'],
            "failed_count":   len(hk_result['failed']),
            "failed_tickers": hk_result['failed'],
        })
    except Exception as e:
        logger.warning("港股数据更新失败", extra={"error": str(e)})


# backward-compat alias
_ensure_hsi_data = _ensure_hk_data
