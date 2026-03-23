"""
数据质量管道
===========
对下载后的 OHLCV 数据执行质量检查:
  - 交易日缺口检测
  - 异常值检测 (单日涨跌幅 / OHLC 逻辑)
  - 复权嫌疑检测 (价格跳变 >50%)
  - 零成交量告警
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import timedelta
from pathlib import Path
from typing import List, Optional

import pandas as pd

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """数据质量报告。"""
    ticker: str = ""
    rows: int = 0
    date_range: str = ""

    # 缺口
    missing_days: List[str] = field(default_factory=list)
    missing_count: int = 0

    # 异常值
    extreme_returns: List[str] = field(default_factory=list)
    ohlc_violations: int = 0
    negative_volume: int = 0
    zero_volume_days: int = 0

    # 复权嫌疑
    split_suspects: List[str] = field(default_factory=list)

    # 修复记录
    repairs: List[str] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return (
            self.missing_count > 0
            or len(self.extreme_returns) > 0
            or self.ohlc_violations > 0
            or len(self.split_suspects) > 0
        )

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        parts = [f"[质量报告] {self.ticker} ({self.rows} 行, {self.date_range})"]
        if self.missing_count:
            parts.append(f"  ⚠ 缺口: {self.missing_count} 天")
        if self.extreme_returns:
            parts.append(f"  ⚠ 极端涨跌幅: {len(self.extreme_returns)} 天")
        if self.ohlc_violations:
            parts.append(f"  ⚠ OHLC 逻辑异常: {self.ohlc_violations} 行")
        if self.split_suspects:
            parts.append(f"  ⚠ 复权嫌疑: {len(self.split_suspects)} 天")
        if self.zero_volume_days:
            parts.append(f"  ℹ 零成交量: {self.zero_volume_days} 天")
        if self.repairs:
            parts.append(f"  🔧 已修复: {len(self.repairs)} 项")
        if not self.has_issues and not self.repairs:
            parts.append("  ✅ 无异常")
        return "\n".join(parts)


def check_quality(
    df: pd.DataFrame,
    ticker: str = "",
    *,
    return_threshold: float = 0.30,
    split_threshold: float = 0.50,
    calendar_check: bool = True,
) -> QualityReport:
    """
    对 OHLCV DataFrame 执行质量检查。

    Args:
        df: 标准化后的 DataFrame (需含 Close 列, datetime index)
        ticker: 股票代码 (用于报告)
        return_threshold: 单日涨跌幅告警阈值 (默认 30%)
        split_threshold: 复权嫌疑阈值 (默认 50%)
        calendar_check: 是否用交易日历做缺口检测

    Returns:
        QualityReport
    """
    report = QualityReport(ticker=ticker)

    if df is None or df.empty:
        return report

    report.rows = len(df)
    if pd.api.types.is_datetime64_any_dtype(df.index):
        report.date_range = f"{df.index.min().date()} → {df.index.max().date()}"

    # ── 1) 交易日缺口检测 ───────────────────────────────────────
    if calendar_check and pd.api.types.is_datetime64_any_dtype(df.index):
        try:
            from data.calendar import is_trading_day
            data_dates = set(df.index.date)
            first, last = df.index.min().date(), df.index.max().date()
            d = first
            while d <= last:
                if is_trading_day(d) and d not in data_dates:
                    report.missing_days.append(str(d))
                d += timedelta(days=1)
            report.missing_count = len(report.missing_days)
        except Exception as e:
            logger.debug(f"交易日历检查失败: {e}")

    # ── 2) 单日涨跌幅异常 ───────────────────────────────────────
    if "Close" in df.columns:
        returns = df["Close"].pct_change().dropna()
        extreme = returns[returns.abs() > return_threshold]
        report.extreme_returns = [
            f"{idx.date() if hasattr(idx, 'date') else idx}: {val:+.2%}"
            for idx, val in extreme.items()
        ]

    # ── 3) OHLC 逻辑校验 ────────────────────────────────────────
    if {"Open", "High", "Low", "Close"}.issubset(df.columns):
        violations = (
            (df["High"] < df["Open"]) |
            (df["High"] < df["Close"]) |
            (df["Low"] > df["Open"]) |
            (df["Low"] > df["Close"])
        )
        report.ohlc_violations = int(violations.sum())

    # ── 4) Volume 检查 ──────────────────────────────────────────
    if "Volume" in df.columns:
        report.negative_volume = int((df["Volume"] < 0).sum())
        report.zero_volume_days = int((df["Volume"] == 0).sum())

    # ── 5) 复权嫌疑 (价格跳变 >50%) ────────────────────────────
    if "Close" in df.columns:
        returns = df["Close"].pct_change().dropna()
        suspects = returns[returns.abs() > split_threshold]
        report.split_suspects = [
            f"{idx.date() if hasattr(idx, 'date') else idx}: {val:+.2%}"
            for idx, val in suspects.items()
        ]

    return report


def repair_quality(
    df: pd.DataFrame,
    report: QualityReport,
    *,
    fill_missing: bool = True,
    fix_ohlc: bool = True,
    fix_volume: bool = True,
) -> pd.DataFrame:
    """
    根据质量报告对数据执行自动修复。

    对标 QLib DataHandler.process 管道，将检测结果转化为修复动作。

    修复项:
    1. OHLC 逻辑异常 → clamp High/Low 使其满足 Low ≤ Open/Close ≤ High
    2. 负 Volume → 置 NaN
    3. 零 Volume → 保留（部分港股停牌日正常）
    4. 缺口日 → 插入空行 + forward fill（可选）

    Args:
        df: 标准化后的 OHLCV DataFrame
        report: check_quality() 返回的质量报告
        fill_missing: 是否对缺口日执行前值填充
        fix_ohlc: 是否修正 OHLC 逻辑异常
        fix_volume: 是否修正负 Volume

    Returns:
        修复后的 DataFrame（不修改原始数据）
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    # ── 1) OHLC 逻辑修正 ────────────────────────────────────────
    if fix_ohlc and report.ohlc_violations > 0:
        ohlc_cols = {"Open", "High", "Low", "Close"}
        if ohlc_cols.issubset(df.columns):
            # High 应该 >= Open 和 Close
            row_max = df[["Open", "High", "Low", "Close"]].max(axis=1)
            row_min = df[["Open", "High", "Low", "Close"]].min(axis=1)

            fixed_high = (df["High"] < row_max).sum()
            fixed_low = (df["Low"] > row_min).sum()

            df["High"] = row_max
            df["Low"] = row_min

            total_fixed = int(fixed_high + fixed_low)
            if total_fixed > 0:
                report.repairs.append(f"OHLC clamp 修正: {total_fixed} 行")
                report.ohlc_violations = 0
                logger.info(f"[修复] OHLC 逻辑修正: {total_fixed} 行")

    # ── 2) 负 Volume → NaN ──────────────────────────────────────
    if fix_volume and "Volume" in df.columns and report.negative_volume > 0:
        neg_mask = df["Volume"] < 0
        neg_count = int(neg_mask.sum())
        if neg_count > 0:
            df.loc[neg_mask, "Volume"] = np.nan
            report.repairs.append(f"负 Volume → NaN: {neg_count} 行")
            report.negative_volume = 0
            logger.info(f"[修复] 负 Volume → NaN: {neg_count} 行")

    # ── 3) 缺口日填充 ───────────────────────────────────────────
    if fill_missing and report.missing_count > 0 and pd.api.types.is_datetime64_any_dtype(df.index):
        try:
            missing_dates = [pd.Timestamp(d) for d in report.missing_days]
            # 只添加在数据范围内的缺口日
            existing_dates = set(df.index)
            new_dates = [d for d in missing_dates if d not in existing_dates]
            if new_dates:
                # 创建空行并 reindex
                new_idx = df.index.append(pd.DatetimeIndex(new_dates))
                new_idx = new_idx.sort_values().drop_duplicates()
                df = df.reindex(new_idx)
                # Forward fill OHLC，Volume 填 0
                ohlcv = [c for c in ["Open", "High", "Low", "Close", "Adj Close"] if c in df.columns]
                df[ohlcv] = df[ohlcv].ffill()
                if "Volume" in df.columns:
                    df["Volume"] = df["Volume"].fillna(0)
                df.index.name = "date"
                report.repairs.append(f"缺口日前值填充: {len(new_dates)} 天")
                logger.info(f"[修复] 缺口日前值填充: {len(new_dates)} 天")
        except Exception as e:
            logger.debug(f"缺口填充失败: {e}")

    return df


def save_quality_report(report: QualityReport, log_dir: Optional[Path] = None) -> Path:
    """将质量报告保存到 JSON 文件。"""
    if log_dir is None:
        log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "quality_report.json"

    existing: list[dict] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            existing = []

    existing.append(report.to_dict())
    existing = existing[-500:]
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")
    return path

