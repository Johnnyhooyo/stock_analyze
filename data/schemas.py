"""
OHLCV 数据模型 & 校验
=====================
定义标准 OHLCV 列规范，提供 DataFrame 级别的 schema 校验与列名归一化。
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# ── 标准列名 ────────────────────────────────────────────────────
REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]
OPTIONAL_COLUMNS = ["Adj Close"]
STANDARD_COLUMNS = REQUIRED_COLUMNS  # 向后兼容

# ── 列名同义词映射 ──────────────────────────────────────────────
# key = canonical name, value = lowercase 变体列表
_SYNONYMS: dict[str, list[str]] = {
    "Open":      ["open", "openprice", "开盘", "开盘价"],
    "High":      ["high", "highprice", "最高", "最高价"],
    "Low":       ["low", "lowprice", "最低", "最低价"],
    "Close":     ["close", "closeprice", "收盘", "收盘价"],
    "Volume":    ["volume", "vol", "volumetraded", "成交量"],
    "Adj Close": ["adjclose", "adj_close", "adjcloseprice", "adjustedclose", "复权收盘价"],
}

# 日期列候选
_DATE_CANDIDATES = ["date", "time", "交易日期", "date_time", "日期"]


def _strip(s: str) -> str:
    """小写并去除空白/下划线。"""
    return str(s).lower().strip().replace(" ", "").replace("_", "")


# ── 列名归一化 ──────────────────────────────────────────────────

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 DataFrame 的列名统一为标准 OHLCV 列名。

    处理:
    1. yfinance MultiIndex 列 → 扁平化
    2. 中英文同义词映射
    3. 日期列 → datetime index
    """
    if df is None or df.empty:
        return df

    df = df.copy()

    # --- 1) 处理 MultiIndex 列 (yfinance >=0.2.31 返回 (Price, Ticker)) ---
    if isinstance(df.columns, pd.MultiIndex):
        lv0 = df.columns.get_level_values(0).tolist()
        price_names = {"open", "high", "low", "close", "volume", "adjclose", "adj close"}
        if any(_strip(str(v)) in price_names for v in lv0):
            # Price 在外层 → swap
            df.columns = df.columns.swaplevel()
        try:
            top = df.columns.get_level_values(0).unique()[0]
            df = df[top].copy()
        except Exception:
            # 如果无法选择，直接用 droplevel
            try:
                df.columns = df.columns.droplevel(0)
            except Exception:
                pass

    # --- 2) 映射列名 ---
    col_map: dict[str, str] = {}
    for col in list(df.columns):
        stripped = _strip(str(col))
        for canonical, variants in _SYNONYMS.items():
            if stripped in variants:
                col_map[col] = canonical
                break
    if col_map:
        df = df.rename(columns=col_map)

    # 去重列：如果同一 canonical 出现多次，合并 (fillna)
    for canonical in list(_SYNONYMS.keys()):
        dups = [c for c in df.columns if c == canonical]
        if len(dups) > 1:
            merged = df[canonical].iloc[:, 0].fillna(df[canonical].iloc[:, 1])
            df = df.drop(columns=[canonical])
            df[canonical] = merged

    # --- 3) 日期列 → index ---
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        # 尝试已知日期列名
        date_col = None
        for cand in _DATE_CANDIDATES:
            for col in df.columns:
                if str(col).lower() == cand:
                    date_col = col
                    break
            if date_col:
                break
        # 如果未找到，试探第一个能解析为日期的列
        if date_col is None:
            for col in df.columns:
                sample = df[col].dropna().astype(str)
                if not sample.empty:
                    try:
                        pd.to_datetime(sample.iloc[0])
                        date_col = col
                        break
                    except Exception:
                        continue
        if date_col is not None:
            try:
                df.index = pd.to_datetime(df[date_col])
                df = df.drop(columns=[date_col], errors="ignore")
            except Exception:
                pass

    # 尝试解析 index 为 datetime
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass

    # 移除时区
    if getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)

    df.index.name = "date"
    return df


# ── DataFrame 级别 schema 校验 ──────────────────────────────────

class ValidationResult:
    """校验结果容器。"""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []

    @property
    def ok(self) -> bool:
        return len(self.errors) == 0

    def __repr__(self) -> str:
        status = "✅ PASS" if self.ok else "❌ FAIL"
        parts = [status]
        if self.errors:
            parts.append(f"errors={self.errors}")
        if self.warnings:
            parts.append(f"warnings={self.warnings}")
        return f"ValidationResult({', '.join(parts)})"


def validate_ohlcv(df: pd.DataFrame) -> ValidationResult:
    """
    对 OHLCV DataFrame 执行基础 schema 校验。

    检查项:
    - 必须包含 Close 列
    - index 必须为 datetime
    - Open ≤ High, Low ≤ Close (如果列存在)
    - Volume ≥ 0
    - 无全空行
    """
    result = ValidationResult()

    if df is None or df.empty:
        result.errors.append("DataFrame 为空或 None")
        return result

    # 必须列
    if "Close" not in df.columns:
        result.errors.append("缺少 Close 列")

    # datetime index
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        result.errors.append("index 不是 datetime 类型")

    # OHLC 关系校验
    if {"Open", "High", "Low", "Close"}.issubset(df.columns):
        bad_high = (df["Open"] > df["High"]).sum() + (df["Close"] > df["High"]).sum()
        bad_low = (df["Open"] < df["Low"]).sum() + (df["Close"] < df["Low"]).sum()
        if bad_high > 0:
            result.warnings.append(f"High < Open/Close 的行数: {bad_high}")
        if bad_low > 0:
            result.warnings.append(f"Low > Open/Close 的行数: {bad_low}")

    # Volume 非负
    if "Volume" in df.columns:
        neg_vol = (df["Volume"] < 0).sum()
        if neg_vol > 0:
            result.warnings.append(f"Volume < 0 的行数: {neg_vol}")

    # Adj Close ≤ Close (如果都存在)
    if "Adj Close" in df.columns and "Close" in df.columns:
        adj_gt_close = (df["Adj Close"] > df["Close"] * 1.001).sum()  # 允许微小浮点误差
        if adj_gt_close > 0:
            result.warnings.append(f"Adj Close > Close 的行数: {adj_gt_close}")

    # 全空行
    all_nan_rows = df[REQUIRED_COLUMNS].dropna(how="all").shape[0] if all(
        c in df.columns for c in REQUIRED_COLUMNS
    ) else df.shape[0]
    empty_rows = df.shape[0] - all_nan_rows
    if empty_rows > 0:
        result.warnings.append(f"全空行数: {empty_rows}")

    return result

