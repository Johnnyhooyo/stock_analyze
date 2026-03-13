"""
策略: RSI 均值回归 (RSI Mean Reversion)
信号: RSI < 超卖阈值 → 做多 (1), RSI > 超买阈值 → 观望 (0)
超参: rsi_period / rsi_oversold / rsi_overbought, 可在 config 覆盖
"""
import pandas as pd
import numpy as np


NAME = "rsi_reversion"


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def run(data: pd.DataFrame, config: dict):
    period    = int(config.get("rsi_period", 14))
    oversold  = float(config.get("rsi_oversold", 30))
    overbought = float(config.get("rsi_overbought", 70))

    df = data.copy()
    df["rsi"] = _rsi(df["Close"], period)
    df = df.dropna(subset=["rsi"])

    # 持仓逻辑：RSI < oversold 进场，RSI > overbought 离场，中间维持上一状态
    position = 0
    positions = []
    for rsi_val in df["rsi"]:
        if rsi_val < oversold:
            position = 1
        elif rsi_val > overbought:
            position = 0
        positions.append(position)

    signal = pd.Series(positions, index=df.index, dtype=int)

    meta = {
        "name": NAME,
        "params": {"rsi_period": period, "rsi_oversold": oversold, "rsi_overbought": overbought},
        "feat_cols": [],
    }
    return signal, None, meta

