"""
ATR 突破策略
价格突破N倍ATR区间时买入，跌破时卖出
"""

import numpy as np
import pandas as pd


NAME = "atr_breakout"


def run(data: pd.DataFrame, config: dict):
    atr_period = int(config.get("atr_period", 14))
    atr_multiplier = float(config.get("atr_multiplier", 2.0))

    df = data.copy()

    # 计算 ATR (Average True Range)
    high_low = df["High"] - df["Low"]
    high_close = np.abs(df["High"] - df["Close"].shift())
    low_close = np.abs(df["Low"] - df["Close"].shift())

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df["atr"] = true_range.rolling(window=atr_period).mean()

    # 计算中轨（均线）
    df["ma"] = df["Close"].rolling(window=atr_period).mean()

    # 计算上下轨
    df["upper_band"] = df["ma"] + atr_multiplier * df["atr"]
    df["lower_band"] = df["ma"] - atr_multiplier * df["atr"]

    # 信号：突破上轨买入，跌破下轨卖出
    df["signal"] = 0
    df.loc[df["Close"] > df["upper_band"], "signal"] = 1
    df.loc[df["Close"] < df["lower_band"], "signal"] = 0

    # 平滑信号
    df["signal"] = df["signal"].replace(0, np.nan).ffill().fillna(0).astype(int)

    signal = df["signal"]

    meta = {
        "name": NAME,
        "params": {
            "atr_period": atr_period,
            "atr_multiplier": atr_multiplier,
        },
        "feat_cols": [],
        "indicators": {
            "atr": df["atr"],
            "upper_band": df["upper_band"],
            "lower_band": df["lower_band"],
        },
    }
    return signal, None, meta
