"""
随机指标策略 (Stochastic Oscillator)
%K 穿越 %D 时买入/卖出
"""

import numpy as np
import pandas as pd


NAME = "stochastic_oscillator"


def run(data: pd.DataFrame, config: dict):
    k_period = int(config.get("k_period", 14))
    d_period = int(config.get("d_period", 3))
    k_low = float(config.get("k_low", 20))
    k_high = float(config.get("k_high", 80))

    df = data.copy()

    # 计算 Stochastic Oscillator
    low_min = df["Low"].rolling(window=k_period).min()
    high_max = df["High"].rolling(window=k_period).max()

    df["k"] = 100 * (df["Close"] - low_min) / (high_max - low_min)
    df["d"] = df["k"].rolling(window=d_period).mean()

    # 信号：%K 穿越 %D
    df["k_above_d"] = df["k"] > df["d"]
    df["k_cross_up"] = (df["k"] > df["d"]) & (df["k"].shift(1) <= df["d"].shift(1))
    df["k_cross_down"] = (df["k"] < df["d"]) & (df["k"].shift(1) >= df["d"].shift(1))

    # 信号生成
    df["signal"] = 0
    # 买入：%K 从下穿越 %D 且在超卖区域
    df.loc[df["k_cross_up"] & (df["k"] < k_low), "signal"] = 1
    # 卖出：%K 从上穿越 %D 且在超买区域
    df.loc[df["k_cross_down"] & (df["k"] > k_high), "signal"] = 0

    # 平滑信号
    df["signal"] = df["signal"].replace(0, np.nan).ffill().fillna(0).astype(int)

    signal = df["signal"]

    meta = {
        "name": NAME,
        "params": {
            "k_period": k_period,
            "d_period": d_period,
            "k_low": k_low,
            "k_high": k_high,
        },
        "feat_cols": [],
        "indicators": {
            "k": df["k"],
            "d": df["d"],
        },
    }
    return signal, None, meta
