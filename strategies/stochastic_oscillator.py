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

    # ⚠️ 前视偏差修复：去掉 ffill() 平滑，改为状态机。
    # 进场：%K 上穿 %D 且处于超卖区；离场：%K 下穿 %D 且处于超买区。
    df = df.dropna(subset=["k", "d"])

    buy_signal  = (df["k_cross_up"]   & (df["k"] < k_low)).values
    sell_signal = (df["k_cross_down"] & (df["k"] > k_high)).values

    position = 0
    positions = []
    for i in range(len(df)):
        if position == 0 and buy_signal[i]:
            position = 1
        elif position == 1 and sell_signal[i]:
            position = 0
        positions.append(position)

    signal = pd.Series(positions, index=df.index, dtype=int)

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


def predict(model, data: pd.DataFrame, config: dict, meta: dict) -> pd.Series:
    """规则策略独立推断接口：重新运行策略，返回信号序列（无需 model）。"""
    signal, _, _ = run(data, config)
    return signal
