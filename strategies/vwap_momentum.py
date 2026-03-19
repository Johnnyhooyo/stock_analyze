"""
VWAP 动量策略
价格站上VWAP均线买入，跌破VWAP卖出
"""

import numpy as np
import pandas as pd


NAME = "vwap_momentum"


def run(data: pd.DataFrame, config: dict):
    vwap_period = int(config.get("vwap_period", 20))
    momentum_period = int(config.get("momentum_period", 10))

    df = data.copy()

    # 计算典型价
    df["typical_price"] = (df["High"] + df["Low"] + df["Close"]) / 3

    # 计算 VWAP (成交量加权平均价)
    df["vwap"] = (df["typical_price"] * df["Volume"]).rolling(window=vwap_period).sum() / df["Volume"].rolling(window=vwap_period).sum()

    # 计算动量
    df["momentum"] = df["Close"].pct_change(periods=momentum_period)

    # 信号：价格站上VWAP且动量为正买入，反之卖出
    df["above_vwap"] = df["Close"] > df["vwap"]
    df["momentum_positive"] = df["momentum"] > 0

    # ⚠️ 前视偏差修复：去掉 ffill() 平滑，改为状态机。
    # 进场：价格 > VWAP 且动量为正；离场：价格跌破 VWAP 或动量转负。
    df = df.dropna(subset=["vwap", "momentum"])

    buy_signal  = (df["above_vwap"] & df["momentum_positive"]).values
    sell_signal = (~df["above_vwap"]).values  # 价格跌破 VWAP 即离场

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
            "vwap_period": vwap_period,
            "momentum_period": momentum_period,
        },
        "feat_cols": [],
        "indicators": {
            "vwap": df["vwap"],
            "momentum": df["momentum"],
        },
    }
    return signal, None, meta


def predict(model, data: pd.DataFrame, config: dict, meta: dict) -> pd.Series:
    """规则策略独立推断接口：重新运行策略，返回信号序列（无需 model）。"""
    signal, _, _ = run(data, config)
    return signal
