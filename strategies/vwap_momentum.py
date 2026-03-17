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

    df["signal"] = 0
    df.loc[df["above_vwap"] & df["momentum_positive"], "signal"] = 1

    # 平滑信号
    df["signal"] = df["signal"].replace(0, np.nan).ffill().fillna(0).astype(int)

    signal = df["signal"]

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
