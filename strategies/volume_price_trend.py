"""
成交量价格趋势策略
结合成交量和价格动量判断趋势
"""

import numpy as np
import pandas as pd


NAME = "volume_price_trend"


def run(data: pd.DataFrame, config: dict):
    volume_ma_period = int(config.get("volume_ma_period", 20))
    price_period = int(config.get("price_period", 10))

    df = data.copy()

    # 成交量均线
    df["volume_ma"] = df["Volume"].rolling(window=volume_ma_period).mean()

    # 成交量放大
    df["volume_increase"] = df["Volume"] > df["volume_ma"]

    # 价格动量
    df["price_return"] = df["Close"].pct_change(periods=price_period)

    # 价格趋势
    df["price_ma"] = df["Close"].rolling(window=price_period).mean()
    df["price_above_ma"] = df["Close"] > df["price_ma"]

    # ⚠️ 前视偏差修复：去掉 ffill() 平滑，改为状态机。
    # 进场：成交量放大 + 价格站上均线；离场：任一条件不满足。
    df = df.dropna(subset=["volume_ma", "price_ma"])

    buy_signal  = (df["volume_increase"] & df["price_above_ma"]).values
    sell_signal = (~df["price_above_ma"]).values  # 价格跌破均线即离场

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
            "volume_ma_period": volume_ma_period,
            "price_period": price_period,
        },
        "feat_cols": [],
        "indicators": {
            "volume_ma": df["volume_ma"],
            "price_return": df["price_return"],
        },
    }
    return signal, None, meta


def predict(model, data: pd.DataFrame, config: dict, meta: dict) -> pd.Series:
    """规则策略独立推断接口：重新运行策略，返回信号序列（无需 model）。"""
    signal, _, _ = run(data, config)
    return signal
