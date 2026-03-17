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

    # 信号：成交量放大 + 价格站上均线 = 买入
    # 成交量萎缩 + 价格跌破均线 = 卖出
    df["signal"] = 0
    df.loc[df["volume_increase"] & df["price_above_ma"], "signal"] = 1

    # 平滑信号
    df["signal"] = df["signal"].replace(0, np.nan).ffill().fillna(0).astype(int)

    signal = df["signal"]

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
