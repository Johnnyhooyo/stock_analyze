"""
RSI 背离策略
当价格创新低但RSI未创新低时买入（金叉），当价格创新高但RSI未创新高时卖出（死叉）
"""

import numpy as np
import pandas as pd


NAME = "rsi_divergence"


def run(data: pd.DataFrame, config: dict):
    rsi_period = int(config.get("rsi_period", 14))
    rsi_oversold = float(config.get("rsi_oversold", 30))
    rsi_overbought = float(config.get("rsi_overbought", 70))

    df = data.copy()

    # 计算 RSI
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=rsi_period).mean()
    avg_loss = loss.rolling(window=rsi_period).mean()
    rs = avg_gain / avg_loss
    df["rsi"] = 100 - (100 / (1 + rs))

    # 计算价格动量
    df["price_change"] = df["Close"].diff(10)  # 10日涨跌

    # RSI 背离信号
    # 超卖金叉：RSI < 30 且开始回升
    df["rsi_oversold"] = df["rsi"] < rsi_oversold
    df["rsi_rebound"] = df["rsi"] > df["rsi"].shift(1)

    # 超买死叉：RSI > 70 且开始回落
    df["rsi_overbought"] = df["rsi"] > rsi_overbought
    df["rsi_decline"] = df["rsi"] < df["rsi"].shift(1)

    # 信号：超卖+反弹买入，超买+回落卖出
    df["signal"] = 0
    df.loc[df["rsi_oversold"] & df["rsi_rebound"], "signal"] = 1
    df.loc[df["rsi_overbought"] & df["rsi_decline"], "signal"] = 0

    # 平滑信号：连续持有信号
    df["signal"] = df["signal"].replace(0, np.nan).ffill().fillna(0).astype(int)

    signal = df["signal"]

    meta = {
        "name": NAME,
        "params": {
            "rsi_period": rsi_period,
            "rsi_oversold": rsi_oversold,
            "rsi_overbought": rsi_overbought,
        },
        "feat_cols": [],
        "indicators": {
            "rsi": df["rsi"],
        },
    }
    return signal, None, meta
