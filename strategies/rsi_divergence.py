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

    # ⚠️ 前视偏差修复：去掉 ffill() 平滑，改为状态机。
    # 旧做法：replace(0,nan).ffill() 会把买入触发信号无限延续，掩盖"无信号=离场"语义，虚增持仓时间。
    # 新做法：进场后持仓，直到收到明确离场信号（超买+回落）才离场。
    buy_signal  = (df["rsi_oversold"] & df["rsi_rebound"]).values
    sell_signal = (df["rsi_overbought"] & df["rsi_decline"]).values

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


def predict(model, data: pd.DataFrame, config: dict, meta: dict) -> pd.Series:
    """规则策略独立推断接口：重新运行策略，返回信号序列（无需 model）。"""
    signal, _, _ = run(data, config)
    return signal
