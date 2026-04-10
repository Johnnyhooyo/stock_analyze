"""
ATR 突破策略
价格突破N倍ATR区间时买入，跌破时卖出
"""

import pandas as pd

from strategies.indicators import atr as _atr

NAME = "atr_breakout"


def run(data: pd.DataFrame, config: dict):
    atr_period = int(config.get("atr_period", 14))
    atr_multiplier = float(config.get("atr_multiplier", 2.0))

    df = data.copy()

    df["atr"] = _atr(df["High"], df["Low"], df["Close"], atr_period)

    # 计算中轨（均线）
    df["ma"] = df["Close"].rolling(window=atr_period).mean()

    # 计算上下轨
    df["upper_band"] = df["ma"] + atr_multiplier * df["atr"]
    df["lower_band"] = df["ma"] - atr_multiplier * df["atr"]

    # ⚠️ 前视偏差修复：去掉 ffill() 平滑，改为状态机。
    # 突破上轨进场，跌破下轨离场，中间维持持仓。
    df = df.dropna(subset=["atr", "ma", "upper_band", "lower_band"])

    buy_signal  = (df["Close"] > df["upper_band"]).values
    sell_signal = (df["Close"] < df["lower_band"]).values

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


def predict(model, data: pd.DataFrame, config: dict, meta: dict) -> pd.Series:
    """规则策略独立推断接口：重新运行策略，返回信号序列（无需 model）。"""
    signal, _, _ = run(data, config)
    return signal
