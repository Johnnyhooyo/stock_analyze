"""
策略: 布林带均值回归 + RSI 确认 (Bollinger Bands Reversion with RSI)
------------------------------------------------------
信号逻辑:
  进场 (1): 价格触及下轨或跌破下轨 + RSI 超卖 → 逢低买入
  离场 (0): 价格触及上轨或突破上轨 + RSI 超买 → 逢高卖出
  中间状态: 维持上一持仓

超参数:
  bb_period       (default 20)   布林带计算周期
  bb_std          (default 2.0)  标准差倍数
  rsi_period      (default 14)   RSI 计算周期
  rsi_oversold    (default 30)   RSI 超卖阈值
  rsi_overbought  (default 70)   RSI 超买阈值
  mean_reversion  (default true) 是否启用均值回归模式

布林带计算:
  middle = SMA(close, period)
  upper  = middle + std * bb_std
  lower  = middle - std * bb_std

进场条件 (均值回归模式):
  价格 <= 下轨 且 RSI < rsi_oversold → 超卖反弹
离场条件:
  价格 >= 上轨 或 RSI > rsi_overbought → 超买回落
"""

import pandas as pd
import numpy as np

NAME = "bollinger_breakout"


def _bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0):
    """返回 (upper, middle, lower) 三条布林带。"""
    middle = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def run(data: pd.DataFrame, config: dict):
    period = int(config.get("bb_period", 20))
    num_std = float(config.get("bb_std", 2.0))
    rsi_period = int(config.get("rsi_period", 14))
    rsi_oversold = float(config.get("rsi_oversold", 30))
    rsi_overbought = float(config.get("rsi_overbought", 70))

    df = data.copy()

    # 计算布林带
    df["bb_upper"], df["bb_middle"], df["bb_lower"] = _bollinger_bands(
        df["Close"], period, num_std
    )

    # 计算 RSI
    df["rsi"] = _rsi(df["Close"], rsi_period)

    # 标记位置
    df["at_lower"] = (df["Close"] <= df["bb_lower"]).astype(int)
    df["at_upper"] = (df["Close"] >= df["bb_upper"]).astype(int)

    # 前期数据不足时填充
    df = df.dropna(subset=["bb_upper", "bb_middle", "bb_lower", "rsi"])

    position = 0
    positions = []
    for idx, row in df.iterrows():
        close = row["Close"]
        upper = row["bb_upper"]
        lower = row["bb_lower"]
        rsi_val = row["rsi"]
        at_lower = row.get("at_lower", 0)
        at_upper = row.get("at_upper", 0)

        if position == 0:
            # 价格触及下轨且 RSI 超卖 → 进场（均值回归）
            if at_lower == 1 and rsi_val < rsi_oversold:
                position = 1
            # 或者：价格突破上轨且 RSI 处于中性区域 → 顺势进场（趋势跟踪）
            elif at_upper == 1 and rsi_val < rsi_overbought:
                position = 1
        else:
            # 价格触及上轨且 RSI 超买 → 离场
            if at_upper == 1 and rsi_val > rsi_overbought:
                position = 0
            # 或者：价格跌破中轨 → 离场
            elif close < row["bb_middle"]:
                position = 0

        positions.append(position)

    signal = pd.Series(positions, index=df.index, dtype=int)

    meta = {
        "name": NAME,
        "params": {
            "bb_period": period,
            "bb_std": num_std,
            "rsi_period": rsi_period,
            "rsi_oversold": rsi_oversold,
            "rsi_overbought": rsi_overbought,
        },
        "feat_cols": [],
        "indicators": {
            "bb_upper": df["bb_upper"],
            "bb_middle": df["bb_middle"],
            "bb_lower": df["bb_lower"],
            "rsi": df["rsi"],
        },
    }
    return signal, None, meta
