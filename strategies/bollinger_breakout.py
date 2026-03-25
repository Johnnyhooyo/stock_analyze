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

from strategies.indicators import bollinger_bands, rsi

NAME = "bollinger_breakout"


def run(data: pd.DataFrame, config: dict):
    period = int(config.get("bb_period", 20))
    num_std = float(config.get("bb_std", 2.0))
    rsi_period = int(config.get("rsi_period", 14))
    rsi_oversold = float(config.get("rsi_oversold", 30))
    rsi_overbought = float(config.get("rsi_overbought", 70))

    df = data.copy()

    # 计算布林带
    df["bb_upper"], df["bb_middle"], df["bb_lower"] = bollinger_bands(
        df["Close"], period, num_std
    )

    # 计算 RSI
    df["rsi"] = rsi(df["Close"], rsi_period)

    # 标记位置
    df["at_lower"] = (df["Close"] <= df["bb_lower"]).astype(int)
    df["at_upper"] = (df["Close"] >= df["bb_upper"]).astype(int)

    # 前期数据不足时填充
    df = df.dropna(subset=["bb_upper", "bb_middle", "bb_lower", "rsi"])

    position = 0
    positions = []
    close_arr   = df["Close"].values
    upper_arr   = df["bb_upper"].values
    lower_arr   = df["bb_lower"].values
    middle_arr  = df["bb_middle"].values
    rsi_arr     = df["rsi"].values
    at_lower_arr = df["at_lower"].values
    at_upper_arr = df["at_upper"].values

    for i in range(len(df)):
        at_lower_v = at_lower_arr[i]
        at_upper_v = at_upper_arr[i]
        rsi_val    = rsi_arr[i]
        close_v    = close_arr[i]

        if position == 0:
            if at_lower_v == 1 and rsi_val < rsi_oversold:
                position = 1
            elif at_upper_v == 1 and rsi_val < rsi_overbought:
                position = 1
        else:
            if at_upper_v == 1 and rsi_val > rsi_overbought:
                position = 0
            elif close_v < middle_arr[i]:
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


def predict(model, data: pd.DataFrame, config: dict, meta: dict) -> pd.Series:
    """规则策略独立推断接口：重新运行策略，返回信号序列（无需 model）。"""
    signal, _, _ = run(data, config)
    return signal

