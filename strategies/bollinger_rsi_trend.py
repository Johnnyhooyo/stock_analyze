"""
策略: 布林带 + RSI + 热度因子组合
------------------------------------------------------
信号逻辑:
  进场 (1): 布林带触及下轨 + RSI超卖 + 热度上升
  离场 (0): 布林带触及上轨 + RSI超买 或 热度下降

超参数:
  bb_period       (default 20)   布林带周期
  bb_std          (default 2.0)  标准差倍数
  rsi_period      (default 14)   RSI周期
  rsi_oversold   (default 30)   RSI超卖阈值
  rsi_overbought  (default 70)   RSI超买阈值
  trend_period    (default 5)    热度均线周期
  trend_confirm   (default True) 是否需要热度确认
"""

import pandas as pd
import numpy as np

NAME = "bollinger_rsi_trend"


def _bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0):
    """布林带计算"""
    middle = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def _rsi(series: pd.Series, period: int) -> pd.Series:
    """RSI计算"""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def run(data: pd.DataFrame, config: dict):
    # 导入热度数据
    try:
        from google_trends import get_trends_with_price
    except ImportError:
        pass

    period = int(config.get("bb_period", 20))
    num_std = float(config.get("bb_std", 2.0))
    rsi_period = int(config.get("rsi_period", 14))
    rsi_oversold = float(config.get("rsi_oversold", 30))
    rsi_overbought = float(config.get("rsi_overbought", 70))
    trend_period = int(config.get("trend_period", 5))

    df = data.copy()

    # 合并热度数据
    try:
        from google_trends import get_trends_with_price as _gt
        df = _gt(df)
    except Exception:
        pass

    # 布林带
    df["bb_upper"], df["bb_middle"], df["bb_lower"] = _bollinger_bands(
        df["Close"], period, num_std
    )

    # RSI
    df["rsi"] = _rsi(df["Close"], rsi_period)

    # 热度均线
    if "trend" in df.columns:
        df["trend_ma"] = df["trend"].rolling(trend_period).mean()
        df["trend_up"] = (df["trend"] > df["trend_ma"]).astype(int)
    else:
        df["trend_up"] = 1  # 无热度数据时默认

    df = df.dropna(subset=["bb_upper", "bb_middle", "bb_lower", "rsi"])

    position = 0
    positions = []
    close_arr  = df["Close"].values
    upper_arr  = df["bb_upper"].values
    lower_arr  = df["bb_lower"].values
    middle_arr = df["bb_middle"].values
    rsi_arr    = df["rsi"].values
    trend_arr  = df["trend_up"].values

    for i in range(len(df)):
        close_v    = close_arr[i]
        upper_v    = upper_arr[i]
        lower_v    = lower_arr[i]
        middle_v   = middle_arr[i]
        rsi_val    = rsi_arr[i]
        trend_up   = trend_arr[i]

        if position == 0:
            if close_v <= lower_v and rsi_val < rsi_oversold:
                position = 1
            elif close_v >= upper_v and rsi_val < rsi_overbought and trend_up == 1:
                position = 1
        else:
            if close_v >= upper_v and rsi_val > rsi_overbought:
                position = 0
            elif close_v < middle_v:
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
            "trend_period": trend_period,
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

