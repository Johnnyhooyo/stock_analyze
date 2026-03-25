"""
策略: MACD + RSI + 热度因子组合
------------------------------------------------------
信号逻辑:
  进场 (1): MACD金叉 + RSI偏弱 + 热度上升确认
  离场 (0): MACD死叉 + RSI超买 或 热度下降

超参数:
  macd_fast    (default 12)   MACD快速线周期
  macd_slow    (default 26)   MACD慢速线周期
  macd_signal  (default 9)    MACD信号线周期
  rsi_period   (default 14)   RSI周期
  rsi_oversold (default 40)   RSI偏弱阈值
  rsi_overbought (default 65) RSI超买阈值
  trend_period (default 5)    热度均线周期
"""

import pandas as pd

from strategies.indicators import macd, rsi

NAME = "macd_rsi_trend"


def run(data: pd.DataFrame, config: dict):
    macd_fast = int(config.get("macd_fast", 12))
    macd_slow = int(config.get("macd_slow", 26))
    macd_signal = int(config.get("macd_signal", 9))
    rsi_period = int(config.get("rsi_period", 14))
    rsi_oversold = float(config.get("rsi_oversold", 40))
    rsi_overbought = float(config.get("rsi_overbought", 65))
    trend_period = int(config.get("trend_period", 5))

    df = data.copy()

    # 合并热度数据
    try:
        from google_trends import get_trends_with_price as _gt
        df = _gt(df)
    except Exception:
        pass

    # MACD
    df["macd_line"], df["signal_line"], df["histogram"] = macd(
        df["Close"], macd_fast, macd_slow, macd_signal
    )

    # RSI
    df["rsi"] = rsi(df["Close"], rsi_period)

    # 热度均线
    if "trend" in df.columns:
        df["trend_ma"] = df["trend"].rolling(trend_period).mean()
        df["trend_up"] = (df["trend"] > df["trend_ma"]).astype(int)
    else:
        df["trend_up"] = 1

    # MACD金叉/死叉
    df["histogram_prev"] = df["histogram"].shift(1)
    df["golden_cross"] = ((df["histogram"] > 0) & (df["histogram_prev"] <= 0)).astype(int)
    df["death_cross"] = ((df["histogram"] < 0) & (df["histogram_prev"] >= 0)).astype(int)

    df = df.dropna(subset=["macd_line", "signal_line", "rsi"])

    position = 0
    positions = []
    golden_arr = df["golden_cross"].values
    death_arr  = df["death_cross"].values
    rsi_arr    = df["rsi"].values
    trend_arr  = df["trend_up"].values

    for i in range(len(df)):
        golden   = golden_arr[i]
        death    = death_arr[i]
        rsi_val  = rsi_arr[i]
        trend_up = trend_arr[i]

        if position == 0:
            if golden == 1 and rsi_val < rsi_overbought and trend_up == 1:
                position = 1
        else:
            if death == 1 or rsi_val > rsi_overbought or trend_up == 0:
                position = 0

        positions.append(position)

    signal = pd.Series(positions, index=df.index, dtype=int)

    meta = {
        "name": NAME,
        "params": {
            "macd_fast": macd_fast,
            "macd_slow": macd_slow,
            "macd_signal": macd_signal,
            "rsi_period": rsi_period,
            "rsi_oversold": rsi_oversold,
            "rsi_overbought": rsi_overbought,
            "trend_period": trend_period,
        },
        "feat_cols": [],
        "indicators": {
            "macd_line": df["macd_line"],
            "signal_line": df["signal_line"],
            "histogram": df["histogram"],
            "rsi": df["rsi"],
        },
    }
    return signal, None, meta


def predict(model, data: pd.DataFrame, config: dict, meta: dict) -> pd.Series:
    """规则策略独立推断接口：重新运行策略，返回信号序列（无需 model）。"""
    signal, _, _ = run(data, config)
    return signal

