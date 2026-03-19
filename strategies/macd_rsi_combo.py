"""
策略: MACD + RSI 组合 (MACD + RSI Combo)
------------------------------------------------------
信号逻辑:
  进场 (1): MACD 金叉 (MACD线从下向上穿过信号线) 且 RSI 处于中性偏多区域
  离场 (0): MACD 死叉 (MACD线从上向下穿过信号线) 或 RSI 进入超买区域
  中间状态: 维持上一持仓

超参数:
  macd_fast    (default 12)   MACD 快速 EMA 周期
  macd_slow    (default 26)   MACD 慢速 EMA 周期
  macd_signal  (default 9)    MACD 信号线周期
  rsi_period   (default 14)   RSI 计算周期
  rsi_oversold (default 40)   RSI 做多阈值（中性偏多）
  rsi_overbought (default 65) RSI 超买阈值

MACD 计算:
  ema_fast  = EMA(close, macd_fast)
  ema_slow  = EMA(close, macd_slow)
  macd_line = ema_fast - ema_slow
  signal_line = EMA(macd_line, macd_signal)
  histogram   = macd_line - signal_line

进场条件:
  1. MACD 金叉 (histogram 从负转正)
  2. RSI < rsi_overbought 且 RSI > rsi_oversold（保持在中性区域）
离场条件:
  1. MACD 死叉 (histogram 从正转负)
  2. RSI > rsi_overbought（进入超买区域）
"""

import pandas as pd
import numpy as np

NAME = "macd_rsi_combo"


def _ema(series: pd.Series, period: int) -> pd.Series:
    """计算指数移动平均线。"""
    return series.ewm(span=period, adjust=False).mean()


def _macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """返回 (macd_line, signal_line, histogram)。"""
    ema_fast = _ema(close, fast)
    ema_slow = _ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def run(data: pd.DataFrame, config: dict):
    macd_fast = int(config.get("macd_fast", 12))
    macd_slow = int(config.get("macd_slow", 26))
    macd_signal = int(config.get("macd_signal", 9))
    rsi_period = int(config.get("rsi_period", 14))
    rsi_oversold = float(config.get("rsi_oversold", 40))
    rsi_overbought = float(config.get("rsi_overbought", 65))

    df = data.copy()

    # 计算 MACD
    df["macd_line"], df["signal_line"], df["histogram"] = _macd(
        df["Close"], macd_fast, macd_slow, macd_signal
    )

    # 计算 RSI
    df["rsi"] = _rsi(df["Close"], rsi_period)

    # 标记 MACD 金叉/死叉
    df["histogram_prev"] = df["histogram"].shift(1)
    df["golden_cross"] = ((df["histogram"] > 0) & (df["histogram_prev"] <= 0)).astype(int)
    df["death_cross"] = ((df["histogram"] < 0) & (df["histogram_prev"] >= 0)).astype(int)

    df = df.dropna(subset=["macd_line", "signal_line", "rsi"])

    # P3-B: 用 .values 数组替代 iterrows()，避免逐行 pd.Series 访问开销
    position = 0
    positions = []
    golden_arr = df["golden_cross"].values
    death_arr  = df["death_cross"].values
    rsi_arr    = df["rsi"].values

    for i in range(len(df)):
        golden  = golden_arr[i]
        death   = death_arr[i]
        rsi_val = rsi_arr[i]

        if position == 0:
            if golden == 1 and rsi_oversold < rsi_val < rsi_overbought:
                position = 1
        else:
            if death == 1 or rsi_val > rsi_overbought:
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

