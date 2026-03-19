"""
策略: RSI + OBV + 斐波那契回撤 量价组合 (RSI + On-Balance Volume + Fibonacci Retracement)
------------------------------------------------------
信号逻辑:
  进场 (1): RSI < rsi_oversold  且  OBV > OBV移动均线
            且  Close <= Fib_0.618支撑位  （超卖+量能上升+处于斐波那契强支撑区）
  离场 (0): RSI > rsi_overbought  或  OBV < OBV移动均线
            或  Close >= Fib_0.382阻力位  （超买 或 量能走弱 或 价格触及斐波那契阻力）
  中间状态: 维持上一持仓

超参数:
  rsi_period      (default 14)   RSI 计算周期
  rsi_oversold    (default 30)   RSI 超卖阈值（触发做多）
  rsi_overbought  (default 70)   RSI 超买阈值（触发观望）
  obv_ma_period   (default 20)   OBV 平滑移动平均周期（用于判断 OBV 趋势方向）
  fib_period      (default 60)   斐波那契回撤计算窗口（取最近 N 根K线的最高/最低价）

OBV 计算:
  OBV = cumsum( sign(Close.diff()) * Volume )
  OBV 上穿均线 → 买盘主导，确认做多信号
  OBV 下穿均线 → 卖盘主导，触发离场

斐波那契回撤:
  swing_high = rolling_max(High, fib_period)
  swing_low  = rolling_min(Low,  fib_period)
  Fib_0.618  = swing_high - 0.618 * (swing_high - swing_low)  ← 支撑位（进场过滤）
  Fib_0.382  = swing_high - 0.382 * (swing_high - swing_low)  ← 阻力位（离场过滤）
"""

import pandas as pd
import numpy as np

NAME = "rsi_obv"


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def _fibonacci(high: pd.Series, low: pd.Series, period: int):
    """返回 (fib_0618_support, fib_0382_resistance) 两条 Series。"""
    swing_high = high.rolling(period).max()
    swing_low  = low.rolling(period).min()
    diff       = swing_high - swing_low
    fib_0618   = swing_high - 0.618 * diff   # 支撑位
    fib_0382   = swing_high - 0.382 * diff   # 阻力位
    return fib_0618, fib_0382


def run(data: pd.DataFrame, config: dict):
    if "Volume" not in data.columns:
        raise ValueError("RSI+OBV 策略需要 'Volume' 列，当前数据中未找到。")

    period        = int(config.get("rsi_period", 14))
    oversold      = float(config.get("rsi_oversold", 30))
    overbought    = float(config.get("rsi_overbought", 70))
    obv_ma_period = int(config.get("obv_ma_period", 20))
    fib_period    = int(config.get("fib_period", 60))

    df = data.copy()
    df["rsi"]    = _rsi(df["Close"], period)
    df["obv"]    = _obv(df["Close"], df["Volume"])
    df["obv_ma"] = df["obv"].rolling(obv_ma_period).mean()
    df["fib_0618"], df["fib_0382"] = _fibonacci(df["High"], df["Low"], fib_period)
    df = df.dropna(subset=["rsi", "obv_ma", "fib_0618", "fib_0382"])

    position  = 0
    positions = []
    for rsi_val, obv_val, obv_ma_val, close_val, f618, f382 in zip(
        df["rsi"], df["obv"], df["obv_ma"], df["Close"], df["fib_0618"], df["fib_0382"]
    ):
        if position == 0:
            # 超卖 + OBV 上升确认 + 价格处于斐波那契 0.618 支撑区 → 进场
            if rsi_val < oversold and obv_val > obv_ma_val and close_val <= f618:
                position = 1
        else:
            # 超买 或 OBV 走弱 或 价格触及斐波那契 0.382 阻力位 → 离场
            if rsi_val > overbought or obv_val < obv_ma_val or close_val >= f382:
                position = 0
        positions.append(position)

    signal = pd.Series(positions, index=df.index, dtype=int)

    meta = {
        "name": NAME,
        "params": {
            "rsi_period":     period,
            "rsi_oversold":   oversold,
            "rsi_overbought": overbought,
            "obv_ma_period":  obv_ma_period,
            "fib_period":     fib_period,
        },
        "feat_cols": [],
        "indicators": {
            "rsi":        df["rsi"],
            "rsi_oversold":  oversold,
            "rsi_overbought": overbought,
            "obv":        df["obv"],
            "obv_ma":     df["obv_ma"],
            "fib_0618":   df["fib_0618"],
            "fib_0382":   df["fib_0382"],
        },
    }
    return signal, None, meta


def predict(model, data: pd.DataFrame, config: dict, meta: dict) -> pd.Series:
    """规则策略独立推断接口：重新运行策略，返回信号序列（无需 model）。"""
    signal, _, _ = run(data, config)
    return signal

