"""
策略: KDJ + OBV + 斐波那契回撤 量价组合 (KDJ + On-Balance Volume + Fibonacci Retracement)
------------------------------------------------------
信号逻辑:
  进场 (1): K < kdj_oversold 且 D < kdj_oversold 且 J < kdj_oversold
            且 OBV > OBV移动均线
            且 Close <= Fib_0.618支撑位  （KDJ三线全超卖 + 量能上升 + 处于斐波那契强支撑区）
  离场 (0): K > kdj_overbought 或 J > kdj_overbought
            或 OBV < OBV移动均线
            或 Close >= Fib_0.382阻力位  （KDJ超买 或 量能走弱 或 价格触及斐波那契阻力）
  中间状态: 维持上一持仓

超参数:
  kdj_period      (default 9)    KDJ 随机指标计算周期（RSV 窗口）
  kdj_oversold    (default 20)   KDJ 超卖阈值（K/D/J 均低于此值时触发做多）
  kdj_overbought  (default 80)   KDJ 超买阈值（K 或 J 高于此值时触发观望）
  obv_ma_period   (default 20)   OBV 平滑移动平均周期
  fib_period      (default 60)   斐波那契回撤计算窗口（取最近 N 根K线的最高/最低价）

KDJ 计算 (中国券商标准，1/3 指数平滑):
  RSV = (Close - LowestLow_n) / (HighestHigh_n - LowestLow_n) * 100
  K   = RSV 的 EWM 平滑 (alpha=1/3)
  D   = K 的 EWM 平滑 (alpha=1/3)
  J   = 3*K - 2*D

斐波那契回撤:
  swing_high = rolling_max(High, fib_period)
  swing_low  = rolling_min(Low,  fib_period)
  Fib_0.618  = swing_high - 0.618 * (swing_high - swing_low)  ← 支撑位（进场过滤）
  Fib_0.382  = swing_high - 0.382 * (swing_high - swing_low)  ← 阻力位（离场过滤）
"""

import pandas as pd
import numpy as np

NAME = "kdj_obv"


def _kdj(high: pd.Series, low: pd.Series, close: pd.Series, period: int):
    lowest_low    = low.rolling(period).min()
    highest_high  = high.rolling(period).max()
    denom         = (highest_high - lowest_low).replace(0, np.nan)
    rsv           = (close - lowest_low) / denom * 100
    K = rsv.ewm(alpha=1/3, adjust=False).mean()
    D = K.ewm(alpha=1/3, adjust=False).mean()
    J = 3 * K - 2 * D
    return K, D, J


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
        raise ValueError("KDJ+OBV 策略需要 'Volume' 列，当前数据中未找到。")

    kdj_period     = int(config.get("kdj_period", 9))
    oversold       = float(config.get("kdj_oversold", 20))
    overbought     = float(config.get("kdj_overbought", 80))
    obv_ma_period  = int(config.get("obv_ma_period", 20))
    fib_period     = int(config.get("fib_period", 60))

    df = data.copy()
    df["K"], df["D"], df["J"] = _kdj(df["High"], df["Low"], df["Close"], kdj_period)
    df["obv"]    = _obv(df["Close"], df["Volume"])
    df["obv_ma"] = df["obv"].rolling(obv_ma_period).mean()
    df["fib_0618"], df["fib_0382"] = _fibonacci(df["High"], df["Low"], fib_period)
    df = df.dropna(subset=["K", "D", "J", "obv_ma", "fib_0618", "fib_0382"])

    position  = 0
    positions = []
    for k, d, j, obv_val, obv_ma_val, close_val, f618, f382 in zip(
        df["K"], df["D"], df["J"], df["obv"], df["obv_ma"], df["Close"], df["fib_0618"], df["fib_0382"]
    ):
        if position == 0:
            # KDJ 三线全超卖 + OBV 上升确认 + 价格处于斐波那契 0.618 支撑区 → 进场
            if k < oversold and d < oversold and j < oversold and obv_val > obv_ma_val and close_val <= f618:
                position = 1
        else:
            # KDJ 超买（K 或 J 任一超买）或 OBV 走弱 或 价格触及斐波那契 0.382 阻力位 → 离场
            if k > overbought or j > overbought or obv_val < obv_ma_val or close_val >= f382:
                position = 0
        positions.append(position)

    signal = pd.Series(positions, index=df.index, dtype=int)

    meta = {
        "name": NAME,
        "params": {
            "kdj_period":     kdj_period,
            "kdj_oversold":   oversold,
            "kdj_overbought": overbought,
            "obv_ma_period":  obv_ma_period,
            "fib_period":     fib_period,
        },
        "feat_cols": [],
        "indicators": {
            "K":              df["K"],
            "D":              df["D"],
            "J":              df["J"],
            "kdj_oversold":   oversold,
            "kdj_overbought": overbought,
            "obv":            df["obv"],
            "obv_ma":         df["obv_ma"],
            "fib_0618":       df["fib_0618"],
            "fib_0382":       df["fib_0382"],
        },
    }
    return signal, None, meta


def predict(model, data: pd.DataFrame, config: dict, meta: dict) -> pd.Series:
    """规则策略独立推断接口：重新运行策略，返回信号序列（无需 model）。"""
    signal, _, _ = run(data, config)
    return signal

