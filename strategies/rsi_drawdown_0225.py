"""
策略: RSI 均值回归 + 回撤风控 (RSI Mean Reversion with Drawdown Risk Control) [0225]
信号: RSI < 超卖阈值 → 做多 (1), RSI > 超买阈值 → 观望 (0)
风控: 持仓期间，若价格相对持仓后最高点回撤超过 drawdown_pct (默认 2%) → 强制平仓 (0)
超参:
  rsi_period      (default 14)   RSI 计算周期
  rsi_oversold    (default 30)   RSI 超卖阈值（触发做多）
  rsi_overbought  (default 70)   RSI 超买阈值（触发观望）
  drawdown_pct    (default 0.02) 持仓期间允许的最大回撤比例（相对于持仓后最高点）
"""

import pandas as pd
import numpy as np

NAME = "rsi_drawdown_0225"


def _rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def run(data: pd.DataFrame, config: dict):
    period       = int(config.get("rsi_period", 14))
    oversold     = float(config.get("rsi_oversold", 30))
    overbought   = float(config.get("rsi_overbought", 70))
    drawdown_pct = float(config.get("drawdown_pct", 0.02))  # 默认 2% 回撤止损

    df = data.copy()
    df["rsi"] = _rsi(df["Close"], period)
    df = df.dropna(subset=["rsi"])

    positions  = []
    position   = 0       # 当前持仓状态
    peak_price = None    # 持仓期间最高价

    for rsi_val, close_val in zip(df["rsi"], df["Close"]):
        if position == 0:
            # 未持仓：RSI 超卖则进场
            if rsi_val < oversold:
                position   = 1
                peak_price = close_val
        else:
            # 持仓中：更新最高价
            if close_val > peak_price:
                peak_price = close_val

            # 风控：相对最高点回撤超过阈值 → 强制平仓
            drawdown = (peak_price - close_val) / peak_price
            if drawdown >= drawdown_pct:
                position   = 0
                peak_price = None
            # RSI 超买也离场
            elif rsi_val > overbought:
                position   = 0
                peak_price = None

        positions.append(position)

    signal = pd.Series(positions, index=df.index, dtype=int)

    meta = {
        "name": NAME,
        "params": {
            "rsi_period":     period,
            "rsi_oversold":   oversold,
            "rsi_overbought": overbought,
            "drawdown_pct":   drawdown_pct,
        },
        "feat_cols": [],
    }
    return signal, None, meta

