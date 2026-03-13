# strategies package – each module must expose:
#   run(data: pd.DataFrame, config: dict) -> (signal: pd.Series, model, meta: dict)
#
# signal : int Series aligned to data.index, values 1 (long) / 0 (flat)
# model  : any serialisable object (or None)
# meta   : dict with at least {"name": str, "params": dict}
#
# ── 新增策略 (2024) ────────────────────────────────────────────────
#
#   bollinger_breakout : 布林带突破策略
#               config keys: bb_period, bb_std, breakout_days
#
#   macd_rsi_combo    : MACD + RSI 组合策略
#               config keys: macd_fast, macd_slow, macd_signal, rsi_period, rsi_oversold, rsi_overbought
#
# ── 量价组合策略 (Volume-Confirmed Strategies) ──
#
#   rsi_obv   : RSI 超卖/超买 + OBV 趋势确认
#               config keys: rsi_period, rsi_oversold, rsi_overbought, obv_ma_period
#
#   rsi_pvt   : RSI 超卖/超买 + PVT 趋势确认
#               config keys: rsi_period, rsi_oversold, rsi_overbought, pvt_ma_period
#
#   kdj_obv   : KDJ 超卖/超买 + OBV 趋势确认
#               config keys: kdj_period, kdj_oversold, kdj_overbought, obv_ma_period
#
#   kdj_pvt   : KDJ 超卖/超买 + PVT 趋势确认
#               config keys: kdj_period, kdj_oversold, kdj_overbought, pvt_ma_period
