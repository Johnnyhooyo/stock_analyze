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
#
# ── 技术指标模块 ───────────────────────────────────────────────
#
#   indicators.py : ta-lib 封装，提供标准化技术指标计算
#               - add_ta_features() : 使用 ta-lib 计算所有指标
#               - add_ta_features_fallback() : 纯 pandas fallback
#
#   xgboost_enhanced 支持 use_ta_lib 配置项启用 ta-lib

import os as _os


def ml_thread_budget() -> int:
    """
    返回 ML 模型(XGBoost/LightGBM/RandomForest)应使用的线程数。

    遵循环境变量 STOCK_ML_THREADS(由 optimize_with_optuna 在启动 study
    前根据 outer n_jobs 设定);未设置时退回 os.cpu_count()。

    目的:避免外层 Optuna n_jobs 与内层 GBDT 多线程相乘导致线程超订。
    """
    try:
        v = _os.environ.get('STOCK_ML_THREADS')
        if v:
            n = int(v)
            return n if n > 0 else 1
    except Exception:
        pass
    return _os.cpu_count() or 1
