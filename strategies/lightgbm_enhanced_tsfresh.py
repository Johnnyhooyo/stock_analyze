"""
策略: LightGBM + tsfresh 特征
------------------------------------------------------
基于 lightgbm_enhanced，但启用 tsfresh 自动特征提取。

特征工程:
  - 收益特征 (ret_1 ~ ret_20)
  - 技术指标 (RSI, MACD, Bollinger Bands, KDJ, ATR, OBV)
  - 波动率特征
  - 成交量特征
  - 趋势特征
  - tsfresh 自动特征 (滚动窗口 10, 20)

模型: LightGBM Classifier

这是 lightgbm_enhanced 的 tsfresh 增强版本，
内部调用 lightgbm_enhanced.run() 并设置 use_tsfresh_features=True
"""

import numpy as np
import pandas as pd
from typing import Tuple

NAME = "lightgbm_enhanced_tsfresh"

# 导入基础策略
from strategies.lightgbm_enhanced import run as lgbm_run
from strategies.lightgbm_enhanced import predict as lgbm_predict


def run(data: pd.DataFrame, config: dict):
    """
    运行 LightGBM + tsfresh 策略

    内部调用 lightgbm_enhanced.run()，但强制启用 tsfresh 特征
    """
    # 强制启用 tsfresh 特征，禁用 ta-lib
    config = config.copy()
    config['use_tsfresh_features'] = True
    config['tsfresh_window_sizes'] = config.get('tsfresh_window_sizes', [10, 20])
    config['use_ta_lib'] = False  # tsfresh 版本不使用 ta-lib

    # 调用基础 lightgbm_enhanced
    return lgbm_run(data, config)


def predict(model, data: pd.DataFrame, config: dict, meta: dict) -> pd.Series:
    """独立推断：委托给 lightgbm_enhanced.predict()，并强制启用 tsfresh 特征。"""
    config = config.copy()
    config['use_tsfresh_features'] = True
    config['tsfresh_window_sizes'] = config.get('tsfresh_window_sizes', [10, 20])
    config['use_ta_lib'] = False
    return lgbm_predict(model, data, config, meta)
