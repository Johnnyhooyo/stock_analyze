"""
策略: LightGBM + ta-lib + tsfresh
------------------------------------------------------
基于 lightgbm_enhanced，同时启用 ta-lib 技术指标和 tsfresh 自动特征。

特征工程:
  - ta-lib 技术指标 (RSI, MACD, Bollinger, KDJ, ATR, OBV)
  - tsfresh 自动特征 (滚动窗口 10, 20)

模型: LightGBM Classifier

配置选项:
  use_ta_lib: true (强制启用)
  use_tsfresh_features: true (强制启用)
"""

import numpy as np
import pandas as pd
from typing import Tuple

NAME = "lightgbm_enhanced_ta_tsfresh"

# 导入基础策略
from strategies.lightgbm_enhanced import run as lgbm_run
from strategies.lightgbm_enhanced import predict as lgbm_predict


def run(data: pd.DataFrame, config: dict):
    """
    运行 LightGBM + ta-lib + tsfresh 策略
    """
    # 强制启用 ta-lib 和 tsfresh
    config = config.copy()
    config['use_ta_lib'] = True
    config['use_tsfresh_features'] = True
    config['tsfresh_window_sizes'] = config.get('tsfresh_window_sizes', [10, 20])

    # 调用基础 lightgbm_enhanced
    return lgbm_run(data, config)


def predict(model, data: pd.DataFrame, config: dict, meta: dict) -> pd.Series:
    """独立推断：委托给 lightgbm_enhanced.predict()，并强制启用 ta-lib + tsfresh。"""
    config = config.copy()
    config['use_ta_lib'] = True
    config['use_tsfresh_features'] = True
    config['tsfresh_window_sizes'] = config.get('tsfresh_window_sizes', [10, 20])
    return lgbm_predict(model, data, config, meta)
