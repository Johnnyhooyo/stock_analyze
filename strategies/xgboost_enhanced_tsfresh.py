"""
策略: XGBoost + tsfresh 特征
------------------------------------------------------
基于 xgboost_enhanced，但启用 tsfresh 自动特征提取。

特征工程:
  - 收益特征 (ret_1 ~ ret_20)
  - 技术指标 (RSI, MACD, Bollinger Bands, KDJ, ATR, OBV)
  - 波动率特征
  - 成交量特征
  - 趋势特征
  - tsfresh 自动特征 (滚动窗口 10, 20)

模型: XGBoost Classifier

这是 xgboost_enhanced 的 tsfresh 增强版本，
内部调用 xgboost_enhanced.run() 并设置 use_tsfresh_features=True
"""

import numpy as np
import pandas as pd
from typing import Tuple

NAME = "xgboost_enhanced_tsfresh"

# 导入基础策略
from strategies.xgboost_enhanced import run as xgboost_run


def run(data: pd.DataFrame, config: dict):
    """
    运行 XGBoost + tsfresh 策略

    内部调用 xgboost_enhanced.run()，但强制启用 tsfresh 特征
    """
    # 强制启用 tsfresh 特征
    config = config.copy()
    config['use_tsfresh_features'] = True
    config['tsfresh_window_sizes'] = config.get('tsfresh_window_sizes', [10, 20])

    # 调用基础 xgboost_enhanced
    return xgboost_run(data, config)
