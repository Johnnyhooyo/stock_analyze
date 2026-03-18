"""
策略: 技术指标增强版 LightGBM
------------------------------------------------------
特征工程:
  - 收益特征 (ret_1 ~ ret_20)
  - 技术指标 (RSI, MACD, Bollinger Bands, KDJ, ATR, OBV)
  - 波动率特征
  - 成交量特征
  - 趋势特征

模型: LightGBM Classifier
"""

import numpy as np
import pandas as pd
from typing import Tuple

NAME = "lightgbm_enhanced"

# 导入特征工程函数
from strategies.xgboost_enhanced import (
    add_features,
    prepare_data,
    _calculate_rsi,
    _calculate_macd,
    _calculate_bollinger_bands,
    _calculate_kdj,
    _calculate_atr,
    _calculate_obv,
    _calculate_pvt,
)


def run(data: pd.DataFrame, config: dict):
    """
    运行 LightGBM 增强版策略

    训练: 使用前 80% 的数据训练模型
    预测: 在后 20% 的数据上生成交易信号
    """
    # 参数
    test_days = int(config.get('test_days', 5))
    n_estimators = int(config.get('lgbm_n_estimators', 100))
    max_depth = int(config.get('lgbm_max_depth', 5))
    learning_rate = float(config.get('lgbm_learning_rate', 0.1))
    num_leaves = int(config.get('lgbm_num_leaves', 31))
    label_period = int(config.get('label_period', 1))

    # 添加特征
    df = add_features(data)

    # 准备数据
    X, y, feat_cols = prepare_data(df, test_days, label_period)

    # 分割训练/测试（80% 训练，20% 测试）
    # 如果 config 中设置了 no_internal_split，则使用全部数据训练
    no_split = config.get('no_internal_split', False)
    if no_split:
        # 使用全部数据训练
        if len(X) < 10:
            raise ValueError(f"数据不足: 需要 > 10 个样本")
        X_train = X
        y_train = y
        X_test = X
        y_test = y
    else:
        if len(X) < 10:
            raise ValueError(f"数据不足: 需要 > 10 个样本")
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_test = y.iloc[split_idx:]

    # 训练模型
    try:
        import lightgbm as lgb
        # 尝试使用 GPU
        try:
            model = lgb.LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                random_state=42,
                verbose=-1,
                device='gpu',
            )
        except Exception:
            # GPU 不可用，回退到 CPU
            model = lgb.LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                num_leaves=num_leaves,
                random_state=42,
                verbose=-1,
            )
        model_name = 'LightGBM'
    except ImportError:
        # 降级使用 XGBoost
        try:
            from xgboost import XGBClassifier
            model = XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0,
            )
            model_name = 'XGBoost'
        except ImportError:
            # 最后降级到 sklearn
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
            )
            model_name = 'GradientBoosting'

    model.fit(X_train, y_train)

    # 预测
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    test_proba = model.predict_proba(X_test)[:, 1]

    # 生成完整信号（在原始数据上生成信号，与 X 的索引对齐）
    # 创建与 X 等长的信号序列
    signal = pd.Series(0, index=X.index)

    # 将预测结果填充到对应位置
    if no_split:
        # 无内部分割时，使用全部预测
        signal.iloc[:] = test_pred
    else:
        if len(X_test) > 0:
            signal.iloc[split_idx:] = test_pred

    # 元数据
    meta = {
        'name': NAME,
        'params': {
            'test_days': test_days,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'num_leaves': num_leaves,
            'label_period': label_period,
        },
        'feat_cols': feat_cols,
        'model': model_name,
        'feature_importances': dict(zip(feat_cols, model.feature_importances_.round(4))) if hasattr(model, 'feature_importances_') else {},
        'indicators': {
            'pred_proba': pd.Series(test_proba, index=X_test.index),
        },
        'train_acc': (train_pred == y_train).mean(),
        'test_acc': (test_pred == y_test).mean() if len(y_test) > 0 else None,
        'train_size': len(X_train),
        'test_size': len(X_test),
    }

    return signal, model, meta
