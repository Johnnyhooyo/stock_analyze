"""
策略: tsfresh 特征提取 + XGBoost
------------------------------------------------------
利用 tsfresh 自动从时间序列中提取数百个特征，
结合传统技术指标，使用 XGBoost 进行预测。

特征来源:
  1. tsfresh 自动特征 (滚动窗口 10/20/30)
  2. 传统技术指标 (RSI, MACD, Bollinger, KDJ, ATR)

模型: XGBoost Classifier

优势:
  - tsfresh 自动提取特征，减少人工特征工程
  - 特征选择去除无关特征，降低过拟合风险
  - 结合领域知识(技术指标)和自动特征提取
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional

NAME = "tsfresh_xgboost"

# 尝试导入 tsfresh 特征提取器
try:
    from strategies.tsfresh_features import (
        extract_tsfresh_features,
        extract_simple_ts_features,
        TSFRESH_AVAILABLE,
    )
except ImportError:
    TSFRESH_AVAILABLE = False
    extract_tsfresh_features = None
    extract_simple_ts_features = None

# 导入技术指标计算
from strategies.xgboost_enhanced import (
    add_features as add_technical_features,
    _calculate_rsi,
    _calculate_macd,
    _calculate_bollinger_bands,
    _calculate_kdj,
    _calculate_atr,
)


def prepare_data(
    df: pd.DataFrame,
    config: dict,
) -> Tuple[pd.DataFrame, pd.Series, list]:
    """
    准备训练数据

    Args:
        df: 原始 OHLCV 数据
        config: 配置字典

    Returns:
        X, y, feature_columns
    """
    label_period = int(config.get('label_period', 1))

    # ===== 1. 添加传统技术指标特征 =====
    df_feat = add_technical_features(df)

    # ===== 2. 提取 tsfresh 特征 =====
    tsfresh_features = pd.DataFrame()

    if TSFRESH_AVAILABLE and extract_tsfresh_features is not None:
        # 创建标签 (与 df_feat 对齐)
        label = np.where(
            df_feat['Close'].shift(-label_period) > df_feat['Close'],
            1, 0
        )
        y_full = pd.Series(label, index=df_feat.index)

        # 提取 tsfresh 特征 (使用较短窗口避免数据不足)
        window_sizes = [10, 20]
        tsfresh_features, tsfresh_cols = extract_tsfresh_features(
            df,
            window_sizes=window_sizes,
            extraction_level='efficient',
            with_selection=True,
            y=y_full,
        )

        if not tsfresh_features.empty:
            # 对齐索引
            tsfresh_features = tsfresh_features.reindex(df_feat.index)
            print(f"  [tsfresh_xgboost] tsfresh 特征数: {len(tsfresh_cols)}")

    # 如果 tsfresh 不可用或失败，使用简化版特征
    if tsfresh_features.empty and extract_simple_ts_features is not None:
        print(f"  [tsfresh_xgboost] 使用简化版时间序列特征")
        tsfresh_features = extract_simple_ts_features(df, windows=[5, 10, 20])
        if not tsfresh_features.empty:
            tsfresh_features = tsfresh_features.reindex(df_feat.index)

    # ===== 3. 合并特征 =====
    # 传统技术指标特征列
    exclude_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume', 'label',
        'Dividends', 'dividends', 'Stock Splits', 'Adj Close', 'adjclose',
        'returns',
    ]
    tech_cols = [
        c for c in df_feat.columns
        if c not in exclude_cols
        and not c.lower().startswith('adj')
        and df_feat[c].notna().sum() > 0
    ]

    # 合并特征
    if not tsfresh_features.empty:
        combined = pd.concat([df_feat[tech_cols], tsfresh_features], axis=1)
    else:
        combined = df_feat[tech_cols]

    # 去除全 NaN 列
    combined = combined.dropna(axis=1, how='all')

    # ===== 4. 创建标签 =====
    df_feat['label'] = np.where(
        df_feat['Close'].shift(-label_period) > df_feat['Close'],
        1, 0
    )

    # ===== 5. 清理数据 =====
    feat_cols = [c for c in combined.columns if combined[c].notna().sum() > 0]

    if not feat_cols:
        return pd.DataFrame(), pd.Series(dtype=int), []

    df_clean = combined[feat_cols].join(df_feat['label']).dropna()

    if len(df_clean) < 20:
        return pd.DataFrame(), pd.Series(dtype=int), []

    X = df_clean[feat_cols]
    y = df_clean['label']

    return X, y, feat_cols


def run(data: pd.DataFrame, config: dict):
    """
    运行 tsfresh + XGBoost 策略

    训练: 使用前 80% 数据
    预测: 在后 20% 数据上生成信号
    """
    # 参数
    test_days = int(config.get('test_days', 5))
    n_estimators = int(config.get('xgb_n_estimators', 100))
    max_depth = int(config.get('xgb_max_depth', 5))
    learning_rate = float(config.get('xgb_learning_rate', 0.1))
    label_period = int(config.get('label_period', 1))

    # 准备数据
    X, y, feat_cols = prepare_data(data, config)

    if X.empty:
        print(f"  [tsfresh_xgboost] 特征提取失败，返回零信号")
        signal = pd.Series(0, index=data.index)
        meta = {
            'name': NAME,
            'params': {
                'test_days': test_days,
                'label_period': label_period,
            },
            'feat_cols': [],
            'model': None,
            'error': '特征提取失败',
        }
        return signal, None, meta

    print(f"  [tsfresh_xgboost] 特征数: {len(feat_cols)}, 样本数: {len(X)}")

    # 分割训练/测试
    no_split = config.get('no_internal_split', False)
    if no_split:
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
    model = None
    model_name = None
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
        try:
            import lightgbm as lgb
            model = lgb.LGBMClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
                verbose=-1,
            )
            model_name = 'LightGBM'
        except ImportError:
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
    test_proba = model.predict_proba(X_test)[:, 1] if len(X_test) > 0 else np.array([])

    # 生成信号
    signal = pd.Series(0, index=X.index)

    if no_split:
        signal.iloc[:] = test_pred
    else:
        if len(X_test) > 0:
            signal.iloc[split_idx:] = test_pred

    # 特征重要性
    if hasattr(model, 'feature_importances_'):
        importance = dict(zip(feat_cols, model.feature_importances_.round(4)))
        # 显示前 10 重要特征
        top_features = sorted(importance.items(), key=lambda x: -x[1])[:10]
        print(f"  [tsfresh_xgboost] Top 10 特征:")
        for fname, fimp in top_features:
            print(f"      {fname}: {fimp:.4f}")
    else:
        importance = {}

    # 元数据
    meta = {
        'name': NAME,
        'params': {
            'test_days': test_days,
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'label_period': label_period,
        },
        'feat_cols': feat_cols,
        'feat_count': len(feat_cols),
        'model': model_name,
        'feature_importances': importance,
        'indicators': {
            'pred_proba': pd.Series(test_proba, index=X_test.index) if len(X_test) > 0 else pd.Series(),
        },
        'train_acc': float((train_pred == y_train).mean()),
        'test_acc': float((test_pred == y_test).mean()) if len(y_test) > 0 else None,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'tsfresh_available': TSFRESH_AVAILABLE,
    }

    return signal, model, meta
