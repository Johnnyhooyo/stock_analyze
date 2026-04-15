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
from typing import Tuple
from log_config import get_logger

logger = get_logger(__name__)

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
from strategies.xgboost_enhanced import add_features as add_technical_features


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

    # ===== 0. 去除重复列名，防止列访问返回 DataFrame =====
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep='first')]

    # ===== 1. 添加传统技术指标特征 =====
    df_feat = add_technical_features(df)

    # ===== 2. 提取 tsfresh 特征 =====
    tsfresh_features = pd.DataFrame()

    if TSFRESH_AVAILABLE and extract_tsfresh_features is not None:
        # ⚠️ 前视偏差修复：特征选择标签仅用 df_feat（即传入的数据切片）的标签，
        # 调用方必须只传入训练集数据，不得包含验证/测试集数据。
        # squeeze() 确保 Close 列是 Series（防止重复列名导致 DataFrame 返回）
        close_series = df_feat['Close'].squeeze() if isinstance(df_feat['Close'], pd.DataFrame) else df_feat['Close']
        label = np.where(
            close_series.shift(-label_period) > close_series,
            1, 0
        )
        y_train_for_selection = pd.Series(label, index=df_feat.index)

        # 提取 tsfresh 特征 (使用较短窗口避免数据不足)
        window_sizes = [10, 20]
        tsfresh_features, tsfresh_cols = extract_tsfresh_features(
            df,
            window_sizes=window_sizes,
            extraction_level='efficient',
            with_selection=True,
            y=y_train_for_selection,
        )

        if not tsfresh_features.empty:
            # 对齐索引
            tsfresh_features = tsfresh_features.reindex(df_feat.index)
            logger.info("tsfresh特征提取成功", extra={"tsfresh_feature_count": len(tsfresh_cols)})

    # 如果 tsfresh 不可用或失败，使用简化版特征
    if tsfresh_features.empty and extract_simple_ts_features is not None:
        logger.info("使用简化版时间序列特征")
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

    def _col_has_data(df_check, col):
        s = df_check[col].notna().sum()
        return bool(s.any()) if isinstance(s, pd.Series) else bool(s > 0)

    tech_cols = [
        c for c in df_feat.columns
        if c not in exclude_cols
        and not c.lower().startswith('adj')
        and _col_has_data(df_feat, c)
    ]

    # 合并特征
    if not tsfresh_features.empty:
        combined = pd.concat([df_feat[tech_cols], tsfresh_features], axis=1)
    else:
        combined = df_feat[tech_cols]

    # 去除全 NaN 列
    combined = combined.dropna(axis=1, how='all')
    # 防止 tsfresh 生成名为 'label' 的特征列，污染 y 标签
    if 'label' in combined.columns:
        combined = combined.drop(columns=['label'])

    # ===== 4. 创建标签 =====
    # 先删除 df_feat 中可能存在的 'label' 列，防止赋值产生重复列
    if 'label' in df_feat.columns:
        df_feat = df_feat.drop(columns=['label'])
    close_s = df_feat['Close']
    if isinstance(close_s, pd.DataFrame):
        close_s = close_s.iloc[:, 0]
    df_feat['label'] = np.where(
        close_s.shift(-label_period) > close_s,
        1, 0
    )

    # ===== 5. 清理数据 =====
    # 显式排除 'label'，防止 tsfresh 产生同名特征导致 join 后出现重复列
    feat_cols = [c for c in combined.columns if _col_has_data(combined, c) and c != 'label']

    if not feat_cols:
        return pd.DataFrame(), pd.Series(dtype=int), []

    df_clean = combined[feat_cols].join(df_feat['label']).replace([np.inf, -np.inf], np.nan).dropna()

    if len(df_clean) < 20:
        return pd.DataFrame(), pd.Series(dtype=int), []

    X = df_clean[feat_cols]
    y = df_clean['label']
    # 确保 y 是一维 Series（防止重复列导致 DataFrame 流入 model.fit）
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

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

    no_split = config.get('no_internal_split', False)

    # ⚠️ 前视偏差修复：先确定分割点，tsfresh 特征选择只能用训练集数据
    # 粗算分割行数（基于原始数据长度），prepare_data 后精确切分
    prelim_split = int(len(data) * 0.8) if not no_split else len(data)
    train_data_raw = data.iloc[:prelim_split]
    test_data_raw  = data.iloc[prelim_split:]

    # 分别为训练集和测试集提取特征（tsfresh 特征选择仅用训练集标签）
    X_train, y_train, feat_cols = prepare_data(train_data_raw, config)
    if X_train.empty:
        logger.warning("tsfresh_xgboost特征提取失败，返回零信号")
        signal = pd.Series(0, index=data.index)
        meta = {
            'name': NAME, 'params': {'test_days': test_days, 'label_period': label_period},
            'feat_cols': [], 'model': None, 'error': '特征提取失败',
        }
        return signal, None, meta

    # 测试集特征（对齐到训练集选出的列）
    X_test = pd.DataFrame()
    if not no_split and len(test_data_raw) >= 10:
        try:
            df_test_feat = add_technical_features(test_data_raw)
            exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'label',
                            'Dividends', 'dividends', 'Stock Splits', 'Adj Close', 'adjclose', 'returns']
            tech_cols_test = [c for c in df_test_feat.columns if c not in exclude_cols
                              and not c.lower().startswith('adj')]
            # tsfresh 特征（无特征选择，对齐到训练集列）
            tsfresh_test = pd.DataFrame()
            if TSFRESH_AVAILABLE and extract_tsfresh_features is not None:
                ts_test, _ = extract_tsfresh_features(
                    test_data_raw, window_sizes=[10, 20],
                    extraction_level='efficient', with_selection=False, y=None)
                if not ts_test.empty:
                    tsfresh_test = ts_test.reindex(df_test_feat.index)

            if not tsfresh_test.empty:
                combined_test = pd.concat([df_test_feat[tech_cols_test], tsfresh_test], axis=1)
            else:
                combined_test = df_test_feat[tech_cols_test]

            # 补齐训练集有但测试集缺失的列
            for c in feat_cols:
                if c not in combined_test.columns:
                    combined_test[c] = 0.0
            X_test = combined_test[feat_cols].dropna()
        except Exception as e:
            logger.warning("tsfresh_xgboost测试集特征提取失败", extra={"error": str(e)})

    logger.info("tsfresh_xgboost数据集准备完成", extra={
        "feature_count": len(feat_cols),
        "train_samples": len(X_train),
        "test_samples": len(X_test)
    })

    # 训练模型
    model = None
    model_name = None
    try:
        from xgboost import XGBClassifier
        from strategies import ml_thread_budget
        _xgb_kwargs = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0,
            tree_method='hist',
            n_jobs=ml_thread_budget(),
        )
        try:
            model = XGBClassifier(**_xgb_kwargs, device='cuda')
        except Exception as e:
            _msg = str(e).lower()
            if any(x in _msg for x in ('cuda', 'gpu', 'device', 'memory')):
                import logging
                logging.getLogger(__name__).warning(f"tsfresh_xgboost GPU不可用，回退CPU: {e}")
                model = XGBClassifier(**_xgb_kwargs)
            else:
                raise
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
                n_jobs=ml_thread_budget(),
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

    # ── Optuna 中途剪枝(仅 XGBoost/LightGBM 路径,有验证集时才能挂)──
    _fit_kwargs = {}
    _trial = config.get('_optuna_trial')
    if _trial is not None and not no_split and len(X_test) > 0 and model_name in ('XGBoost', 'LightGBM'):
        try:
            if model_name == 'XGBoost':
                from optuna.integration import XGBoostPruningCallback
                _fit_kwargs = {
                    'eval_set': [(X_test, y_test)],
                    'verbose': False,
                    'callbacks': [XGBoostPruningCallback(_trial, 'validation_0-logloss')],
                }
            else:
                import lightgbm as lgb
                from optuna.integration import LightGBMPruningCallback
                _fit_kwargs = {
                    'eval_set': [(X_test, y_test)],
                    'callbacks': [
                        lgb.early_stopping(20, verbose=False),
                        LightGBMPruningCallback(_trial, 'binary_logloss', valid_name='valid_0'),
                    ],
                }
        except Exception as _e:
            logger.debug(f"tsfresh_xgboost PruningCallback 不可用: {_e}")
            _fit_kwargs = {}

    model.fit(X_train, y_train, **_fit_kwargs)

    # 生成信号（只填入测试集的预测结果）
    signal = pd.Series(0, index=data.index)
    test_pred = np.array([])
    test_proba = np.array([])
    y_test = pd.Series(dtype=int)

    if no_split:
        # no_internal_split：对全部数据推断（val 阶段调用）
        full_X, full_y, _ = prepare_data(data, config)
        if not full_X.empty:
            full_pred = model.predict(full_X)
            signal = pd.Series(full_pred.astype(int), index=full_X.index)
            test_pred = full_pred
            test_proba = model.predict_proba(full_X)[:, 1]
            y_test = full_y
        else:
            test_pred = np.array([])
            test_proba = np.array([])
            y_test = pd.Series(dtype=int)
    elif not X_test.empty:
        test_pred = model.predict(X_test)
        test_proba = model.predict_proba(X_test)[:, 1]
        signal.loc[X_test.index] = test_pred.astype(int)
        y_test = y_train.iloc[0:0]  # placeholder

    train_pred = model.predict(X_train)

    # 特征重要性
    if hasattr(model, 'feature_importances_'):
        importance = dict(zip(feat_cols, model.feature_importances_.round(4)))
        top_features = sorted(importance.items(), key=lambda x: -x[1])[:10]
        logger.info("tsfresh_xgboost Top10特征重要性", extra={
            "top_features": {fname: fimp for fname, fimp in top_features}
        })
    else:
        importance = {}

    meta = {
        'name': NAME,
        'params': {
            'test_days': test_days, 'n_estimators': n_estimators,
            'max_depth': max_depth, 'learning_rate': learning_rate,
            'label_period': label_period,
        },
        'feat_cols': feat_cols,
        'feat_count': len(feat_cols),
        'model': model_name,
        'feature_importances': importance,
        'indicators': {
            'pred_proba': pd.Series(test_proba, index=X_test.index) if len(test_proba) and not X_test.empty else pd.Series(),
        },
        'train_acc': float((train_pred == y_train).mean()),
        'test_acc': float((test_pred == y_test).mean()) if len(test_pred) and len(y_test) else None,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'tsfresh_available': TSFRESH_AVAILABLE,
    }

    return signal, model, meta


def predict(model, data: pd.DataFrame, config: dict, meta: dict) -> pd.Series:
    """独立推断：用已训练模型在新数据上生成信号，不重新 fit。"""
    feat_cols = meta.get('feat_cols', [])
    if not feat_cols:
        return pd.Series(0, index=data.index)

    df_feat = add_technical_features(data)
    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'label',
                    'Dividends', 'dividends', 'Stock Splits', 'Adj Close', 'adjclose', 'returns']
    tech_cols = [c for c in df_feat.columns if c not in exclude_cols and not c.lower().startswith('adj')]

    tsfresh_df = pd.DataFrame()
    if TSFRESH_AVAILABLE and extract_tsfresh_features is not None:
        try:
            ts_feats, _ = extract_tsfresh_features(
                data, window_sizes=[10, 20],
                extraction_level='efficient', with_selection=False, y=None)
            if not ts_feats.empty:
                tsfresh_df = ts_feats.reindex(df_feat.index)
        except Exception:
            pass

    if not tsfresh_df.empty:
        combined = pd.concat([df_feat[tech_cols], tsfresh_df], axis=1)
    else:
        combined = df_feat[tech_cols]

    for c in feat_cols:
        if c not in combined.columns:
            combined[c] = 0.0

    X = combined[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    predictions = model.predict(X)
    return pd.Series(predictions.astype(int), index=X.index)

