"""
策略: 技术指标增强版 LightGBM
------------------------------------------------------
特征工程:
  - 收益特征 (ret_1 ~ ret_20)
  - 技术指标 (RSI, MACD, Bollinger Bands, KDJ, ATR, OBV)
  - 波动率特征
  - 成交量特征
  - 趋势特征
  - (可选) ta-lib 技术指标
  - (可选) tsfresh 自动特征

模型: LightGBM Classifier

配置选项:
  use_ta_lib: 是否使用 ta-lib 计算技术指标 (默认 False)
  use_tsfresh_features: 是否添加 tsfresh 自动特征 (默认 False)
"""

import numpy as np
import pandas as pd
from typing import Tuple
from log_config import get_logger

logger = get_logger(__name__)

NAME = "lightgbm_enhanced"

# ── 策略元数据 ───────────────────────────────────────────────────
MIN_BARS = 100  # 运行此策略所需的最少数据行数

# 超参数空间（供 optimize_with_optuna.py 读取，避免在两处维护）
PARAM_SPACE = {
    'test_days':            (3, 15),
    'lgbm_n_estimators':   (50, 200),
    'lgbm_max_depth':      (3, 8),
    'lgbm_learning_rate':  (0.01, 0.3),
    'lgbm_num_leaves':     (15, 63),
    'label_period':        (1, 5),
    'lgb_feature_fraction': (0.6, 1.0),
    'lgb_bagging_fraction': (0.6, 1.0),
    'lgb_reg_alpha':       (0.0, 1.0),
    'lgb_reg_lambda':      (0.0, 1.0),
    'lgb_min_child_samples': (10, 50),
}

# 导入特征工程函数（指标计算已统一到 strategies.indicators）
from strategies.xgboost_enhanced import (
    add_features,
    prepare_data,
    TSFRESH_AVAILABLE,
)


def run(data: pd.DataFrame, config: dict):
    """
    运行 LightGBM 增强版策略

    训练: 使用前 80% 的数据训练模型
    预测: 在后 20% 的数据上生成交易信号

    可选配置:
      use_ta_lib: 是否使用 ta-lib 计算技术指标 (默认 False)
      use_tsfresh_features: 是否添加 tsfresh 自动特征 (默认 False)
    """
    # 参数
    test_days = int(config.get('test_days', 5))
    n_estimators = int(config.get('lgbm_n_estimators', 100))
    max_depth = int(config.get('lgbm_max_depth', 5))
    learning_rate = float(config.get('lgbm_learning_rate', 0.1))
    num_leaves = int(config.get('lgbm_num_leaves', 31))
    label_period = int(config.get('label_period', 1))
    use_ta_lib = config.get('use_ta_lib', False)
    use_tsfresh = config.get('use_tsfresh_features', False)
    # ── 修复项4：LightGBM 正则化参数（使用 LightGBM 原生参数名） ──
    feature_fraction    = float(config.get('lgb_feature_fraction', 0.8))
    bagging_fraction    = float(config.get('lgb_bagging_fraction', 0.8))
    reg_alpha           = float(config.get('lgb_reg_alpha', 0.0))
    reg_lambda          = float(config.get('lgb_reg_lambda', 0.0))
    min_child_samples   = int(config.get('lgb_min_child_samples', 20))

    # ── 多股票路径（有 ticker 列）：per-ticker 特征，避免跨 ticker 指标计算 ──
    if 'ticker' in data.columns and data['ticker'].nunique() > 1:
        from train_multi_stock import create_multi_stock_dataset
        _ts_windows = config.get('tsfresh_window_sizes', [10, 20]) if use_tsfresh else None
        X, y, feat_cols = create_multi_stock_dataset(
            data,
            test_days,
            label_period,
            use_tsfresh=bool(use_tsfresh),
            tsfresh_window_sizes=_ts_windows,
        )
        if X.empty or len(X) < 10:
            raise ValueError("多股票特征数据不足: 需要 > 10 个样本")
        import re as _re
        _ts_re = _re.compile(r"_w\d+$")
        selected_tsfresh_cols = [c for c in feat_cols if _ts_re.search(str(c))] if use_tsfresh else []
        tsfresh_feat_count = len(selected_tsfresh_cols)
        no_split = False
        split_idx = len(X)
        X_train, y_train = X, y
        X_test = X.iloc[:0]
        y_test = y.iloc[:0]
    else:
        # ── 单股票路径（原有逻辑）────────────────────────────────────────────────
        # 添加技术指标特征
        df = add_features(data, use_ta_lib=use_ta_lib)

        # ===== 可选: 添加 tsfresh 特征 =====
        # ⚠️ 关键：tsfresh 特征选择只能使用训练集数据，不得泄露测试期标签
        no_split = config.get('no_internal_split', False)
        _prelim_split_idx = int(len(df) * 0.8) if not no_split else len(df)

        tsfresh_feat_count = 0
        selected_tsfresh_cols = []
        if use_tsfresh:
            try:
                from strategies.tsfresh_features import (
                    extract_tsfresh_features,
                    extract_simple_ts_features,
                )
            except ImportError:
                extract_tsfresh_features = None
                extract_simple_ts_features = None

            if TSFRESH_AVAILABLE and extract_tsfresh_features is not None:
                window_sizes = config.get('tsfresh_window_sizes', [10, 20])

                # ---- 训练集部分：提取特征 + 特征选择（只用训练期标签） ----
                train_data_for_ts = data.iloc[:_prelim_split_idx]
                train_label = np.where(
                    train_data_for_ts['Close'].shift(-label_period) > train_data_for_ts['Close'],
                    1, 0
                )
                y_train_for_selection = pd.Series(train_label, index=train_data_for_ts.index)

                train_tsfresh, train_tsfresh_cols = extract_tsfresh_features(
                    train_data_for_ts,
                    window_sizes=window_sizes,
                    extraction_level='efficient',
                    with_selection=True,
                    y=y_train_for_selection,
                )
                selected_tsfresh_cols = list(train_tsfresh_cols)

                # ---- 测试集部分：只提取特征，对齐到训练集的列 ----
                if not no_split and len(data) > _prelim_split_idx:
                    test_data_for_ts = data.iloc[_prelim_split_idx:]
                    test_tsfresh, _ = extract_tsfresh_features(
                        test_data_for_ts,
                        window_sizes=window_sizes,
                        extraction_level='efficient',
                        with_selection=False,
                        y=None,
                    )
                    if not test_tsfresh.empty and selected_tsfresh_cols:
                        test_tsfresh = test_tsfresh.reindex(columns=selected_tsfresh_cols, fill_value=0)

                    if not train_tsfresh.empty and not test_tsfresh.empty:
                        tsfresh_features = pd.concat([train_tsfresh, test_tsfresh])
                    elif not train_tsfresh.empty:
                        tsfresh_features = train_tsfresh.reindex(columns=selected_tsfresh_cols, fill_value=0)
                    else:
                        tsfresh_features = pd.DataFrame()
                else:
                    tsfresh_features = train_tsfresh

                if not tsfresh_features.empty:
                    tsfresh_features = tsfresh_features.reindex(df.index)
                    tsfresh_feat_count = len(selected_tsfresh_cols)
                    logger.info("lightgbm_enhanced tsfresh特征提取成功", extra={
                        "tsfresh_feature_count": tsfresh_feat_count,
                        "note": "已修复前视偏差"
                    })
                    df = pd.concat([df, tsfresh_features], axis=1)
            elif extract_simple_ts_features is not None:
                logger.info("lightgbm_enhanced使用简化版时间序列特征")
                simple_features = extract_simple_ts_features(data, windows=[5, 10, 20])
                if not simple_features.empty:
                    simple_features = simple_features.reindex(df.index)
                    df = pd.concat([df, simple_features], axis=1)
                    tsfresh_feat_count = len(simple_features.columns)
                    logger.info("lightgbm_enhanced简化版特征数", extra={"simplified_feature_count": tsfresh_feat_count})

        # 准备数据
        X, y, feat_cols = prepare_data(df, test_days, label_period)

        if tsfresh_feat_count > 0:
            logger.info("lightgbm_enhanced总特征数", extra={"total_features": len(feat_cols)})

        # 分割训练/测试（80% 训练，20% 测试）
        split_idx = len(X)  # 默认值，no_split 时使用全部数据
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
        from strategies import ml_thread_budget
        # 正则化参数（修复项4）传入模型，使用 LightGBM 原生参数名
        _lgbm_kwargs = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            feature_fraction=feature_fraction,
            bagging_fraction=bagging_fraction,
            bagging_freq=1,        # bagging_fraction 生效需要 bagging_freq > 0
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_samples=min_child_samples,
            random_state=42,
            verbose=-1,
            n_jobs=ml_thread_budget(),
        )
        # LightGBM pip 版本未编译 CUDA，支持 OpenCL GPU（需额外驱动）
        # CPU 已足够快，直接使用
        model = lgb.LGBMClassifier(**_lgbm_kwargs)
        model_name = 'LightGBM'
    except ImportError:
        # 降级使用 XGBoost
        try:
            from xgboost import XGBClassifier
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
                    logger.warning(f"lightgbm XGBoost GPU不可用，回退CPU: {e}")
                    model = XGBClassifier(**_xgb_kwargs)
                else:
                    raise
            model_name = 'XGBoost'
        except ImportError:
            from sklearn.ensemble import GradientBoostingClassifier
            model = GradientBoostingClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                random_state=42,
            )
            model_name = 'GradientBoosting'

    # ── 修复项3a：Early Stopping，复用已有 X_test/y_test，零额外数据消耗 ──
    _fit_kwargs = {}
    if not no_split and len(X_test) > 0 and model_name == 'LightGBM':
        _fit_kwargs = {
            'eval_set': [(X_test, y_test)],
            'callbacks': [lgb.early_stopping(20, verbose=False)],
        }
        # ── Optuna 中途剪枝:LGBM 每轮 boosting 后报告 valid binary_logloss,
        # 同期表现差的 trial 直接 TrialPruned。──
        _trial = config.get('_optuna_trial')
        if _trial is not None:
            try:
                from optuna.integration import LightGBMPruningCallback
                _fit_kwargs['callbacks'].append(
                    LightGBMPruningCallback(_trial, 'binary_logloss', valid_name='valid_0')
                )
            except Exception as _e:
                logger.debug(f"LightGBMPruningCallback 不可用,跳过中途剪枝: {_e}")

    # ⚠️ GPU 修复：GPU 错误在 fit() 时才实际触发，在此捕获并静默回退 CPU
    try:
        model.fit(X_train, y_train, **_fit_kwargs)
    except Exception as gpu_err:
        _msg = str(gpu_err).lower()
        if 'gpu' in _msg or 'cuda' in _msg or 'opencl' in _msg or 'device' in _msg:
            logger.warning("lightgbm GPU不可用，回退CPU", extra={"gpu_error": str(gpu_err)})
            try:
                import lightgbm as lgb
                model = lgb.LGBMClassifier(**_lgbm_kwargs)
                model.fit(X_train, y_train, **_fit_kwargs)
            except Exception:
                raise
        else:
            raise

    # 预测
    train_pred = model.predict(X_train)
    if len(X_test) > 0:
        test_pred = model.predict(X_test)
        test_proba = model.predict_proba(X_test)[:, 1]
    else:
        test_pred = np.array([], dtype=int)
        test_proba = np.array([], dtype=float)

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

    # ── 修复项3b：TimeSeriesSplit CV（仅在 use_cv=True 且非 no_split 模式下执行） ──
    cv_val_acc_mean = float('nan')
    cv_overfit_gap  = float('nan')
    if config.get('use_cv', False) and not no_split and len(X) >= 50:
        try:
            from sklearn.model_selection import TimeSeriesSplit
            from sklearn.metrics import accuracy_score as _acc
            tscv = TimeSeriesSplit(n_splits=5)
            cv_train_accs, cv_val_accs = [], []
            for _tr_idx, _val_idx in tscv.split(X):
                _Xtr, _Xvl = X.iloc[_tr_idx], X.iloc[_val_idx]
                _ytr, _yvl = y.iloc[_tr_idx], y.iloc[_val_idx]
                _cv_model = model.__class__(**model.get_params())
                _cv_model.fit(_Xtr, _ytr)
                cv_train_accs.append(_acc(_ytr, _cv_model.predict(_Xtr)))
                cv_val_accs.append(_acc(_yvl, _cv_model.predict(_Xvl)))
            cv_val_acc_mean = float(np.mean(cv_val_accs))
            cv_overfit_gap  = float(np.mean(np.array(cv_train_accs) - np.array(cv_val_accs)))
        except Exception:
            pass

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
            'use_ta_lib': use_ta_lib,
            'use_tsfresh_features': use_tsfresh,
        },
        'feat_cols': feat_cols,
        'feat_count': len(feat_cols),
        'tsfresh_feat_count': tsfresh_feat_count,
        'selected_tsfresh_cols': selected_tsfresh_cols,  # 训练集选出的 tsfresh 特征列
        'model': model_name,
        'feature_importances': dict(zip(feat_cols, model.feature_importances_.round(4))) if hasattr(model, 'feature_importances_') else {},
        'indicators': {
            'pred_proba': pd.Series(test_proba, index=X_test.index),
        },
        'train_acc': float((train_pred == y_train).mean()),
        'test_acc': float((test_pred == y_test).mean()) if len(y_test) > 0 else None,
        # 修复项3b：CV 评估指标
        'cv_val_acc_mean': cv_val_acc_mean,
        'cv_overfit_gap':  cv_overfit_gap,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'tsfresh_available': TSFRESH_AVAILABLE,
    }

    return signal, model, meta


def predict(model, data: pd.DataFrame, config: dict, meta: dict) -> pd.Series:
    """
    独立推断函数：只做特征工程 + model.predict()，不做 model.fit()。

    供 validate_strategy.py 的 Walk-Forward 和 out_of_sample_test() 调用，
    确保模型在训练集上 fit 后，可以在测试集上独立推断，不泄露未来信息。

    Args:
        model: 已训练的模型对象（来自 run() 返回的第二个值）
        data: 待推断的 OHLCV 数据（测试集）
        config: 配置字典
        meta: run() 返回的 meta 字典（包含 feat_cols、selected_tsfresh_cols 等）

    Returns:
        信号序列 (0/1)，索引与 data 对齐
    """
    use_ta_lib = config.get('use_ta_lib', False)
    use_tsfresh = config.get('use_tsfresh_features', False)
    label_period = int(config.get('label_period', 1))

    feat_cols = meta.get('feat_cols', [])
    selected_tsfresh_cols = meta.get('selected_tsfresh_cols', [])

    # 添加技术指标特征（与训练时相同流程）
    df = add_features(data, use_ta_lib=use_ta_lib)

    # 添加 tsfresh 特征（仅变换，不做特征选择，对齐到训练集选出的列）
    if use_tsfresh and selected_tsfresh_cols and TSFRESH_AVAILABLE:
        try:
            from strategies.tsfresh_features import extract_tsfresh_features
            window_sizes = config.get('tsfresh_window_sizes', [10, 20])
            ts_features, _ = extract_tsfresh_features(
                data,
                window_sizes=window_sizes,
                extraction_level='efficient',
                with_selection=False,  # 推断时不做特征选择
                y=None,
            )
            if not ts_features.empty:
                ts_features = ts_features.reindex(columns=selected_tsfresh_cols, fill_value=0)
                ts_features = ts_features.reindex(df.index)
                df = pd.concat([df, ts_features], axis=1)
        except Exception:
            pass

    # 补充训练时有但推断时缺失的列（填 0）
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0.0

    X = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    predictions = model.predict(X)
    signal = pd.Series(predictions, index=X.index, dtype=int)
    return signal


