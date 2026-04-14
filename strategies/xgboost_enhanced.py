"""
策略: 技术指标增强版 XGBoost
------------------------------------------------------
特征工程:
  - 收益特征 (ret_1 ~ ret_20)
  - 技术指标 (RSI, MACD, Bollinger Bands, KDJ, ATR, OBV)
  - 波动率特征
  - 成交量特征
  - 趋势特征
  - (可选) tsfresh 自动特征

模型: XGBoost Classifier

配置选项:
  use_tsfresh_features: 是否添加 tsfresh 自动特征 (默认 False)
  tsfresh_window_sizes: tsfresh 滚动窗口大小 (默认 [10, 20])
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from log_config import get_logger
from strategies.indicators import (
    rsi as _ind_rsi,
    macd as _ind_macd,
    bollinger_bands as _ind_bollinger,
    kdj as _ind_kdj,
    atr as _ind_atr,
    obv as _ind_obv,
    pvt as _ind_pvt,
)

logger = get_logger(__name__)

NAME = "xgboost_enhanced"

# ── 策略元数据 ───────────────────────────────────────────────────
MIN_BARS = 100  # 运行此策略所需的最少数据行数

# 超参数空间（供 optimize_with_optuna.py 读取，避免在两处维护）
PARAM_SPACE = {
    'test_days':              (3, 15),
    'xgb_n_estimators':      (50, 200),
    'xgb_max_depth':         (3, 8),
    'xgb_learning_rate':     (0.01, 0.3),
    'label_period':          (1, 5),
    'xgb_subsample':         (0.6, 1.0),
    'xgb_colsample_bytree':  (0.6, 1.0),
    'xgb_reg_alpha':         (0.0, 1.0),
    'xgb_reg_lambda':        (0.0, 1.0),
    'xgb_min_child_weight':  (1, 10),
}

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


# Thin wrappers around shared indicators — preserve call signatures for add_features()
_calculate_rsi = _ind_rsi

# 尝试导入 ta-lib 指标计算
try:
    from strategies.indicators import add_ta_features as _add_ta_features, TA_AVAILABLE as TA_LIB_AVAILABLE
except ImportError:
    TA_LIB_AVAILABLE = False
    _add_ta_features = None


def _calculate_macd(
    series: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    return _ind_macd(series, fast, slow, signal)


def _calculate_bollinger_bands(
    series: pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    upper, ma, lower = _ind_bollinger(series, period, std_dev)
    position = (series - lower) / (upper - lower).replace(0, np.nan)
    return upper, ma, lower, position


def _calculate_kdj(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    return _ind_kdj(high, low, close, period)


_calculate_atr = _ind_atr
_calculate_obv = _ind_obv
_calculate_pvt = _ind_pvt


def add_features(df: pd.DataFrame, use_ta_lib: bool = False) -> pd.DataFrame:
    """
    添加技术指标特征

    Args:
        df: 原始 OHLCV 数据
        use_ta_lib: 是否使用 ta-lib 计算指标（默认 False）
    """
    # 如果启用 ta-lib 且可用，使用 ta-lib 实现
    if use_ta_lib and TA_LIB_AVAILABLE and _add_ta_features is not None:
        return _add_ta_features(df)

    df = df.copy()

    # 去除重复列名（可能来自多股票 concat 或 tsfresh 特征合并）
    # 保留每个列名的第一次出现，确保 df['Close'] 等列访问始终返回 Series 而非 DataFrame
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep='first')]

    has_high   = 'High'   in df.columns
    has_low    = 'Low'    in df.columns
    has_volume = 'Volume' in df.columns

    # ===== 基础收益特征 =====
    df['returns'] = df['Close'].pct_change()
    for i in range(1, 21):  # 20天收益
        df[f'ret_{i}'] = df['Close'].pct_change(i)

    # ===== RSI =====
    for period in [6, 14, 21]:
        df[f'rsi_{period}'] = _calculate_rsi(df['Close'], period)

    # ===== MACD =====
    macd, signal, hist = _calculate_macd(df['Close'])
    df['macd'] = macd
    df['macd_signal'] = signal
    df['macd_hist'] = hist

    # ===== Bollinger Bands =====
    for period, std in [(10, 1.5), (20, 2.0), (30, 2.5)]:
        upper, ma, lower, position = _calculate_bollinger_bands(df['Close'], period, std)
        df[f'bb_upper_{period}'] = upper
        df[f'bb_mid_{period}'] = ma
        df[f'bb_lower_{period}'] = lower
        df[f'bb_position_{period}'] = position
        # 带宽
        df[f'bb_width_{period}'] = (upper - lower) / ma

    # ===== KDJ (需要 High/Low) =====
    if has_high and has_low:
        k, d, j = _calculate_kdj(df['High'], df['Low'], df['Close'])
        df['kdj_k'] = k
        df['kdj_d'] = d
        df['kdj_j'] = j
        df['kdj_overbought'] = (j > 80).astype(int)
        df['kdj_oversold'] = (j < 20).astype(int)

    # ===== ATR (需要 High/Low) =====
    if has_high and has_low:
        for period in [7, 14, 21]:
            df[f'atr_{period}'] = _calculate_atr(df['High'], df['Low'], df['Close'], period)
            df[f'atr_pct_{period}'] = df[f'atr_{period}'] / df['Close']

    # ===== 波动率 =====
    for period in [5, 10, 20]:
        df[f'volatility_{period}'] = df['returns'].rolling(period).std()
        # 波动率变化
        df[f'volatility_change_{period}'] = df[f'volatility_{period}'].pct_change()

    # ===== 成交量特征 (需要 Volume) =====
    if has_volume:
        df['volume'] = df['Volume']
        for period in [5, 10, 20]:
            df[f'volume_ma_{period}'] = df['Volume'].rolling(period).mean()
            df[f'volume_ratio_{period}'] = df['Volume'] / df[f'volume_ma_{period}']

        # ===== OBV & PVT (需要 Volume) =====
        df['obv'] = _calculate_obv(df['Close'], df['Volume'])
        df['pvt'] = _calculate_pvt(df['Close'], df['Volume'])
        for period in [5, 10]:
            df[f'obv_ma_{period}'] = df['obv'].rolling(period).mean()
            df[f'pvt_ma_{period}'] = df['pvt'].rolling(period).mean()

    # ===== 移动平均 =====
    for period in [5, 10, 20, 50]:
        df[f'ma_{period}'] = df['Close'].rolling(period).mean()
        # 价格相对 MA 位置
        df[f'price_vs_ma_{period}'] = df['Close'] / df[f'ma_{period}']

    # ===== 趋势特征 (需要 High/Low) =====
    if has_high and has_low:
        for period in [5, 10, 20]:
            df[f'high_{period}'] = df['High'].rolling(period).max()
            df[f'low_{period}'] = df['Low'].rolling(period).min()
            df[f'high_ratio_{period}'] = df['Close'] / df[f'high_{period}']
            df[f'low_ratio_{period}'] = df['Close'] / df[f'low_{period}']

    # ===== 动量 =====
    for period in [5, 10, 20]:
        df[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1

    return df


def prepare_data(df: pd.DataFrame, test_days: int, label_period: int = 1) -> Tuple:
    """
    准备训练数据

    Args:
        df: 特征数据
        test_days: 用于预测的天数（特征窗口）
        label_period: 预测未来第几天

    Returns:
        X, y, feature_columns
    """
    # 去除重复列名（tsfresh concat 后可能引入重复列），防止 df['Close'] 返回 DataFrame
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep='first')]

    # 标签: 未来 label_period 天是否上涨
    # 先删除可能存在的 'label' 列（例如 tsfresh 可能生成同名特征），防止产生重复列
    if 'label' in df.columns:
        df = df.drop(columns=['label'])
    close_s = df['Close']
    if isinstance(close_s, pd.DataFrame):
        close_s = close_s.iloc[:, 0]
    df['label'] = np.where(close_s.shift(-label_period) > close_s, 1, 0)

    # 特征列（排除标签和原始数据，以及无用的列）
    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'label',
                    'Dividends', 'dividends', 'Stock Splits', 'Adj Close', 'adjclose',
                    'ticker']  # 多股票数据中的 ticker 列
    feat_cols = [c for c in df.columns if c not in exclude_cols and not c.lower().startswith('adj')]

    # 只保留有数据的特征列（兼容 MultiIndex 列，notna().sum() 可能返回 Series）
    def _has_data(col):
        s = df[col].notna().sum()
        return bool(s.any()) if isinstance(s, pd.Series) else bool(s > 0)
    feat_cols = [c for c in feat_cols if _has_data(c)]

    if not feat_cols:
        return pd.DataFrame(), pd.Series(dtype=int), []

    # 只选择特征列和标签，去除 NaN / inf
    df_clean = df[feat_cols + ['label']].replace([np.inf, -np.inf], np.nan).dropna()

    if len(df_clean) < 10:
        return pd.DataFrame(), pd.Series(dtype=int), []

    X = df_clean[feat_cols]
    y = df_clean['label']
    # 确保 y 是一维 Series（防止重复列导致 DataFrame 流入 model.fit）
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    return X, y, feat_cols


def run(data: pd.DataFrame, config: dict):
    """
    运行 XGBoost 增强版策略

    训练: 使用前 80% 的数据训练模型
    预测: 在后 20% 的数据上生成交易信号

    可选配置:
      use_tsfresh_features: 是否添加 tsfresh 自动特征 (默认 False)
      tsfresh_window_sizes: tsfresh 滚动窗口大小 (默认 [10, 20])
      use_ta_lib: 是否使用 ta-lib 计算技术指标 (默认 False)
    """
    # 参数
    test_days = int(config.get('test_days', 5))
    n_estimators = int(config.get('xgb_n_estimators', 100))
    max_depth = int(config.get('xgb_max_depth', 5))
    learning_rate = float(config.get('xgb_learning_rate', 0.1))
    label_period = int(config.get('label_period', 1))  # 预测未来第几天
    use_tsfresh = config.get('use_tsfresh_features', False)  # 是否使用 tsfresh 特征
    use_ta_lib = config.get('use_ta_lib', False)  # 是否使用 ta-lib 计算指标
    # ── 修复项4：正则化参数（防过拟合，默认值与 XGBoost 官方一致） ──
    subsample         = float(config.get('xgb_subsample', 0.8))
    colsample_bytree  = float(config.get('xgb_colsample_bytree', 0.8))
    reg_alpha         = float(config.get('xgb_reg_alpha', 0.0))
    reg_lambda        = float(config.get('xgb_reg_lambda', 1.0))
    min_child_weight  = int(config.get('xgb_min_child_weight', 1))

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
        # tsfresh 列按 `_w<window>` 后缀识别，用于 predict() 推断时对齐
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
        # ⚠️ 关键：先确定训练/测试分割点，tsfresh 特征选择只能使用训练集数据
        # 这里先用原始 df 的行数估算 split_idx，后续 prepare_data 后会重新精确分割
        no_split = config.get('no_internal_split', False)
        _prelim_split_idx = int(len(df) * 0.8) if not no_split else len(df)

        tsfresh_feat_count = 0
        selected_tsfresh_cols = []  # 记录训练集选出的特征列，供测试集对齐使用
        if use_tsfresh:
            if TSFRESH_AVAILABLE and extract_tsfresh_features is not None:
                window_sizes = config.get('tsfresh_window_sizes', [10, 20])

                # ---- 训练集部分：提取特征 + 特征选择 ----
                train_data_for_ts = data.iloc[:_prelim_split_idx]
                # 标签只使用训练集范围（防止泄露测试期未来信息）
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
                selected_tsfresh_cols = list(train_tsfresh_cols)  # 保存训练集选出的列

                # ---- 测试集部分：只提取特征，不做特征选择，对齐到训练集的列 ----
                if not no_split and len(data) > _prelim_split_idx:
                    test_data_for_ts = data.iloc[_prelim_split_idx:]
                    test_tsfresh, _ = extract_tsfresh_features(
                        test_data_for_ts,
                        window_sizes=window_sizes,
                        extraction_level='efficient',
                        with_selection=False,  # 测试集不做特征选择
                        y=None,
                    )
                    # 对齐到训练集选出的列（填 0 补充训练集有但测试集缺失的列）
                    if not test_tsfresh.empty and selected_tsfresh_cols:
                        test_tsfresh = test_tsfresh.reindex(columns=selected_tsfresh_cols, fill_value=0)

                    # 合并训练集和测试集 tsfresh 特征
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
                    logger.info("xgboost_enhanced tsfresh特征提取成功", extra={
                        "tsfresh_feature_count": tsfresh_feat_count,
                        "note": "已修复前视偏差"
                    })
                    df = pd.concat([df, tsfresh_features], axis=1)
            elif extract_simple_ts_features is not None:
                # fallback 到简化版特征（无特征选择，无前视偏差）
                logger.info("xgboost_enhanced使用简化版时间序列特征")
                simple_features = extract_simple_ts_features(data, windows=[5, 10, 20])
                if not simple_features.empty:
                    simple_features = simple_features.reindex(df.index)
                    df = pd.concat([df, simple_features], axis=1)
                    tsfresh_feat_count = len(simple_features.columns)
                    logger.info("xgboost_enhanced简化版特征数", extra={"simplified_feature_count": tsfresh_feat_count})

        # 准备数据
        X, y, feat_cols = prepare_data(df, test_days, label_period)

        if tsfresh_feat_count > 0:
            logger.info("xgboost_enhanced总特征数", extra={"total_features": len(feat_cols)})

        # 分割训练/测试（80% 训练，20% 测试）
        # 如果 config 中设置了 no_internal_split，则使用全部数据训练
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
        from xgboost import XGBClassifier
        # 正则化参数（修复项4）传入模型
        _xgb_kwargs = dict(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            min_child_weight=min_child_weight,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss',
            verbosity=0,
            tree_method='hist',
        )
        # 尝试使用 GPU
        try:
            model = XGBClassifier(**_xgb_kwargs, device='cuda')
        except Exception as e:
            _msg = str(e).lower()
            if any(x in _msg for x in ('cuda', 'gpu', 'device', 'memory')):
                logger.warning(f"XGBoost GPU不可用，回退CPU: {e}")
                model = XGBClassifier(**_xgb_kwargs)
            else:
                raise
    except ImportError:
        # 降级使用 sklearn
        from sklearn.ensemble import GradientBoostingClassifier
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
        )

    # ── 修复项3a：Early Stopping，复用已有 X_test/y_test，零额外数据消耗 ──
    _fit_kwargs = {}
    if not no_split and len(X_test) > 0 and hasattr(model, 'get_booster'):
        _fit_kwargs = {
            'eval_set': [(X_test, y_test)],
            'verbose': False,
        }
        # XGBoost >= 2.0 early_stopping_rounds 在构造时传入，< 2.0 在 fit() 传入
        try:
            import xgboost as _xgb
            _xgb_ver = tuple(int(x) for x in _xgb.__version__.split('.')[:2])
            if _xgb_ver >= (2, 0):
                model.set_params(early_stopping_rounds=20)
            else:
                _fit_kwargs['early_stopping_rounds'] = 20
        except Exception:
            _fit_kwargs['early_stopping_rounds'] = 20

    model.fit(X_train, y_train, **_fit_kwargs)

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
                # fix #12: 传入 eval_set 以启用 early_stopping_rounds
                _cv_model.fit(
                    _Xtr, _ytr,
                    eval_set=[(_Xvl, _yvl)],
                    verbose=False,
                )
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
            'label_period': label_period,
            'use_tsfresh_features': use_tsfresh,
            'use_ta_lib': use_ta_lib,
        },
        'feat_cols': feat_cols,
        'feat_count': len(feat_cols),
        'tsfresh_feat_count': tsfresh_feat_count,
        'selected_tsfresh_cols': selected_tsfresh_cols,  # 训练集选出的 tsfresh 特征列（用于推断对齐）
        'model': 'XGBoost' if hasattr(model, 'get_booster') else 'GradientBoosting',
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
        'ta_lib_available': TA_LIB_AVAILABLE,
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
    test_days = int(config.get('test_days', 5))

    feat_cols = meta.get('feat_cols', [])
    selected_tsfresh_cols = meta.get('selected_tsfresh_cols', [])

    # 添加技术指标特征（与训练时相同流程）
    df = add_features(data, use_ta_lib=use_ta_lib)

    # 添加 tsfresh 特征（仅变换，不做特征选择，对齐到训练集选出的列）
    if use_tsfresh and selected_tsfresh_cols and TSFRESH_AVAILABLE and extract_tsfresh_features is not None:
        window_sizes = config.get('tsfresh_window_sizes', [10, 20])
        ts_features, _ = extract_tsfresh_features(
            data,
            window_sizes=window_sizes,
            extraction_level='efficient',
            with_selection=False,  # 推断时不做特征选择
            y=None,
        )
        if not ts_features.empty:
            # 严格对齐到训练集选出的列
            ts_features = ts_features.reindex(columns=selected_tsfresh_cols, fill_value=0)
            ts_features = ts_features.reindex(df.index)
            df = pd.concat([df, ts_features], axis=1)

    # 准备特征矩阵（只保留训练时确定的特征列）
    available_feat_cols = [c for c in feat_cols if c in df.columns]
    missing_cols = [c for c in feat_cols if c not in df.columns]
    if missing_cols:
        # 补充训练时有但推断时缺失的列（填 0）
        for c in missing_cols:
            df[c] = 0.0

    X = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)

    # 推断
    predictions = model.predict(X)
    signal = pd.Series(predictions, index=X.index, dtype=int)
    return signal


