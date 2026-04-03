"""
多股票数据训练模块
使用所有恒生指数成分股的数据来训练模型，解决单一股票数据量不足的问题
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

# 导入策略
from strategies.xgboost_enhanced import add_features, prepare_data
from strategies.lightgbm_enhanced import run as run_lgbm


# ── 黑名单追踪（同一进程内跨多次调用）────────────────────────────────
_session_failures: dict[str, int] = {}


def _check_stock_quality(df: pd.DataFrame, ticker: str) -> Tuple[bool, str]:
    """
    检查单只股票的数据质量。

    Returns:
        (pass, reason): pass=True 表示合格，reason 为不合格原因
    """
    close = df['Close']

    nan_pct = close.isna().mean()
    if nan_pct > 0.2:
        return False, f"NaN_pct {nan_pct:.1%} > 20%"

    if close.nunique() < 5:
        return False, f"Close_nunique {close.nunique()} < 5"

    if 'Volume' in df.columns and len(df) > 0:
        volume = df['Volume']
        zero_vol_pct = (volume == 0).mean()
        if zero_vol_pct > 0.3:
            return False, f"volume_zero_pct {zero_vol_pct:.1%} > 30%"

        if (volume == 0).all():
            return False, "volume all zero"

    return True, ""


def _update_blacklist(ticker: str, reason: str) -> None:
    """检查失败次数，连续2次失败则加入黑名单。"""
    failures = _session_failures.get(ticker, 0) + 1
    _session_failures[ticker] = failures
    if failures >= 2:
        try:
            from data.hk_stocks import add_to_blacklist
            add_to_blacklist(ticker, reason, ["quality_check"])
        except Exception:
            pass
        _session_failures[ticker] = 0


def load_all_hsi_data(period: str = '5y', min_days: int = 300) -> pd.DataFrame:
    """
    加载所有恒生指数成分股数据并合并

    Args:
        period: 数据周期
        min_days: 最小天数要求

    Returns:
        合并后的 DataFrame
    """
    data_dir = Path(__file__).parent / 'data' / 'historical'

    all_data = []
    stock_info = []
    skipped_quality = 0

    for csv_file in data_dir.glob(f'*_{period}.csv'):
        try:
            df = pd.read_csv(csv_file, index_col=0)

            # 转换索引为datetime并去除时区
            df.index = pd.to_datetime(df.index)
            if df.index.tz is not None:
                df.index = df.index.tz_convert(None)

            # 确保列名标准化 (首字母大写)
            df.columns = [c[0].upper() + c[1:] if isinstance(c, str) and len(c) > 1 else c.upper() for c in df.columns]

            # 确保必要的列存在
            if 'Close' not in df.columns and 'close' in df.columns:
                df = df.rename(columns={'close': 'Close'})
            if 'Volume' not in df.columns and 'volume' in df.columns:
                df = df.rename(columns={'volume': 'Volume'})

            # 提取股票代码
            ticker = csv_file.stem.split('_')[0] + '.HK'

            if len(df) >= min_days:
                # 捕获日期范围（必须在 reset_index() 之前，否则 index 变成整数）
                start_date = df.index.min().date()
                end_date   = df.index.max().date()

                # 添加股票标识
                df = df.reset_index()  # 重置索引，将日期变成列
                df['ticker'] = ticker

                # 去除重复日期（同一天多条记录只保留第一条）
                df = df.drop_duplicates(subset=['date'], keep='first')

                # 标准化列名
                if 'Close' not in df.columns:
                    continue

                # 质量检查
                ok, reason = _check_stock_quality(df, ticker)
                if not ok:
                    print(f"  SKIP {ticker}: {reason}")
                    _update_blacklist(ticker, reason)
                    skipped_quality += 1
                    continue

                all_data.append(df)
                stock_info.append({
                    'ticker': ticker,
                    'days':   len(df),
                    'start':  start_date,
                    'end':    end_date,
                })
        except Exception as e:
            print(f"  [load_all_hsi_data] 跳过 {csv_file.name}: {e}")
            continue

    if not all_data:
        return pd.DataFrame()

    # 合并所有数据
    combined = pd.concat(all_data, ignore_index=True)

    # 设置日期为索引
    combined = combined.set_index('date')

    # 排序
    combined = combined.sort_index()

    print(f"加载了 {len(stock_info)} 只股票的数据（质量过滤跳过 {skipped_quality} 只）")
    print(f"总记录数: {len(combined)}")
    print(f"时间范围: {combined.index.min().date()} ~ {combined.index.max().date()}")

    return combined


def create_multi_stock_dataset(
    combined_data: pd.DataFrame,
    test_days: int = 5,
    label_period: int = 1
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    创建多股票训练数据集

    Args:
        combined_data: 合并后的数据
        test_days: 用于预测的天数
        label_period: 预测未来第几天

    Returns:
        X, y, feature_columns
    """
    # 按股票分别添加特征
    tickers = combined_data['ticker'].unique()
    all_X = []
    all_y = []
    feat_cols = []  # 初始化，避免循环体内未赋值就被返回

    for ticker in tickers:
        stock_data = combined_data[combined_data['ticker'] == ticker].copy()

        if len(stock_data) < 100:
            continue

        # 添加特征
        stock_with_features = add_features(stock_data)

        # ── 修复项5：截断末尾 label_period 行，消除 shift(-label_period) 边界泄露 ──
        if len(stock_with_features) > label_period:
            stock_with_features = stock_with_features.iloc[:-label_period]

        # 准备数据
        try:
            X, y, feat_cols = prepare_data(stock_with_features, test_days, label_period)
            all_X.append(X)
            all_y.append(y)
        except Exception as e:
            continue

    if not all_X:
        return pd.DataFrame(), pd.Series([]), []

    # 合并所有股票的数据
    X = pd.concat(all_X, ignore_index=True)
    y = pd.concat(all_y, ignore_index=True)

    # 删除 ticker 列（如果存在）
    if 'ticker' in X.columns:
        X = X.drop(columns=['ticker'])

    # 去除 NaN
    mask = ~(X.isna().any(axis=1) | y.isna())
    X = X[mask]
    y = y[mask]

    print(f"训练数据: {len(X)} 样本, {X.shape[1]} 特征")
    print(f"正样本比例: {y.mean():.2%}")

    return X, y, feat_cols


def train_multi_stock_model(
    combined_data: pd.DataFrame,
    test_days: int = 5,
    label_period: int = 1,
    model_type: str = 'lightgbm',
    n_estimators: int = 100,
    max_depth: int = 5,
    learning_rate: float = 0.1,
    test_size: float = 0.2,
) -> Dict:
    """
    使用多股票数据训练模型

    Args:
        combined_data: 合并后的股票数据
        test_days: 特征窗口天数
        label_period: 预测天数
        model_type: 模型类型 (lightgbm, xgboost)
        n_estimators: 树的数量
        max_depth: 最大深度
        learning_rate: 学习率
        test_size: 测试集比例

    Returns:
        训练结果字典
    """
    from sklearn.metrics import accuracy_score, classification_report

    print("=" * 60)
    print(f"多股票模型训练 ({model_type})")
    print("=" * 60)

    # 创建数据集
    X, y, feat_cols = create_multi_stock_dataset(
        combined_data, test_days, label_period
    )

    if len(X) == 0:
        return {'error': '没有足够的数据'}

    # 按时间顺序分割训练/测试（禁止 shuffle，防止未来数据泄露到训练集）
    split_idx = int(len(X) * (1 - test_size))
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]
    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    print(f"\n训练集: {len(X_train)} 样本")
    print(f"测试集: {len(X_test)} 样本")

    # 训练模型
    if model_type == 'lightgbm':
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
    else:
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

    print(f"\n训练 {model_name} 模型...")
    model.fit(X_train, y_train)

    # 预测
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    test_proba = model.predict_proba(X_test)[:, 1]

    # 评估
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)

    print(f"\n训练准确率: {train_acc:.2%}")
    print(f"测试准确率: {test_acc:.2%}")

    # 特征重要性
    importance = dict(zip(feat_cols, model.feature_importances_))
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]

    print(f"\nTop 15 重要特征:")
    for name, imp in top_features:
        print(f"  {name}: {imp:.4f}")

    return {
        'model': model,
        'model_name': model_name,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'feature_importances': importance,
        'feat_cols': feat_cols,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'test_proba': test_proba,
    }


def optimize_multi_stock_params(
    combined_data: pd.DataFrame,
    n_trials: int = 20,
    model_type: str = 'lightgbm'
) -> Dict:
    """
    使用 Optuna 优化多股票模型参数

    ⚠️ 前视偏差修复：先按时间切分原始数据，再分别提取特征（含 shift(-label_period) 标签），
    确保测试集的未来标签不会泄露到训练集的特征/标签生成过程中。
    """
    import optuna
    from sklearn.metrics import accuracy_score

    print("=" * 60)
    print("Optuna 多股票模型参数优化")
    print("=" * 60)

    # ⚠️ 前视偏差修复：先按时间对原始行切分，再分别调用 create_multi_stock_dataset
    # 旧做法（错误）：对全量数据提取特征（含 shift(-1) 标签）后再切分 → 测试集标签已参与训练集特征计算
    # 新做法（正确）：先用原始价格数据按 80/20 时间切分，再分别提取特征，测试集标签与训练集完全隔离
    if combined_data.empty:
        return {'error': '没有足够的数据'}

    # 按时间排序，确保切分有意义
    df_sorted = combined_data.sort_index()
    unique_dates = df_sorted.index.unique().sort_values()
    split_date = unique_dates[int(len(unique_dates) * 0.8)]

    train_raw = df_sorted[df_sorted.index < split_date]
    test_raw  = df_sorted[df_sorted.index >= split_date]

    # 分别提取特征（标签 shift(-label_period) 只在各自子集内计算）
    X_train, y_train, feat_cols = create_multi_stock_dataset(train_raw, test_days=5, label_period=1)
    X_test,  y_test,  _         = create_multi_stock_dataset(test_raw,  test_days=5, label_period=1)

    # 对齐特征列（测试集对齐到训练集的列）
    for c in feat_cols:
        if c not in X_test.columns:
            X_test[c] = 0.0
    if feat_cols:
        X_test = X_test[feat_cols]

    if len(X_train) == 0 or len(X_test) == 0:
        return {'error': '没有足够的数据'}

    print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")

    def objective(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 200),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
            'num_leaves': trial.suggest_int('num_leaves', 15, 63) if model_type == 'lightgbm' else 31,
            'random_state': 42,
            'verbose': -1 if model_type == 'lightgbm' else 0,
        }

        if model_type == 'lightgbm':
            import lightgbm as lgb
            model = lgb.LGBMClassifier(**params)
        else:
            from xgboost import XGBClassifier
            model = XGBClassifier(**params, use_label_encoder=False, eval_metric='logloss')

        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        return accuracy_score(y_test, pred)

    # 优化
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    print(f"\n最佳准确率: {study.best_value:.4f}")
    print(f"最佳参数: {study.best_params}")

    return {
        'best_value': study.best_value,
        'best_params': study.best_params,
        'study': study,
    }


if __name__ == "__main__":
    # 加载数据
    print("加载股票数据...")
    data = load_all_hsi_data(period='5y')

    # 使用 Optuna 优化参数
    result = optimize_multi_stock_params(
        data,
        n_trials=30,
        model_type='xgboost'
    )

    print("\n" + "=" * 60)
    print("优化完成!")
    print("=" * 60)
