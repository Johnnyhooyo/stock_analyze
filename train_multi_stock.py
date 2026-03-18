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


def load_all_hsi_data(period: str = '3y', min_days: int = 300) -> pd.DataFrame:
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
                # 添加股票标识
                df = df.reset_index()  # 重置索引，将日期变成列
                df['ticker'] = ticker

                # 标准化列名
                if 'Close' not in df.columns:
                    continue

                all_data.append(df)
                stock_info.append({
                    'ticker': ticker,
                    'days': len(df),
                    'start': df.index.min().date(),
                    'end': df.index.max().date(),
                })
        except Exception as e:
            continue

    if not all_data:
        return pd.DataFrame()

    # 合并所有数据
    combined = pd.concat(all_data, ignore_index=True)

    # 设置日期为索引
    combined = combined.set_index('date')

    # 排序
    combined = combined.sort_index()

    print(f"加载了 {len(stock_info)} 只股票的数据")
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

    for ticker in tickers:
        stock_data = combined_data[combined_data['ticker'] == ticker].copy()

        if len(stock_data) < 100:
            continue

        # 添加特征
        stock_with_features = add_features(stock_data)

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
    from sklearn.model_selection import train_test_split
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

    # 分割训练/测试
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, shuffle=True
    )

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
    """
    import optuna
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    print("=" * 60)
    print("Optuna 多股票模型参数优化")
    print("=" * 60)

    # 创建数据集
    X, y, feat_cols = create_multi_stock_dataset(combined_data, test_days=5, label_period=1)

    if len(X) == 0:
        return {'error': '没有足够的数据'}

    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

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
    data = load_all_hsi_data(period='3y')

    # 使用 Optuna 优化参数
    result = optimize_multi_stock_params(
        data,
        n_trials=30,
        model_type='xgboost'
    )

    print("\n" + "=" * 60)
    print("优化完成!")
    print("=" * 60)
