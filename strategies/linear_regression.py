import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


NAME = "linear_regression"


def _build_features(data: pd.DataFrame, test_days: int) -> tuple:
    """构建特征矩阵，返回 (df_clean, feat_cols)"""
    df = data.copy()
    df["returns"] = df["Close"].pct_change()
    for i in range(1, test_days + 1):
        df[f"ret_{i}"] = df["returns"].shift(i)
    df = df.dropna()
    return df, [f"ret_{i}" for i in range(1, test_days + 1)]


def run(data: pd.DataFrame, config: dict):
    test_days = int(config.get("test_days", 5))

    df, feat_cols = _build_features(data, test_days)
    X = df[feat_cols]
    y = df["returns"]

    # ⚠️ 前视偏差修复：80/20 拆分，仅在训练集 fit，测试集预测
    no_split = config.get("no_internal_split", False)
    if no_split or len(X) < 20:
        X_train, y_train = X, y
        split_idx = len(X)
    else:
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]

    model = LinearRegression()
    model.fit(X_train, y_train)

    # 只对测试集（或全集，若 no_split）生成信号
    signal = pd.Series(0, index=X.index)
    if no_split:
        pred = model.predict(X)
        signal.iloc[:] = np.where(pred > 0, 1, 0)
    else:
        if split_idx < len(X):
            pred_test = model.predict(X.iloc[split_idx:])
            signal.iloc[split_idx:] = np.where(pred_test > 0, 1, 0)
            pred = np.concatenate([model.predict(X_train), pred_test])
        else:
            pred = model.predict(X_train)

    meta = {
        "name": NAME,
        "params": {"test_days": test_days},
        "coef": dict(zip(feat_cols, model.coef_.round(4))),
        "intercept": round(float(model.intercept_), 6),
        "feat_cols": feat_cols,
        "indicators": {
            "pred_return": pd.Series(pred, index=X.index),
        },
    }
    return signal, model, meta


def predict(model, data: pd.DataFrame, config: dict, meta: dict) -> pd.Series:
    """独立推断：用已训练模型在新数据上生成信号，不重新 fit。"""
    test_days = int(config.get("test_days", 5))
    feat_cols = meta.get("feat_cols", [f"ret_{i}" for i in range(1, test_days + 1)])

    df, _ = _build_features(data, test_days)
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0.0

    X = df[feat_cols].fillna(0)
    pred = model.predict(X)
    return pd.Series(np.where(pred > 0, 1, 0).astype(int), index=X.index)
