import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


NAME = "random_forest"


def _build_features(data: pd.DataFrame, test_days: int) -> tuple:
    """构建特征矩阵，返回 (df_clean, feat_cols)"""
    df = data.copy()
    df["returns"] = df["Close"].pct_change()
    for i in range(1, test_days + 1):
        df[f"ret_{i}"] = df["returns"].shift(i)
    # 标签：使用下一日收益方向（shift(-1）），避免使用当日收益造成泄露
    df["label"] = np.where(df["Close"].shift(-1) > df["Close"], 1, 0)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df, [f"ret_{i}" for i in range(1, test_days + 1)]


def run(data: pd.DataFrame, config: dict):
    test_days    = int(config.get("test_days", 5))
    n_estimators = int(config.get("rf_n_estimators", 100))
    max_depth_raw = config.get("rf_max_depth", None)
    max_depth: "int | None" = int(max_depth_raw) if max_depth_raw is not None else None

    df, feat_cols = _build_features(data, test_days)
    X = df[feat_cols]
    y = df["label"]

    # ⚠️ 前视偏差修复：80/20 拆分，仅在训练集 fit，测试集预测
    no_split = config.get("no_internal_split", False)
    if no_split or len(X) < 20:
        X_train, y_train = X, y
        split_idx = len(X)
    else:
        split_idx = int(len(X) * 0.8)
        X_train = X.iloc[:split_idx]
        y_train = y.iloc[:split_idx]

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # 只对测试集（或全集，若 no_split）生成信号
    signal = pd.Series(0, index=X.index)
    if no_split:
        signal.iloc[:] = model.predict(X)
    else:
        if split_idx < len(X):
            signal.iloc[split_idx:] = model.predict(X.iloc[split_idx:])

    pred_proba_test = model.predict_proba(X.iloc[split_idx:])[:, 1] if split_idx < len(X) else np.array([])

    meta = {
        "name": NAME,
        "params": {"test_days": test_days, "n_estimators": n_estimators, "max_depth": max_depth},
        "feature_importances": dict(zip(feat_cols, model.feature_importances_.round(4))),
        "feat_cols": feat_cols,
        "indicators": {
            "pred_proba": pd.Series(pred_proba_test, index=X.index[split_idx:]),
        },
    }
    return signal, model, meta


def predict(model, data: pd.DataFrame, config: dict, meta: dict) -> pd.Series:
    """独立推断：用已训练模型在新数据上生成信号，不重新 fit。"""
    test_days = int(config.get("test_days", 5))
    feat_cols = meta.get("feat_cols", [f"ret_{i}" for i in range(1, test_days + 1)])

    df, _ = _build_features(data, test_days)
    available = [c for c in feat_cols if c in df.columns]
    for c in feat_cols:
        if c not in df.columns:
            df[c] = 0.0

    X = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    predictions = model.predict(X)
    return pd.Series(predictions.astype(int), index=X.index)
