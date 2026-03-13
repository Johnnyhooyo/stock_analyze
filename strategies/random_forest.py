import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


NAME = "random_forest"


def run(data: pd.DataFrame, config: dict):
    test_days    = int(config.get("test_days", 5))
    n_estimators = int(config.get("rf_n_estimators", 100))
    max_depth_raw = config.get("rf_max_depth", None)
    max_depth: "int | None" = int(max_depth_raw) if max_depth_raw is not None else None

    df = data.copy()
    df["returns"] = df["Close"].pct_change()
    for i in range(1, test_days + 1):
        df[f"ret_{i}"] = df["returns"].shift(i)
    df["label"] = np.where(df["returns"] > 0, 1, 0)
    df = df.dropna()

    feat_cols = [f"ret_{i}" for i in range(1, test_days + 1)]
    X = df[feat_cols]
    y = df["label"]

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)

    signal = pd.Series(model.predict(X).astype(int), index=df.index)
    pred_proba = model.predict_proba(X)[:, 1]   # 上涨概率

    meta = {
        "name": NAME,
        "params": {"test_days": test_days, "n_estimators": n_estimators, "max_depth": max_depth},
        "feature_importances": dict(zip(feat_cols, model.feature_importances_.round(4))),
        "feat_cols": feat_cols,
        "indicators": {
            "pred_proba": pd.Series(pred_proba, index=df.index),
        },
    }
    return signal, model, meta

