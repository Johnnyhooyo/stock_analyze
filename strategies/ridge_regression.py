import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


NAME = "ridge_regression"


def run(data: pd.DataFrame, config: dict):
    test_days   = int(config.get("test_days", 5))
    ridge_alpha = float(config.get("ridge_alpha", 1.0))

    df = data.copy()
    df["returns"] = df["Close"].pct_change()
    for i in range(1, test_days + 1):
        df[f"ret_{i}"] = df["returns"].shift(i)
    df = df.dropna()

    feat_cols = [f"ret_{i}" for i in range(1, test_days + 1)]
    X = df[feat_cols]
    y = df["returns"]

    model = Ridge(alpha=ridge_alpha)
    model.fit(X, y)

    pred   = model.predict(X)
    signal = pd.Series(np.where(pred > 0, 1, 0), index=df.index)

    meta = {
        "name": NAME,
        "params": {"test_days": test_days, "ridge_alpha": ridge_alpha},
        "coef": dict(zip(feat_cols, model.coef_.round(4))),
        "intercept": round(float(model.intercept_), 6),
        "feat_cols": feat_cols,
        "indicators": {
            "pred_return": pd.Series(pred, index=df.index),
        },
    }
    return signal, model, meta

