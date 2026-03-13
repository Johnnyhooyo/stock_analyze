import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


NAME = "linear_regression"


def run(data: pd.DataFrame, config: dict):
    test_days = int(config.get("test_days", 5))

    df = data.copy()
    df["returns"] = df["Close"].pct_change()
    for i in range(1, test_days + 1):
        df[f"ret_{i}"] = df["returns"].shift(i)
    df = df.dropna()

    feat_cols = [f"ret_{i}" for i in range(1, test_days + 1)]
    X = df[feat_cols]
    y = df["returns"]

    model = LinearRegression()
    model.fit(X, y)

    pred   = model.predict(X)
    signal = pd.Series(np.where(pred > 0, 1, 0), index=df.index)

    # Sharpe ratio: annualised (252 trading days), risk-free rate = 0
    strategy_returns = df["returns"] * signal
    excess_returns   = strategy_returns  # assuming risk-free = 0
    sharpe_ratio     = (
        round(float(excess_returns.mean() / excess_returns.std() * np.sqrt(252)), 4)
        if excess_returns.std() != 0 else np.nan
    )

    meta = {
        "name": NAME,
        "params": {"test_days": test_days},
        "coef": dict(zip(feat_cols, model.coef_.round(4))),
        "intercept": round(float(model.intercept_), 6),
        "feat_cols": feat_cols,
        "sharpe_ratio": sharpe_ratio,
        "indicators": {
            "pred_return": pd.Series(pred, index=df.index),
        },
    }
    return signal, model, meta

