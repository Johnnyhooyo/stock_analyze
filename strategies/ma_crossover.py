import numpy as np
import pandas as pd


NAME = "ma_crossover"


def run(data: pd.DataFrame, config: dict):
    fast = int(config.get("ma_fast", 5))
    slow = int(config.get("ma_slow", 20))

    df = data.copy()
    df["ma_fast"] = df["Close"].rolling(fast).mean()
    df["ma_slow"] = df["Close"].rolling(slow).mean()
    df = df.dropna(subset=["ma_fast", "ma_slow"])

    signal = pd.Series(np.where(df["ma_fast"] > df["ma_slow"], 1, 0), index=df.index)

    meta = {
        "name": NAME,
        "params": {"ma_fast": fast, "ma_slow": slow},
        "feat_cols": [],
        "indicators": {
            "ma_fast": df["ma_fast"],
            "ma_slow": df["ma_slow"],
        },
    }
    return signal, None, meta


def predict(model, data: pd.DataFrame, config: dict, meta: dict) -> pd.Series:
    """规则策略独立推断接口：重新运行策略，返回信号序列（无需 model）。"""
    signal, _, _ = run(data, config)
    return signal
