"""
engine/meta_aggregator.py — Stacking 第二层 meta-learner

使用 Logistic Regression 融合基础策略信号 + 市场状态特征，
在不同市场环境下动态选择信任哪些策略。

训练：
    ma = MetaAggregator(meta_dir=Path("data/meta"))
    metrics = ma.train(ticker, data, artifacts, config)
    ma.save(ticker)

推断：
    ma = MetaAggregator.load(ticker, meta_dir)
    signal, proba = ma.predict(feature_vec)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from log_config import get_logger

logger = get_logger(__name__)


class MetaAggregator:
    """Stacking meta-learner: Logistic Regression over base strategy signals + market state."""

    META_DIR_DEFAULT = Path(__file__).parent.parent / "data" / "meta"

    def __init__(self, meta_dir: Optional[Path] = None):
        self._meta_dir = meta_dir or self.META_DIR_DEFAULT
        self._model: Optional[LogisticRegression] = None
        self._scaler: Optional[StandardScaler] = None
        self._strategy_names: list[str] = []  # fixed ordering, stored in pkl

    # ── Market state ─────────────────────────────────────────────

    @staticmethod
    def _calc_adx(data: pd.DataFrame, period: int = 14) -> float:
        """ADX(period) at the last bar, pure pandas."""
        df = data[["High", "Low", "Close"]].dropna()
        if len(df) < period * 2 + 1:
            return 25.0  # neutral fallback
        high, low, close = df["High"], df["Low"], df["Close"]
        tr = pd.concat([
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ], axis=1).max(axis=1)
        dm_plus = (high - high.shift(1)).clip(lower=0)
        dm_minus = (low.shift(1) - low).clip(lower=0)
        cond = dm_plus >= dm_minus
        dm_plus = dm_plus.where(cond, 0.0)
        dm_minus = dm_minus.where(~cond, 0.0)
        atr = tr.rolling(period).mean()
        di_plus = 100 * dm_plus.rolling(period).mean() / atr.replace(0, np.nan)
        di_minus = 100 * dm_minus.rolling(period).mean() / atr.replace(0, np.nan)
        dx = 100 * (di_plus - di_minus).abs() / (di_plus + di_minus).replace(0, np.nan)
        adx_series = dx.rolling(period).mean().dropna()
        return float(adx_series.iloc[-1]) if not adx_series.empty else 25.0

    @staticmethod
    def _calc_atr_pct_rank(data: pd.DataFrame, period: int = 14, window: int = 252) -> float:
        """ATR(period) percentile rank over last `window` bars."""
        df = data[["High", "Low", "Close"]].dropna()
        if len(df) < period + 2:
            return 0.5
        tr = pd.concat([
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs(),
        ], axis=1).max(axis=1)
        atr = tr.rolling(period).mean().dropna()
        if len(atr) < 2:
            return 0.5
        lookback = atr.tail(window)
        return float((lookback < float(atr.iloc[-1])).mean())

    @staticmethod
    def _calc_volume_ratio(data: pd.DataFrame, ma_period: int = 20) -> float:
        """Today's volume / 20-day average volume."""
        vol = data["Volume"].dropna()
        if len(vol) < ma_period + 1:
            return 1.0
        ma = vol.rolling(ma_period).mean()
        last_ma = float(ma.iloc[-1])
        return float(vol.iloc[-1]) / last_ma if last_ma > 0 else 1.0

    def _market_state_at(self, data: pd.DataFrame, idx: int) -> np.ndarray:
        """Market state features for the bar at position `idx`."""
        sub = data.iloc[: idx + 1]
        return np.array([
            self._calc_adx(sub),
            self._calc_atr_pct_rank(sub),
            self._calc_volume_ratio(sub),
        ], dtype=float)

    # ── Feature vector ────────────────────────────────────────────

    def build_feature_vector(
        self,
        base_signals: dict[str, int],
        data: pd.DataFrame,
    ) -> np.ndarray:
        """
        Feature = [strategy signals (ordered)] + [adx_14, atr_pct_rank, volume_ratio].
        Strategies not in `self._strategy_names` are ignored;
        names not in `base_signals` are filled with 0 (bearish).
        """
        sig_vec = np.array(
            [float(base_signals.get(name, 0)) for name in self._strategy_names],
            dtype=float,
        )
        market = self._market_state_at(data, len(data) - 1)
        return np.concatenate([sig_vec, market])

    # ── Training (stub — implemented in Task 2) ───────────────────

    def train(
        self,
        ticker: str,
        data: pd.DataFrame,
        artifacts: list[dict],
        config: dict,
        n_splits: int = 5,
        label_days: int = 5,
    ) -> dict:
        raise NotImplementedError("train() implemented in Task 2")

    # ── Save / Load ───────────────────────────────────────────────

    def save(self, ticker: str) -> Path:
        """Atomic save to data/meta/meta_model_{TICKER_SAFE}.pkl."""
        self._meta_dir.mkdir(parents=True, exist_ok=True)
        ticker_safe = ticker.replace(".", "_").upper()
        path = self._meta_dir / f"meta_model_{ticker_safe}.pkl"
        payload = {
            "model": self._model,
            "scaler": self._scaler,
            "strategy_names": self._strategy_names,
        }
        tmp = path.with_suffix(".tmp")
        joblib.dump(payload, tmp)
        tmp.replace(path)
        logger.info("meta-model 已保存", extra={"path": str(path), "ticker": ticker})
        return path

    @classmethod
    def load(cls, ticker: str, meta_dir: Optional[Path] = None) -> Optional["MetaAggregator"]:
        """Load saved meta-model. Returns None if file missing or corrupt."""
        meta_dir = meta_dir or cls.META_DIR_DEFAULT
        ticker_safe = ticker.replace(".", "_").upper()
        path = meta_dir / f"meta_model_{ticker_safe}.pkl"
        if not path.exists():
            return None
        try:
            payload = joblib.load(path)
            ma = cls(meta_dir=meta_dir)
            ma._model = payload["model"]
            ma._scaler = payload["scaler"]
            ma._strategy_names = payload["strategy_names"]
            return ma
        except Exception as e:
            logger.warning("meta-model 加载失败 %s: %s", path, e)
            return None

    # ── Inference ─────────────────────────────────────────────────

    def predict(self, feature_vec: np.ndarray) -> tuple[int, float]:
        """Returns (signal: 0/1, bullish_proba: float in [0, 1])."""
        if self._model is None or self._scaler is None:
            return 0, 0.5
        X = self._scaler.transform(feature_vec.reshape(1, -1))
        proba = float(self._model.predict_proba(X)[0][1])
        return (1 if proba >= 0.5 else 0), proba
