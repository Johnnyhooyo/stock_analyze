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

    # ── Training ──────────────────────────────────────────────────

    def _build_historical_signals(
        self,
        data: pd.DataFrame,
        artifacts: list[dict],
    ) -> tuple[list[str], np.ndarray]:
        """
        Re-run each artifact's strategy on full historical data.
        Returns (strategy_names, signal_matrix) where signal_matrix shape = (n_bars, n_strategies).
        Strategies that fail to produce signals are silently skipped.
        """
        from analyze_factor import _discover_strategies
        strategy_modules = {mod.NAME: mod for mod in _discover_strategies()}

        names: list[str] = []
        series_list: list[pd.Series] = []

        for art in artifacts:
            meta = art.get("meta", {})
            name = meta.get("name", "")
            if not name:
                continue
            mod = strategy_modules.get(name)
            if mod is None:
                continue
            try:
                model = art.get("model")
                feat_cols = meta.get("feat_cols", [])
                is_ml = model is not None and len(feat_cols) > 0
                art_cfg = dict(art.get("config", {}))

                if is_ml and hasattr(mod, "predict"):
                    sig = mod.predict(model, data.copy(), art_cfg, meta)
                else:
                    sig, _, _ = mod.run(data.copy(), art_cfg)

                if sig is not None and not sig.empty:
                    names.append(name)
                    series_list.append(sig.reindex(data.index).fillna(0).astype(float))
            except Exception as e:
                logger.debug("历史信号生成失败 %s: %s", name, e)

        if not series_list:
            return [], np.empty((len(data), 0), dtype=float)

        return names, np.column_stack([s.values for s in series_list])

    def train(
        self,
        ticker: str,
        data: pd.DataFrame,
        artifacts: list[dict],
        config: dict,
        n_splits: int = 5,
        label_days: int = 5,
    ) -> dict:
        """
        Walk-Forward training (expanding window, TimeSeriesSplit).

        Steps:
          1. Re-run each artifact's strategy on full history → signal matrix
          2. Compute market state features for each bar
          3. Labels: 1 if close[t+label_days] > close[t] else 0
          4. TimeSeriesSplit CV to compute accuracy
          5. Train final model on all valid samples (for inference)
        """
        if len(artifacts) < 2:
            raise ValueError(f"至少需要 2 个因子才能训练 meta-model，当前: {len(artifacts)}")

        strategy_names, sig_matrix = self._build_historical_signals(data, artifacts)

        if len(strategy_names) < 2:
            raise ValueError(
                f"至少需要 2 个策略的历史信号，实际产生信号的策略数: {len(strategy_names)}"
            )

        n = len(data)

        # Market state features for each bar (uses data up to that bar only)
        market_rows = [self._market_state_at(data, i) for i in range(n)]
        market_matrix = np.array(market_rows, dtype=float)

        # Labels: look-forward label_days bars
        closes = data["Close"].values
        labels = np.zeros(n, dtype=int)
        for i in range(n - label_days):
            labels[i] = 1 if closes[i + label_days] > closes[i] else 0

        # Mask out last label_days rows (no future data available)
        valid_idx = np.arange(n - label_days)
        X_all = np.hstack([sig_matrix, market_matrix])
        X_valid = X_all[valid_idx]
        y_valid = labels[valid_idx]

        # Walk-Forward CV
        from sklearn.model_selection import TimeSeriesSplit
        tscv = TimeSeriesSplit(n_splits=n_splits)
        fold_accs: list[float] = []

        for tr_idx, te_idx in tscv.split(X_valid):
            if len(tr_idx) < 10 or len(te_idx) < 5:
                continue
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_valid[tr_idx])
            X_te = scaler.transform(X_valid[te_idx])
            lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
            try:
                lr.fit(X_tr, y_valid[tr_idx])
                fold_accs.append(float((lr.predict(X_te) == y_valid[te_idx]).mean()))
            except ValueError:
                logger.debug("跳过 CV fold：训练集标签为单一类别")
                continue

        if not fold_accs:
            logger.warning(
                "所有 CV fold 已跳过（数据不足或标签单一），meta-model 精度设为默认 0.5",
                extra={"ticker": ticker},
            )
        mean_acc = float(np.mean(fold_accs)) if fold_accs else 0.5

        # Final model: all valid samples
        if np.unique(y_valid).size < 2:
            logger.warning(
                "训练标签全为单一类别，跳过 meta-model 训练",
                extra={"ticker": ticker},
            )
            return {"accuracy": 0.5, "n_samples": len(y_valid), "n_features": X_all.shape[1]}

        scaler_final = StandardScaler()
        X_scaled = scaler_final.fit_transform(X_valid)
        lr_final = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
        lr_final.fit(X_scaled, y_valid)

        self._model = lr_final
        self._scaler = scaler_final
        self._strategy_names = strategy_names

        return {"accuracy": mean_acc, "n_samples": len(y_valid), "n_features": X_all.shape[1]}

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
