# Ensemble/Stacking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Logistic Regression meta-learner (`engine/meta_aggregator.py`) that fuses base strategy signals + market state features, integrated into `SignalAggregator` as an opt-in `aggregation_method="stacking"` mode.

**Architecture:** `MetaAggregator` is trained offline by `main.py` after base strategies are saved, stored as `data/meta/meta_model_{TICKER_SAFE}.pkl`. `SignalAggregator` lazily loads the meta-model per ticker; when unavailable it silently falls back to the existing Sharpe-weighted vote.

**Tech Stack:** scikit-learn (`LogisticRegression`, `StandardScaler`, `TimeSeriesSplit`), joblib, pandas, numpy

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `engine/meta_aggregator.py` | Create | MetaAggregator: feature construction, Walk-Forward train, save/load, predict |
| `engine/signal_aggregator.py` | Modify | Add `aggregation_method` + `meta_dir` params; `_load_meta()`; stacking branch in `aggregate()` |
| `engine/position_analyzer.py` | Modify | Pass `aggregation_method` from config when constructing `SignalAggregator` |
| `engine/__init__.py` | Modify | Export `MetaAggregator` |
| `main.py` | Modify | Add `_train_meta_model()` function; call from `main()` and `train_portfolio_tickers()` |
| `config.yaml` | Modify | Add `stacking:` block |
| `tests/test_meta_aggregator.py` | Create | Unit tests for MetaAggregator |
| `tests/test_signal_aggregator.py` | Modify | Add stacking + fallback tests |
| `tests/test_integration.py` | Modify | Add end-to-end meta train→save→load→predict test |

---

## Task 1: MetaAggregator — feature vector + save/load/predict

**Files:**
- Create: `engine/meta_aggregator.py`
- Create: `tests/test_meta_aggregator.py`

- [ ] **Step 1.1: Write failing tests for feature vector + save/load/predict**

```python
# tests/test_meta_aggregator.py
"""tests/test_meta_aggregator.py — MetaAggregator unit tests"""
import numpy as np
import pytest
from pathlib import Path
from engine.meta_aggregator import MetaAggregator


class TestBuildFeatureVector:
    def test_shape_is_n_strategies_plus_3(self, synthetic_ohlcv, tmp_path):
        """Feature vector = n strategy signals + 3 market state indicators."""
        ma = MetaAggregator(meta_dir=tmp_path)
        ma._strategy_names = ["s1", "s2", "s3"]
        feat = ma.build_feature_vector({"s1": 1, "s2": 0, "s3": 1}, synthetic_ohlcv)
        assert feat.shape == (6,)   # 3 strategies + 3 market state

    def test_missing_strategy_fills_zero(self, synthetic_ohlcv, tmp_path):
        """Strategy not in base_signals → treated as bearish (0)."""
        ma = MetaAggregator(meta_dir=tmp_path)
        ma._strategy_names = ["known", "unknown_in_signals"]
        feat = ma.build_feature_vector({"known": 1}, synthetic_ohlcv)
        # index 1 is "unknown_in_signals" → should be 0.0
        assert feat[0] == 1.0
        assert feat[1] == 0.0

    def test_market_state_values_in_reasonable_range(self, synthetic_ohlcv, tmp_path):
        """ADX in [0, 100], ATR rank in [0, 1], volume ratio > 0."""
        ma = MetaAggregator(meta_dir=tmp_path)
        ma._strategy_names = []
        feat = ma.build_feature_vector({}, synthetic_ohlcv)
        adx, atr_rank, vol_ratio = feat[0], feat[1], feat[2]
        assert 0.0 <= adx <= 100.0
        assert 0.0 <= atr_rank <= 1.0
        assert vol_ratio > 0.0


class TestSaveLoadPredict:
    def _make_trained_meta(self, tmp_path, synthetic_ohlcv):
        """Helper: create a MetaAggregator with a minimal fitted model."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        ma = MetaAggregator(meta_dir=tmp_path)
        ma._strategy_names = ["s1", "s2"]
        # Fit on dummy data (5 features = 2 strategies + 3 market state)
        X = np.random.RandomState(0).randn(50, 5)
        y = (X[:, 0] > 0).astype(int)
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)
        lr = LogisticRegression(max_iter=100)
        lr.fit(X_s, y)
        ma._model = lr
        ma._scaler = scaler
        return ma

    def test_save_creates_pkl_file(self, synthetic_ohlcv, tmp_path):
        ma = self._make_trained_meta(tmp_path, synthetic_ohlcv)
        path = ma.save("0700.HK")
        assert path.exists()
        assert path.name == "meta_model_0700_HK.pkl"

    def test_load_returns_none_when_file_missing(self, tmp_path):
        result = MetaAggregator.load("9999.HK", tmp_path)
        assert result is None

    def test_save_load_round_trip_preserves_strategy_names(self, synthetic_ohlcv, tmp_path):
        ma = self._make_trained_meta(tmp_path, synthetic_ohlcv)
        ma.save("0700.HK")
        loaded = MetaAggregator.load("0700.HK", tmp_path)
        assert loaded is not None
        assert loaded._strategy_names == ["s1", "s2"]

    def test_predict_signal_in_0_or_1(self, synthetic_ohlcv, tmp_path):
        ma = self._make_trained_meta(tmp_path, synthetic_ohlcv)
        ma._strategy_names = ["s1", "s2"]
        feat = ma.build_feature_vector({"s1": 1, "s2": 0}, synthetic_ohlcv)
        signal, proba = ma.predict(feat)
        assert signal in (0, 1)
        assert 0.0 <= proba <= 1.0

    def test_predict_after_save_load(self, synthetic_ohlcv, tmp_path):
        ma = self._make_trained_meta(tmp_path, synthetic_ohlcv)
        ma.save("0700.HK")
        loaded = MetaAggregator.load("0700.HK", tmp_path)
        feat = loaded.build_feature_vector({"s1": 1, "s2": 0}, synthetic_ohlcv)
        signal, proba = loaded.predict(feat)
        assert signal in (0, 1)
        assert 0.0 <= proba <= 1.0
```

- [ ] **Step 1.2: Run tests to verify they fail**

```bash
cd /home/thenine/projects/stock_analyze
pytest tests/test_meta_aggregator.py -v 2>&1 | head -30
```

Expected: `ImportError: cannot import name 'MetaAggregator'`

- [ ] **Step 1.3: Create `engine/meta_aggregator.py`**

```python
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

    # ── Training (implemented in Task 2) ─────────────────────────

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
```

- [ ] **Step 1.4: Run tests to verify they pass**

```bash
pytest tests/test_meta_aggregator.py -v
```

Expected: all 8 tests PASS (train() tests will run in Task 2)

- [ ] **Step 1.5: Commit**

```bash
git add engine/meta_aggregator.py tests/test_meta_aggregator.py
git commit -m "feat: add MetaAggregator scaffold with feature vector, save/load, predict"
```

---

## Task 2: MetaAggregator — train()

**Files:**
- Modify: `engine/meta_aggregator.py` (implement `train()`)
- Modify: `tests/test_meta_aggregator.py` (add train tests)

- [ ] **Step 2.1: Write failing tests for train()**

Add the following class to `tests/test_meta_aggregator.py`:

```python
class TestTrain:
    def _make_rule_artifact(self, strategy_name: str) -> dict:
        return {
            "meta": {"name": strategy_name, "params": {}, "feat_cols": []},
            "model": None,
            "sharpe_ratio": 1.2,
            "config": {},
        }

    def test_train_returns_expected_keys(self, synthetic_ohlcv, tmp_path):
        """train() returns dict with accuracy, n_samples, n_features."""
        ma = MetaAggregator(meta_dir=tmp_path)
        artifacts = [
            self._make_rule_artifact("ma_crossover"),
            self._make_rule_artifact("rsi_reversion"),
        ]
        result = ma.train("0700.HK", synthetic_ohlcv, artifacts, config={}, n_splits=2, label_days=5)
        assert "accuracy" in result
        assert "n_samples" in result
        assert "n_features" in result

    def test_train_accuracy_in_valid_range(self, synthetic_ohlcv, tmp_path):
        """Accuracy must be a float in [0, 1]."""
        ma = MetaAggregator(meta_dir=tmp_path)
        artifacts = [
            self._make_rule_artifact("ma_crossover"),
            self._make_rule_artifact("rsi_reversion"),
        ]
        result = ma.train("0700.HK", synthetic_ohlcv, artifacts, config={}, n_splits=2, label_days=5)
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_train_sets_strategy_names(self, synthetic_ohlcv, tmp_path):
        """After train(), _strategy_names matches the strategies that produced signals."""
        ma = MetaAggregator(meta_dir=tmp_path)
        artifacts = [
            self._make_rule_artifact("ma_crossover"),
            self._make_rule_artifact("rsi_reversion"),
        ]
        ma.train("0700.HK", synthetic_ohlcv, artifacts, config={}, n_splits=2, label_days=5)
        assert len(ma._strategy_names) >= 1  # at least one strategy produced signals

    def test_train_raises_with_fewer_than_2_artifacts(self, synthetic_ohlcv, tmp_path):
        """train() raises ValueError when fewer than 2 artifacts provided."""
        ma = MetaAggregator(meta_dir=tmp_path)
        with pytest.raises(ValueError, match="至少需要"):
            ma.train("0700.HK", synthetic_ohlcv, [self._make_rule_artifact("ma_crossover")],
                     config={}, n_splits=2, label_days=5)

    def test_train_predict_after_train(self, synthetic_ohlcv, tmp_path):
        """After train(), predict() returns valid output."""
        ma = MetaAggregator(meta_dir=tmp_path)
        artifacts = [
            self._make_rule_artifact("ma_crossover"),
            self._make_rule_artifact("rsi_reversion"),
        ]
        ma.train("0700.HK", synthetic_ohlcv, artifacts, config={}, n_splits=2, label_days=5)
        feat = ma.build_feature_vector(
            {name: 1 for name in ma._strategy_names}, synthetic_ohlcv
        )
        signal, proba = ma.predict(feat)
        assert signal in (0, 1)
        assert 0.0 <= proba <= 1.0
```

- [ ] **Step 2.2: Run tests to verify they fail**

```bash
pytest tests/test_meta_aggregator.py::TestTrain -v 2>&1 | head -20
```

Expected: `NotImplementedError: train() implemented in Task 2`

- [ ] **Step 2.3: Implement `train()` in `engine/meta_aggregator.py`**

Replace the `train()` stub with:

```python
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
        lr.fit(X_tr, y_valid[tr_idx])
        fold_accs.append(float((lr.predict(X_te) == y_valid[te_idx]).mean()))

    mean_acc = float(np.mean(fold_accs)) if fold_accs else 0.5

    # Final model: all valid samples
    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X_valid)
    lr_final = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    lr_final.fit(X_scaled, y_valid)

    self._model = lr_final
    self._scaler = scaler_final
    self._strategy_names = strategy_names

    return {"accuracy": mean_acc, "n_samples": len(y_valid), "n_features": X_all.shape[1]}

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
```

- [ ] **Step 2.4: Run tests to verify they pass**

```bash
pytest tests/test_meta_aggregator.py -v
```

Expected: all tests PASS

- [ ] **Step 2.5: Commit**

```bash
git add engine/meta_aggregator.py tests/test_meta_aggregator.py
git commit -m "feat: implement MetaAggregator.train() with Walk-Forward cross-validation"
```

---

## Task 3: Modify SignalAggregator to support stacking

**Files:**
- Modify: `engine/signal_aggregator.py`
- Modify: `tests/test_signal_aggregator.py`

- [ ] **Step 3.1: Write failing tests for stacking in SignalAggregator**

Add to `tests/test_signal_aggregator.py`:

```python
class TestSignalAggregatorStacking:
    """SignalAggregator stacking mode: uses meta-model when available, fallback to vote."""

    def _make_rule_factor(self, factors_dir: Path, run_id: int = 1):
        import joblib
        factors_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "meta": {"name": "ma_crossover", "params": {}, "feat_cols": []},
            "model": None, "sharpe_ratio": 1.2, "config": {},
        }, factors_dir / f"factor_{run_id:04d}.pkl")

    def _make_and_save_meta(self, ticker: str, strategy_names: list, meta_dir: Path, synthetic_ohlcv):
        """Create a minimal fitted meta-model saved to meta_dir."""
        import joblib
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        import numpy as np
        meta_dir.mkdir(parents=True, exist_ok=True)
        n_feats = len(strategy_names) + 3
        X = np.random.RandomState(0).randn(60, n_feats)
        y = (X[:, 0] > 0).astype(int)
        scaler = StandardScaler()
        lr = LogisticRegression(max_iter=100)
        lr.fit(scaler.fit_transform(X), y)
        ticker_safe = ticker.replace(".", "_").upper()
        joblib.dump(
            {"model": lr, "scaler": scaler, "strategy_names": strategy_names},
            meta_dir / f"meta_model_{ticker_safe}.pkl",
        )

    def test_stacking_uses_meta_model_when_available(self, synthetic_ohlcv, tmp_path):
        """aggregation_method='stacking' with meta-model present → result is AggregatedSignal."""
        factors_dir = tmp_path / "factors"
        meta_dir = tmp_path / "meta"
        self._make_rule_factor(factors_dir)
        self._make_and_save_meta("0700.HK", ["ma_crossover"], meta_dir, synthetic_ohlcv)

        agg = SignalAggregator(
            factors_dir=factors_dir,
            aggregation_method="stacking",
            meta_dir=meta_dir,
            use_registry=False,
        )
        result = agg.aggregate("0700.HK", synthetic_ohlcv, {"ticker": "0700.HK"})
        assert isinstance(result, AggregatedSignal)
        assert result.consensus_signal in (0, 1)
        assert 0.0 <= result.confidence_pct <= 1.0

    def test_stacking_fallback_to_vote_when_no_meta_model(self, synthetic_ohlcv, tmp_path):
        """aggregation_method='stacking' without meta-model → fallback to vote, no error."""
        factors_dir = tmp_path / "factors"
        self._make_rule_factor(factors_dir)

        agg = SignalAggregator(
            factors_dir=factors_dir,
            aggregation_method="stacking",
            meta_dir=tmp_path / "nonexistent_meta",
            use_registry=False,
        )
        result = agg.aggregate("0700.HK", synthetic_ohlcv, {"ticker": "0700.HK"})
        assert isinstance(result, AggregatedSignal)
        assert 0.0 <= result.confidence_pct <= 1.0

    def test_vote_method_unchanged_by_new_params(self, synthetic_ohlcv, tmp_path):
        """aggregation_method='vote' (default) behaves identically to original."""
        factors_dir = tmp_path / "factors"
        self._make_rule_factor(factors_dir)
        agg = SignalAggregator(factors_dir=factors_dir, use_registry=False)
        result = agg.aggregate("0700.HK", synthetic_ohlcv, {})
        assert isinstance(result, AggregatedSignal)
        assert 0.0 <= result.confidence_pct <= 1.0
```

- [ ] **Step 3.2: Run tests to verify they fail**

```bash
pytest tests/test_signal_aggregator.py::TestSignalAggregatorStacking -v 2>&1 | head -20
```

Expected: `TypeError: __init__() got an unexpected keyword argument 'aggregation_method'`

- [ ] **Step 3.3: Modify `engine/signal_aggregator.py`**

**3.3a — Add import at top of file (after existing imports):**

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from engine.meta_aggregator import MetaAggregator
```

**3.3b — Modify `__init__` signature and body:**

Replace the existing `__init__` signature and its body up to (but not including) `# ── 内部：加载因子列表`) with:

```python
def __init__(
    self,
    factors_dir: Optional[Path] = None,
    min_sharpe_weight: float = 0.0,
    max_factors: int = 20,
    use_registry: bool = True,
    aggregation_method: str = "vote",
    meta_dir: Optional[Path] = None,
):
    """
    Args:
        factors_dir:          factor_*.pkl 所在目录（默认 data/factors/）
        min_sharpe_weight:    参与投票的最低 sharpe 阈值（默认 0，全部参与）
        max_factors:          最多加载的因子数量（取编号最大的 N 个）
        use_registry:         是否使用因子注册表过滤（默认 True，只加载 active 因子）
        aggregation_method:   "vote"（默认，Sharpe 加权投票）或 "stacking"（meta-learner）
        meta_dir:             meta-model 目录（默认 data/meta/）
    """
    self.factors_dir = factors_dir or (
        Path(__file__).parent.parent / "data" / "factors"
    )
    self.min_sharpe_weight = min_sharpe_weight
    self.max_factors = max_factors
    self._use_registry = use_registry
    self._registry = None
    self._strategy_modules: dict = {}
    self._aggregation_method = aggregation_method
    self._meta_dir = meta_dir or (Path(__file__).parent.parent / "data" / "meta")
    self._meta_cache: dict[str, Optional["MetaAggregator"]] = {}
```

**3.3c — Add `_load_meta()` method** (add after `_filter_by_registry`, before `aggregate`):

```python
def _load_meta(self, ticker: str) -> Optional["MetaAggregator"]:
    """Lazily load meta-model for ticker. Returns None if unavailable."""
    if ticker not in self._meta_cache:
        try:
            from engine.meta_aggregator import MetaAggregator
            self._meta_cache[ticker] = MetaAggregator.load(ticker, self._meta_dir)
        except Exception as e:
            logger.debug("meta-model 加载异常: %s", e)
            self._meta_cache[ticker] = None
    return self._meta_cache[ticker]
```

**3.3d — Modify `aggregate()` method.** Find these two consecutive lines in `aggregate()`:

```python
consensus = 1 if bull_w >= bear_w else 0
confidence = (max(bull_w, bear_w) / total_w) if total_w > 0 else 0.5
```

Insert the stacking branch **immediately after** them, before the `# 分别统计 ML 和规则策略` comment:

```python
# Stacking override (only when votes are available)
if self._aggregation_method == "stacking" and votes:
    meta = self._load_meta(ticker)
    if meta is not None:
        try:
            base_signals = {v["strategy_name"]: v["signal"] for v in votes}
            feat = meta.build_feature_vector(base_signals, data)
            consensus, confidence = meta.predict(feat)
        except Exception as e:
            logger.warning(
                "%s: stacking predict 失败，fallback 到 vote: %s", ticker, e,
                extra={"ticker": ticker}
            )
    else:
        logger.debug("%s: meta-model 不存在，fallback 到 vote", ticker,
                     extra={"ticker": ticker})
```

- [ ] **Step 3.4: Run tests to verify they pass**

```bash
pytest tests/test_signal_aggregator.py -v
```

Expected: all existing tests PASS, all 3 new `TestSignalAggregatorStacking` tests PASS

- [ ] **Step 3.5: Commit**

```bash
git add engine/signal_aggregator.py tests/test_signal_aggregator.py
git commit -m "feat: add stacking aggregation_method to SignalAggregator with fallback"
```

---

## Task 4: Modify PositionAnalyzer + update engine/__init__.py + config.yaml

**Files:**
- Modify: `engine/position_analyzer.py`
- Modify: `engine/__init__.py`
- Modify: `config.yaml`

- [ ] **Step 4.1: Modify `engine/position_analyzer.py`**

In `PositionAnalyzer.__init__`, replace:

```python
self._aggregator = SignalAggregator(
    factors_dir=self.factors_dir,
    max_factors=max_factors,
)
```

with:

```python
stacking_cfg = config.get("stacking", {})
_agg_method = stacking_cfg.get("aggregation_method", "vote")
_meta_dir_str = stacking_cfg.get("meta_dir", "data/meta")
_meta_dir = Path(_meta_dir_str) if Path(_meta_dir_str).is_absolute() else (
    Path(__file__).parent.parent / _meta_dir_str
)
self._aggregator = SignalAggregator(
    factors_dir=self.factors_dir,
    max_factors=max_factors,
    aggregation_method=_agg_method,
    meta_dir=_meta_dir,
)
```

Also add `from pathlib import Path` if not already imported at the top of the file (it is already imported via `from pathlib import Path` in the existing file).

- [ ] **Step 4.2: Update `engine/__init__.py`**

Add `MetaAggregator` to the imports and `__all__`:

```python
from .meta_aggregator import MetaAggregator
```

And in `__all__`:

```python
"MetaAggregator",
```

- [ ] **Step 4.3: Add `stacking` block to `config.yaml`**

After the `sentiment_weight: 0.03` line, add:

```yaml
stacking:
  aggregation_method: vote    # "vote" | "stacking"
  meta_dir: data/meta
  label_days: 5
  n_splits: 5
```

- [ ] **Step 4.4: Run existing tests to verify no regressions**

```bash
pytest tests/test_signal_aggregator.py tests/test_integration.py tests/test_position_manager.py -v
```

Expected: all PASS (PositionAnalyzer defaults to `aggregation_method="vote"`, no behavior change)

- [ ] **Step 4.5: Commit**

```bash
git add engine/position_analyzer.py engine/__init__.py config.yaml
git commit -m "feat: wire stacking config into PositionAnalyzer; export MetaAggregator"
```

---

## Task 5: Add `_train_meta_model()` to main.py

**Files:**
- Modify: `main.py`

- [ ] **Step 5.1: Add `_train_meta_model()` function to `main.py`**

Add this function after the `_print_portfolio_summary()` function (around line 750 area, before the `_resolve_artifact` function):

```python
# ══════════════════════════════════════════════════════════════════
#  Meta-model 训练（Stacking 第二层）
# ══════════════════════════════════════════════════════════════════

def _train_meta_model(
    ticker: str,
    data: pd.DataFrame,
    config: dict,
    factors_dir: Path,
    meta_dir: Path,
) -> None:
    """
    训练并保存 meta-model（Stacking 第二层 Logistic Regression）。
    在基础策略训练完成、factor_*.pkl 已保存后调用。
    失败时记录 warning，不中断主流程。
    """
    try:
        from engine.meta_aggregator import MetaAggregator
        from engine.signal_aggregator import SignalAggregator

        agg = SignalAggregator(factors_dir=factors_dir, use_registry=False)
        artifacts = agg._load_factors()

        if len(artifacts) < 2:
            logger.warning(
                "meta-model 训练跳过：active 因子数量不足（需要 >= 2，当前 %d）",
                len(artifacts),
                extra={"ticker": ticker},
            )
            return

        stacking_cfg = config.get("stacking", {})
        n_splits = int(stacking_cfg.get("n_splits", 5))
        label_days = int(stacking_cfg.get("label_days", 5))

        ma = MetaAggregator(meta_dir=meta_dir)
        metrics = ma.train(
            ticker=ticker,
            data=data,
            artifacts=artifacts,
            config=config,
            n_splits=n_splits,
            label_days=label_days,
        )
        ma.save(ticker)
        logger.info(
            "meta-model 训练完成",
            extra={"ticker": ticker, **metrics},
        )
    except Exception as e:
        logger.warning(
            "meta-model 训练失败，不影响基础策略: %s", e,
            extra={"ticker": ticker},
        )
```

- [ ] **Step 5.2: Call `_train_meta_model()` from `main()`**

In `main()`, `ticker` is first defined at line ~1563 (`ticker = config.get('ticker', '0700.HK').upper()`).
Place the `_train_meta_model()` call **immediately after that line**, before `_print_portfolio_summary`:

```python
ticker = config.get('ticker', '0700.HK').upper()   # already exists

# ── Meta-model 训练（Stacking）─────────────────────────────────
_train_meta_model(
    ticker=ticker,
    data=hist_data,
    config=config,
    factors_dir=Path(__file__).parent / "data" / "factors",
    meta_dir=Path(__file__).parent / "data" / "meta",
)

result = {   # already exists, keep as-is
    ...
```

- [ ] **Step 5.3: Call `_train_meta_model()` from `train_portfolio_tickers()`**

In `train_portfolio_tickers()`, after the per-ticker `step2_train()` succeeds and `results.append(...)` adds an "ok" entry, add:

```python
# Meta-model 训练（per-ticker，训练完基础策略后立即训练）
meta_dir = Path(__file__).parent / "data" / "meta"
_train_meta_model(
    ticker=ticker,
    data=hist_data,
    config=config,
    factors_dir=ticker_factors_dir,
    meta_dir=meta_dir,
)
```

Place this after `results.append({...})` and before `logger.info("规则训练完成: %s", ...)`.

- [ ] **Step 5.4: Run smoke test to verify main.py parses without errors**

```bash
python3 -c "import main; print('main.py OK')"
```

Expected: `main.py OK`

- [ ] **Step 5.5: Commit**

```bash
git add main.py
git commit -m "feat: add _train_meta_model() to main.py; call after base strategy training"
```

---

## Task 6: Integration test + full test suite

**Files:**
- Modify: `tests/test_integration.py`

- [ ] **Step 6.1: Write integration test**

Add the following class to `tests/test_integration.py`:

```python
class TestMetaAggregatorEndToEnd:
    """Offline end-to-end: train → save → load → predict."""

    def _make_rule_artifact(self, strategy_name: str) -> dict:
        return {
            "meta": {"name": strategy_name, "params": {}, "feat_cols": []},
            "model": None,
            "sharpe_ratio": 1.2,
            "config": {},
        }

    def test_train_save_load_predict_pipeline(self, synthetic_ohlcv, tmp_path):
        """Full offline pipeline: train on synthetic data, save, load, predict."""
        from engine.meta_aggregator import MetaAggregator

        meta_dir = tmp_path / "meta"
        artifacts = [
            self._make_rule_artifact("ma_crossover"),
            self._make_rule_artifact("rsi_reversion"),
        ]

        # Train
        ma = MetaAggregator(meta_dir=meta_dir)
        metrics = ma.train(
            ticker="0700.HK",
            data=synthetic_ohlcv,
            artifacts=artifacts,
            config={},
            n_splits=2,
            label_days=5,
        )
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert metrics["n_samples"] > 0

        # Save
        path = ma.save("0700.HK")
        assert path.exists()

        # Load
        loaded = MetaAggregator.load("0700.HK", meta_dir)
        assert loaded is not None
        assert loaded._strategy_names == ma._strategy_names

        # Predict
        feat = loaded.build_feature_vector(
            {name: 1 for name in loaded._strategy_names}, synthetic_ohlcv
        )
        signal, proba = loaded.predict(feat)
        assert signal in (0, 1)
        assert 0.0 <= proba <= 1.0

    def test_signal_aggregator_stacking_e2e(self, synthetic_ohlcv, tmp_path):
        """SignalAggregator in stacking mode uses trained meta-model correctly."""
        import joblib
        from engine.meta_aggregator import MetaAggregator
        from engine.signal_aggregator import SignalAggregator, AggregatedSignal

        # Setup: factor + meta-model
        factors_dir = tmp_path / "factors"
        meta_dir = tmp_path / "meta"
        factors_dir.mkdir()
        joblib.dump({
            "meta": {"name": "ma_crossover", "params": {}, "feat_cols": []},
            "model": None, "sharpe_ratio": 1.5, "config": {},
        }, factors_dir / "factor_0001.pkl")

        # Train and save meta-model
        ma = MetaAggregator(meta_dir=meta_dir)
        artifacts = [
            {"meta": {"name": "ma_crossover", "params": {}, "feat_cols": []},
             "model": None, "sharpe_ratio": 1.5, "config": {}},
            {"meta": {"name": "rsi_reversion", "params": {}, "feat_cols": []},
             "model": None, "sharpe_ratio": 1.2, "config": {}},
        ]
        ma.train("0700.HK", synthetic_ohlcv, artifacts, config={}, n_splits=2, label_days=5)
        ma.save("0700.HK")

        # Aggregate with stacking
        agg = SignalAggregator(
            factors_dir=factors_dir,
            aggregation_method="stacking",
            meta_dir=meta_dir,
            use_registry=False,
        )
        result = agg.aggregate("0700.HK", synthetic_ohlcv, {"ticker": "0700.HK"})
        assert isinstance(result, AggregatedSignal)
        assert result.consensus_signal in (0, 1)
        assert 0.0 <= result.confidence_pct <= 1.0
```

- [ ] **Step 6.2: Run integration tests**

```bash
pytest tests/test_integration.py::TestMetaAggregatorEndToEnd -v
```

Expected: both tests PASS

- [ ] **Step 6.3: Run full test suite**

```bash
pytest tests/ -v --tb=short 2>&1 | tail -20
```

Expected: all tests PASS (count should be previous 223 + new ~16 = ~239)

- [ ] **Step 6.4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add MetaAggregator end-to-end integration tests"
```

---

## Task 7: Final validation

- [ ] **Step 7.1: Verify daily_run still works with vote mode**

```bash
python3 daily_run.py --dry-run --skip-notify --skip-sentiment 2>&1 | tail -10
```

Expected: prints recommendations without errors (stacking config is `vote` by default)

- [ ] **Step 7.2: Verify config.yaml unchanged after test run**

```bash
pytest tests/ -v -q && head -5 config.yaml
```

Expected: `ticker: 0700.HK` still on line 1 (config.yaml not mutated by tests)

- [ ] **Step 7.3: Final commit with updated docs**

```bash
git add docs/superpowers/specs/2026-04-07-ensemble-stacking-design.md
git commit -m "docs: mark ensemble/stacking spec as implemented"
```
