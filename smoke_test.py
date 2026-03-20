"""
Offline smoke test for the stock_analyze project.
Creates a synthetic OHLCV price series and exercises the core pipeline:
  data → strategy.run() → backtest → signal values

Run with: python3 smoke_test.py
"""
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ── Synthetic data ─────────────────────────────────────────────────
np.random.seed(42)
n = 300
dates = pd.bdate_range(end=datetime.today(), periods=n)
close = np.cumprod(1 + np.random.normal(0, 0.01, n)) * 100
data = pd.DataFrame({
    'Open':   close * (1 + np.random.normal(0, 0.002, n)),
    'High':   close * (1 + np.abs(np.random.normal(0, 0.005, n))),
    'Low':    close * (1 - np.abs(np.random.normal(0, 0.005, n))),
    'Close':  close,
    'Volume': np.random.randint(500_000, 5_000_000, n).astype(float),
}, index=dates)

print(f"Synthetic data: {len(data)} rows, {data.index[0].date()} ~ {data.index[-1].date()}")
print(data[['Open', 'High', 'Low', 'Close', 'Volume']].tail(3).to_string())
print()

# ── Strategy smoke tests ────────────────────────────────────────────
from strategies.ma_crossover      import run as ma_run
from strategies.rsi_reversion     import run as rsi_run
from strategies.bollinger_breakout import run as bb_run
from strategies.macd_rsi_combo    import run as macd_run
from strategies.atr_breakout      import run as atr_run
from strategies.stochastic_oscillator import run as stoch_run
from strategies.vwap_momentum     import run as vwap_run
from strategies.volume_price_trend import run as vpt_run
from strategies.rsi_divergence    import run as rsidiv_run

strategy_tests = [
    ("ma_crossover",          ma_run),
    ("rsi_reversion",         rsi_run),
    ("bollinger_breakout",    bb_run),
    ("macd_rsi_combo",        macd_run),
    ("atr_breakout",          atr_run),
    ("stochastic_oscillator", stoch_run),
    ("vwap_momentum",         vwap_run),
    ("volume_price_trend",    vpt_run),
    ("rsi_divergence",        rsidiv_run),
]

cfg = {}
passed = True
for name, fn in strategy_tests:
    try:
        sig, model, meta = fn(data, cfg)
        assert isinstance(sig, pd.Series), "signal not a Series"
        assert set(sig.unique()).issubset({0, 1}), f"unexpected values: {sig.unique()}"
        assert meta.get("name") == name.replace("_run", ""), f"meta name mismatch"
        print(f"  ✅ {name:30s}  hold={int(sig.sum()):4d}/{len(sig)}d  model={'yes' if model else 'no'}")
    except Exception as e:
        print(f"  ❌ {name:30s}  FAIL: {e}")
        passed = False

# ── Backtest smoke test ─────────────────────────────────────���───────
print()
try:
    from analyze_factor import backtest
    sig_ma, _, _ = ma_run(data, {})
    result = backtest(data, sig_ma, {'initial_capital': 100_000, 'fees_rate': 0.00088, 'stamp_duty': 0.001, 'lookback_months': 3})
    assert 'cum_return' in result
    assert 'sharpe_ratio' in result
    print(f"  ✅ analyze_factor.backtest()  cum_return={result['cum_return']:.2%}  sharpe={result.get('sharpe_ratio', float('nan')):.4f}")
except Exception as e:
    print(f"  ❌ analyze_factor.backtest()  FAIL: {e}")
    passed = False

# ── predict() interface smoke test ─────────────────────────────────
print()
from strategies.ma_crossover import predict as ma_predict
try:
    pred_sig = ma_predict(None, data, {}, {})
    assert isinstance(pred_sig, pd.Series)
    assert set(pred_sig.unique()).issubset({0, 1})
    print(f"  ✅ predict() interface (ma_crossover)  ok")
except Exception as e:
    print(f"  ❌ predict() interface (ma_crossover)  FAIL: {e}")
    passed = False

# ── ML strategy smoke tests ────────────────────────────────────────
print()
print("ML strategy smoke tests (synthetic, n=300)")

ml_tests = []
try:
    from strategies.xgboost_enhanced import run as xgb_run, NAME as xgb_name
    ml_tests.append(("xgboost_enhanced", xgb_run, xgb_name))
except ImportError as e:
    print(f"  ⚠️  xgboost_enhanced import skipped: {e}")

try:
    from strategies.lightgbm_enhanced import run as lgbm_run, NAME as lgbm_name
    ml_tests.append(("lightgbm_enhanced", lgbm_run, lgbm_name))
except ImportError as e:
    print(f"  ⚠️  lightgbm_enhanced import skipped: {e}")

ml_cfg = {
    'test_days': 5,
    'label_period': 1,
    'xgb_n_estimators': 30,
    'xgb_max_depth': 3,
    'xgb_learning_rate': 0.1,
    'lgbm_n_estimators': 30,
    'lgbm_max_depth': 3,
    'lgbm_learning_rate': 0.1,
    'lgbm_num_leaves': 15,
}

for name, fn, expected_name in ml_tests:
    try:
        sig, model, meta = fn(data, ml_cfg)
        assert isinstance(sig, pd.Series), "signal not a Series"
        assert set(sig.dropna().unique()).issubset({0, 1}), f"unexpected values: {sig.unique()}"
        assert meta.get("name") == expected_name, f"meta name mismatch: {meta.get('name')}"
        print(f"  ✅ {name:30s}  hold={int(sig.sum()):4d}/{len(sig)}d  model={'yes' if model else 'no'}")
    except Exception as e:
        print(f"  ❌ {name:30s}  FAIL: {e}")
        passed = False

# ── run_factor_analysis smoke test ────────────────────────────────
print()
try:
    from analyze_factor import run_factor_analysis
    sig_fa, _, _ = ma_run(data, {})
    fa_result = run_factor_analysis(data, sig_fa, {})
    assert 'ic_mean' in fa_result, "missing ic_mean"
    assert 'quintile_df' in fa_result, "missing quintile_df"
    assert 'decay_series' in fa_result, "missing decay_series"
    ic = fa_result.get('ic_mean', float('nan'))
    print(f"  ✅ run_factor_analysis()  ic_mean={ic:.4f}  icir={fa_result.get('icir', float('nan')):.4f}")
except Exception as e:
    print(f"  ❌ run_factor_analysis()  FAIL: {e}")
    passed = False

print()
print("=" * 50)
print(f"  RESULT: {'PASS ✅' if passed else 'FAIL ❌'}")
print("=" * 50)
sys.exit(0 if passed else 1)

