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

# ── Backtest smoke test ─────────────────────────────────────────────
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

# ── ATR 止损回测模拟 smoke test（Issue #9）──────────────────────────
print()
print("ATR stop-loss backtest simulation smoke tests (Issue #9)")

# 构造先涨后暴跌的测试数据（确保会触发 ATR 止损）
n_atr = 120
dates_atr = pd.bdate_range(end=datetime.today(), periods=n_atr)
prices_up   = np.linspace(100, 130, 60)
prices_down = np.linspace(129, 85, 60)
atr_closes  = np.concatenate([prices_up, prices_down])
atr_highs   = atr_closes * 1.003
atr_lows    = atr_closes * 0.997
data_atr    = pd.DataFrame({
    'Open':   atr_closes * 1.001,
    'High':   atr_highs,
    'Low':    atr_lows,
    'Close':  atr_closes,
    'Volume': np.ones(n_atr) * 1_000_000,
}, index=dates_atr)
# 全程持仓信号
sig_always_hold = pd.Series(1, index=dates_atr)

# test A: simulate_atr_stoploss 纯函数
try:
    from position_manager import simulate_atr_stoploss
    modified_sig = simulate_atr_stoploss(data_atr, sig_always_hold, atr_period=5, atr_multiplier=2.0)
    n_stops = (modified_sig == 0).sum()
    assert n_stops > 0, "ATR 止损未触发任何平仓信号"
    print(f"  ✅ simulate_atr_stoploss()  止损 bar 数={n_stops}/{n_atr}")
except Exception as e:
    print(f"  ❌ simulate_atr_stoploss()  FAIL: {e}")
    passed = False

# test B: native backtest 开启 ATR 止损 vs 关闭对比
try:
    cfg_on  = {'initial_capital': 100_000, 'fees_rate': 0.00088, 'stamp_duty': 0.001,
               'lookback_months': 3, 'risk_management': {
                   'simulate_in_backtest': True, 'use_atr_stop': True,
                   'atr_period': 5, 'atr_multiplier': 2.0, 'trailing_stop': True, 'cooldown_bars': 0}}
    cfg_off = {'initial_capital': 100_000, 'fees_rate': 0.00088, 'stamp_duty': 0.001,
               'lookback_months': 3, 'risk_management': {'simulate_in_backtest': False}}
    r_on  = backtest(data_atr, sig_always_hold.copy(), cfg_on)
    r_off = backtest(data_atr, sig_always_hold.copy(), cfg_off)
    dd_on  = r_on['max_drawdown']
    dd_off = r_off['max_drawdown']
    # 止损开启时：交易次数更多（有止损卖出）
    assert r_on['sell_cnt'] >= r_off['sell_cnt'], "开启止损后 sell_cnt 应 ≥ 关闭时"
    print(f"  ✅ native backtest ATR on/off  max_dd: on={dd_on:.2%}  off={dd_off:.2%}"
          f"  sell_cnt: on={r_on['sell_cnt']}  off={r_off['sell_cnt']}")
except Exception as e:
    print(f"  ❌ native backtest ATR on/off  FAIL: {e}")
    passed = False

# test C: vectorbt backtest 开启 ATR 止损
try:
    from backtest_vectorbt import backtest_vectorbt
    vbt_cfg_on  = {'initial_capital': 100_000, 'fees_rate': 0.00088, 'stamp_duty': 0.001,
                   'risk_management': {
                       'simulate_in_backtest': True, 'use_atr_stop': True,
                       'atr_period': 5, 'atr_multiplier': 2.0, 'trailing_stop': True, 'cooldown_bars': 0}}
    vbt_cfg_off = {'initial_capital': 100_000, 'fees_rate': 0.00088, 'stamp_duty': 0.001,
                   'risk_management': {'simulate_in_backtest': False}}
    vr_on  = backtest_vectorbt(data_atr, sig_always_hold.copy(), vbt_cfg_on)
    vr_off = backtest_vectorbt(data_atr, sig_always_hold.copy(), vbt_cfg_off)
    print(f"  ✅ vectorbt backtest ATR on/off  max_dd: on={vr_on['max_drawdown']:.2%}  off={vr_off['max_drawdown']:.2%}"
          f"  trades: on={vr_on['total_trades']}  off={vr_off['total_trades']}")
except Exception as e:
    print(f"  ❌ vectorbt backtest ATR on/off  FAIL: {e}")
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

