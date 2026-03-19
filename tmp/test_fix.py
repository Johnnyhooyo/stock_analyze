import sys
sys.path.insert(0, '/Users/Enzo/PycharmProjects/stock_analyze')
import numpy as np
import pandas as pd
from strategies import xgboost_enhanced, lightgbm_enhanced

np.random.seed(42)
n = 200
dates = pd.date_range('2022-01-01', periods=n, freq='B')
close = 100 + np.cumsum(np.random.randn(n) * 0.5)
df = pd.DataFrame({
    'Open': close * 0.999, 'High': close * 1.005,
    'Low': close * 0.995, 'Close': close,
    'Volume': np.random.randint(1000000, 5000000, n).astype(float),
}, index=dates)

# --- XGBoost ---
cfg = {'xgb_n_estimators': 10, 'xgb_max_depth': 3, 'use_tsfresh_features': False}
sig, model, meta = xgboost_enhanced.run(df, cfg)
assert 'selected_tsfresh_cols' in meta, "missing selected_tsfresh_cols"
print(f"XGBoost run OK | test_acc={meta['test_acc']:.2%} | feat={meta['feat_count']}")

pred = xgboost_enhanced.predict(model, df.iloc[-40:], cfg, meta)
assert len(pred) == 40
print(f"XGBoost predict OK | signals={int(pred.sum())}")

# --- LightGBM ---
cfg2 = {'lgbm_n_estimators': 10, 'lgbm_max_depth': 3, 'use_tsfresh_features': False}
sig2, model2, meta2 = lightgbm_enhanced.run(df, cfg2)
assert 'selected_tsfresh_cols' in meta2, "missing selected_tsfresh_cols"
print(f"LightGBM run OK | test_acc={meta2['test_acc']:.2%} | feat={meta2['feat_count']}")

pred2 = lightgbm_enhanced.predict(model2, df.iloc[-40:], cfg2, meta2)
assert len(pred2) == 40
print(f"LightGBM predict OK | signals={int(pred2.sum())}")

# --- 验证训练集大小约为 80% ---
expected_train = int(len(sig) * 0.8)
assert abs(meta['train_size'] - expected_train) <= 5, f"train_size={meta['train_size']} expected~{expected_train}"
print(f"Train/test split OK | train={meta['train_size']} test={meta['test_size']}")

print("\n=== All tests passed! ===")
