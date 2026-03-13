"""Offline smoke test for the stock_analyze project.
Creates a synthetic price series, runs the factor test and plotting functions.
Run with: python3 smoke_test.py
"""
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from analyze_factor import test_factor
from visualize import plot_trades

# Create 30 days of synthetic closing prices with small noise
dates = pd.date_range(end=datetime.today(), periods=30)
np.random.seed(42)
prices = np.cumprod(1 + np.random.normal(0, 0.01, size=len(dates))) * 100

data = pd.DataFrame({'Close': prices}, index=dates)

print('Synthetic data created:')
print(data.head())

# Run factor analysis (should not error)
res_data, factor_path, total_return = test_factor(data)
print('test_factor returned:', type(res_data), factor_path, total_return)

# Run plotting (should produce trade_signals.png)
plot_path = plot_trades(res_data)
print('plot saved to:', plot_path)

