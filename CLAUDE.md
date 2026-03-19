# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

腾讯控股 (0700.hk) 股票智能分析系统 - Automated stock analysis with multi-strategy backtesting, hyperparameter optimization, signal prediction, and position management.

## Running the Project

```bash
python3 main.py                    # Full analysis with random search
python3 main.py --use-optuna       # Use Optuna Bayesian optimization
python3 main.py --use-optuna --optuna-trials 100
python3 main.py --skip-train       # Skip training, use existing factors
python3 main.py --n-days 5         # Prediction horizon
```

## Architecture

### Core Flow (main.py)
1. **Data Check** - Load local CSV or download from yfinance/yahooquery
2. **Strategy Search** - Random/Optuna hyperparameter search across all strategies
3. **Validation** - Out-of-sample testing with multi-dimensional thresholds
4. **Prediction** - Generate signals for next N trading days
5. **Reporting** - Save reports to `data/reports/`, optionally send to Feishu

### Key Modules

| Module | Purpose |
|--------|---------|
| `main.py` | Entry point, orchestrates full pipeline |
| `analyze_factor.py` | Core backtesting engine, strategy discovery, hyperparam search |
| `validate_strategy.py` | Out-of-sample and Walk-Forward validation |
| `backtest_vectorbt.py` | Optional Vectorbt-based backtesting |
| `fetch_data.py` | Data download with 2-day stale check |
| `position_manager.py` | Position state and trade suggestions |
| `sentiment_analysis.py` | News sentiment scoring with caching |
| `google_trends.py` | Google Trends热度数据 with caching |
| `feishu_notify.py` | Feishu webhook notifications |
| `optimize_with_optuna.py` | Bayesian hyperparameter optimization |
| `train_multi_stock.py` | Loads HSI stocks for multi-stock training |

### Strategy Interface

All strategies in `strategies/` must expose **both** functions:
```python
def run(data: pd.DataFrame, config: dict) -> (signal: pd.Series, model, meta: dict)
def predict(model, data: pd.DataFrame, config: dict, meta: dict) -> pd.Series
```
- `signal`: int Series (1=long, 0=flat), aligned to data.index
- `model`: serializable object or None (rule strategies return None)
- `meta`: `{"name": str, "params": dict, "feat_cols": list, "indicators": dict}`
- `predict()`: for ML strategies, uses the trained model to infer on new data without refitting; for rule strategies, re-runs `run()` (no model needed, `model` arg ignored)

**Rule strategies must not use `ffill()` signal smoothing** — use explicit state machines (enter/hold/exit logic) to avoid inflated holding periods and win rates.

### Strategy Discovery

Strategies are auto-discovered via `_discover_strategies()` in `analyze_factor.py`:
- Reads from `config.yaml` → `strategy_training` (single/multi/custom lists)
- Imports modules dynamically from `strategies/` package
- Module must have both `run` function and `NAME` attribute

### Strategy Training Types

| Type | Training Data | Validation Data |
|------|--------------|-----------------|
| `single` | Target stock only | Target stock (lookback_months) |
| `multi` | Multiple HSI stocks | Target stock (lookback_months) |
| `custom` | Strategy-defined | Strategy-defined |

**Rule-based strategies** (RSI, MACD, Bollinger, etc.) → use `single` training
**ML strategies** (XGBoost, LightGBM) → use `multi` training with HSI stocks

### Look-Ahead Bias Prevention

All backtesting is designed to be free of look-ahead bias:

1. **Signal execution delay**: Both `analyze_factor.backtest()` and `backtest_vectorbt()` apply `signal.shift(1)` internally — signals generated on day T execute on day T+1. Do not shift signals before passing them in.
2. **Train/val split**: `run_trial()` trains on `[train_start, val_start)` and validates on `[val_start, end]`. The validation set is **never** used during training.
3. **ML prediction**: `predict()` is called on the validation set with the already-fitted model — never re-fit on validation data.
4. **Multi-stock optimization** (`train_multi_stock.optimize_multi_stock_params`): Raw price data is split by time **before** `create_multi_stock_dataset()` is called, so `shift(-label_period)` labels in the test set cannot leak into training features.
5. **Rule strategy signals**: State-machine loops (not `ffill()`) — only explicit entry/exit conditions change position.

### Walk-Forward Validation

`validate_strategy.py` provides:
- `out_of_sample_test()`: Trains on 12 months, tests on 3 months
- `walk_forward_analysis()`: Rolling windows, all windows (including losing ones) are included in win-rate statistics
- `generate_test_report()`: Generates markdown report combining both

### Standalone Modules

`train_multi_stock.py` is a **standalone script** — it is not called by `main.py` at runtime. Its `load_all_hsi_data()` is used indirectly via `analyze_factor._load_multi_stock_data()`. Run it directly for multi-stock dataset exploration:
```bash
python3 train_multi_stock.py   # optimize params with Optuna (standalone)
```

### Multi-Stock Data Cache

`analyze_factor._load_multi_stock_data()` uses `joblib.Memory` (disk cache at `data/cache/`) so multiple Optuna worker processes share the same cached data and avoid repeated disk I/O.

### tsfresh Parallelism

`strategies/tsfresh_features.py` limits `n_jobs` to `cpu_count // 2` to prevent process explosion when Optuna runs parallel trials alongside tsfresh's internal parallelism.



Configured via `config.yaml` → `backtest_engine`:
- `native` (recommended): Pure Python backtester in `analyze_factor.py:backtest()` - handles HK fees (0.088% + 0.1% stamp duty)
- `vectorbt`: Vectorbt-based backtesting - results differ slightly

### Multi-Dimensional Threshold Validation

Strategies must pass all thresholds to be saved:
- `cum_return > min_return`
- `sharpe_ratio > min_sharpe_ratio`
- `max_drawdown >= max_drawdown` (negative value)
- `total_trades >= min_total_trades`

## Configuration

### config.yaml
| Key | Default | Description |
|-----|---------|-------------|
| `ticker` | 0700.hk | Stock code |
| `period` | 5y | Backtest period |
| `lookback_months` | 3 | Validation window |
| `train_years` | 5 | Training window |
| `backtest_engine` | vectorbt | native or vectorbt |
| `use_optuna` | true | Enable Bayesian optimization |
| `optuna_trials` | 100 | Optuna search iterations |
| `max_tries` | 300 | Random search iterations |
| `min_return` | 0.10 | Validation threshold |
| `min_sharpe_ratio` | 1.0 | Validation threshold |
| `max_drawdown` | -0.15 | Maximum drawdown (negative) |
| `min_total_trades` | 5 | Minimum trade count |

### keys.yaml (not committed)
```yaml
alpha_vantage_key: null
feishu_webhook: https://open.feishu.cn/...
```

## Data Storage

- Historical data: `data/historical/*.csv`
- Factors: `data/factors/factor_*.pkl`
- Reports: `data/reports/report_*.md`
- Trending data: `data/trends/tencent_trends.csv`
- Sentiment cache: `data/sentiment/sentiment_cache.csv`
- HSI stocks: `data/historical/*_HK_*.csv` (via `fetch_hsi_stocks.py`)
- Multi-stock cache: `data/cache/` (joblib.Memory, auto-managed)

## Testing

```bash
python3 smoke_test.py   # Offline smoke test — no network required, uses synthetic data
```

Covers: all major rule strategies, `analyze_factor.backtest()`, and the `predict()` interface.

## ML Strategies

| Strategy | Model | Features | Training Type |
|----------|-------|----------|---------------|
| `xgboost_enhanced` | XGBoost | 技术指标 (pandas) | multi |
| `xgboost_enhanced_tsfresh` | XGBoost | 技术指标 + tsfresh | multi |
| `xgboost_enhanced_ta_tsfresh` | XGBoost | ta-lib 技术指标 + tsfresh | multi |
| `lightgbm_enhanced` | LightGBM | 技术指标 (pandas) | multi |
| `lightgbm_enhanced_tsfresh` | LightGBM | 技术指标 + tsfresh | multi |
| `lightgbm_enhanced_ta_tsfresh` | LightGBM | ta-lib 技术指标 + tsfresh | multi |

### tsfresh Integration

tsfresh (Time Series Feature extraction) automatically extracts thousands of features from time series data:

- **Automatic feature extraction**: Statistical, trend, seasonality, FFT, entropy features
- **Rolling window**: Configurable windows (default 10, 20 days)
- **Feature selection**: FDR correction (p < 0.05) to filter significant features
- **Hybrid mode**: `xgboost_enhanced` can use `use_tsfresh_features: true` to combine technical indicators + tsfresh

**Config example** (`config.yaml`):
```yaml
ml_strategies:
  xgboost_enhanced:
    use_tsfresh_features: false  # set true to add tsfresh features
  xgboost_enhanced_tsfresh:
    use_tsfresh_features: true
    tsfresh_window_sizes: [10, 20]
```

## Dependencies

- pandas, numpy, scikit-learn - data processing
- xgboost, lightgbm - ML strategies
- vectorbt - optional backtesting engine
- optuna - Bayesian optimization
- pytrends - Google Trends
- tsfresh, pyts - automatic time series feature extraction
- ta - technical analysis library (200+ indicators, wraps TA-Lib C library)
- yfinance, akshare - market data

### ta-lib Integration

ta-lib (`pip install ta`) provides standardized technical indicator calculations:

- **strategies/indicators.py** - Central module wrapping ta-lib
- **xgboost_enhanced** supports `use_ta_lib: true` config to enable
- Falls back to pandas implementation if ta not available
- Same feature output as pandas version for compatibility
