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

All strategies in `strategies/` must expose:
```python
def run(data: pd.DataFrame, config: dict) -> (signal: pd.Series, model, meta: dict)
```
- `signal`: int Series (1=long, 0=flat), aligned to data.index
- `model`: serializable object or None
- `meta`: `{"name": str, "params": dict, "feat_cols": list, "indicators": dict}`

### Strategy Training Types (analyze_factor.py)

| Type | Training Data | Validation Data |
|------|--------------|-----------------|
| `single` | Target stock only | Target stock (lookback_months) |
| `multi` | Multiple HSI stocks | Target stock (lookback_months) |
| `custom` | Strategy-defined | Strategy-defined |

Configured in `config.yaml` under `strategy_training`.

### Backtesting Engine

Configured via `config.yaml` → `backtest_engine`:
- `native` (default): Pure Python backtester in `analyze_factor.py:backtest()` - handles HK fees (0.088% + 0.1% stamp duty)
- `vectorbt`: Vectorbt-based backtesting - results differ slightly

**Recommendation**: Use `native` engine for accurate Hong Kong market fee simulation.

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
| `optuna_trials` | 50 | Optuna search iterations |
| `max_tries` | 300 | Random search iterations |
| `min_return` | 0.10 | Validation threshold |
| `min_sharpe_ratio` | 1.0 | Validation threshold |

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

## ML Strategies

| Strategy | Model | Features | Training Type |
|----------|-------|----------|---------------|
| `xgboost_enhanced` | XGBoost | 95 (RSI, MACD, Bollinger, KDJ, ATR, OBV, returns) | multi |
| `xgboost_enhanced_tsfresh` | XGBoost | 技术指标 + tsfresh 自动特征 (~7000+) | multi |
| `lightgbm_enhanced` | LightGBM | 95 (same features) | multi |
| `tsfresh_xgboost` | XGBoost | tsfresh auto-features only | multi |

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
- yfinance, akshare - market data
