# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

港股智能分析系统 — 自动化多策略回测、超参数优化、信号预测与持仓管理。默认标的腾讯控股 (0700.HK)，支持 HSI 全部成分股。

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
1. **Data Check** — Load local CSV or download via vendor chain (yahooquery → yfinance → akshare …)
2. **Strategy Search** — Random/Optuna hyperparameter search across all strategies
3. **Validation** — Out-of-sample testing with multi-dimensional thresholds
4. **Prediction** — Generate signals for next N trading days
5. **Reporting** — Save reports to `data/reports/`, optionally send to Feishu

### Key Modules

| Module | Purpose |
|--------|---------|
| `main.py` | Entry point, orchestrates full pipeline |
| `analyze_factor.py` | Core backtesting engine, strategy discovery, hyperparam search |
| `validate_strategy.py` | Out-of-sample and Walk-Forward validation |
| `backtest_vectorbt.py` | Optional Vectorbt-based backtesting |
| `fetch_data.py` | **Backward-compatible shim** → delegates to `data/` package |
| `fetch_hsi_stocks.py` | **Backward-compatible shim** → delegates to `data/` package |
| `position_manager.py` | Position state, ATR stop-loss, trade suggestions |
| `sentiment_analysis.py` | News sentiment scoring with caching |
| `google_trends.py` | Google Trends 热度数据 with caching |
| `feishu_notify.py` | Feishu webhook notifications |
| `optimize_with_optuna.py` | Bayesian hyperparameter optimization |
| `train_multi_stock.py` | Loads HSI stocks for multi-stock training |
| `oms.py` | Broker API integration for live order submission |
| `visualize.py` | Strategy plotting and chart generation |
| `easy_quptation.py` | 实时行情工具 (easyquotation wrapper, standalone) |
| `time_kline.py` | 港股分时 K 线获取 (standalone) |

### Data Package (`data/`)

数据模块经过 Phase 2 重构后采用分层架构：

```
data/
├── __init__.py              # 包入口 (导出所有公开 API)
├── manager.py               # DataManager — 核心入口：vendor 链 → 校验 → 质量 → 存储
├── vendor_base.py           # DataVendor 抽象基类 + fetch_with_retry() 统一重试
├── vendors/                 # 各数据源适配器
│   ├── yfinance_vendor.py   # yfinance (单次请求，重试由基类处理)
│   ├── yahooquery_vendor.py # yahooquery
│   ├── pandas_datareader_vendor.py
│   ├── akshare_vendor.py    # akshare 中文列名映射
│   └── alpha_vantage_vendor.py
├── schemas.py               # OHLCV 列归一化 + schema 校验 (含 Adj Close)
├── calendar.py              # 港股交易日历 (纯 exchange_calendars XHKG，无硬编码假期)
├── quality.py               # 数据质量管道 (检测 + 自动修复)
├── storage.py               # 存储后端 (CSV / Parquet, 含迁移工具)
├── rate_limiter.py          # per-vendor 令牌桶限速器
├── config.py                # DataConfig 配置模型 (从 config.yaml 加载)
└── hsi_stocks.py            # HSI 成分股列表
```

**使用方式**：

```python
# 新 API（推荐）
from data import DataManager
mgr = DataManager()
df, path = mgr.download("0700.HK", period="3y")

# 纯读取（不触发网络请求）
df = mgr.load("0700.HK", period="3y")

```python
from data import DataManager
mgr = DataManager()
df, path = mgr.download("0700.HK", period="3y")
```

**关键特性**：
- **Vendor 链**：数据源按 config.yaml 优先级自动回退
- **统一重试**：`DataVendor.fetch_with_retry()` 基类模板方法，指数退避 + rate limiting
- **Rate Limiter**：per-vendor 令牌桶限速，防止批量下载触发限流
- **exchange_calendars**：纯 XHKG 日历（不再维护硬编码假期表）
- **Schema 校验**：自动列名归一化 (中英文同义词 + Adj Close) + OHLCV 校验
- **质量管道**：检测 + 自动修复（OHLC clamp / 负 Volume → NaN / 缺口前值填充）
- **存储后端**：支持 CSV / Parquet，`AutoBackend` 读取时自动检测格式
- **并发下载**：`download_hsi_incremental()` 支持 `ThreadPoolExecutor` 并发（config.batch_max_workers）
- **纯读取接口**：`DataManager.load()` 不触发网络请求，供策略模块直接调用
- **原子写入**：tempfile → os.replace，防止中断损坏文件
- **元数据追踪**：每个数据文件附带 `.meta.json` (来源/时间戳/SHA-256/last_bar_date)
- **高效缓存判断**：`_is_stale()` 优先读 `.meta.json`，避免全量解析 CSV

### Strategy Interface

All strategies in `strategies/` must expose **both** functions and a `NAME` attribute:
```python
NAME: str = "strategy_name"   # Must match config key

def run(data: pd.DataFrame, config: dict) -> (signal: pd.Series, model, meta: dict)
def predict(model, data: pd.DataFrame, config: dict, meta: dict) -> pd.Series
```
- `signal`: int Series (1=long, 0=flat), aligned to data.index
- `model`: serializable object or None (rule strategies return None)
- `meta`: `{"name": str, "params": dict, "feat_cols": list, "indicators": dict}`
- `predict()`: for ML strategies, uses the trained model to infer on new data without refitting; for rule strategies, re-runs `run()` (no model needed, `model` arg ignored)

**Rule strategies must not use `ffill()` signal smoothing** — use explicit state machines (enter/hold/exit logic) to avoid inflated holding periods and win rates.

**Strategy parameter search**: `run()` receives a `config` dict with strategy-specific params sampled by Optuna/random search. `meta["params"]` stores the final selected params.

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

**Rule-based strategies** (RSI, MACD, Bollinger, KDJ, ATR, VWAP, etc.) → use `single` training on target stock only. Includes: `bollinger_rsi_trend`, `macd_rsi_trend`, `rsi_divergence`, `stochastic_oscillator`, `vwap_momentum`, `atr_breakout`, `volume_price_trend`, `ma_crossover`, `rsi_reversion`, `kdj_obv`, `kdj_pvt`, `rsi_obv`, `rsi_pvt`, `bollinger_breakout`, `macd_rsi_combo`, `rsi_drawdown_0225` (with stop-loss logic).

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

`easy_quptation.py` and `time_kline.py` are **standalone utility modules** for realtime quotes and intraday K-line data. They are not part of the core pipeline.

### Multi-Stock Data Cache

`analyze_factor._load_multi_stock_data()` uses `joblib.Memory` (disk cache at `data/cache/`) so multiple Optuna worker processes share the same cached data and avoid repeated disk I/O.

### tsfresh Parallelism

`strategies/tsfresh_features.py` limits `n_jobs` to `cpu_count // 2` to prevent process explosion when Optuna runs parallel trials alongside tsfresh's internal parallelism.

### Backtest Engines

Configured via `config.yaml` → `backtest_engine`:
- `native` (recommended): Pure Python backtester in `analyze_factor.py:backtest()` - handles HK fees (0.088% + 0.1% stamp duty)
- `vectorbt`: Vectorbt-based backtesting - results differ slightly

**ATR 止损模拟（Issue #9 已修复）**: 两个引擎均支持在回测中模拟 ATR 动态止损。通过 `config.yaml → risk_management.simulate_in_backtest`（默认 `true`）控制。开启后，`simulate_atr_stoploss()` 会在回测开始前逐 bar 扫描信号，当持仓期间收盘价跌破 `peak - multiplier × ATR` 时将信号强制置 0（平仓）。`cooldown_bars > 0` 可模拟止损后暂停重入的冷却期。设为 `false` 可恢复旧行为（向后兼容）。

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
| `data_sources` | [yahooquery] | Data vendor priority chain |
| `storage_format` | csv | Storage format: "csv" or "parquet" |
| `batch_max_workers` | 4 | Concurrent download threads for HSI batch |
| `lookback_months` | 3 | Validation window |
| `train_years` | 5 | Training window |
| `backtest_engine` | vectorbt | native or vectorbt |
| `use_optuna` | true | Enable Bayesian optimization |
| `optuna_trials` | 100 | Optuna search iterations |
| `max_tries` | 300 | Random search iterations |
| `min_return` | 0.10 | Validation threshold |
| `min_sharpe_ratio` | 1.0 | Validation threshold |
| `max_drawdown` | -0.15 | Maximum drawdown (negative) |
| `min_total_trades` | 4 | Minimum trade count |
| `test_months` | 6 | Hold-out period (months, not used in search) |
| `test_days` | 5 | Prediction horizon |
| `early_stop_threshold` | 0.03 | Skip trial if val return below this (trial >= 10) |
| `wf_min_window_win_rate` | 0.5 | Walk-forward minimum window win rate |
| `use_cv` | false | Enable TimeSeriesSplit CV (keep false for tsfresh variants) |
| `sentiment_weight` | 0.03 | Sentiment signal weight (0=display only, 0.0-0.3 for light weighting) |

### keys.yaml (not committed)
```yaml
alpha_vantage_key: null
feishu_webhook: https://open.feishu.cn/...
broker_api_key: null
broker_account: null
extra_hk_holidays: []          # 临时停市日 (如台风), 格式: ['2026-09-15']
```

### Backtest Engine Fees
| Key | Default | Description |
|-----|---------|-------------|
| `invest_fraction` | 0.95 | Position size (5% cash buffer) |
| `slippage` | 0.001 | Slippage for both engines |
| `fees_rate` | 0.00088 | Hong Kong trading fee (0.088%) |
| `stamp_duty` | 0.001 | Hong Kong stamp duty (0.1%) |

### Risk Management
| Key | Default | Description |
|-----|---------|-------------|
| `use_atr_stop` | true | Enable ATR dynamic stop-loss |
| `atr_period` | 14 | ATR calculation period |
| `atr_multiplier` | 2.0 | Stop-loss = peak - multiplier × ATR |
| `trailing_stop` | true | Use trailing stop (else fixed entry price) |
| `simulate_in_backtest` | true | Simulate ATR stop-loss in backtest (false = legacy behaviour) |
| `cooldown_bars` | 0 | Bars to pause after a stop-loss (0 = no cooldown) |
| `use_kelly` | false | Enable Kelly position sizing |
| `kelly_fraction` | 0.5 | Kelly scaling (0.5 = half Kelly) |
| `max_position_pct` | 0.25 | Maximum single position (% of portfolio) |
| `portfolio_value` | 200000.0 | Kelly calculation base |
| `daily_loss_limit` | 0.05 | Circuit breaker (% daily loss threshold) |
| `max_consecutive_loss_days` | 3 | Circuit breaker consecutive loss days |

### Position Management
| Key | Default | Description |
|-----|---------|-------------|
| `position_shares` | 200 | Current holding shares (0 = flat) |
| `position_avg_cost` | 500.0 | Average cost per share |
| `position_peak_price` | 0.0 | Peak price during holding (0 = use entry price) |

### OMS / Live Trading
| Key | Default | Description |
|-----|---------|-------------|
| `broker_api_url` | null | Broker API URL (Futu/Tiger/IBKR), null = paper trade |

## Data Storage

```
data/
├── historical/          # OHLCV 数据 (*.csv 或 *.parquet) + 元数据 (*.meta.json)
├── factors/             # 有效因子存储 (factor_*.pkl)
├── reports/             # 回测报告 (report_*.md)
├── trends/              # Google Trends 数据
├── sentiment/           # 情感分析缓存
├── cache/               # joblib.Memory 多股票数据缓存
├── logs/                # 运行日志 (fetch.log, fetch_hsi.log)
│   └── quality_report.json  # 数据质量报告 (自动追加)
├── timekline/           # 分时 K 线缓存
└── plots/               # 图表输出
```

## Testing

```bash
python3 smoke_test.py   # Offline smoke test — no network required, uses synthetic data
```

Covers: all major rule strategies, `analyze_factor.backtest()`, and the `predict()` interface.

## ML Strategies

All ML strategies use **multi** training (trained on HSI stocks, validated on target stock).

| Strategy | Model | Features |
|----------|-------|----------|
| `xgboost_enhanced` | XGBoost | Technical indicators (pandas) + optional ta-lib |
| `xgboost_enhanced_tsfresh` | XGBoost | Technical indicators + tsfresh features |
| `xgboost_enhanced_ta_tsfresh` | XGBoost | ta-lib indicators + tsfresh features |
| `lightgbm_enhanced` | LightGBM | Technical indicators (pandas) + optional ta-lib |
| `lightgbm_enhanced_tsfresh` | LightGBM | Technical indicators + tsfresh features |
| `lightgbm_enhanced_ta_tsfresh` | LightGBM | ta-lib indicators + tsfresh features |

### ML Strategy Configuration
Each ML strategy inherits config from `config.yaml → ml_strategies.<strategy_name>`:
- `use_tsfresh_features`: bool — add tsfresh rolling-window features
- `use_ta_lib`: bool — use ta-lib instead of pandas for indicators
- `tsfresh_window_sizes`: list — rolling windows for tsfresh (default [10, 20])
- XGBoost params: `xgb_n_estimators`, `xgb_max_depth`, `xgb_learning_rate`, `xgb_subsample`, `xgb_colsample_bytree`, `xgb_reg_alpha`, `xgb_reg_lambda`, `xgb_min_child_weight`
- LightGBM params: `lgbm_n_estimators`, `lgbm_max_depth`, `lgbm_learning_rate`, `lgbm_num_leaves`, `lgb_feature_fraction`, `lgb_bagging_fraction`, `lgb_reg_alpha`, `lgb_reg_lambda`, `lgb_min_child_samples`

## Dependencies

Core:
- `yfinance`, `yahooquery`, `akshare` — market data vendors
- `pandas`, `numpy`, `scikit-learn` — data processing
- `pyarrow` — Parquet storage backend
- `tenacity` — retry with exponential backoff (data downloads)
- `exchange_calendars` — HK trading calendar (XHKG, required dependency)
- `PyYAML` — config loading

ML:
- `xgboost`, `lightgbm` — ML strategies
- `tsfresh`, `pyts` — automatic time series feature extraction
- `ta` — technical analysis library (200+ indicators)

Backtesting:
- `vectorbt` — optional backtesting engine
- `optuna` — Bayesian optimization

Other:
- `pytrends` — Google Trends
- `textblob`, `snownlp` — sentiment analysis
- `matplotlib` — visualization
- `requests` — HTTP client
