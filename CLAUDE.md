# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**核心目标：一个专业的量化软件，每日运行，基于持仓给出下一个交易日的操作建议。**

港股智能分析系统 — 每日自动运行，扫描持仓与观察列表，聚合多策略共识信号，结合 ATR 止损/Kelly 仓位/熔断等风控规则，基于持仓详情给出明日操作建议（买入 / 持有 / 卖出 / 观望 / 止损卖出），并通过飞书推送报告。支持 HSI 全部成分股，默认标的腾讯控股 (0700.HK)。

- **每日入口**：`daily_run.py` — 轻量推断，不重新训练，秒级出结果
- **每日训练**：`main.py` — 完整超参数搜索 + 验证，更新 `data/factors/` 因子库
- **持仓管理**：`data/portfolio.yaml` — 手动维护持仓，系统自动更新止损峰值与连续亏损天数

## Running the Project

```bash
# ── 每日运行（核心入口）────────────────────────────────────────
python3 daily_run.py                          # 分析 portfolio.yaml 中的持仓
python3 daily_run.py --tickers 0700.HK 0005.HK  # 指定股票
python3 daily_run.py --watchlist hsi          # 分析 HSI 全部成分股
python3 daily_run.py --skip-notify            # 不发送飞书通知
python3 daily_run.py --skip-sentiment         # 跳过情感分析（更快）
python3 daily_run.py --dry-run                # 打印建议但不保存状态
bash daily_run.sh                             # Shell 封装（含日志、venv 激活）

# ── 每日训练（更新因子库）──────────────────────────────────────
python3 main.py --use-optuna                  # 完整超参数搜索 + 验证
python3 main.py --use-optuna --optuna-trials 100
python3 main.py --skip-train                  # 跳过训练，仅生成信号报告
python3 main.py --n-days 5                    # 预测天数

# ── 测试 ─────────────────────────────────────────────────────────
pytest tests/ -v                              # 100 单元测试（全离线）
python3 smoke_test.py                         # 快速冒烟测试
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
| `config_loader.py` | **唯一配置加载入口** — `load_config(include_keys=True)` |
| `log_config.py` | **唯一日志初始化入口** — `get_logger(__name__)` |
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

### Engine Package (`engine/`)

每日推荐引擎核心包，Phase 1 已完整实现：

```
engine/
├── __init__.py          # 导出 PortfolioState, PortfolioPosition, load_portfolio,
│                        #        SignalAggregator, AggregatedSignal,
│                        #        PositionAnalyzer, RecommendationResult
├── portfolio_state.py   # 多持仓状态加载 / 保存（data/portfolio.yaml）
├── signal_aggregator.py # 多策略共识信号聚合（Sharpe 加权投票）
└── position_analyzer.py # 单只股票推荐生成（信号 + 风控 + 建议）
```

**`PortfolioState` / `PortfolioPosition`**：
- `PortfolioPosition` — 持仓快照，含 `has_position`、`to_position_manager_position()` 桥接、`to_dict()` 序列化
- `PortfolioState` — 全持仓管理，含 `get_position()` / `all_tickers()` / `held_tickers()` / `summary()` / `save()` 原子写回
- `load_portfolio()` — 从 `data/portfolio.yaml` 加载，文件不存在或解析失败时优雅降级

### Data Package (`data/`)

数据模块经过 Phase 2 重构后采用分层架构：

```
data/
├── __init__.py              # 包入口 (导出所有公开 API，含 FactorRegistry/FactorRecord)
├── manager.py               # DataManager — 核心入口：vendor 链 → 校验 → 质量 → 存储
├── factor_registry.py       # FactorRegistry — 因子生命周期注册表（TTL / 归档 / Sharpe 劣化检测）
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

### Factor Registry (`data/factor_registry.py`)

因子生命周期注册表，索引文件为 `data/factors/factor_registry.json`。

```python
from data.factor_registry import FactorRegistry

registry = FactorRegistry()
registry.register(factor_id, filename, subdir, strategy_name, ticker,
                  training_type, sharpe_ratio, cum_return, max_drawdown, total_trades)
registry.expire_stale()                      # active → expired（TTL 到期）
registry.archive_old()                       # expired > 90 天 → 移动 pkl 到 archive/
active = registry.active_records(ticker="0700.HK")  # 仅返回 status=active 且未过 TTL
```

TTL 规则：`single` = 30 天，`multi` = 60 天，`custom` = 45 天。新因子 Sharpe 低于历史最优 0.2 时记录劣化警告至 `notes` 字段。

**迁移存量因子**（一次性）：
```bash
python3 data/factor_registry.py --migrate
```

**集成状态（截至 2026-04-04）**：`signal_aggregator.py` 已支持 `use_registry` 参数（默认 `True`），但 `analyze_factor.py` 中 `_save_factor()` 调用 `registry.register()` 的集成点尚未实现。在集成完成前，`use_registry=True`（默认）会将所有因子过滤为空列表，造成功能性回归——详见 `docs/review_factor_registry.md` [C1]。**当前应手动确保 `use_registry=False` 或完成集成点 A**。

### Logging

全项目统一使用 `log_config.py` 中的 `get_logger(__name__)`，**禁止在核心管线中使用 `print()`**：

```python
from log_config import get_logger
logger = get_logger(__name__)
logger.info("消息", extra={"ticker": "0700.HK", "elapsed_ms": 120})
```

- **控制台**：彩色文本（INFO=蓝, WARNING=黄, ERROR=红）
- **文件**：JSON Lines → `data/logs/app.log`，10MB 滚动，保留 5 份
- **级别控制**：`LOG_LEVEL=DEBUG python3 daily_run.py`

> `easy_quptation.py`、`time_kline.py`、`oms.py` 等 standalone 脚本可保留 `print()`，不影响核心管线。

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

### Common Indicators (`strategies/indicators.py`)

所有规则策略共享公共指标库，**禁止在策略文件中重复实现指标函数**：

```python
from strategies.indicators import rsi, bollinger_bands, ema, macd, kdj, obv, fibonacci
```

已提供的公共指标：`rsi`, `bollinger_bands`, `ema`, `macd`, `kdj`, `obv`, `fibonacci`

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

All backtesting is designed to be free of look-ahead bias：

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

`train_multi_stock.py` is a **standalone script** — it is not called by `main.py` at runtime. Its `load_all_hsi_data()` is used indirectly via `analyze_factor._load_multi_stock_data()`. Run it directly for multi-stock dataset exploration：
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
├── factors/             # 有效因子存储 (factor_*.pkl) + factor_registry.json（因子索引）
│   └── archive/         # 过期 > 90 天的因子（自动归档）
├── reports/             # 回测报告 (report_*.md)
├── trends/              # Google Trends 数据
├── sentiment/           # 情感分析缓存
├── cache/               # joblib.Memory 多股票数据缓存
├── logs/                # 运行日志 (app.log JSON Lines, fetch.log, fetch_hsi.log)
│   └── quality_report.json  # 数据质量报告 (自动追加)
├── timekline/           # 分时 K 线缓存
└── plots/               # 图表输出
```

## Testing

```bash
pytest tests/ -v          # 全离线，当前 219 个测试
pytest tests/test_factor_registry.py -v  # 仅跑因子注册表测试（24 个）
python3 smoke_test.py     # 离线冒烟测试，无需网络，使用合成数据
```

### 测试文件结构

| 文件 | 测试数 | 覆盖内容 |
|------|--------|----------|
| `tests/conftest.py` | — | Fixtures: `synthetic_ohlcv`, `atr_plunge_ohlcv`, `default_config`, `minimal_config` |
| `tests/test_schemas.py` | 13 | 列名归一化（中英文/MultiIndex）、OHLCV 校验 |
| `tests/test_oms.py` | 9 | PaperOMS 买卖/拒绝/round-trip，`create_oms()` 工厂 |
| `tests/test_portfolio_state.py` | 11 | `PortfolioPosition` 属性、`PortfolioState` 持久化、ticker 转换 |
| `tests/test_position_manager.py` | 12 | Kelly 边界、熔断触发、ATR 止损、冷却期 |
| `tests/test_backtest.py` | 5 | 必填字段、全持/全空信号、ATR 止损、前视偏差修复 |
| `tests/test_strategies.py` | 35 | 全部 16 个规则策略 `run()` + `predict()`、ML 信号边界 |
| `tests/test_data_manager.py` | 5 | schemas 集成、原子 CSV 写入 |
| `tests/test_signal_aggregator.py` | 7 | 空目录降级、置信度边界、`AggregatedSignal` 属性 |
| `tests/test_integration.py` | 3 | 策略→回测全流程、`PortfolioPosition` round-trip、`get_recommendation` |
| `tests/test_factor_registry.py` | 24 | TTL 边界、Sharpe 劣化、归档、迁移脚本 |

> **全程离线**，无任何网络请求。核心路径覆盖率：`position_manager` ~60%，`xgboost_enhanced` ~64%，规则策略 90-100%。

## Phase 1 完成状态（截至 2026-03-25）

所有 Phase 1 基础债务清理任务已全部完成：

| # | 任务 | 状态 | 关键变更 |
|---|------|------|---------|
| 1 | 补全 `engine/portfolio_state.py` | ✅ 完成 | `PortfolioState` / `PortfolioPosition` / `load_portfolio()` 完整实现 |
| 2 | 统一配置加载 | ✅ 完成 | 新建 `config_loader.py`，全项目唯一入口 `load_config()` |
| 3 | 公共指标库整合 | ✅ 完成 | `strategies/indicators.py` 提供 7 个公共函数，10 个策略文件已迁移 |
| 4 | 测试框架搭建 | ✅ 完成 | `pytest tests/` → 100 passed，核心路径覆盖率 > 60% |
| 5 | 依赖版本升级 | ✅ 完成 | 全库钉死版本；`pip-audit` → No known vulnerabilities |
| 6 | 结构化日志 | ✅ 完成 | `log_config.py` + `get_logger()`，核心管线零 `print()` |

**Phase 2（策略与风控增强）现已可启动**，详见 `docs/upgrade_plan.md`。

## Phase 2 待办（策略与风控增强）

> 参见 `docs/upgrade_plan.md` 获取完整设计方案。

| # | 任务 | 优先级 | 状态 | 关键影响文件 |
|---|------|--------|------|------------|
| 1 | Ensemble / Stacking 策略 | P1 | ⬜ 待做 | `engine/signal_aggregator.py`，新建 `strategies/ensemble_stacking.py` |
| 2 | Portfolio-Level 风控 | P1 | ⬜ 待做 | 新建 `engine/portfolio_risk.py`，修改 `daily_run.py` |
| 3 | 因子生命周期管理 | P1 | 🚧 进行中 | `data/factor_registry.py` ✅，集成点 A（`analyze_factor.py`）⚠️ 待修复，详见 `docs/review_factor_registry.md` |
| 4 | PnL 历史追踪 | P1 | ✅ 完成 | `data/pnl_tracker.py`，修改 `daily_run.py` |
| 5 | OMS 实盘对接（Futu） | P1 | ⬜ 待做 | `oms.py`，新建 `oms_futu.py` |
| 6 | 深度学习策略（可选） | P1 | ✅ 完成 | `strategies/rnn_trend.py`（GRU/LSTM） |

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
- `yfinance>=1.2.0`, `yahooquery`, `akshare` — market data vendors
- `pandas>=2.3`, `numpy>=1.26`, `scikit-learn>=1.8` — data processing
- `pyarrow` — Parquet storage backend
- `tenacity` — retry with exponential backoff (data downloads)
- `exchange_calendars>=4.13` — HK trading calendar (XHKG, required dependency)
- `PyYAML` — config loading

ML:
- `xgboost>=3.2`, `lightgbm>=4.6` — ML strategies
- `tsfresh`, `pyts` — automatic time series feature extraction
- `ta` — technical analysis library (200+ indicators)

Backtesting:
- `vectorbt` — optional backtesting engine
- `optuna>=4.8` — Bayesian optimization

Other:
- `pytrends` — Google Trends
- `textblob`, `snownlp` — sentiment analysis
- `matplotlib` — visualization
- `requests` — HTTP client
- `pip-audit` — dependency vulnerability scanning (`pip-audit -r requirements.txt`)
