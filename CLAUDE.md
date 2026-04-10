# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

港股智能分析系统 — 每日扫描持仓与观察列表，聚合多策略共识信号，结合风控规则，给出明日操作建议（买入/持有/卖出/观望/止损卖出），并通过飞书推送报告。默认标的腾讯控股 (0700.HK)。

**两条管线，职责严格分离：**
- `daily_run.py` — 每日推断入口，轻量推断，不重新训练，秒级出结果
- `main.py` — 每日训练入口，完整超参数搜索 + 验证，更新 `data/factors/` 因子库

## Commands

```bash
# 每日运行
python3 daily_run.py                             # 分析 portfolio.yaml 中的持仓
python3 daily_run.py --tickers 0700.HK 0005.HK   # 指定股票
python3 daily_run.py --watchlist hsi              # HSI 全部成分股
python3 daily_run.py --skip-notify --dry-run      # 不发飞书、不保存状态

# 训练
python3 main.py --use-optuna                      # 完整超参数搜索
python3 main.py --portfolio                       # 分层混合：ML全局 + 每只股票规则策略

# 测试（全离线，284 个，无网络请求）
pytest tests/ -v
pytest tests/test_factor_registry.py -v           # 单个文件
pytest tests/test_strategies.py::TestRuleStrategiesSignalValues -v  # 单个类
python3 smoke_test.py                             # 快速冒烟测试（合成数据）
```

## Architecture

### Two Pipelines

**Training** (`main.py` → `pipeline/`): Data download → strategy hyperparameter search → out-of-sample validation → save best factor to `data/factors/` → Feishu notification.

**Daily inference** (`daily_run.py` → `engine/`): Load saved factors → run each strategy's `predict()` → weighted vote aggregation (`SignalAggregator`) → per-stock risk analysis (`PositionAnalyzer`) → portfolio-level risk check → generate recommendation → Feishu push.

### Key Architectural Boundaries

| Layer | Entry point | Responsibility |
|-------|-------------|---------------|
| `pipeline/` | `main.py` | Training only: `data_prep` → `train` → `select` → `train_portfolio` |
| `engine/` | `daily_run.py` | Inference only: `signal_aggregator` → `position_analyzer` → `portfolio_risk` |
| `strategies/` | Both | Strategy implementations (rule-based + ML), auto-discovered by `NAME` attribute |
| `data/` | Both | Data download (vendor chain), storage, quality, factor registry |

### Signal Flow (daily_run.py)

```
data/factors/*.pkl  →  SignalAggregator.aggregate()  →  AggregatedSignal
                           ↓ (per factor)                    ↓
                    strategy.predict(model, data)      PositionAnalyzer.analyze()
                                                             ↓
                                                    Action recommendation
```

## Mandatory Conventions

### Shared Utilities — Use These, Don't Reimplement

| Need | Use | Import |
|------|-----|--------|
| Config loading | `config_loader.py` | `from config_loader import load_config` |
| Logging | `log_config.py` | `from log_config import get_logger` |
| Technical indicators | `strategies/indicators.py` | `from strategies.indicators import rsi, macd, kdj, obv, atr, pvt, ...` |
| Ticker → filesystem name | `config_loader.py` | `from config_loader import ticker_to_safe` |
| Data download/load | `data/manager.py` | `from data import DataManager` |

- **No `print()` in core pipeline** — use `get_logger(__name__)`. Standalone scripts (`easy_quptation.py`, `time_kline.py`, `oms.py`) may use print.
- **No duplicate indicator implementations** — `strategies/indicators.py` provides: `rsi`, `bollinger_bands`, `ema`, `macd`, `kdj`, `obv`, `fibonacci`, `atr`, `pvt`. ML strategies use these via wrapper functions in `xgboost_enhanced.py`.
- **Ticker normalization** — always use `ticker_to_safe("0700.HK")` → `"0700_HK"` for filesystem paths. Never inline `.replace(".", "_").upper()`.

### Strategy Interface

Every strategy in `strategies/` must expose:
```python
NAME: str = "strategy_name"

def run(data: pd.DataFrame, config: dict) -> (signal: pd.Series, model, meta: dict)
def predict(model, data: pd.DataFrame, config: dict, meta: dict) -> pd.Series
```
- `signal`: int Series (1=long, 0=flat), aligned to `data.index`
- Rule strategies return `model=None`; `predict()` just re-runs `run()`
- ML strategies return a fitted model; `predict()` does inference without refitting

**Rule strategies must use explicit state machines** (enter/hold/exit loop), not `ffill()` signal smoothing.

### Look-Ahead Bias Prevention (Critical)

1. `backtest()` and `backtest_vectorbt()` apply `signal.shift(1)` internally — **do not shift before passing signals in**
2. Train/val split: validation data is **never** used during training
3. Multi-stock data: raw prices are split by time **before** `create_multi_stock_dataset()` is called
4. Rule strategies: state-machine loops only — `ffill()` is forbidden

### Strategy Training Types

| Type | Strategies | Training data |
|------|-----------|---------------|
| `single` | All rule-based (RSI, MACD, Bollinger, KDJ, ATR, VWAP, etc.) | Target stock only |
| `multi` | ML (XGBoost, LightGBM, RNN) | Multiple HSI stocks |
| `custom` | Strategy-defined | Strategy-defined |

### Factor Lifecycle

Factors are saved as `data/factors/factor_*.pkl` (or in per-ticker subdirectories like `data/factors/0700_HK/`). The `FactorRegistry` (`data/factor_registry.py`) tracks TTL (single=30d, multi=60d, custom=45d), auto-expires stale factors, and archives after 90 days. `signal_aggregator.py` defaults to `use_registry=True` but falls back to all disk factors if the registry filters to empty.

### Validation Thresholds

A strategy must pass all of these to be saved:
- `cum_return > min_return` (default 0.10)
- `sharpe_ratio > min_sharpe_ratio` (default 1.0)
- `max_drawdown >= max_drawdown` (default -0.15)
- `total_trades >= min_total_trades` (default 4)

## Configuration

Two YAML files at project root:
- `config.yaml` — all business config (ticker, period, thresholds, backtest engine, risk management, ML hyperparams)
- `keys.yaml` (not committed) — API keys, Feishu webhook, broker credentials, extra HK holidays

Load via: `from config_loader import load_config; cfg = load_config(include_keys=True)`

## Testing

All 284 tests run fully offline with synthetic data. Key fixtures in `tests/conftest.py`: `synthetic_ohlcv`, `atr_plunge_ohlcv`, `default_config`, `minimal_config`.

When adding new strategies: add to the parametrized list in `tests/test_strategies.py` — it auto-discovers all rule strategies with valid `NAME` + `run` + `predict`.
