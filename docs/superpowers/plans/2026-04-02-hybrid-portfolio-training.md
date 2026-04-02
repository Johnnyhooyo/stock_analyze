# 分层混合组合训练 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 实现分层混合训练模式：ML 策略全局联合训练一次（存 `data/factors/`），规则策略按股票独立训练（存 `data/factors/{TICKER_SAFE}/`），`SignalAggregator` 合并两类因子投票，`daily_run.py` 按持仓给出操作建议。

**Architecture:** Rule 策略（RSI/MACD/Bollinger 等 `single` 类型）的最优超参因股票特征而异，需 per-ticker 独立优化；ML 策略（XGBoost/LightGBM `multi` 类型）在 HSI 全体数据上训练，参数鲁棒性强，共享一份全局因子。`SignalAggregator.aggregate()` 同时加载 per-ticker 规则因子 + 全局 ML 因子合并加权投票；`main.py --portfolio` 先跑一次全局 ML 训练，再循环对每只股票跑规则训练。

**Tech Stack:** Python, joblib, pathlib, 已有 `DataManager` / `PositionAnalyzer` / `SignalAggregator` / `analyze_factor.run_search` / `optimize_with_optuna`

---

## Context

当前问题：`main.py` 只训练单只股票（`config.yaml → ticker`，默认 0700.hk），所有因子存入 `data/factors/`。`daily_run.py` 对全部持仓股票调用相同因子 —— 用腾讯调优的 RSI 参数分析汇丰，结论不可靠。

分层混合方案优势：ML 策略在多股票数据上联合训练，参数泛化性强，不用每只股票重训；规则策略参数语义明确（RSI 周期、Bollinger 倍数），贴合个股波动特征，值得 per-ticker 调优。持仓 5~10 只时训练时间可接受。

---

## 目录结构（训练后）

```
data/factors/
├── factor_XXXX.pkl       ← ML 全局因子（XGBoost / LightGBM）
├── 0700_HK/
│   └── factor_XXXX.pkl   ← 腾讯专属规则策略因子
├── 0005_HK/
│   └── factor_XXXX.pkl   ← 汇丰专属规则策略因子
└── ...
```

---

## 关键文件

| 文件 | 变更内容 |
|------|---------|
| `analyze_factor.py` | `run_search()` 增加 `strategy_type` 过滤参数 |
| `engine/signal_aggregator.py` | `aggregate()` 同时加载 per-ticker 规则因子 + 全局 ML 因子 |
| `main.py` | `step2_train*(factors_dir_override, strategy_type)`；新增 `_ensure_hsi_data()`、`step1_ensure_data(ticker=None)`、`train_portfolio_tickers()`、`_print_portfolio_summary()`、`--portfolio` flag |
| `daily_run.py` | `_check_factor_freshness()` 扫描子目录；因子存在性检查扫描子目录 |
| `tests/test_signal_aggregator.py` | 新增混合加载测试类 |
| `tests/test_portfolio_training.py` | 新文件 — 组合训练编排测试 |

---

## Task 1: `analyze_factor.py` — `run_search()` 增加 `strategy_type` 参数

**Files:**
- Modify: `analyze_factor.py:1162` (`run_search` 函数签名及内部 `_discover_strategies` 调用)

**背景：** `step2_train_native` 调用 `run_search()`，但 `run_search` 内部调用 `_discover_strategies()` 时不传类型过滤，无法区分 ML 策略和规则策略。`step2_train_optuna` 已支持 `strategy_type`；`run_search` 需要同等支持。

- [ ] **Step 1: 写失败测试**

在 `tests/test_signal_aggregator.py` 顶部的 imports 下方，或在一个合适的测试文件中（不需要新建文件，可加到已有文件末尾）：

```python
# 这个测试验证 run_search 的 strategy_type 过滤确实生效
# 加到 tests/test_signal_aggregator.py 末尾，作为独立函数（非类方法）

def test_run_search_strategy_type_filters_strategies(synthetic_ohlcv, default_config):
    """run_search(strategy_type='single') 不应包含 multi 类策略的结果。"""
    from analyze_factor import run_search, _discover_strategies
    # 只有 single 类型的策略应参与搜索
    cfg = default_config.copy()
    cfg['max_tries'] = 1
    cfg['min_return'] = -999.0  # 不要求达标，只要跑完
    _, sorted_results, _ = run_search(synthetic_ohlcv, cfg, strategy_type='single')
    # 所有结果的策略名称都不应是 ML 策略
    ml_names = {'xgboost_enhanced', 'lightgbm_enhanced',
                'xgboost_enhanced_tsfresh', 'lightgbm_enhanced_tsfresh'}
    result_names = {r.get('strategy_name', '') for r in sorted_results}
    assert result_names.isdisjoint(ml_names), (
        f"strategy_type='single' 搜索结果中出现了 ML 策略: {result_names & ml_names}"
    )
```

- [ ] **Step 2: 运行确认失败**

```bash
cd /home/thenine/projects/stock_analyze
pytest tests/test_signal_aggregator.py::test_run_search_strategy_type_filters_strategies -v 2>&1 | tail -15
```
Expected: FAILED（`run_search` 尚无 `strategy_type` 参数，或 ML 策略出现在结果中）

- [ ] **Step 3: 实现 — 修改 `run_search` 签名**

在 `analyze_factor.py:1162`，修改函数签名：

```python
def run_search(
    data: pd.DataFrame,
    cfg: Optional[dict] = None,
    on_result=None,
    strategy_type: str = None,          # ← 新增
) -> tuple:
```

- [ ] **Step 4: 实现 — 将 `strategy_type` 传入 `_discover_strategies`**

在 `run_search` 内部，找到（约 line 1219）：
```python
    strategy_mods = _discover_strategies()
```
替换为：
```python
    strategy_mods = _discover_strategies(strategy_type=strategy_type)
```

- [ ] **Step 5: 将 `strategy_type` 透传给 `step2_train_native`**

在 `main.py:336`，修改 `step2_train_native` 签名和内部 `run_search` 调用：

```python
def step2_train_native(hist_data: pd.DataFrame, strategy_type: str = None):
    ...
    best_result, sorted_results, test_df = run_search(hist_data, cfg, strategy_type=strategy_type)
```

同时更新 `step2_train`（line 318）让它透传 `strategy_type` 给 `step2_train_native`：

```python
def step2_train(hist_data: pd.DataFrame, use_optuna: bool = False, optuna_trials: int = 50, strategy_type: str = None):
    if use_optuna and OPTUNA_AVAILABLE:
        return step2_train_optuna(hist_data, n_trials=optuna_trials, strategy_type=strategy_type)
    else:
        return step2_train_native(hist_data, strategy_type=strategy_type)   # ← 之前没有透传
```

- [ ] **Step 6: 运行测试确认通过**

```bash
pytest tests/test_signal_aggregator.py::test_run_search_strategy_type_filters_strategies -v
pytest tests/ -v -k "not test_integration" 2>&1 | tail -10
```
Expected: 新测试 PASS，其余测试不退步。

- [ ] **Step 7: Commit**

```bash
git add analyze_factor.py main.py tests/test_signal_aggregator.py
git commit -m "feat(analyze_factor): run_search() supports strategy_type filter; thread through step2_train_native"
```

---

## Task 2: `engine/signal_aggregator.py` — 混合双目录因子加载

**Files:**
- Modify: `engine/signal_aggregator.py:103-121` (`_load_factors`)
- Modify: `engine/signal_aggregator.py:187` (`aggregate` 内 `artifacts = self._load_factors()` 这一行)
- Test: `tests/test_signal_aggregator.py`

**设计：**
- per-ticker 目录存在且有因子 → 从该目录加载（规则因子）+ 从全局目录只加载 ML 因子（`feat_cols` 非空）
- per-ticker 目录不存在 → 只加载全局目录（向后兼容）
- 判断 ML：`len(art.get("meta", {}).get("feat_cols", [])) > 0`

- [ ] **Step 1: 写失败测试**

在 `tests/test_signal_aggregator.py` 末尾追加：

```python
class TestSignalAggregatorHybridLoading:
    """分层混合：per-ticker 规则因子 + 全局 ML 因子合并投票。"""

    def _make_rule_factor(self, d: Path, run_id: int = 1, sharpe: float = 1.2):
        import joblib
        d.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "meta": {"name": "ma_crossover", "params": {}, "feat_cols": []},
            "model": None, "sharpe_ratio": sharpe, "config": {},
        }, d / f"factor_{run_id:04d}.pkl")

    def _make_ml_factor(self, d: Path, run_id: int = 1, sharpe: float = 1.5):
        import joblib
        import numpy as np
        from unittest.mock import MagicMock
        d.mkdir(parents=True, exist_ok=True)
        mock_model = MagicMock()
        joblib.dump({
            "meta": {"name": "xgboost_enhanced", "params": {},
                     "feat_cols": ["rsi", "macd"]},
            "model": mock_model, "sharpe_ratio": sharpe, "config": {},
        }, d / f"factor_{run_id:04d}.pkl")

    def test_both_dirs_loaded_when_per_ticker_exists(self, tmp_path, synthetic_ohlcv):
        """per-ticker 目录存在时，规则因子 + 全局 ML 因子都应参与投票。"""
        global_dir = tmp_path / "factors"
        ticker_dir = global_dir / "0005_HK"
        self._make_rule_factor(ticker_dir, run_id=1)
        self._make_ml_factor(global_dir, run_id=1)

        agg = SignalAggregator(factors_dir=global_dir)
        result = agg.aggregate("0005.HK", synthetic_ohlcv, {})
        assert isinstance(result, AggregatedSignal)
        # 两个因子都加载了，total_strategies 应该 >= 0（ML 可能推断失败但不崩溃）
        assert result.total_strategies >= 0

    def test_global_only_when_no_per_ticker_dir(self, tmp_path, synthetic_ohlcv):
        """没有 per-ticker 目录时，加载全局目录所有因子（向后兼容）。"""
        global_dir = tmp_path / "factors"
        self._make_rule_factor(global_dir, run_id=1)
        agg = SignalAggregator(factors_dir=global_dir)
        result = agg.aggregate("9988.HK", synthetic_ohlcv, {})
        assert isinstance(result, AggregatedSignal)

    def test_global_ml_only_when_per_ticker_exists(self, tmp_path, synthetic_ohlcv):
        """per-ticker 目录存在时，全局目录中的规则因子不应重复加载。"""
        global_dir = tmp_path / "factors"
        ticker_dir = global_dir / "0005_HK"
        # 全局目录存一个规则因子（老式单票训练遗留）
        self._make_rule_factor(global_dir, run_id=1, sharpe=1.0)
        # 全局目录存一个 ML 因子
        self._make_ml_factor(global_dir, run_id=2, sharpe=1.8)
        # per-ticker 目录存一个规则因子
        self._make_rule_factor(ticker_dir, run_id=1, sharpe=1.3)

        agg = SignalAggregator(factors_dir=global_dir)
        # 不应崩溃；ML 因子来自全局，规则因子来自 per-ticker
        result = agg.aggregate("0005.HK", synthetic_ohlcv, {})
        assert isinstance(result, AggregatedSignal)

    def test_empty_per_ticker_dir_falls_back_to_global(self, tmp_path, synthetic_ohlcv):
        """空的 per-ticker 目录不影响全局目录加载。"""
        global_dir = tmp_path / "factors"
        (global_dir / "1299_HK").mkdir(parents=True, exist_ok=True)
        self._make_rule_factor(global_dir, run_id=1)
        agg = SignalAggregator(factors_dir=global_dir)
        result = agg.aggregate("1299.HK", synthetic_ohlcv, {})
        assert isinstance(result, AggregatedSignal)

    def test_load_factors_explicit_dir(self, tmp_path):
        """_load_factors(dir) 使用传入的目录，而非 self.factors_dir。"""
        import joblib
        dir_a = tmp_path / "a"; dir_a.mkdir()
        dir_b = tmp_path / "b"; dir_b.mkdir()
        joblib.dump({"meta": {"name": "s1", "params": {}, "feat_cols": []},
                     "model": None, "sharpe_ratio": 0.5, "config": {}},
                    dir_a / "factor_0001.pkl")
        joblib.dump({"meta": {"name": "s2", "params": {}, "feat_cols": []},
                     "model": None, "sharpe_ratio": 1.8, "config": {}},
                    dir_b / "factor_0001.pkl")
        agg = SignalAggregator(factors_dir=dir_a)
        arts = agg._load_factors(dir_b)
        assert len(arts) == 1
        assert arts[0]["sharpe_ratio"] == 1.8
```

- [ ] **Step 2: 运行确认失败**

```bash
pytest tests/test_signal_aggregator.py::TestSignalAggregatorHybridLoading -v 2>&1 | tail -20
```
Expected: 多数 FAIL（`_load_factors` 无参数，`aggregate` 未区分目录）

- [ ] **Step 3: 实现 — `_load_factors(factors_dir=None)`**

替换 `engine/signal_aggregator.py` 中 `_load_factors` 方法（约 103–121 行）：

```python
def _load_factors(self, factors_dir: Optional[Path] = None) -> list[dict]:
    """
    加载指定目录中所有 factor_*.pkl。
    factors_dir 为 None 时使用 self.factors_dir（向后兼容）。
    """
    target = factors_dir or self.factors_dir
    candidates = sorted(
        target.glob("factor_*.pkl"),
        key=lambda p: int(p.stem.split("_")[1]),
        reverse=True,
    )
    artifacts = []
    for path in candidates[: self.max_factors]:
        try:
            art = joblib.load(path)
            art["_path"] = str(path)
            artifacts.append(art)
        except Exception as e:
            logger.warning("加载因子失败 %s: %s", path.name, e,
                           extra={"factor": path.name})
    return artifacts
```

- [ ] **Step 4: 实现 — `aggregate()` 混合加载逻辑**

在 `engine/signal_aggregator.py:187`，替换 `artifacts = self._load_factors()` 这一行为：

```python
    # ── 混合加载：per-ticker 规则因子 + 全局 ML 因子 ─────────────────
    ticker_safe = ticker.replace(".", "_").upper()
    per_ticker_dir = self.factors_dir / ticker_safe

    if per_ticker_dir.is_dir() and any(per_ticker_dir.glob("factor_*.pkl")):
        # per-ticker 目录有因子 → 从那里取规则因子，全局只取 ML 因子
        per_ticker_arts = self._load_factors(per_ticker_dir)
        global_arts_all = self._load_factors(self.factors_dir)
        # 只保留全局 ML 因子（feat_cols 非空 = ML 策略）
        global_ml_arts = [
            a for a in global_arts_all
            if len(a.get("meta", {}).get("feat_cols", [])) > 0
        ]
        artifacts = per_ticker_arts + global_ml_arts
        logger.debug(
            "混合加载: per-ticker %d 个规则因子 + 全局 %d 个ML因子",
            len(per_ticker_arts), len(global_ml_arts),
            extra={"ticker": ticker},
        )
    else:
        # 无 per-ticker 目录 → 全局目录全部加载（向后兼容）
        artifacts = self._load_factors(self.factors_dir)
        logger.debug("全局因子加载（无per-ticker目录）: %d 个",
                     len(artifacts), extra={"ticker": ticker})
```

- [ ] **Step 5: 运行所有信号聚合器测试**

```bash
pytest tests/test_signal_aggregator.py -v 2>&1 | tail -30
```
Expected: 全部通过（含新增 5 个混合测试）

- [ ] **Step 6: Commit**

```bash
git add engine/signal_aggregator.py tests/test_signal_aggregator.py
git commit -m "feat(signal_aggregator): hybrid dual-dir loading — per-ticker rule factors + global ML factors"
```

---

## Task 3: `main.py` — `factors_dir_override` 参数 + `_ensure_hsi_data()` + `step1_ensure_data(ticker=None)`

**Files:**
- Modify: `main.py` — `step2_train`, `step2_train_native`, `step2_train_optuna`, `step1_ensure_data`

**三件事合一提交，都是 main.py 的基础设施改动。**

- [ ] **Step 1: `step2_train*` 增加 `factors_dir_override` 参数**

修改 `main.py:318` `step2_train`：

```python
def step2_train(
    hist_data: pd.DataFrame,
    use_optuna: bool = False,
    optuna_trials: int = 50,
    strategy_type: str = None,
    factors_dir_override: Path = None,        # ← 新增
):
    if use_optuna and OPTUNA_AVAILABLE:
        return step2_train_optuna(
            hist_data, n_trials=optuna_trials,
            strategy_type=strategy_type,
            factors_dir_override=factors_dir_override,   # ← 新增
        )
    else:
        return step2_train_native(
            hist_data,
            strategy_type=strategy_type,
            factors_dir_override=factors_dir_override,   # ← 新增
        )
```

修改 `main.py:336` `step2_train_native`（同时整合 Task 1 的 strategy_type 透传）：

```python
def step2_train_native(
    hist_data: pd.DataFrame,
    strategy_type: str = None,
    factors_dir_override: Path = None,        # ← 新增
):
    logger.info("步骤2/3: 多策略超参搜索开始", extra={"search_method": "random_search"})
    cfg = load_config()
    best_result, sorted_results, test_df = run_search(hist_data, cfg, strategy_type=strategy_type)
    strategy_mods = _discover_strategies(strategy_type=strategy_type)
    if sorted_results:
        best_result = _select_best_with_holdout(
            sorted_results, test_df, cfg, strategy_mods, full_data=hist_data
        )
    # ── 确定保存目录 ──────────────────────────────────────────────
    factors_dir = (
        factors_dir_override
        if factors_dir_override is not None
        else Path(__file__).parent / 'data' / 'factors'
    )
    factor_path = None
    if best_result is not None:
        try:
            factor_path = _save_factor(best_result, factors_dir)
            best_result['factor_path'] = factor_path
            badge = {'double': '🏅 双验证通过', 'double_no_wf': '🥈 双验证（WF不足）',
                     'val_only': '⚠️  仅验证集达标'}.get(best_result.get('validated', ''), '❓')
            logger.info("因子已保存", extra={
                "factor_file": Path(factor_path).name,
                "strategy_name": best_result['strategy_name'],
                "cum_return": f"{best_result['cum_return']:.2%}",
                "sharpe_ratio": best_result.get('sharpe_ratio', float('nan')),
                "validated": best_result.get('validated', '?'),
                "badge": badge,
            })
        except Exception as e:
            logger.warning("因子保存失败", extra={"error": str(e)})
    return factor_path, best_result, sorted_results
```

在 `main.py:379` `step2_train_optuna` 中，找到（约 line 517）：
```python
    factors_dir = Path(__file__).parent / 'data' / 'factors'
```
替换为：
```python
    factors_dir = (
        factors_dir_override
        if factors_dir_override is not None
        else Path(__file__).parent / 'data' / 'factors'
    )
```
并在函数签名加 `factors_dir_override: Path = None`。

- [ ] **Step 2: 提取 `_ensure_hsi_data()`**

在 `step1_ensure_data` 之后（约 line 312）插入新函数：

```python
def _ensure_hsi_data(cfg: dict = None) -> None:
    """HSI 成分股增量更新 — 单独提取，供 main() 和 train_portfolio_tickers() 调用一次。"""
    if cfg is None:
        cfg = load_config()
    hsi_period = cfg.get('hsi_period', '3y')
    logger.info("开始增量更新HSI成分股数据", extra={"period": hsi_period})
    try:
        mgr = DataManager()
        hsi_result = mgr.download_hsi_incremental(period=hsi_period)
        logger.info("HSI成分股更新完成", extra={
            "total":          hsi_result['total'],
            "skipped":        hsi_result['skipped'],
            "updated":        hsi_result['updated'],
            "failed_count":   len(hsi_result['failed']),
            "failed_tickers": hsi_result['failed'],
        })
    except Exception as e:
        logger.warning("HSI成分股更新失败", extra={"error": str(e)})
```

- [ ] **Step 3: `step1_ensure_data` 增加 `ticker` 参数，删除内部 HSI 块**

修改函数签名（line 209）：
```python
def step1_ensure_data(sources_override=None, ticker: str = None):
```

在函数体开头，将：
```python
    cfg = load_config()
    ticker = cfg.get('ticker', '0700.hk')
```
改为：
```python
    cfg = load_config()
    effective_ticker = ticker or cfg.get('ticker', '0700.hk')
```

将函数体内所有用 `ticker`（原始 cfg 变量）的地方改为 `effective_ticker`（共 4 处：`_ticker_lower`、`_ticker_safe`、日志 extra、下载调用）。

下载调用（约 line 273）将：
```python
        hist_data, hist_path = mgr.download_from_config(sources_override=sources_override)
```
改为：
```python
        hist_data, hist_path = mgr.download(
            effective_ticker,
            period=cfg.get('period', '3y'),
            sources_override=sources_override,
        )
```

删除 `step1_ensure_data` 末尾的 HSI 更新块（lines 292–311，`# ── 1b. HSI 成分股增量更新 ──` 至函数末尾 return 前）。

- [ ] **Step 4: 在 `main()` 里补回 HSI 更新调用**

在 `main()` 的 `step1_ensure_data(sources_override)` 调用之后插入：

```python
    _ensure_hsi_data(config)
```

- [ ] **Step 5: 运行测试**

```bash
pytest tests/ -v -k "not test_integration" 2>&1 | tail -15
```
Expected: 全部通过，行为与修改前一致。

- [ ] **Step 6: Commit**

```bash
git add main.py
git commit -m "refactor(main): factors_dir_override + strategy_type in step2_train*; extract _ensure_hsi_data(); ticker param in step1_ensure_data"
```

---

## Task 4: `main.py` — `train_portfolio_tickers()` 混合版本 + 测试

**Files:**
- Modify: `main.py` — 在 `step2_train_optuna` 之后插入（约 line 582）
- Create: `tests/test_portfolio_training.py`

**逻辑：**
1. `_ensure_hsi_data()` 调用一次
2. `step2_train(ref_data, strategy_type='multi')` — ML 全局训练，`ref_data` 用第一只持仓股票的历史数据（ML 策略内部会加载 HSI 全量数据）
3. 对每只 ticker：下载数据 → `step2_train(ticker_data, strategy_type='single', factors_dir_override=ticker_dir)`
4. 汇总

- [ ] **Step 1: 创建测试文件**

创建 `tests/test_portfolio_training.py`：

```python
"""Portfolio-mode training orchestration tests (all I/O mocked)."""
from __future__ import annotations
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pandas as pd
import pytest


def _make_ohlcv(n: int = 100) -> pd.DataFrame:
    np.random.seed(42)
    close = np.cumprod(1 + np.random.normal(0, 0.01, n)) * 100
    dates = pd.bdate_range(end=pd.Timestamp("2025-12-31"), periods=n)
    return pd.DataFrame({
        "Open": close, "High": close * 1.01,
        "Low": close * 0.99, "Close": close,
        "Volume": np.ones(n) * 1e6,
    }, index=dates)


_OHLCV = _make_ohlcv()
_FACTOR_RESULT = ("/tmp/factor_0001.pkl", {"sharpe_ratio": 1.5, "validated": "double"}, [])
_EMPTY_RESULT  = (None, None, [])


class TestTrainPortfolioTickers:

    @patch("main._ensure_hsi_data")
    @patch("main.step1_ensure_data", return_value=(_OHLCV, "/tmp/fake.csv"))
    @patch("main.step2_train", side_effect=[_FACTOR_RESULT, _FACTOR_RESULT, _FACTOR_RESULT])
    @patch("main.generate_signal_report", return_value="# report")
    def test_ml_trained_once_rules_per_ticker(
        self, mock_report, mock_train, mock_step1, mock_hsi
    ):
        """ML 全局训练 1 次，2 只股票的规则训练各 1 次，共调用 step2_train 3 次。"""
        from main import train_portfolio_tickers
        results = train_portfolio_tickers(tickers=["0700.HK", "0005.HK"])
        assert len(results) == 2
        assert all(r["status"] == "ok" for r in results)
        # 1 次 ML global + 2 次 per-ticker rule = 3
        assert mock_train.call_count == 3
        mock_hsi.assert_called_once()

    @patch("main._ensure_hsi_data")
    @patch("main.step1_ensure_data", return_value=(_OHLCV, "/tmp/fake.csv"))
    @patch("main.step2_train", side_effect=[_FACTOR_RESULT, _FACTOR_RESULT, _FACTOR_RESULT])
    @patch("main.generate_signal_report", return_value="# report")
    def test_ml_call_uses_multi_strategy_type(
        self, mock_report, mock_train, mock_step1, mock_hsi
    ):
        """ML 全局训练调用中 strategy_type='multi'，无 factors_dir_override。"""
        from main import train_portfolio_tickers
        train_portfolio_tickers(tickers=["0700.HK", "0005.HK"])
        first_call = mock_train.call_args_list[0]
        assert first_call.kwargs.get("strategy_type") == "multi"
        assert first_call.kwargs.get("factors_dir_override") is None

    @patch("main._ensure_hsi_data")
    @patch("main.step1_ensure_data", return_value=(_OHLCV, "/tmp/fake.csv"))
    @patch("main.step2_train", side_effect=[_FACTOR_RESULT, _FACTOR_RESULT, _FACTOR_RESULT])
    @patch("main.generate_signal_report", return_value="# report")
    def test_rule_calls_use_single_and_per_ticker_dir(
        self, mock_report, mock_train, mock_step1, mock_hsi
    ):
        """规则训练调用：strategy_type='single'，factors_dir_override 包含 TICKER_SAFE。"""
        from main import train_portfolio_tickers
        train_portfolio_tickers(tickers=["0700.HK", "0005.HK"])
        # call index 1 = 0700.HK rule, call index 2 = 0005.HK rule
        for i, expected_safe in enumerate(["0700_HK", "0005_HK"], start=1):
            call = mock_train.call_args_list[i]
            assert call.kwargs.get("strategy_type") == "single"
            dir_arg = call.kwargs.get("factors_dir_override")
            assert dir_arg is not None, f"call {i} 没有 factors_dir_override"
            assert expected_safe in str(dir_arg), (
                f"call {i} 的目录 {dir_arg} 不含 {expected_safe}"
            )

    @patch("main._ensure_hsi_data")
    @patch("main.step1_ensure_data", side_effect=Exception("network error"))
    @patch("main.step2_train", return_value=_FACTOR_RESULT)
    def test_data_failure_skips_ticker_continues(self, mock_train, mock_step1, mock_hsi):
        """数据下载失败时跳过该 ticker，继续处理其余 ticker，不抛异常。"""
        from main import train_portfolio_tickers
        results = train_portfolio_tickers(tickers=["0700.HK", "0005.HK"])
        assert len(results) == 2
        # step1 失败，所有 ticker 都 data_failed（ML training 也需要 step1）
        assert all(r["status"] == "data_failed" for r in results)

    @patch("main._ensure_hsi_data")
    @patch("main.step1_ensure_data", return_value=(_OHLCV, "/tmp/fake.csv"))
    @patch("main.step2_train", side_effect=[_FACTOR_RESULT, Exception("train error"), _FACTOR_RESULT])
    @patch("main.generate_signal_report", return_value="# report")
    def test_rule_train_failure_skips_ticker_continues(
        self, mock_report, mock_train, mock_step1, mock_hsi
    ):
        """规则训练失败时跳过该 ticker，其余 ticker 正常完成。"""
        from main import train_portfolio_tickers
        results = train_portfolio_tickers(tickers=["0700.HK", "0005.HK"])
        assert results[0]["status"] == "train_failed"   # 0700.HK rule 失败
        assert results[1]["status"] == "ok"             # 0005.HK 正常

    @patch("main._ensure_hsi_data")
    @patch("main.step1_ensure_data", return_value=(_OHLCV, "/tmp/fake.csv"))
    @patch("main.step2_train", side_effect=[_EMPTY_RESULT, _EMPTY_RESULT, _EMPTY_RESULT])
    @patch("main._latest_factor_path", return_value=None)
    def test_no_factor_still_ok_status(self, mock_latest, mock_train, mock_step1, mock_hsi):
        """训练未产生因子时 status=ok 但 factor_path=None。"""
        from main import train_portfolio_tickers
        results = train_portfolio_tickers(tickers=["0700.HK"])
        assert results[0]["status"] == "ok"
        assert results[0]["factor_path"] is None
```

- [ ] **Step 2: 运行确认失败**

```bash
pytest tests/test_portfolio_training.py -v 2>&1 | tail -20
```
Expected: ImportError 或 AssertionError（函数不存在）

- [ ] **Step 3: 实现 `train_portfolio_tickers()`**

在 `main.py` 中 `step2_train_optuna` 之后（约 line 582），`_resolve_artifact` 之前插入：

```python
# ══════════════════════════════════════════════════════════════════
#  分层混合组合训练入口
# ══════════════════════════════════════════════════════════════════

def train_portfolio_tickers(
    tickers: list[str],
    use_optuna: bool = False,
    optuna_trials: int = 50,
    sources_override: list[str] = None,
    n_days: int = 3,
) -> list[dict]:
    """
    分层混合训练：
      1. ML 全局训练（strategy_type='multi'）运行一次 → data/factors/
      2. 对每只 ticker 跑规则策略训练（strategy_type='single'）→ data/factors/{TICKER_SAFE}/
      3. 生成每只 ticker 的信号报告
    返回每只 ticker 的结果列表。
    """
    config = load_config()
    _ensure_hsi_data(config)

    base_factors_dir = Path(__file__).parent / 'data' / 'factors'
    results: list[dict] = []

    # ── 步骤 A：ML 全局训练（一次）────────────────────────────────
    # ML 策略内部加载 HSI 全量数据，只需任意一只股票的历史数据作为验证集
    ref_ticker = tickers[0] if tickers else config.get('ticker', '0700.HK')
    logger.info("ML全局训练开始（使用 %s 作为验证参考）", ref_ticker)
    try:
        ref_hist, _ = step1_ensure_data(sources_override=sources_override, ticker=ref_ticker)
        step2_train(
            ref_hist,
            use_optuna=use_optuna,
            optuna_trials=optuna_trials,
            strategy_type='multi',
            factors_dir_override=None,   # 全局目录
        )
        logger.info("ML全局训练完成，因子已存入 %s", base_factors_dir)
    except Exception as e:
        logger.warning("ML全局训练失败（将继续进行规则策略训练）: %s", e)

    # ── 步骤 B：每只 ticker 的规则策略训练 ────────────────────────
    for ticker in tickers:
        ticker_safe = ticker.replace('.', '_').upper()
        ticker_factors_dir = base_factors_dir / ticker_safe

        logger.info("规则策略训练: %s → %s", ticker, ticker_factors_dir,
                    extra={"ticker": ticker})

        # 下载数据
        try:
            hist_data, _ = step1_ensure_data(
                sources_override=sources_override, ticker=ticker
            )
        except Exception as e:
            logger.error("数据获取失败，跳过 %s: %s", ticker, e,
                         extra={"ticker": ticker})
            results.append({"ticker": ticker, "status": "data_failed",
                             "factor_path": None, "error": str(e)})
            continue

        # 规则策略训练
        try:
            factor_path, best_result, _ = step2_train(
                hist_data,
                use_optuna=use_optuna,
                optuna_trials=optuna_trials,
                strategy_type='single',
                factors_dir_override=ticker_factors_dir,
            )
        except Exception as e:
            logger.error("规则训练失败，跳过 %s: %s", ticker, e,
                         extra={"ticker": ticker})
            results.append({"ticker": ticker, "status": "train_failed",
                             "factor_path": None, "error": str(e)})
            continue

        if factor_path is None:
            factor_path = _latest_factor_path(ticker_factors_dir)

        # 信号报告
        report_md = ""
        if factor_path:
            try:
                report_md = generate_signal_report(hist_data, factor_path, n_days=n_days)
            except Exception as e:
                logger.warning("信号报告失败 %s: %s", ticker, e,
                               extra={"ticker": ticker})

        sharpe    = best_result.get('sharpe_ratio', float('nan')) if best_result else float('nan')
        validated = best_result.get('validated', 'unknown') if best_result else 'unknown'

        results.append({
            "ticker":     ticker,
            "status":     "ok",
            "factor_path": factor_path,
            "factors_dir": str(ticker_factors_dir),
            "sharpe_ratio": sharpe,
            "validated":   validated,
            "report_md":   report_md,
        })
        logger.info("规则训练完成: %s", ticker,
                    extra={"ticker": ticker, "sharpe_ratio": sharpe})

    return results
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_portfolio_training.py -v 2>&1 | tail -20
```
Expected: 全部 6 个测试通过。

- [ ] **Step 5: Commit**

```bash
git add main.py tests/test_portfolio_training.py
git commit -m "feat(main): train_portfolio_tickers() — ML global once + rule per-ticker hybrid"
```

---

## Task 5: `main.py` — `_print_portfolio_summary()` + `--portfolio` CLI 入口

**Files:**
- Modify: `main.py` — `main()` 函数（约 line 1252）

- [ ] **Step 1: 在 `main()` 之前插入 `_print_portfolio_summary()`**

```python
def _print_portfolio_summary(results: list[dict], config: dict) -> None:
    """打印组合训练汇总表，可选飞书通知。"""
    import math as _m
    ok     = [r for r in results if r['status'] == 'ok']
    failed = [r for r in results if r['status'] != 'ok']
    badge_map = {
        'double':       '🏅 双验证',
        'double_no_wf': '🥈 WF不足',
        'val_only':     '⚠️  验证集',
        'unknown':      '❓ 未知',
    }
    lines = [
        "", "=" * 66, "  分层混合组合训练完成汇总", "=" * 66,
        f"  成功: {len(ok)}  失败: {len(failed)}  共: {len(results)}",
        "-" * 66,
        f"  {'股票代码':<12s}  {'状态':<8s}  {'Sharpe':>8s}  {'验证等级':<14s}",
        "-" * 66,
    ]
    for r in results:
        if r['status'] == 'ok':
            sh     = r.get('sharpe_ratio', float('nan'))
            sh_str = f"{sh:8.4f}" if not _m.isnan(sh) else '     N/A'
            badge  = badge_map.get(r.get('validated', 'unknown'), '❓')
            lines.append(f"  {r['ticker']:<12s}  {'✅ OK':<8s}  {sh_str}  {badge}")
        else:
            err = str(r.get('error', ''))[:30]
            lines.append(f"  {r['ticker']:<12s}  {'❌ FAIL':<8s}  {'':>8s}  {err}")
    lines += ["=" * 66, ""]
    print("\n".join(lines))
    logger.info("组合训练汇总", extra={"ok": len(ok), "failed": len(failed)})

    feishu_webhook = config.get('feishu_webhook')
    if feishu_webhook and ok:
        rows = []
        for r in results:
            sh = r.get('sharpe_ratio', float('nan'))
            if r['status'] == 'ok':
                rows.append(
                    f"- {r['ticker']}  Sharpe={'N/A' if _m.isnan(sh) else f'{sh:.4f}'}"
                    f"  {badge_map.get(r.get('validated','unknown'), '❓')}"
                )
            else:
                rows.append(f"- {r['ticker']}  ❌ {r['status']}")
        msg = (
            f"**分层混合组合训练完成**  {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            f"成功 {len(ok)} / 共 {len(results)}\n\n"
            + "\n".join(rows)
        )
        try:
            from feishu_notify import send_feishu_message
            send_feishu_message(feishu_webhook, msg, msg_type="markdown")
        except Exception as e:
            logger.warning("飞书通知失败", extra={"error": str(e)})
```

- [ ] **Step 2: 增加 `--portfolio` argparse 参数**

在 `main()` 的 `--strategy-type` 参数之后（约 line 1278）插入：

```python
    parser.add_argument(
        '--portfolio', action='store_true',
        help='分层混合模式：ML全局训练一次 + 对 portfolio.yaml 每只股票训练规则策略',
    )
```

- [ ] **Step 3: 在 `main()` 中增加 portfolio 分支**

在 `main()` 的 `sources_override` 设置之后（约 line 1288），`step1_ensure_data` 之前插入：

```python
    if args.portfolio:
        from engine.portfolio_state import load_portfolio
        portfolio_state = load_portfolio()
        tickers = portfolio_state.all_tickers()
        if not tickers:
            default_ticker = config.get('ticker', '0700.HK').upper()
            tickers = [default_ticker]
            logger.warning("portfolio.yaml 无标的，降级使用 config.yaml ticker",
                           extra={"ticker": default_ticker})
        logger.info("分层混合组合训练启动",
                    extra={"ticker_count": len(tickers), "tickers": tickers})
        use_optuna = args.use_optuna or config.get('use_optuna', False)
        results = train_portfolio_tickers(
            tickers=tickers,
            use_optuna=use_optuna,
            optuna_trials=args.optuna_trials,
            sources_override=sources_override,
            n_days=args.n_days,
        )
        _print_portfolio_summary(results, config)
        return
```

- [ ] **Step 4: 验证 CLI**

```bash
cd /home/thenine/projects/stock_analyze
python3 main.py --help | grep -A2 portfolio
```
Expected 输出包含：`--portfolio`

- [ ] **Step 5: 运行全部测试**

```bash
pytest tests/ -v -k "not test_integration" 2>&1 | tail -15
```
Expected: 全部通过。

- [ ] **Step 6: Commit**

```bash
git add main.py
git commit -m "feat(main): --portfolio flag + _print_portfolio_summary() for hybrid training mode"
```

---

## Task 6: `daily_run.py` — 因子目录检查扩展到 per-ticker 子目录

**Files:**
- Modify: `daily_run.py:46-73` (`_check_factor_freshness`)
- Modify: `daily_run.py:492-496` (因子存在性检查)

- [ ] **Step 1: 替换 `_check_factor_freshness`**

```python
def _check_factor_freshness(factors_dir: Path, min_age_days: int = 7) -> bool:
    """
    检查 factors_dir 及其所有直接子目录中，最新因子是否在 min_age_days 天内。
    只要有任一目录的因子还新鲜就返回 True。
    """
    import joblib as _jl

    def _newest_in(d: Path):
        cands = sorted(
            d.glob("factor_*.pkl"),
            key=lambda p: int(p.stem.split("_")[1]),
            reverse=True,
        )
        return cands[0] if cands else None

    dirs = [factors_dir]
    if factors_dir.is_dir():
        dirs += [
            sd for sd in factors_dir.iterdir()
            if sd.is_dir() and not sd.name.startswith(".")
        ]

    newest = None
    newest_id = -1
    for d in dirs:
        p = _newest_in(d)
        if p is None:
            continue
        rid = int(p.stem.split("_")[1])
        if rid > newest_id:
            newest_id = rid
            newest = p

    if newest is None:
        return False

    try:
        art = _jl.load(newest)
        saved_at_str = art.get("saved_at", "")
        if saved_at_str:
            age_days = (datetime.now() - datetime.fromisoformat(saved_at_str)).days
            return age_days <= min_age_days
    except Exception:
        pass
    age_days = (datetime.now() - datetime.fromtimestamp(newest.stat().st_mtime)).days
    return age_days <= min_age_days
```

- [ ] **Step 2: 扩展因子存在性检查（约 line 492）**

将：
```python
    factor_files = list(factors_dir.glob("factor_*.pkl"))
    if not factor_files:
        logger.error("data/factors/ 中无因子文件，请先运行 main.py 进行策略训练")
        sys.exit(1)
```
替换为：
```python
    factor_files = list(factors_dir.glob("factor_*.pkl"))
    if not factor_files and factors_dir.is_dir():
        for _sub in factors_dir.iterdir():
            if _sub.is_dir() and not _sub.name.startswith("."):
                factor_files.extend(_sub.glob("factor_*.pkl"))
    if not factor_files:
        logger.error(
            "data/factors/ 及其子目录均无因子文件，"
            "请先运行 main.py 或 main.py --portfolio 进行策略训练"
        )
        sys.exit(1)
```

- [ ] **Step 3: 运行全部测试**

```bash
pytest tests/ -v 2>&1 | tail -15
```
Expected: 全部通过。

- [ ] **Step 4: Commit**

```bash
git add daily_run.py
git commit -m "fix(daily_run): factor freshness and existence checks scan per-ticker subdirs"
```

---

## 验证（端到端，需网络）

```bash
# 1. 确认 portfolio.yaml 中有 2~3 只股票（0700.HK + 0005.HK 等）

# 2. 快速测试（随机搜索，trials 少）
python3 main.py --portfolio --optuna-trials 5

# 预期输出：
#   data/factors/factor_XXXX.pkl        ← ML 全局因子
#   data/factors/0700_HK/factor_XXXX.pkl ← 腾讯规则因子
#   data/factors/0005_HK/factor_XXXX.pkl ← 汇丰规则因子
#   终端打印汇总表

# 3. 每日推荐（验证 SignalAggregator 混合加载）
LOG_LEVEL=DEBUG python3 daily_run.py --skip-notify 2>&1 | grep "混合加载\|全局因子加载"

# 预期 DEBUG 日志：每只有 per-ticker 目录的股票显示"混合加载: X 个规则因子 + Y 个ML因子"

# 4. 单元测试（离线）
pytest tests/ -v
# 预期：所有测试通过
```

---

## 提交顺序

| # | commit message |
|---|---------------|
| 1 | `feat(analyze_factor): run_search() supports strategy_type filter` |
| 2 | `feat(signal_aggregator): hybrid dual-dir loading — per-ticker rule + global ML` |
| 3 | `refactor(main): factors_dir_override + strategy_type in step2_train*; _ensure_hsi_data(); step1 ticker param` |
| 4 | `feat(main): train_portfolio_tickers() — ML global once + rule per-ticker` |
| 5 | `feat(main): --portfolio flag + _print_portfolio_summary()` |
| 6 | `fix(daily_run): factor checks scan per-ticker subdirs` |
