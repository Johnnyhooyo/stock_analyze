# main.py 职责收敛 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 main.py 职责收敛到纯训练 + 训练结果通知，预测与操作建议全部交由 daily_run.py。

**Architecture:** 删除 `train_portfolio_tickers` 和 `main()` 中对 `generate_signal_report` 的调用；用已有的 `_print_portfolio_summary` 统一承担训练完成通知（终端输出 + 飞书推送）；将 sentiment/feishu 重型 import 下沉到函数体内，保持模块启动轻量。

**Tech Stack:** Python, unittest.mock (tests)

---

## File Map

| 文件 | 变更类型 | 说明 |
|------|---------|------|
| `main.py` | Modify | 4 处变更（见各任务） |
| `tests/test_portfolio_training.py` | Modify | 移除 `generate_signal_report` mock |

---

### Task 1: 从 `train_portfolio_tickers` 移除报告生成，同步更新测试

**Files:**
- Modify: `main.py:657-763`（`train_portfolio_tickers` 函数）
- Modify: `tests/test_portfolio_training.py`

- [ ] **Step 1: 更新测试 — 移除 `generate_signal_report` mock**

打开 `tests/test_portfolio_training.py`，对以下 4 个测试方法做相同修改：
`test_ml_trained_once_rules_per_ticker`、`test_ml_call_uses_multi_strategy_type`、
`test_rule_calls_use_single_and_per_ticker_dir`、`test_rule_train_failure_skips_ticker_continues`。

每个方法删除 `@patch("main.generate_signal_report", return_value="# report")` 装饰器，
并将方法签名中的 `mock_report` 参数删除。

修改前（以 `test_ml_trained_once_rules_per_ticker` 为例）：
```python
@patch("main._ensure_hk_data")
@patch("main.step1_ensure_data", return_value=(_OHLCV, "/tmp/fake.csv"))
@patch("main.step2_train", side_effect=[_FACTOR_RESULT, _FACTOR_RESULT, _FACTOR_RESULT])
@patch("main.generate_signal_report", return_value="# report")
def test_ml_trained_once_rules_per_ticker(
    self, mock_report, mock_train, mock_step1, mock_hk
):
```

修改后：
```python
@patch("main._ensure_hk_data")
@patch("main.step1_ensure_data", return_value=(_OHLCV, "/tmp/fake.csv"))
@patch("main.step2_train", side_effect=[_FACTOR_RESULT, _FACTOR_RESULT, _FACTOR_RESULT])
def test_ml_trained_once_rules_per_ticker(
    self, mock_train, mock_step1, mock_hk
):
```

对另外 3 个方法做同样处理（`mock_report` 参数和对应装饰器一起删除）。

- [ ] **Step 2: 运行测试确认当前状态**

```bash
cd /home/thenine/projects/stock_analyze
pytest tests/test_portfolio_training.py -v
```

预期：6 个测试全部通过（mock 只是抑制调用，移除后如果函数还在被调用会失败 — 用于验证下一步）

- [ ] **Step 3: 修改 `train_portfolio_tickers` — 删除 `n_days` 参数和 `generate_signal_report` 调用**

在 `main.py` 中修改 `train_portfolio_tickers` 函数签名，删除 `n_days: int = 3,` 参数：

```python
def train_portfolio_tickers(
    tickers: list[str],
    use_optuna: bool = False,
    optuna_trials: int = 50,
    sources_override: list[str] = None,
    skip_download: bool = False,
) -> list[dict]:
    """
    分层混合训练：
      1. ML 全局训练（strategy_type='multi'）运行一次 → data/factors/
      2. 对每只 ticker 跑规则策略训练（strategy_type='single'）→ data/factors/{TICKER_SAFE}/
    返回每只 ticker 的结果列表。
    """
```

- [ ] **Step 4: 删除 `train_portfolio_tickers` 内的 `report_md` 块**

找到以下代码块（ticker 循环内，step2_train 调用之后）并完整删除：

```python
        # 信号报告
        report_md = ""
        if factor_path:
            try:
                report_md = generate_signal_report(hist_data, factor_path, n_days=n_days)
            except Exception as e:
                logger.warning("信号报告失败 %s: %s", ticker, e,
                               extra={"ticker": ticker})
```

同时从 `results.append({...})` 字典中删除 `"report_md": report_md,` 一行：

```python
        results.append({
            "ticker":     ticker,
            "status":     "ok",
            "factor_path": factor_path,
            "factors_dir": str(ticker_factors_dir),
            "sharpe_ratio": sharpe,
            "validated":   validated,
            "ml_status":   ml_status,
        })
```

- [ ] **Step 5: 运行测试确认通过**

```bash
pytest tests/test_portfolio_training.py -v
```

预期：6 个测试全部 PASS

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_portfolio_training.py
git commit -m "refactor: remove generate_signal_report from train_portfolio_tickers"
```

---

### Task 2: 更新 `main()` 单股路径 — 删除 step 3，加训练通知

**Files:**
- Modify: `main.py:1489-1600`（`main` 函数）

- [ ] **Step 1: 删除 `--skip-train` 和 `--n-days` 参数**

找到 `main()` 中以下两段 `add_argument` 调用并删除：

```python
    parser.add_argument(
        '--skip-train', action='store_true',
        help='跳过超参搜索，直接使用 data/factors/ 中最新的因子做预测'
    )
    parser.add_argument(
        '--n-days', type=int, default=3,
        help='日线预测天数（默认 3）'
    )
```

- [ ] **Step 2: 删除 `--portfolio` 路径中的 `n_days` 参数**

找到 `train_portfolio_tickers` 调用，删除 `n_days=args.n_days,`：

```python
        results = train_portfolio_tickers(
            tickers=tickers,
            use_optuna=use_optuna,
            optuna_trials=args.optuna_trials,
            sources_override=sources_override,
            skip_download=args.skip_data_download,
        )
```

- [ ] **Step 3: 重写单股训练 + 通知逻辑**

找到从 `logger.info("腾讯股票分析流程启动")` 到函数末尾的代码，替换为：

```python
    logger.info("腾讯股票分析流程启动")

    # ── 步骤 1 ────────────────────────────────────────────────────
    hist_data, hist_path = step1_ensure_data(sources_override, skip_download=args.skip_data_download)

    if not args.skip_data_download:
        _ensure_hk_data(config)

    # ── 步骤 2 ────────────────────────────────────────────────────
    use_optuna = args.use_optuna if args.use_optuna else config.get('use_optuna', False)
    factor_path, best_result, _ = step2_train(
        hist_data,
        use_optuna=use_optuna,
        optuna_trials=args.optuna_trials,
        strategy_type=args.strategy_type,
    )
    if factor_path is None:
        factor_path = _latest_factor_path(Path(__file__).parent / 'data' / 'factors')
        if factor_path:
            logger.info("使用已有最新因子", extra={"factor_file": Path(factor_path).name})

    # ── 训练完成通知 ──────────────────────────────────────────────
    ticker = config.get('ticker', '0700.HK').upper()
    result = {
        "ticker":      ticker,
        "status":      "ok" if best_result is not None else "no_factor",
        "sharpe_ratio": best_result.get('sharpe_ratio', float('nan')) if best_result else float('nan'),
        "validated":   best_result.get('validated', 'unknown') if best_result else 'unknown',
        "ml_status":   "n/a",
    }
    _print_portfolio_summary([result], config)

    logger.info("训练流程完成")
```

- [ ] **Step 4: 运行冒烟测试确认 import 不报错**

```bash
cd /home/thenine/projects/stock_analyze
python3 -c "from main import train_portfolio_tickers, _print_portfolio_summary; print('OK')"
```

预期：打印 `OK`，无报错

- [ ] **Step 5: Commit**

```bash
git add main.py
git commit -m "refactor: main() single-stock path — remove step3 prediction, add training summary notify"
```

---

### Task 3: 清理顶层 imports + 加废弃注释

**Files:**
- Modify: `main.py:1-53`（文件头部 imports）
- Modify: `main.py:926`（`generate_signal_report` 函数定义处）

- [ ] **Step 1: 将重型 import 下沉到 `generate_signal_report` 函数体内**

找到文件顶层的两行 import：
```python
from feishu_notify import send_full_report_to_feishu
from sentiment_analysis import analyze_stock_sentiment, get_sentiment_signal
```
将这两行从顶层删除。

然后在 `generate_signal_report` 函数体的最开头（`if 'Close' not in data.columns:` 之前）添加：
```python
    from feishu_notify import send_full_report_to_feishu
    from sentiment_analysis import analyze_stock_sentiment, get_sentiment_signal
```

- [ ] **Step 2: 更新文件顶部 docstring**

找到文件开头的 docstring：
```python
"""
main.py — 腾讯股票分析一体化流程

步骤 1 : 数据就绪检查
        - 历史日线数据（本地不存在则下载）
步骤 2 : 多策略 × 100 次超参搜索
        - 复用 analyze_factor.__main__ 中的逻辑
        - 每个策略最多 max_tries 次随机参数组合
        - 保存所有满足阈值的因子，并输出排行榜
步骤 3 : 预测
        - 基于最优模型预测未来 n 个交易日（日线）
"""
```

替换为：
```python
"""
main.py — 腾讯股票分析训练流程

步骤 1 : 数据就绪检查
        - 历史日线数据（本地不存在则下载）
步骤 2 : 多策略超参搜索 + 因子保存
        - 每个策略最多 max_tries 次随机参数组合（或 Optuna 贝叶斯优化）
        - 保存所有满足阈值的因子到 data/factors/
        - 训练完成后推送飞书摘要

推断 / 操作建议请运行: python3 daily_run.py
"""
```

- [ ] **Step 3: 在 `generate_signal_report` 函数顶部加废弃注释**

找到 `def generate_signal_report(` 行，在其上方紧接着的 docstring 或函数体开头前加注释：

```python
def generate_signal_report(data: pd.DataFrame, factor_path: str, n_days: int = 3) -> str:
    # 暂停使用：预测职责已移至 daily_run.py
    # 本函数保留供调试，不在主流程中调用。
```

- [ ] **Step 4: 验证 import 干净**

```bash
python3 -c "import main; print('import OK')"
```

预期：打印 `import OK`，无 ImportError

- [ ] **Step 5: Commit**

```bash
git add main.py
git commit -m "refactor: move heavy imports into generate_signal_report body, update docstring"
```

---

### Task 4: 全量测试验证

**Files:** 无新改动

- [ ] **Step 1: 运行完整测试套件**

```bash
cd /home/thenine/projects/stock_analyze
pytest tests/ -v 2>&1 | tail -20
```

预期：所有测试通过（当前 219 个），无新失败

- [ ] **Step 2: 运行冒烟测试**

```bash
python3 smoke_test.py
```

预期：通过，无报错

- [ ] **Step 3: 如有失败，检查原因**

常见原因：
- `args.skip_train` 或 `args.n_days` 仍被引用 → 检查 Task 2 Step 1/3 是否完整
- `report_md` 仍被断言 → 检查测试中是否有遗漏的 assert

- [ ] **Step 4: Final commit（如 Step 3 有修复）**

```bash
git add -p
git commit -m "fix: address test failures from main.py convergence refactor"
```
