# 港股量化分析系统 — 2026 Q2 执行计划

> 基于 2026-04-05 全面代码审计生成  
> 当前状态：229 测试全通过 | Phase 1 完成 | Phase 2.5 全部完成（BUG-1/2/3/4/5/6/7 + OPT-1/3 已修复）  
> 分支：`main`

---

## 目录

- [一、现状总结](#一现状总结)
- [二、Bug 修复清单](#二bug-修复清单)
- [三、代码优化清单](#三代码优化清单)
- [四、执行计划](#四执行计划)
  - [Phase 2.5 — 稳定性与可靠性（1-2 周）](#phase-25--稳定性与可靠性12-周)
  - [Phase 3 — 策略与性能增强（3-4 周）](#phase-3--策略与性能增强34-周)
  - [Phase 4 — 平台化与实盘（6-8 周）](#phase-4--平台化与实盘68-周)
- [五、风险与依赖](#五风险与依赖)

---

## 一、现状总结

### 已完成

| 阶段 | 内容 | 状态 |
|------|------|------|
| Phase 1 | 基础债务清理（portfolio_state、统一配置、公共指标、测试框架、结构化日志） | ✅ 全部完成 |
| Phase 2-4 | PnL 历史追踪 | ✅ `data/pnl_tracker.py` |
| Phase 2-6 | 深度学习策略 | ✅ `strategies/rnn_trend.py` (GRU/LSTM) |
| Phase 2-3 | 因子生命周期管理 | ✅ `data/factor_registry.py` + `signal_aggregator.py` + `main.py` 集成完整 |
| Phase 2-7 | main.py 职责收敛 | ✅ 预测步骤移除，仅保留训练 + 飞书摘要；`generate_signal_report` 暂停调用 |
| 新增 | 全市场选股引擎 | ✅ `engine/stock_screener.py` |
| 新增 | 数据质量过滤 + 黑名单 | ✅ `data/hk_stocks.py` + `hk_stocks_blacklist.json` |

### 核心指标

- **测试**：223 个单元测试全通过，核心覆盖率 > 60%
- **策略**：9 个规则策略 + 3 个 ML 策略 + 1 个深度学习策略 = 13 个
- **数据源**：6 个 vendor（yahooquery → yfinance → akshare → ...）
- **股票池**：全量港股 2767+ 只（含质量过滤 + 黑名单）

### 已知阻塞性问题

~~1. `config.yaml` 关键字段被删减，训练阈值失效~~ → ✅ 已修复（BUG-1，2026-04-07）  
~~2. 因子注册表集成不完整，`daily_run` 可能返回空信号~~ → ✅ 已修复（BUG-2，2026-04-07）

**当前无阻塞性问题。**

---

## 二、Bug 修复清单

### BUG-1: config.yaml 关键字段丢失（P0 — 阻塞性）✅ 已修复 2026-04-07

**文件**: `config.yaml`

**现象**: 当前工作区的 `config.yaml` 丢失了大量关键配置项：

| 丢失字段 | 影响 |
|----------|------|
| `ticker`, `period`, `train_years` | 训练流程无法正确识别目标股票和数据范围 |
| `backtest_engine`, `data_sources` | 回退到默认值，行为不可预测 |
| `feishu_webhook` | 飞书通知完全失效 |
| `min_sharpe_ratio: 0.5`, `max_drawdown: -0.30`, `min_total_trades: 3` | 质量阈值丢失 |
| `test_days`, `test_months`, `early_stop_threshold`, `wf_min_window_win_rate` | 验证流程参数缺失 |
| `rnn_trend` (multi), 6个 single 策略 | 13 个策略中仅剩 5 个参与训练 |
| 整个 `daily_run` 配置块 | 选股模块、并发数、因子过期等配置丢失 |

**特别危险**: `min_return: -999.0` 意味着任何亏损策略都能通过验证，完全丧失质量把关。

**修复方案**: 从 git 历史恢复完整 config.yaml，按 CLAUDE.md 文档默认值重建，同时修复 `analyze_factor.run_search()` 中 `_save_config(base_cfg)` 副作用（每次测试调用会把 stripped config 写回磁盘）。

**验证**: `python3 daily_run.py --dry-run --skip-notify --skip-sentiment` 正常输出建议，223 个测试运行后 config.yaml 内容不变。

---

### BUG-2: 因子注册表集成不完整导致空信号（P0 — 阻塞性）✅ 已修复 2026-04-07

**文件**: `engine/signal_aggregator.py:88-203`, `main.py:195-222`

**现象**: `SignalAggregator` 默认 `use_registry=True`。当注册表存在但 active 记录为空时，`_filter_by_registry()` 过滤掉所有因子，导致 `aggregate()` 返回默认空仓信号。

**根因**: `_save_factor()` 中的 `registry.register()` 已实现（`main.py:207-219`），但 `subdir` 参数的计算逻辑与 `_filter_by_registry()` 中的匹配逻辑可能不一致：

```python
# main.py:202-205 — 保存时的 subdir 逻辑
if factors_dir.name.endswith('_HK') or factors_dir.name.endswith('_HK_s'):
    subdir = factors_dir.name
elif factors_dir != (Path(__file__).parent / 'data' / 'factors'):
    subdir = factors_dir.name

# signal_aggregator.py:219 — 过滤时的 subdir
ticker_safe = ticker.replace(".", "_").upper()  # "0700_HK"
per_ticker_dir = self.factors_dir / ticker_safe
# 但 _filter_by_registry 传入的 subdir=ticker_safe（大写）
# 而注册时 factors_dir.name 可能是小写或不同格式
```

**修复方案**:
1. 统一 subdir 命名规范：`ticker.replace(".", "_").upper()` 作为唯一标准
2. 在 `_filter_by_registry` 中增加大小写不敏感匹配
3. 增加 fallback：当注册表过滤后为空但目录中有文件时，记录警告并返回原始列表
4. 补充集成测试：`test_save_factor_then_aggregate` 端到端验证

**验证**: `tests/test_integration.py::TestFactorRegistryAggregateIntegration` 4 个端到端测试全通过，覆盖：注册→聚合、per-ticker subdir 匹配、subdir 不匹配 fallback、空注册表降级。

---

### BUG-3: `_discover_strategies()` 在信号生成时重复调用（P1）

**文件**: `engine/signal_aggregator.py:148`

**现象**: `_get_signal_from_artifact()` 中每个因子推断都调用 `_discover_strategies()` 全量扫描策略模块。20 个因子 x 6 只股票 = 120 次重复的 `importlib.import_module()` + 配置解析。

**修复方案**: 在 `SignalAggregator.__init__` 中一次性加载策略模块列表，存为 `self._strategy_modules` dict：

```python
# __init__ 中
self._strategy_modules = {
    mod.NAME: mod for mod in _discover_strategies()
}

# _get_signal_from_artifact 中
strategy_mod = self._strategy_modules.get(strategy_name)
```

**验证**: `daily_run --dry-run` 耗时对比（预期 3-5x 提速）。

---

### BUG-4: Sortino Ratio 计算错误（P1）

**文件**: `analyze_factor.py:534-538`

**现象**: 分子为日均收益率 `mean_ret`，分母为年化下行标准差 `downside_returns.std() * sqrt(252)`，量纲不一致，Sortino 被系统性低估约 252 倍。

**当前代码**:
```python
sortino_ratio = mean_ret / (downside_returns.std(ddof=1) * np.sqrt(252))
```

**修复**:
```python
sortino_ratio = (mean_ret * 252) / (downside_returns.std(ddof=1) * np.sqrt(252))
```

**验证**: 单元测试 `test_backtest.py` 增加 Sortino 断言。

---

### BUG-5: `compute_ic` 常量输入警告（P2）

**文件**: `analyze_factor.py:119-123`

**现象**: 测试输出 `ConstantInputWarning` — 当因子序列或收益率序列为常量时，`spearmanr` 返回 NaN 并产生警告。

**修复**:
```python
def compute_ic(factor, forward_returns):
    combined = pd.concat([factor, forward_returns], axis=1).dropna()
    if len(combined) < 5:
        return float('nan')
    # 检查常量输入
    if combined.iloc[:, 0].nunique() < 2 or combined.iloc[:, 1].nunique() < 2:
        return float('nan')
    r, _ = _scipy_stats.spearmanr(combined.iloc[:, 0], combined.iloc[:, 1])
    return float(r)
```

**验证**: `pytest tests/test_integration.py -v` 无 warning。

---

### BUG-6: 风控状态文件非原子写入 + 全局共享（P2）

**文件**: `position_manager.py:95-116`

**现象**:
1. `_save_risk_state()` 直接 `json.dump()`，非原子写入
2. 状态文件 `data/logs/risk_state.json` 是全局单文件，但 `PositionManager` 是 per-ticker 实例
3. 多股票并发分析时可能互相覆盖状态

**修复方案**:
1. 改为 per-ticker 状态文件：`data/logs/risk_state_{ticker_safe}.json`
2. 使用 tempfile + os.replace 原子写入（与 `_atomic_write_csv` 一致）
3. 或者整合到 `portfolio.yaml` 的 `consecutive_loss_days` 字段（已有）

**验证**: 并发分析 3 只股票，检查各自状态文件独立。

---

### BUG-7: 回测胜率计算遗漏未平仓交易（P2）

**文件**: `analyze_factor.py:500-525`

**现象**: 嵌套循环 O(n^2) 匹配买卖对，如果最后一笔是买入但未卖出，该笔交易被忽略。对于趋势策略（常在牛市末尾仍持仓），胜率可能偏高（幸存者偏差）。

**修复方案**: 将最后未平仓交易以当前价格虚拟平仓，纳入胜率统计：

```python
# 在 trade_returns 循环后
if trades.iloc[-1]['trade'] == 1:  # 最后一笔是买入，未平仓
    entry_price = trades.iloc[-1]['Close']
    exit_price = bt.iloc[-1]['Close']  # 用回测最后一天收盘价
    ret = (exit_price - entry_price) / entry_price
    trade_returns.append(ret)
```

**验证**: `test_backtest.py` 增加「末尾持仓」场景断言。

---

## 三、代码优化清单

### OPT-1: 过多 `except Exception: pass` 静默吞异常（P1）

**范围**: 全项目 193 处 `except Exception`，其中 26+ 处直接 `pass`

**策略**: 分三层处理：
- **关键路径**（数据下载、因子保存、风控状态）：改为 `logger.warning` + 具体异常类型
- **降级路径**（情感分析、Google Trends）：保留 `except`，但加 `logger.debug`
- **初始化路径**（可选依赖导入）：保持现状（合理的 try/except ImportError）

**估计工作量**: 审计 + 修改约 40 处关键路径。

---

### OPT-2: `backtest()` 逐行循环性能瓶颈（P2）

**文件**: `analyze_factor.py:423-459`

**现象**: `for idx, row in bt.iterrows()` 逐行模拟。Optuna 100 trials x 9 策略 = 900 次调用。

**优化方案**: 向量化重写核心循环：
1. 用 `np.where` + `diff()` 检测信号切换点
2. 批量计算买入/卖出价格（含滑点和费用）
3. 用 `cumsum()` / `cumprod()` 计算持仓市值序列

**预期**: 10-50x 加速（从 ~5ms/次 → ~0.1ms/次）。

**风险**: 向量化回测需处理边界条件（资金不足、部分成交），需充分测试。

---

### OPT-3: `_discover_strategies()` 缺乏缓存（P1）

**文件**: `analyze_factor.py:67-105`

**修复**: 添加模块级缓存：

```python
_strategy_cache: dict[str, list] = {}

def _discover_strategies(strategy_type=None, cfg=None):
    cache_key = strategy_type or "__all__"
    if cache_key in _strategy_cache:
        return _strategy_cache[cache_key]
    # ... 现有逻辑 ...
    _strategy_cache[cache_key] = modules
    return modules
```

---

### OPT-4: `main.py` 过于庞大（P2）— 🚧 部分完成

**现状（2026-04-05 更新）**: 预测/报告职责已从 `main.py` 移除（`generate_signal_report` 暂停，飞书改为推送训练摘要）。`main.py` 现在仅负责数据检查 + 训练 + 因子保存 + 训练通知，约 1400 行。

**剩余重构方案**（Phase 4 延续）:
```
pipeline/
├── __init__.py
├── data_prep.py       # step1_ensure_data, _ensure_hk_data
├── train.py           # step2_train_native, step2_train_optuna
└── select.py          # _select_best_with_holdout, _save_factor
```

`main.py` 缩减为 ~100 行的入口协调器。

---

### OPT-5: `train_multi_stock.py` 硬编码策略导入（P3）

**文件**: `train_multi_stock.py:14-15`

```python
from strategies.xgboost_enhanced import add_features, prepare_data
from strategies.lightgbm_enhanced import run as run_lgbm
```

**问题**: 与策略自动发现机制矛盾，新增 ML 策略需手动修改此文件。

**修复**: 改用 `_discover_strategies(strategy_type='multi')` 动态获取。

---

### OPT-6: DataManager 重复实例化（P3）

**现象**: `main.py`、`daily_run.py`、`_analyze_one_ticker()` 等多处 `DataManager()`。

**修复**: 在入口函数中创建一次，通过参数传递。`daily_run.py` 已部分实现（`data_mgr` 参数），`main.py` 尚未统一。

---

## 四、执行计划

### Phase 2.5 — 稳定性与可靠性（1-2 周）

> 目标：修复阻塞性 Bug，确保系统可正确运行

```
Week 1（2026-04-05 ~ 2026-04-07 完成）
├── Day 1-2: BUG-1 恢复 config.yaml + BUG-2 因子注册表集成         ✅
│   ├── [1] 恢复完整 config.yaml，修复 min_return: -999.0           ✅
│   ├── [2] 移除 run_search() 中 _save_config() 副作用调用          ✅
│   ├── [3] _filter_by_registry 大小写不敏感 + fallback             ✅
│   ├── [4] 端到端测试：TestFactorRegistryAggregateIntegration      ✅
│   └── [5] 验证：daily_run --dry-run 正常输出                      ✅
│
├── Day 3: BUG-3 策略发现缓存 + BUG-4 Sortino 修复                  ✅（f36d4ae）
│   ├── [1] SignalAggregator.__init__ 缓存策略模块                  ✅
│   ├── [2] 修复 Sortino 年化分子                                   ✅
│   ├── [3] compute_ic 常量输入检查（BUG-5）                        ✅
│   └── [4] 回测未平仓交易处理（BUG-7）                             ✅
│
└── Day 4-5: BUG-6 + OPT-1 异常处理审计                             ✅（2026-04-07）

Week 2
├── Day 1-2: OPT-1 关键路径异常处理审计 + 代码清理                   ✅（2026-04-07）
│
└── Day 3: 发布 & 文档更新                                           ✅（2026-04-07）
```

**交付物**:
- config.yaml 完整恢复 ✅
- 因子注册表端到端可用 ✅
- Sortino/胜率/IC 计算正确 ✅
- `_discover_strategies` 缓存，daily_run 耗时降低 ✅
- 测试数量 219 → 223 ✅
- BUG-6（风控状态 per-ticker 隔离 + 原子写入 + 6 个隔离测试）✅
- OPT-1（关键路径异常处理：position_manager/analyze_factor/main/data/manager 等 16 处）✅

---

### Phase 3 — 策略与性能增强（3-4 周）

> 目标：提升策略质量和系统性能

#### Step 3.1: 回测引擎向量化（Week 3）

**影响文件**: `analyze_factor.py:377-556`

**实施步骤**:
1. 新建 `backtest_vectorized()` 函数，与 `backtest()` 并行存在
2. 用 `np.where` + 信号 diff 检测交易点
3. 向量化计算持仓市值和费用
4. 交叉验证：对比 `backtest()` 和 `backtest_vectorized()` 在相同输入下的输出
5. 性能达标后（>10x 加速），替换默认引擎
6. 保留 `backtest()` 作为参考实现（`_backtest_reference()`）

**测试**: 在 `test_backtest.py` 中增加 `test_vectorized_matches_reference` 对比测试。

#### Step 3.2: Ensemble/Stacking 策略（Week 4）

**新建文件**: `strategies/ensemble_stacking.py`

**设计**:
```
第一层: 各基础策略独立产生信号 (已有)
第二层: Meta-learner (新增)
  - 输入: 各策略信号 + 各策略 Sharpe/IC + 市场状态指标
  - 模型: LightGBM 或 Logistic Regression
  - 输出: 最终共识信号 + 置信度
  - 训练: Walk-Forward，避免 look-ahead
```

**集成点**: `signal_aggregator.py` 增加 `aggregation_method` 参数：
- `"vote"` — 当前的 Sharpe 加权投票（默认）
- `"stacking"` — Meta-learner 融合

#### Step 3.3: Portfolio-Level 风控（Week 5）

**新建文件**: `engine/portfolio_risk.py`

**功能清单**:

| 功能 | 描述 | 触发条件 |
|------|------|---------|
| 行业集中度限制 | 同板块持仓不超过总资产 40% | 买入前检查 |
| 总仓位上限 | 总持仓市值不超过总资产 80% | 买入前检查 |
| 相关性监控 | 持仓股票间 20 日滚动相关系数 | 相关性 > 0.8 时警告 |
| 组合级止损 | 组合总亏损超过 10% 时触发全面减仓 | 每日检查 |
| VaR 预警 | 95% VaR 超过阈值时警告 | 每日检查 |

**集成**: `daily_run.py` 在个股分析后增加组合风控检查，可覆盖个股建议。

#### Step 3.4: Walk-Forward 集成到 Optuna（Week 5-6）

**影响文件**: `optimize_with_optuna.py`, `validate_strategy.py`

**方案**: 在 Optuna 目标函数中集成 Walk-Forward：

```python
def objective(trial):
    # ... 现有参数采样 + 回测 ...
    bt_result = backtest(data, signal, config)
    
    # 新增：WF 验证作为多目标
    wf = walk_forward_analysis(data, strategy_mod, config, params)
    
    # Optuna 多目标优化
    return bt_result['sharpe_ratio'], wf['window_win_rate']
```

WF 胜率不达标的 trial 直接被 pruner 剪枝，减少无效计算。

#### Step 3.5: 因子组合优化（Week 6）

**影响文件**: `engine/signal_aggregator.py`

**方案**: 用 Rank IC 替代 Sharpe 作为投票权重：

```python
# 当前
weight = max(sharpe, 0.01)

# 优化后
weight = max(abs(factor_ic), 0.01)  # IC 越高权重越大
```

IC 直接衡量预测能力，比 Sharpe（回测拟合度）更能反映因子真实价值。需要在因子保存时同时存储 IC 值。

---

### Phase 4 — 平台化与实盘（6-8 周）

> 目标：从研究工具过渡到生产系统

#### Step 4.1: main.py 重构（Week 7）— 🚧 部分完成

**2026-04-05 进展**: `predict.py` / `report.py` 职责已通过「main.py 职责收敛」从主流程中移除（`generate_signal_report` 暂停调用，预测交由 `daily_run.py`）。

**剩余重构方案**:
```
pipeline/
├── __init__.py          # 导出 run_training_pipeline
├── data_prep.py         # step1_ensure_data, _ensure_hk_data, _hist_data_is_stale
├── train.py             # step2_train_native, step2_train_optuna, run_search
├── select.py            # _select_best_with_holdout, _save_factor, _next_factor_run_id
└── train_portfolio.py   # train_portfolio_tickers（从 main.py 提取）
```

**原则**:
- `main.py` 缩减为 ~100 行入口
- 各子模块可独立测试
- 保持 CLI 接口不变（向后兼容）

#### Step 4.2: OMS 实盘对接（Week 8-9）

**新建文件**: `oms_futu.py`（Futu OpenD 适配器）

**功能**:
1. 继承 `oms.py` 的 `OrderManagementSystem` 接口
2. 通过 Futu OpenD API 提交真实订单
3. 订单状态回调 → 自动更新 `portfolio.yaml`
4. 限价单 / 市价单支持
5. 安全保护：单笔金额上限、日内总金额上限

**切换方式**: `config.yaml → broker_api_url` 从 `null`（PaperOMS）改为实际 URL。

#### Step 4.3: Web Dashboard（Week 10-11）

**技术选型**: Streamlit（快速原型）或 Grafana + InfluxDB（生产级）

**页面规划**:
1. **持仓总览**: 各股票持仓/盈亏/止损位
2. **信号面板**: 各策略最新信号 + 共识投票详情
3. **PnL 曲线**: 每日 PnL 追踪（已有 `pnl_tracker.py` 数据源）
4. **因子健康度**: 注册表状态、TTL 倒计时、Sharpe 趋势
5. **选股推荐**: 当日 Top-N + 板块强弱热力图

#### Step 4.4: A/B 测试框架（Week 12）

**设计**:
```
ab_test/
├── manager.py      # 创建/管理 A/B 实验
├── splitter.py     # 将资金按比例分配到不同策略组合
└── evaluator.py    # 对比各组 PnL、Sharpe、最大回撤
```

**流程**: 新策略上线前，先用 10% 资金（虚拟或真实）运行 30 天，与基线策略对比。

#### Step 4.5: 多市场扩展（Week 13+）

**抽象层**:
```python
# markets/base.py
class MarketConfig:
    calendar_id: str       # "XHKG", "XSHG", "XNYS"
    fees: FeeStructure     # 佣金、印花税、过户费
    lot_size: int          # 港股100, A股100, 美股1
    currency: str          # HKD, CNY, USD

# markets/hk.py, markets/cn.py, markets/us.py
```

**优先级**: A 股（XSHG/XSHE） > 美股（XNYS/XNAS）

#### Step 4.6: MLOps 管道（Week 14+）

**组件**:
1. **模型版本管理**: MLflow 或 Weights & Biases 记录每次训练的参数/指标/模型文件
2. **自动重训触发**: 因子 TTL 过期或 Sharpe 劣化 > 20% 时自动触发 `main.py --use-optuna`
3. **模型退化监控**: 每日对比预测信号 vs 实际收益，IC 连续 5 天低于 0.02 时告警
4. **数据漂移检测**: 特征分布 KS 检验，漂移时强制重训

---

## 五、风险与依赖

### 技术风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 向量化回测引入精度差异 | 高 | 中 | 保留参考实现，交叉验证 |
| Futu API 限速/断连 | 中 | 高 | 重试 + 降级到 PaperOMS |
| Optuna 多目标优化收敛慢 | 中 | 低 | 设置 trial 上限 + early pruning |
| 全量港股扫描耗时过长 | 低 | 中 | 增量扫描 + 预过滤 |

### 外部依赖

| 依赖 | 用途 | 替代方案 |
|------|------|---------|
| Yahoo Finance API | 主数据源 | akshare / Alpha Vantage |
| Futu OpenD | 实盘交易 | Tiger / IBKR |
| exchange_calendars | 港股日历 | 自维护假期表 |
| Optuna | 超参优化 | 原生随机搜索 |

### 里程碑

| 日期 | 里程碑 | 验收标准 |
|------|--------|---------|
| Week 2 末 | Phase 2.5 完成 | 全部 Bug 修复，daily_run 端到端正确 |
| Week 6 末 | Phase 3 完成 | Ensemble 策略上线，回测提速 10x+ |
| Week 9 末 | OMS 实盘可用 | Futu 纸上交易模式跑通 |
| Week 12 末 | Phase 4 核心完成 | Dashboard + A/B 框架可用 |

---

*本文档由代码审计自动生成，随项目进展持续更新。*
