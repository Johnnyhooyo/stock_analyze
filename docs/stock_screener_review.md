# 选股模块设计方案 Review

> **Review 日期**：2026-04-03
> **Review 对象**：`docs/stock_screener_design.md`
> **当前实现状态**：`engine/stock_screener.py`、`engine/screener_factors.py`、`engine/screener_backtest.py`、`tests/test_stock_screener.py`、`tests/test_screener_backtest.py` 已存在，`daily_run.py` 集成已完成。

---

## 重要问题（实施前需解决）

### 1. `print()` 违反日志规范（Section 3.4）

设计文档中的集成示例代码使用 `print()`：

```python
print("\n  🔍 选股模块: 全市场扫描...")
print(f"  ✅ 选股完成: ...")
```

CLAUDE.md 明确：**禁止在核心管线中使用 `print()`**。集成代码应改用 `logger = get_logger(__name__)`。

> **现状**：实际 `daily_run.py` 实现已使用 `logger`，问题仅存在于设计文档示例代码中，可更新文档或保留作为"示意"说明。

---

### 2. `portfolio_state.add_watchlist_ticker()` 在设计文档中未定义

Section 3.4 的集成代码调用了 `portfolio_state.add_watchlist_ticker(pick.ticker)`，但 Section 3.3 的类设计中没有说明此方法需新增。

> **现状**：`engine/portfolio_state.py` 第 139 行已实现该方法，`daily_run.py` 已正常调用。设计文档未来修订时应在 `PortfolioState` 接口说明中补充此方法。

---

### 3. `ScreenerResult.sector` 填充来源不明（Section 3.2/3.3）

`sector_ranking()` 依赖 `result.sector` 字段，但 `screen()` 的伪代码没有说明 sector 如何赋值。Config 中的 `sectors` 是 `sector → [tickers]` 正向映射，需要在运行时反转查找。

> **现状**：`engine/stock_screener.py` 中已有 `_get_sector(ticker)` 私有方法实现反转查找，`_evaluate_ticker()` 调用并赋值到 `ScreenerResult.sector`。设计文档 Section 3.3 补充 `_get_sector()` 方法签名会更完整。

---

## 次要问题

### 4. 数据加载是串行的（Section 3.4）

集成示例代码用 `for t in candidate_tickers` 串行加载，而 `data/manager.py` 已有 `ThreadPoolExecutor` 并发能力。

> **现状**：`daily_run.py` 实际实现（第 626 行附近）在大规模扫描时有并发处理路径，但选股前的数据加载仍为串行。后续可考虑用 `ThreadPoolExecutor` 优化。

---

### 5. VWAP 因子对日线数据意义有限（Section 3.2 Tier 2）

Tier 2 列出的 "VWAP 偏离" 因子用于跨日趋势判断时，日线 VWAP 等同于 `(H+L+C)/3`，不如 `Close / MA20 - 1` 等偏离指标有意义。

> **现状**：`screener_factors.py` 中未实现 VWAP 因子（已用量比/OBV替代），设计文档 Tier 2 表格可删除或替换该条目。

---

### 6. 权重之和无校验逻辑（Section 3.3）

注释写 `# 总和应为 1.0`，但 `__init__` 里没有运行时断言，配置错误时会静默产生偏差。

> **建议**：在 `StockScreener.__init__` 中增加：
> ```python
> active_weights = {k: v for k, v in self.weights.items()
>                   if (k != "valuation" or self.enable_valuation)
>                   and (k != "sentiment" or self.enable_sentiment)}
> assert abs(sum(active_weights.values()) - 1.0) < 0.01, \
>     f"screener weights must sum to 1.0, got {sum(active_weights.values())}"
> ```

---

### 7. `top_n` 参数类型注解缺失（Section 3.3）

```python
def top_n(self, results, n=None, exclude_held=True, portfolio_state=None)
```

其他方法都有类型注解，`portfolio_state` 缺少 `Optional[PortfolioState]`。

---

### 8. 缺少测试计划（Section 4.1）

五个 Step 里没有测试。

> **现状**：`tests/test_stock_screener.py` 和 `tests/test_screener_backtest.py` 已实现。建议在 Section 4.1 表格中补充 "测试文件" 列，记录覆盖范围。

---

## 小问题

### 9. `screener_factors.py` 内容空白（Section 3.1）

模块结构列了 `screener_factors.py` 但正文没有对应设计说明，职责边界不清晰。

> **建议**：在 Section 3.1 或新增 Section 3.2.x 中补充 `ScreenerFactors` 类的接口说明（即 `FactorResult` dataclass + `calc_all()` 签名）。

---

### 10. Sector 列表硬编码维护负担

`config.yaml` 中 `sectors` 的 ticker 列表会随 HSI 成分调整而过期。

> **建议**：在 Section 6（风险与注意事项）中补充一条：定期同步 `data/hsi_stocks.py` 与 `sectors` 定义，或后期改为从 `hsi_stocks.py` 动态读取并通过预定义标签匹配。

---

## 总体评估

| 维度 | 评价 |
|------|------|
| 问题诊断 | 清晰，"选股盲区"描述准确 |
| 架构设计 | 漏斗式分层合理，与现有管线解耦良好 |
| 因子体系 | 分层清晰，Tier 1/2 轻量可行；VWAP 因子需替换 |
| 类接口设计 | `ScreenerResult` 数据结构干净；sector 填充逻辑有缺口 |
| 集成设计 | 整体可行；print()/logger 问题需修正 |
| 实施计划 | 分步合理；缺测试计划列 |

**结论**：设计方向正确。重要问题 1-3 均已在实际代码中解决，设计文档可按上述建议补充完善。后续实施可按 Step 2（集成报告）→ Step 3（板块分析）→ Step 5（回测验证）推进。
