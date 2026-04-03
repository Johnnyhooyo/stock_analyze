# PnL Tracker 代码审查报告

> **审查日期**：2026-04-03  
> **审查文件**：`data/pnl_tracker.py`、`tests/test_pnl_tracker.py`、`daily_run.py`（集成部分）  
> **对应设计文档**：`docs/design_pnl_tracker.md`  
> **测试状态**：32/32 passed（0.50s）

---

## 一、总体评价

实现质量扎实。原子写入、幂等性、T+1 回填跳过逻辑均正确；`raw_votes` 字段键名（`strategy_name`）与 `signal_aggregator.py:243` 一致；`daily_run.py` 调用顺序（`fill_t1_returns` → `record_daily`）正确；测试覆盖 32 个用例，超出设计文档预期的 15 个。

**结论：With fixes（两处必修，其余可同批处理）**

---

## 二、发现问题

### 2.1 Critical（上线前必修）

#### C-1 `fill_t1_returns` ZeroDivisionError — `pnl_tracker.py:97`

```python
t1_ret = t1_close / row["last_close"] - 1   # last_close == 0 时崩溃
```

**问题**：港股停牌时部分 vendor 返回 `last_close=0`。一条异常记录会导致整个 `fill_t1_returns` 调用中止，当日所有后续记录永久无法回填（`t1_close is not None` 守卫阻止重填）。

**修复方案**：

```python
if not row.get("last_close"):
    logger.warning("last_close 为 0，跳过 T+1 回填", extra={"ticker": ticker, "date": prev_date})
    continue
```

**需补充测试**：`test_fill_t1_returns_zero_last_close`

---

#### C-2 非交易日运行时 `fill_t1_returns` 回填错误数据 — `daily_run.py:779`

```python
if price_map:
    tracker.fill_t1_returns(prev_date.strftime("%Y-%m-%d"), price_map)  # 未检查 market_is_open
if market_is_open:
    tracker.record_daily(run_date, results)
```

**问题**：在周六或节假日运行 `daily_run.py` 时，`price_map` 来自缓存的旧收盘价（无新数据），`fill_t1_returns` 会将前一交易日记录回填 `t1_return_pct=0.0`，导致 `t1_correct` 计算错误，且因守卫机制无法重填。

**修复方案**：与 `record_daily` 使用相同守卫：

```python
if market_is_open:
    if price_map:
        tracker.fill_t1_returns(prev_date.strftime("%Y-%m-%d"), price_map)
    tracker.record_daily(run_date, results)
```

---

### 2.2 Important（使用摘要报告前修复）

#### I-1 `summary_report` 键名不一致 — `pnl_tracker.py:211`

```python
# 无 T+1 数据时返回：
return {"total": 0}

# 有数据时返回：
return {"total_recommendations": total, ...}
```

**问题**：调用方 `result.get("total_recommendations", 0)` 在前者路径下静默返回 `0`，飞书通知可能产生错误摘要。

**修复方案**：统一键名：

```python
return {"total_recommendations": 0}
```

同步更新 `tests/test_pnl_tracker.py:280` 的断言：

```python
assert result == {"total_recommendations": 0}
```

---

### 2.3 Minor（同批清理）

#### M-1 `attribution_by_strategy` 死代码 — `pnl_tracker.py:133`

```python
correct = record["t1_correct"]   # 赋值后从未使用
```

策略级别的准确性在第 135 行独立计算（`strat_correct`），此变量应删除。

---

#### M-2 设计文档伪代码与实现不一致 — `design_pnl_tracker.md` §3.1

设计文档伪代码：

```python
votes = {v["strategy"]: v["signal"] for v in agg.raw_votes if "strategy" in v}
```

实际实现（正确）：

```python
votes = {v["strategy_name"]: v["signal"] for v in agg.raw_votes}
```

字段名应更新为 `strategy_name`，与 `signal_aggregator.py:243` 保持一致。

---

## 三、改进计划

| 优先级 | 编号 | 文件 | 改动 |
|--------|------|------|------|
| Critical | C-1 | `data/pnl_tracker.py:97` | 在 `fill_t1_returns` 循环内添加 `last_close == 0` 跳过守卫 |
| Critical | C-2 | `daily_run.py:779` | 将 `fill_t1_returns` 移入 `if market_is_open:` 块 |
| Important | I-1 | `data/pnl_tracker.py:211` | 将 `{"total": 0}` 改为 `{"total_recommendations": 0}` |
| Important | I-1 | `tests/test_pnl_tracker.py:280` | 同步更新断言 |
| Minor | M-1 | `data/pnl_tracker.py:133` | 删除 `correct = record["t1_correct"]` |
| Minor | M-2 | `docs/design_pnl_tracker.md` §3.1 | 更新伪代码键名 `"strategy"` → `"strategy_name"` |
| Minor | — | `tests/test_pnl_tracker.py` | 补充 `test_fill_t1_returns_zero_last_close` |

---

## 四、优点记录

- `_existing_keys()` 集合查重：正确且高效
- `_write_all` 原子写入（tmp → replace）：与项目其他存储模块一致
- `fill_t1_returns` 幂等（`t1_close is not None` 守卫）：可安全重复调用
- `_rec_to_row` 使用正确的 `strategy_name` 字段（设计文档有误，代码正确）
- `daily_run.py` 调用顺序：`fill_t1_returns(prev)` → `record_daily(today)`，逻辑正确
- `export_csv` 使用 `utf-8-sig`：Excel 中文兼容正确
- 测试覆盖：32 个用例覆盖全部边界，含幂等、零填充、多 ticker、原子写入

---

## 五、与设计文档对照

| 设计文档需求 | 状态 | 备注 |
|-------------|------|------|
| 追加快照到 `pnl_history.jsonl` | ✅ | |
| date×ticker 幂等 | ✅ | |
| T+1 收益回填 | ✅ 待修 | ZeroDivisionError 未守卫（C-1） |
| 按策略 / 标的 / 操作归因 | ✅ 待修 | 死代码（M-1） |
| CSV 导出 | ✅ | |
| `daily_run.py` 集成 | ✅ 待修 | 非交易日守卫缺失（C-2） |
| `data/__init__.py` 导出 | ✅ | |
| `.gitignore` 条目 | ✅ | |
| `profit_amount` 字段 | ✅ 偏差已修正 | 实现正确使用 `rec.profit`，设计文档有误 |
| 测试数量（预计 15） | ✅ 超出 | 实际 32 个 |
