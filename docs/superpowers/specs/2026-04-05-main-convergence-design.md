# main.py 职责收敛设计

**日期**: 2026-04-05  
**分支**: feature/hybrid-portfolio-training  
**目标**: 将 main.py 职责收敛到纯训练，预测与操作建议全部交由 daily_run.py 负责。

---

## 背景

当前 main.py 承担三个职责：
1. 数据下载（step1）
2. 策略训练 + 因子保存（step2）
3. 信号预测 + IC 分析 + 飞书推送（step3，`generate_signal_report`）

step3 与 daily_run.py 的功能高度重叠：daily_run 通过 `SignalAggregator` + `PositionAnalyzer` 做多策略聚合推荐，质量更高、覆盖整个 portfolio。main.py 的预测只用单因子文件、只处理单只股票，是劣质的重复。

---

## 目标状态

```
main.py     →  数据下载 + 训练 + 保存因子 + 推送训练摘要
daily_run.py →  加载因子 + 聚合信号 + 生成建议 + 推送操作报告
```

---

## 变更范围（方案 A）

### 1. `main()` 函数

**删除：**
- step 3 调用：`generate_signal_report(hist_data, factor_path, n_days=args.n_days)`
- `--skip-train` 参数（此场景属于 daily_run）
- `--n-days` 参数（只服务于预测）

**保留：**
- `--sources`、`--portfolio`、`--use-optuna`、`--optuna-trials`、`--strategy-type`、`--skip-data-download`

**新增：**
- 训练完成后调用 `_notify_training_done(results)`

### 2. `train_portfolio_tickers()`

**删除：**
- 每只 ticker 循环内的 `generate_signal_report` 调用
- `report_md` 变量和 results dict 中的 `report_md` 字段
- 函数签名中的 `n_days` 参数

**新增：**
- 全部 ticker 训练完成后调用 `_notify_training_done(results)`

### 3. 新增 `_notify_training_done(results: list[dict]) -> None`

推送飞书 markdown，内容：

```
训练完成 — {YYYY-MM-DD}
共训练 N 只股票，成功 ok，失败 failed

| 股票 | 最佳 Sharpe | Validated | 状态 |
|------|-------------|-----------|------|
| 0700.HK | 1.42 | yes | ok |
...

ML全局训练：成功 / 失败 / 未运行
```

- 数据来源：`step2_train` 返回的 `best_result` dict（已有字段，无需额外计算）
- 单只股票走 `main()` 时，构造单元素 `results` 列表，格式一致
- 无飞书 webhook 时只打 INFO 日志，不报错

### 4. Import 清理

- `send_full_report_to_feishu` 替换为 `send_feishu_message`（训练通知只需轻量接口）
- `analyze_stock_sentiment` / `get_sentiment_signal`：从顶层 import 移入 `generate_signal_report` 函数体内（避免启动时加载 snownlp/textblob）

### 5. `generate_signal_report` 函数

- **函数体保留**，不删除
- 顶部添加注释：`# 暂停使用：预测职责已移至 daily_run.py`
- 与主流程完全断开（不被 main() 或 train_portfolio_tickers() 调用）

---

## 不在本次范围内

- daily_run.py 的任何改动
- `generate_signal_report` 函数的删除（延后，等确认不再需要后再做）
- IC 分析 / 可视化模块的迁移

---

## 验证标准

1. `python3 main.py --use-optuna` 完成训练后推送训练摘要，不推送操作建议
2. `python3 main.py --portfolio` 完成训练后推送训练摘要，不推送操作建议
3. `python3 daily_run.py` 行为不变
4. `pytest tests/ -v` 全部通过
5. `python3 smoke_test.py` 通过
