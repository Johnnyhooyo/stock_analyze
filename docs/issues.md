# 项目问题整理

> 生成时间: 2026-03-20
> 审查范围: main.py, analyze_factor.py, validate_strategy.py, position_manager.py, optimize_with_optuna.py, backtest_vectorbt.py, strategies/*.py, oms.py, fetch_data.py, smoke_test.py

---

## 🔴 CRITICAL - 必须修复

### 1. 多股票Optuna优化信号被丢弃
**文件:** `optimize_with_optuna.py` 第411-413行

```python
if self.train_type == 'multi' and not self.multi_stock_data.empty:
    signal, _, meta, val_df = self._run_multi_stock(trial_cfg)
    backtest_data = val_df  # ← signal 被赋值但从未使用！
```

**问题:** 所有多股票ML策略（xgboost_enhanced、lightgbm_enhanced及其tsfresh变体）的Optuna优化实际上是在**零信号/原始价格数据**上进行的，返回的 `best_params` 毫无意义。

**影响:** 优化100次 trial 都是在白跑，参数是随机选择的。

---

### 2. entry_price 参数在 ATR 止损中从未使用
**文件:** `position_manager.py` 第163-177行

```python
def check_atr_stop(self, close_price: float, entry_price: float, ...):
    # entry_price 被接收但从未在计算中使用
    stop_price = peak_price - self.atr_multiplier * atr  # 只用peak
```

**问题:** 调用者传入 `entry_price` 期望它影响止损价格计算，但实际计算完全忽略该参数。

---

### 3. Position 的 current_price = -1.0 哨兵值
**文件:** `position_manager.py` 第530-543行

```python
return Position(shares=shares, avg_cost=avg_cost, current_price=-1.0)  # 无效标记
```

**问题:** 如果在 `update_current_price()` 调用前使用此Position对象：
- `market_value` 返回负值 (`-shares * 1.0`)
- `profit` 计算完全错误

**影响:** 持仓状态加载时若顺序出错，会产生静默的负市场价值。

---

### 4. Rule 策略在 warmup+test 数据上重新训练（Look-Ahead Bias）
**文件:** `validate_strategy.py` 第94-98行、216-220行

```python
# 注释说"不参与 fit"，但 rule 策略没有内部 train/test 分割
warmup_df = pd.concat([train_df.iloc[-warmup:], test_df])
sig, _, _ = strategy_mod.run(warmup_df, trial_cfg)  # test 数据被用于指标计算！
sig = sig.reindex(test_df.index).fillna(0)
```

**问题:** Rule 策略（如 ma_crossover、rsi_reversion）的 `run()` 方法计算指标时使用**所有传入数据**，没有内部分割保护。传入 warmup_df 后，test 期间的数据被用于计算 test 期的指标（如 MA、RSI）。

**影响:** Walk-Forward 和 Out-of-Sample 验证结果被人为高估。

---

## 🟠 HIGH - 重要问题

### 5. Config 加载不一致
**文件:** `main.py` 第328行 vs 第393行

| 函数 | 使用的加载函数 | 密钥配置 |
|------|--------------|---------|
| `step2_train_native()` | `_load_config()` | ❌ 不可用 |
| `step2_train_optuna()` | `_load_config_full()` | ✅ 可用 |

**问题:** 运行 native random search 时，`broker_api_key`、`feishu_webhook` 等密钥配置不可用。

---

### 6. 策略发现静默失败
**文件:** `analyze_factor.py` 第116-121行

```python
for name in strategy_list:
    try:
        mod = importlib.import_module(f'strategies.{name}')
    except ImportError:
        pass  # ← 静默吞掉错误！
```

**问题:** 如果 `config.yaml` 中配置了不存在的策略，错误被完全忽略，用户不知道自己的策略根本没在运行。

---

### 7. validate_position_size 验证逻辑错误
**文件:** `position_manager.py` 第373-376行

```python
# Kelly 不使用时，验证的是当前持仓而非建议的新仓位
result["kelly_shares"] = self.validate_position_size(
    self.position.shares, price, capital  # ← 当前shares，不是拟建仓位
)
```

**问题:** 当 `use_kelly=false` 时，这段代码验证的是"当前持仓是否超过上限"而非"建议的新仓位是否合规"，没有实际意义。

---

### 8. Multi-objective 优化使用未合并的 config
**文件:** `optimize_with_optuna.py` 第732-733行

```python
def objective(trial):
    trial_cfg = config.copy()  # ← 使用原始config，未调用 _merge_ml_strategy_config()
    trial_cfg.update(params)
```

**问题:** 多目标优化时，ML策略的特定配置项（`use_tsfresh_features`、`use_ta_lib` 等）没有被正确合并，导致优化使用错误的参数集。

---

### 9. ATR 止损未在回测中模拟
**文件:** `backtest_vectorbt.py`、`analyze_factor.py` vs `position_manager.py`

- `apply_risk_controls()` 在实盘模式下执行 ATR 动态止损和熔断
- `backtest()` 和 `backtest_vectorbt()` **不模拟**这些止损

**影响:**
- 回测的最大回撤可能**显著低于**实盘实际发生的
- 策略在回测中表现虚高

---

## 🟡 MEDIUM - 中等问题

### 10. TrailingStop 的 _peak 未持久化
**文件:** `position_manager.py` 第50-79行

`_STATE_FILE` 保存了 `consecutive_loss_days` 和 `last_trade_date`，但 `_peak`（跟踪的最高价）未保存。

**问题:** 系统重启后丢失峰值价格，可能导致：
- 过早触发止损（如果重启期间价格下跌）
- 过晚触发止损（如果峰值本应更高）

---

### 11. circuit_breaker 重复调用时跳过状态更新
**文件:** `position_manager.py` 第254-264行

```python
if trade_date and state.get("last_trade_date") == trade_date:
    return {  # ← 直接返回，不重新评估
        "tripped": tripped,
        ...
    }
```

**问题:** 同一交易日第二次调用时直接返回，**不重新评估** `today_pnl_pct`。如果第一次调用后价格继续下跌，第二次调用返回的是过时状态。

---

### 12. XGBoost CV 模型未继承 early_stopping_rounds
**文件:** `strategies/xgboost_enhanced.py` 第494-495行

```python
_cv_model = model.__class__(**model.get_params())  # get_params() 不包含 early_stopping_rounds
_cv_model.fit(_Xtr, _ytr)  # ← 无 early stopping
```

**问题:** `early_stopping_rounds` 是 fit 参数而非模型参数，`get_params()` 不返回它。因此 CV 训练时没有 early stopping，CV 结果与主模型训练不一致。

---

### 13. tsfresh 特征选择空 DataFrame 时索引错误
**文件:** `strategies/tsfresh_features.py` 第405-421行

当 FDR 特征选择返回空结果时，代码回退到选择前100列，但此时 `combined` 使用整数索引而非 datetime 索引。

**问题:** 特征矩阵索引与原数据不匹配，后续合并会出错。

---

### 14. HK 节假日集合仅覆盖到 2026 年
**文件:** `fetch_data.py` 第31-62行

今天是 2026-03-20，2026年剩余假期（圣诞节12-25、重阳节10-29等）未录入。

**问题:** 涉及这些假日的交易日判断会错误，可能导致数据过期判断失误。

---

### 15. 缓存失效机制缺失
**文件:** `analyze_factor.py` 第33-48行

```python
@_memory.cache
def _load_multi_stock_data_cached(period: str = '3y', min_days: int = 300):
    data = load_all_hsi_data(period=period, min_days=min_days)
```

`download_hsi_incremental()` 可能更新 CSV 文件，但 joblib.Memory 缓存基于函数参数（不变），缓存不会自动失效。

---

## 🔵 LOW - 改进建议

| # | 文件 | 行 | 问题描述 |
|---|------|---|---------|
| 16 | `analyze_factor.py` | 465, 469 | `mean_ret` 提前赋值但被覆盖，无用代码 |
| 17 | `indicators.py` | 80-88 | ATR变量赋值为ta库结果但完全未使用，手动计算覆盖 |
| 18 | `backtest_vectorbt.py` | 37 | 费用公式 `(fees+fees+stamp)/2` 重复使用fees_rate |
| 19 | `backtest_vectorbt.py` | 187 | `tr_start` 赋值两次但从未使用 |
| 20 | `ma_crossover.py` | 18 | 信号dtype未指定int，与其他策略不一致 |
| 21 | `oms.py` | 102-105 | `PaperOMS.__init__` 接受config参数但完全忽略 |
| 22 | `oms.py` | 多处 | `LiveOMS` 为TODO占位，但已被生产代码实例化，无警告 |
| 23 | `validate_strategy.py` | 53, 172 | `total_months` 计算忽略day分量 |
| 24 | `position_manager.py` | 546-564 | `calc_atr` 缺少列名验证，列缺失时静默返回0.0 |
| 25 | `train_multi_stock.py` | 44 | 列名标准化逻辑脆弱，`OPEN`→`OPEn` |
| 26 | `xgboost_enhanced.py` | 582-583 | `available_feat_cols` 计算后未使用 |
| 27 | `lightgbm_enhanced.py` | 45-56 | 从xgboost_enhanced导入实现细节，紧耦合 |
| 28 | `smoke_test.py` | - | ML策略 `predict()` 接口未测试 |
| 29 | `smoke_test.py` | - | `validate_strategy.py` 函数未覆盖 |
| 30 | `smoke_test.py` | - | tsfresh变体策略未测试 |

---

## 修复优先级建议

### 第一优先级（功能性bug）
1. #1 多股票Optuna信号丢弃 → 修复优化逻辑
2. #4 rule策略look-ahead bias → 传入 `no_internal_split=True`

### 第二优先级（配置/状态问题）
3. #5 config加载不一致 → 统一使用 `_load_config_full()`
4. #6 策略发现静默失败 → 添加日志警告
5. #3 current_price=-1.0 哨兵值 → 添加运行时检查

### 第三优先级（代码质量）
6. #9 ATR回测不一致 → 文档说明或回测中模拟
7. #12 early_stopping_rounds继承 → 修复CV训练
8. #10 TrailingStop持久化 → 添加_peak保存
