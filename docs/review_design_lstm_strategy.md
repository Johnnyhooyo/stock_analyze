# Design Review: design_lstm_strategy.md

> **审阅日期**：2026-04-03  
> **审阅对象**：`docs/design_lstm_strategy.md` + 已有实现 `strategies/rnn_trend.py` + `tests/test_rnn_strategy.py`

---

## 总体评价

设计文档结构清晰，约束条件、选型理由、防过拟合手段、验收标准均有覆盖，可作为实现指引。但对照已有实现（`rnn_trend.py`），存在 **3 个严重缺陷**、**3 个显著问题**和若干小问题，需在合并前修复。

---

## 严重缺陷（Critical）

### C1 — 特征矩阵构建逻辑错误，序列信息全部丢失

**位置**：`rnn_trend.py:222`

```python
X[t] = features   # ← features 是 13 个标量，X[t] 的 shape 是 (window, 13)
```

NumPy 会将 13 个标量广播到 `(window, 13)` 的每一行，导致窗口内所有时间步的特征向量完全相同。GRU/LSTM 处理的是 `window` 帧内容全部一致的序列，等价于一个全连接网络——序列建模能力被彻底消除，与设计初衷背道而驰。

**修复方向**：`build_feature_matrix` 应在每个时间步 `τ ∈ [t-window+1, t]` 分别计算特征，构造真正的时序特征矩阵。

---

### C2 — 实现与设计策略名、配置键不一致

| 维度 | 设计文档 | 实际实现 |
|------|----------|----------|
| `NAME` | `"gru_trend"` | `"rnn_trend"` |
| config 前缀 | `gru_*` | `rnn_*` |
| 文件名（设计） | `strategies/gru_trend.py` | `strategies/rnn_trend.py` |
| 测试文件（设计） | `tests/test_gru_strategy.py` | `tests/test_rnn_strategy.py` |

这意味着：
- 设计文档中的 config.yaml 配置块（`ml_strategies.gru_trend`）无法被实现读取
- `analyze_factor._discover_strategies()` 会发现 `rnn_trend` 而非 `gru_trend`

**建议**：统一使用 `rnn_trend` / `rnn_*`（实现名更通用，支持 GRU+LSTM），同时更新设计文档的 §2～§8 所有相关引用。

---

### C3 — 训练数据来源偏离设计：未使用 HSI 多股票数据

设计 §5.1 明确：

> 训练集 → 所有 HSI 股票 × [train_start, val_start)  
> 通过 `analyze_factor._load_multi_stock_data()` 加载

实际实现 `run()` 直接对传入的单只股票 `data` 做 80/20 时间分割：

```python
split_idx = int(len(df) * 0.8) if not no_split else len(df)
```

既未调用 `_load_multi_stock_data()`，也未设置 `"training_type": "multi"`。这意味着：
- 训练集仅为单股约 4 年数据（≈1000 bar），样本量远低于设计预期（HSI 50 只 × 3 年 ≈ 37,500 样本）
- 泛化能力弱，Walk-Forward 验收指标 Sharpe ≥ 1.0 难以达到
- 与 XGBoost 策略的对比基准不公平

**建议**：在实现中按 XGBoost 策略的模式接入 `_load_multi_stock_data()`，或在设计文档中明确降级为单股训练并说明理由。

---

## 显著问题（Significant）

### S1 — Signal 包含 -1，违反策略接口契约

`CLAUDE.md` 明确：

> `signal`: int Series (1=long, 0=flat)

但 `run()` 在训练集区段将信号置为 -1（`all_signals[valid_mask] = -1`），测试也检验 `{-1, 0, 1}` 是合法值域。将 -1 传入 `backtest()` 会被当作做空信号（或异常值）处理，产生错误回测结果。

**修复**：不确定区（概率在上下阈值之间）应映射为 0（持平）。训练集段也应填 0 而非 -1。

---

### S2 — EMA 指标用 SMA 实现

```python
ema20 = np.mean(w_close[-20:])   # 应为 EWM，实为简单均值
ema60 = np.mean(w_close[-60:])
```

EMA 要求指数加权平均（`ewm`），用 SMA 替代是计算错误，与特征名 `ema_20_dist` / `ema_60_dist` 不符。

---

### S3 — MACD Signal Line 计算错误

```python
macd_signal = pd.Series([macd_line] * 9).ewm(span=9, adjust=False).mean().iloc[-1]
```

此处将同一个标量重复 9 次再做 EWM，结果恒等于 `macd_line`（EWM 对常数序列的结果即常数本身），相当于 `macd_signal == macd_line`，两者差值（直方图）恒为 0，失去 MACD 信号的意义。

正确做法：对窗口内历史 MACD 序列做 EWM-9。

---

## 小问题（Minor）

| # | 位置 | 问题 |
|---|------|------|
| M1 | `rnn_trend.py:77` | `torch.load(buf, map_location="cpu")` 缺少 `weights_only=True`，torch ≥ 2.0 会抛出 FutureWarning |
| M2 | `rnn_trend.py:310-332` | `_infer_signal()` 函数定义后在 `run()` / `predict()` 中均未被调用，是死代码 |
| M3 | 设计 §4.2 | 说明"来自 `strategies/indicators.py`"，但实现完全重写了所有指标，违反 CLAUDE.md "禁止在策略文件中重复实现指标函数" 的规定 |
| M4 | 设计 §8 | Optuna 搜索 `gru_upper/lower_threshold` 时，Early Stopping 的 val_set 与回测的 OOS val_set 是同一段数据，阈值选择实质上在 OOS 数据上做了超参搜索，存在轻微数据集污染 |
| M5 | 设计 §三 | `gru_cell_type: "gru"` 的 config key 前缀用 `gru_*`，但当切换为 LSTM 时语义混乱；建议统一为 `rnn_cell_type` |

---

## 设计文档自身建议

1. **§4.3** 特征矩阵构建的 docstring 描述正确，但缺少伪代码示意每个时间步的特征计算逻辑，导致实现时出现 C1 错误。建议在文档中补充每个 `τ` 时刻计算哪些特征的具体说明。

2. **§5.2** `run()` 骨架中的 `train_df` / `val_df` 未说明来源，与 C3 问题直接相关。建议明确写出 `_load_multi_stock_data()` 调用路径。

3. **§九 验收标准** 缺少对信号值域的约束（`signal ∈ {0, 1}`），测试文件因此允许了 -1，形成 S1 问题。

---

## 优先修复顺序

| 优先级 | 问题 | 影响 |
|--------|------|------|
| P0 | C1：特征矩阵逻辑错误 | 模型完全无效 |
| P0 | S1：-1 信号违反接口 | 回测结果错误 |
| P1 | C3：未使用 multi 训练 | 泛化能力和基准对比失效 |
| P1 | S2/S3：EMA/MACD 计算错误 | 特征质量低 |
| P2 | C2：名称统一 | 集成时策略发现失败 |
| P3 | M1-M5：小问题 | 潜在警告/可维护性 |

---

## C3 详细修复方案：接入 HSI 多股票训练

### 背景：现有 multi 训练机制

`analyze_factor.run_trial()` 已为 `training_type = "multi"` 的策略提供了完整的多股票数据管道，`xgboost_enhanced` 和 `lightgbm_enhanced` 均已接入。整体数据流如下：

```
analyze_factor.run_trial()
│
├─ 加载目标股票数据（整个历史）→ df_target
├─ 计算 val_start = 当前日期 - lookback_months（默认 3 个月）
│
├─ 训练数据（multi 模式）
│   └─ _load_multi_stock_data(period='3y', min_days=300)
│       ├─ 扫描 data/historical/*_3y.csv（HSI 全部成分股）
│       ├─ 追加 ticker 列，concat 为宽 DataFrame
│       ├─ 按 [val_start - train_years, val_start) 截取（防止数据泄漏）
│       └─ 删去尾部 label_period 行（防止 shift(-N) 标签前视）
│   → train_df：MultiIndex 行（date × ticker 混合），约 3K-15K 行
│       列：DatetimeIndex | Open/High/Low/Close/Volume | ticker | ...
│
├─ 调用 strategy_mod.run(train_df, config)
│   → 返回 (signal, model, meta)
│
├─ 验证数据（始终单只目标股票）
│   └─ val_df = df_target.loc[index >= val_start]（约 60-90 行）
│
└─ 调用 strategy_mod.predict(model, val_df, config, meta)
    → val_signal：与 val_df.index 对齐的 0/1 Series
```

**关键约定**：
- `run()` 收到的是**多股票合并 DataFrame**（含 `ticker` 列，约 3K-15K 行）
- `predict()` 收到的是**单股目标数据**（不含 `ticker` 列，约 60-90 行）
- `run_trial()` 负责时间对齐，策略内部无需再次切分 train/val

### rnn_trend.run() 需要的修改

#### Step 1：声明训练类型

在 `config.yaml` 的 `strategy_training.multi` 列表中加入 `rnn_trend`：

```yaml
strategy_training:
  multi:
    - xgboost_enhanced
    - lightgbm_enhanced
    - rnn_trend          # ← 新增
```

同时在 `analyze_factor._get_strategy_training_type()` 的自动识别规则中补充 `rnn`（若不想修改 config.yaml 则可依赖此规则）：

```python
# analyze_factor.py，约 576 行
if 'xgboost' in name or 'lightgbm' in name or 'rnn' in name:
    return 'multi'
```

#### Step 2：run() 处理多股票输入

`run()` 收到 `train_df` 时，其中包含 HSI 全部股票的 OHLCV 数据和 `ticker` 列。RNN 策略需要为**每只股票独立**构建时序滑动窗口，再 concat 为训练样本。

```python
def run(data: pd.DataFrame, config: dict):
    window       = int(config.get("rnn_window", 30))
    label_period = int(config.get("rnn_label_period", 5))
    label_thresh = float(config.get("rnn_label_threshold", 0.02))

    is_multi = "ticker" in data.columns and data["ticker"].nunique() > 1

    if is_multi:
        # ── 多股票路径 ──────────────────────────────────────────
        X_list, y_list = [], []
        for ticker, stock_df in data.groupby("ticker"):
            stock_df = stock_df.sort_index().drop(columns=["ticker"])
            if len(stock_df) < window + label_period + 10:
                continue
            X_stock = _build_feature_matrix(stock_df, window)   # (N, W, F)
            y_stock = _build_labels(stock_df["Close"].values, label_period, label_thresh)
            # 只保留有效样本（有完整窗口且有标签的行）
            valid = (np.arange(len(stock_df)) >= window - 1) & \
                    (np.arange(len(stock_df)) < len(stock_df) - label_period)
            X_list.append(X_stock[valid])
            y_list.append(y_stock[valid])

        X_train = np.concatenate(X_list, axis=0)   # (N_total, W, F)
        y_train = np.concatenate(y_list, axis=0)   # (N_total,)

        # Early stopping 验证集：从训练集末 20% 切割（不是目标股 OOS）
        split = int(len(X_train) * 0.8)
        X_tr, y_tr = X_train[:split], y_train[:split]
        X_es, y_es = X_train[split:], y_train[split:]
    else:
        # ── 单股后备路径（兼容 smoke_test / unit test）──────────
        ...（原有 80/20 逻辑保留）

    model = _GRUModel(input_size=X_tr.shape[-1], ...)
    model = _train(model, X_tr, y_tr, X_es, y_es, config)

    # run() 返回的 signal 对应 train_df 的索引，analyze_factor 不使用它（只用 predict()）
    # 返回一个与 data 等长的零信号即可
    signal = pd.Series(0, index=data.index)

    meta = {
        "name": NAME,
        "params": {k: config.get(k) for k in _PARAM_KEYS if k in config},
        "feat_cols": FEATURE_COLS,
        "indicators": {},
        "state_dict_b64": _serialize_state_dict(model),
        "input_size": len(FEATURE_COLS),
        "hidden_size": int(config.get("rnn_hidden_size", 64)),
        "num_layers": int(config.get("rnn_num_layers", 2)),
        "dropout": float(config.get("rnn_dropout", 0.2)),
        "model_type": config.get("rnn_cell_type", "gru"),
    }
    return signal, model, meta
```

> **为什么 run() 返回零信号**：`run_trial()` 在 multi 模式下只用 `run()` 的返回值获取 model 和 meta，随后调用 `predict(model, val_df, ...)` 在目标股验证集上生成真正的信号。train_df 的 signal 不进入回测，填零不影响结果。

#### Step 3：predict() 不变

`predict()` 接收的是单只目标股票的 `val_df`（无 `ticker` 列），与现有逻辑完全兼容，无需修改。

### 数据规模估算

| 来源 | 行数（约） | 有效序列样本数（window=30, label=5）|
|------|-----------|-------------------------------------|
| HSI 50 只 × 3 年 × 250 日 | 37,500 行 | ≈ 34,000 个 (W,F) 样本 |
| 原单股 80% × 1000 行 | 800 行 | ≈ 760 个样本 |

样本量约提升 **45 倍**，是解决过拟合的核心手段。

### 前视偏差检查点

| 环节 | 由谁保证 | 方式 |
|------|----------|------|
| train_df 不含 val 时段数据 | `run_trial()`（已有） | 按 `val_start` 截断 |
| 标签 `close[t+N]` 不溢出 | `run()` 内 `valid` mask | `< len(df) - label_period` |
| 每个样本窗口仅用 `[t-W+1, t]` | `_build_feature_matrix()` 切片 | `df.iloc[t-window+1 : t+1]` |
| Early Stopping 用训练集末段 | `run()` 80/20 切割 | 不使用目标股 val_df |
| predict() 特征与训练特征一致 | `FEATURE_COLS` 常量 | 同一列表，无顺序差异 |

### 对测试的影响

单元测试（`test_rnn_strategy.py`）传入合成单股数据，属于单股后备路径，无需修改。  
需新增集成测试用例：传入含 `ticker` 列的模拟双股数据，验证多股路径能正常训练并返回有效 meta。

```python
def test_run_multi_stock(gru_mod, minimal_cfg):
    """验证多股票路径（含 ticker 列）"""
    n = 300
    dates = pd.bdate_range(end="2025-12-31", periods=n)
    np.random.seed(0)
    frames = []
    for tk in ["0001.HK", "0700.HK"]:
        close = np.cumprod(1 + np.random.normal(0, 0.01, n)) * 100
        df = pd.DataFrame({"Open": close, "High": close*1.01,
                           "Low": close*0.99, "Close": close,
                           "Volume": 1e6 * np.ones(n), "ticker": tk},
                          index=dates)
        frames.append(df)
    multi_df = pd.concat(frames).sort_index()

    signal, model, meta = gru_mod.run(multi_df, minimal_cfg)
    assert model is not None
    assert "state_dict_b64" in meta
    assert isinstance(signal, pd.Series)
```
