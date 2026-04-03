# 深度学习策略设计方案

> **编制日期**：2026-04-03  
> **对应 upgrade_plan.md**：Phase 2 § 4.2  
> **优先级**：P1（可选）  
> **工作量估算**：XL（3-4 周）

---

## 一、目标与约束

### 1.1 目标
在现有规则策略（16 种）和树模型策略（XGBoost/LightGBM，6 种）的基础上，引入序列建模能力，捕捉 OHLCV 时间序列中的局部趋势与节奏，弥补树模型对顺序信息不敏感的缺陷。

### 1.2 硬性约束

| 约束 | 要求 |
|------|------|
| 策略接口 | 严格遵守 `NAME / run() / predict()` 三件套 |
| 训练类型 | `multi`（同 XGBoost），在 HSI 全体成分股上训练 |
| 推理时间 | CPU 推理 < 1 秒/股票（含特征构建） |
| 无前视偏差 | 特征窗口仅用 `t-W … t`，不含 `t+1` 及以后数据 |
| 离线测试 | `pytest tests/` 全程无网络（mock 模型权重） |
| 新依赖 | 仅增加 `torch`；不引入 `tensorflow` / `keras` |

### 1.3 不做
- **Temporal Fusion Transformer**：HSI 50 只 × 3 年 ≈ 37,500 样本，Transformer 参数量大、容易过拟合；先验证 LSTM/GRU 有效性再考虑升级。
- **盘中推理**：本期只做 EOD（每日收盘后）批量推理。
- **自动超参搜索**：网络结构固定，Optuna 仅搜索阈值和特征窗口，不搜索层数/隐层大小（避免 trial 过慢）。

---

## 二、选型决策：GRU vs LSTM

| 维度 | GRU | LSTM |
|------|-----|------|
| 参数量 | 较少（~75% of LSTM） | 较多 |
| 小数据集表现 | 通常略优 | 相当 |
| 推理速度 | 更快 | 相当 |
| 可解释性 | 相当 | 相当 |

**结论**：默认使用 **GRU**，实现中同时支持 `cell_type: "gru" | "lstm"`（可通过 config 切换）。

---

## 三、模型架构

```
输入 (B, W, F)
   W = 窗口长度，默认 30 天
   F = 特征数，约 18-22

GRU Layer 1  hidden=64, dropout=0.2
GRU Layer 2  hidden=32, dropout=0.2
线性层        32 → 1
Sigmoid       → 概率 p ∈ (0, 1)

分类阈值      p > upper_threshold → 1 (看涨)
              p < lower_threshold → 0 (看跌)
              else                → -1 (忽略此 bar，不产生信号)
```

### 参数说明

```yaml
# config.yaml 中的配置（ml_strategies.gru_trend）
gru_hidden_size: 64        # GRU 隐层大小
gru_num_layers: 2          # GRU 层数
gru_dropout: 0.2           # Dropout 比例
gru_window: 30             # 滑动窗口天数
gru_label_period: 5        # 标签预测未来 N 天
gru_label_threshold: 0.02  # 未来 N 天收益 > 2% → 标签 1
gru_upper_threshold: 0.60  # 信号=1 的概率门槛
gru_lower_threshold: 0.40  # 信号=0 的概率门槛
gru_epochs: 50
gru_batch_size: 128
gru_lr: 0.001
gru_cell_type: "gru"       # "gru" | "lstm"
```

---

## 四、特征工程

所有特征均在窗口内计算，严格无前视。

### 4.1 价格衍生特征（日度变化率，已标准化）

| 特征 | 计算 | 说明 |
|------|------|------|
| `ret_1d` | `close.pct_change(1)` | 日收益率 |
| `ret_5d` | `close.pct_change(5)` | 周收益率 |
| `log_hl` | `log(high/low)` | 日内振幅 |
| `log_co` | `log(close/open)` | 日内涨跌 |
| `volume_z` | `(volume - mean) / std`，rolling 20d | 成交量 Z-score |

### 4.2 技术指标（来自 `strategies/indicators.py`）

| 特征 | 参数 | 说明 |
|------|------|------|
| `rsi_14` | period=14 | RSI |
| `bb_upper_dist` | period=20 | `(close - bb_upper) / close` |
| `bb_lower_dist` | period=20 | `(close - bb_lower) / close` |
| `macd_line` | 12,26 | MACD 值（归一化） |
| `macd_signal` | 9 | Signal 线 |
| `ema_20_dist` | period=20 | `(close - ema20) / close` |
| `ema_60_dist` | period=60 | `(close - ema60) / close` |
| `atr_14_pct` | period=14 | `ATR / close`（波动率代理） |

### 4.3 特征矩阵构建

```python
def build_feature_matrix(df: pd.DataFrame, window: int = 30) -> np.ndarray:
    """
    返回 shape = (N_samples, window, n_features) 的 3D 数组。
    每个样本对应时间 t 时刻的历史窗口 [t-window+1, t]。
    注意：特征归一化使用窗口内统计量，防止未来数据泄漏。
    """
```

### 4.4 标签构建（无前视）

```python
label_t = 1 if (close[t + label_period] / close[t] - 1) > label_threshold else 0
```

**重要**：在 `run()` 中，label 使用 `t+label_period` 的数据，但只对训练集计算。  
调用 backtest 时，`run()` 返回的 signal 已去掉尾部 `label_period` 行（没有标签），保持与 backtest 数据对齐。

---

## 五、训练流程

### 5.1 数据准备（与 XGBoost 一致）

```
训练集  →  所有 HSI 股票  ×  [train_start, val_start)
验证集  →  目标股票       ×  [val_start, end]
```

通过 `analyze_factor._load_multi_stock_data()` 加载（含磁盘缓存，多进程安全）。

### 5.2 训练代码骨架

```python
# strategies/gru_trend.py

NAME = "gru_trend"

def run(data: pd.DataFrame, config: dict):
    X_train, y_train = _build_dataset(train_df, config)
    X_val,   y_val   = _build_dataset(val_df,   config)

    model = _GRUModel(
        input_size=X_train.shape[-1],
        hidden_size=config.get("gru_hidden_size", 64),
        num_layers=config.get("gru_num_layers", 2),
        dropout=config.get("gru_dropout", 0.2),
        cell_type=config.get("gru_cell_type", "gru"),
    )
    _train(model, X_train, y_train, config)

    signal = _infer_signal(model, X_val, config)
    meta = {
        "name": NAME,
        "params": {k: config[k] for k in _PARAM_KEYS if k in config},
        "feat_cols": FEATURE_COLS,
        "indicators": {},
        "state_dict_b64": _serialize_state_dict(model),  # base64 存入 meta
    }
    return signal, model, meta


def predict(model, data: pd.DataFrame, config: dict, meta: dict):
    # 若 model 为 None（从磁盘加载时），先重建
    if model is None:
        model = _restore_model(meta, data.shape[-1])
    return _infer_signal(model, _build_features(data, config), config)
```

### 5.3 模型序列化

PyTorch 模型通过 `state_dict` 序列化后转 base64 存入 `meta["state_dict_b64"]`，随因子 pkl 一起持久化。加载时：

```python
def _restore_model(meta: dict, input_size: int) -> _GRUModel:
    params = meta["params"]
    model = _GRUModel(input_size, ...)
    buf = base64.b64decode(meta["state_dict_b64"])
    model.load_state_dict(torch.load(io.BytesIO(buf), map_location="cpu"))
    model.eval()
    return model
```

---

## 六、防止过拟合策略

| 手段 | 说明 |
|------|------|
| Dropout 0.2 | 训练时随机丢弃 20% 神经元 |
| Early Stopping | 连续 5 个 epoch val_loss 不下降则停止 |
| 小模型优先 | 默认 64→32，参数量约 3 万（XGBoost 对比约 2-5 万估算树） |
| Walk-Forward 验证 | 严格 OOS 评估，与其他策略标准一致 |
| 标签平衡 | 训练时对少数类做上采样（若涨跌比例偏差 > 2:1） |

---

## 七、文件结构与影响范围

```
strategies/
└── gru_trend.py         # 新建（≈ 350 行）

tests/
└── test_gru_strategy.py  # 新建（≈ 80 行，全离线，mock 模型）

config.yaml              # 新增 ml_strategies.gru_trend 块
requirements.txt         # 新增 torch>=2.0（CPU-only 推荐用 torch+cpu wheel）
```

**不修改**的文件：
- `engine/signal_aggregator.py` — 已支持任意策略，无需改动
- `analyze_factor.py` — 训练类型 `multi` 已存在，`gru_trend` 自动进入发现链
- `main.py` — 无需改动

---

## 八、Optuna 集成

GRU 策略参与 Optuna 超参搜索，但**仅搜索阈值和窗口**，不搜索网络结构（防止 trial 超时）：

```python
# 在 Optuna trial 中采样的参数
{
    "gru_window":           trial.suggest_int("gru_window", 15, 60),
    "gru_label_period":     trial.suggest_int("gru_label_period", 3, 10),
    "gru_label_threshold":  trial.suggest_float("gru_label_threshold", 0.01, 0.04),
    "gru_upper_threshold":  trial.suggest_float("gru_upper_threshold", 0.55, 0.75),
    "gru_lower_threshold":  trial.suggest_float("gru_lower_threshold", 0.25, 0.45),
}
```

---

## 九、验收标准

| 指标 | 目标 |
|------|------|
| Walk-Forward Sharpe | ≥ 1.0 |
| Walk-Forward Sharpe vs XGBoost | 不显著劣于（差距 < 0.3） |
| CPU 推理时间（单股） | < 1 秒 |
| `pytest tests/test_gru_strategy.py` | 全通过（全离线） |
| `python3 smoke_test.py` | 无回归 |
| 最大回撤（OOS） | ≥ -0.20 |

---

## 十、实施步骤

```
Step 1  安装并锁定 torch CPU wheel，更新 requirements.txt         [0.5d]
Step 2  实现 strategies/gru_trend.py
         - _GRUModel (torch.nn.Module)
         - build_feature_matrix()
         - _train() with early stopping
         - _infer_signal()
         - run() / predict()                                        [3d]
Step 3  单元测试 tests/test_gru_strategy.py
         - 接口形态（run 返回 tuple，predict 返回 Series）
         - 推理时不需要网络（mock state_dict）
         - signal 值域 ∈ {0, 1}
         - 无前视偏差（window 内特征不包含 t+1）                    [1d]
Step 4  集成测试（本地，需网络数据）
         - 完整 run() + backtest() 流程
         - Walk-Forward 验证通过阈值                                [1d]
Step 5  config.yaml 增加 gru_trend 配置块                         [0.5d]
Step 6  更新 upgrade_plan.md 4.2 节状态为 ✅                        [0.5d]
```

---

## 十一、风险与降级方案

| 风险 | 概率 | 降级方案 |
|------|------|----------|
| Walk-Forward Sharpe < 1.0 | 中 | 不加入每日投票池，仅保留为实验策略 |
| torch 安装包体积过大 | 低 | 使用 `torch-cpu` 精简版（约 200MB） |
| 训练超时（> 30 分钟/run） | 中 | 将 gru_trend 从默认 `strategy_training` 中移出，改为 `--strategy gru_trend` 手动触发 |
| 过拟合（OOS Sharpe 崩溃）| 高 | 加强 Dropout（0.3）+ 缩短 epochs（30）+ 更大窗口（60d）|
