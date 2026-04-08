# Ensemble/Stacking 策略设计文档

> 日期：2026-04-07  
> 对应执行计划：Phase 3 Step 3.2  
> 状态：已实现（2026-04-08）

---

## 背景

现有 `SignalAggregator` 用 Sharpe 加权投票聚合多策略信号，权重是静态的（训练时的历史 Sharpe）。Stacking 引入第二层 meta-learner，让模型在不同市场状态下动态选择相信哪些策略，提升共识信号的准确性。

---

## 架构总览

```
训练阶段（main.py）
  └─ 基础策略训练完成后
       └─ _train_meta_model(ticker, data, config)
            ├─ 加载所有 active factor_*.pkl
            ├─ Walk-Forward 切片（默认 5 折）
            ├─ 每折：重跑各策略 run() → 历史信号序列
            ├─ 拼接市场状态特征（ADX、ATR分位、成交量比率）
            ├─ 标签 = 未来 label_days 日涨跌
            ├─ 训练 Logistic Regression
            └─ 保存 data/meta/meta_model_{ticker_safe}.pkl

推断阶段（daily_run → PositionAnalyzer → SignalAggregator）
  └─ SignalAggregator.aggregate(aggregation_method="stacking")
       ├─ 照常从 factor_*.pkl 收集各策略最新信号（已有逻辑不变）
       ├─ 构造当日特征向量（策略信号 + 市场状态）
       ├─ MetaAggregator.predict(feature_vec) → 最终信号 + 置信度
       └─ fallback：meta-model 不存在时自动退回 "vote"
```

---

## 新增 / 修改文件

| 文件 | 动作 | 说明 |
|------|------|------|
| `engine/meta_aggregator.py` | 新建 | MetaAggregator 类：特征构造、Walk-Forward 训练、保存/加载、推断 |
| `engine/signal_aggregator.py` | 修改 | 增加 `aggregation_method` / `meta_dir` 参数；`aggregate()` 末尾增加 stacking 分支 |
| `engine/__init__.py` | 修改 | 导出 `MetaAggregator` |
| `main.py` | 修改 | 基础策略训练完后调用 `_train_meta_model()` |
| `config.yaml` | 修改 | 新增 `stacking` 配置块 |
| `tests/test_meta_aggregator.py` | 新建 | MetaAggregator 单元测试 |
| `tests/test_signal_aggregator.py` | 修改 | 新增 stacking 分支 + fallback 测试 |
| `tests/test_integration.py` | 修改 | 新增 meta train→save→load→predict 端到端测试 |

---

## MetaAggregator 类接口

```python
class MetaAggregator:
    def __init__(self, meta_dir: Path): ...

    def build_feature_vector(
        self,
        base_signals: dict[str, int],   # {strategy_name: 0/1}
        data: pd.DataFrame,             # 历史 OHLCV
    ) -> np.ndarray:
        """
        特征 = 各策略信号（有序列表） + 3 个市场状态指标：
          - adx_14:       ADX(14) 最新值（趋势强度，< 25 震荡，> 25 趋势）
          - atr_pct_rank: 近 252 日 ATR(14) 百分位（0~1，波动率分位）
          - volume_ratio: 今日成交量 / 20 日均量
        策略信号顺序固定（训练时存入 meta，推断时对齐）。
        """

    def train(
        self,
        ticker: str,
        data: pd.DataFrame,
        artifacts: list[dict],
        config: dict,
        n_splits: int = 5,
        label_days: int = 5,
    ) -> dict:
        """
        Walk-Forward 训练。
        切分示意（5折，expand window）：
          fold 1: train=[0, 12mo)  test=[12, 15mo)
          fold 2: train=[0, 15mo)  test=[15, 18mo)
          ...
          fold 5: train=[0, 24mo)  test=[24, 27mo)
          最终模型：全量数据训练，用于推断

        标签：label[t] = 1 if close[t+label_days] > close[t] else 0
        返回：{"accuracy": float, "n_samples": int, "n_features": int}
        """

    def save(self, ticker: str) -> Path:
        """
        保存到 data/meta/meta_model_{ticker_safe}.pkl。
        pkl 内容：{"model": lr, "strategy_names": [...], "meta_dir": str}
        strategy_names 记录训练时的特征顺序，推断时用于对齐。
        """

    @classmethod
    def load(cls, ticker: str, meta_dir: Path) -> Optional["MetaAggregator"]:
        """加载已保存的 meta-model；文件不存在返回 None。"""

    def predict(self, feature_vec: np.ndarray) -> tuple[int, float]:
        """返回 (signal: 0/1, proba: 看涨概率 ∈ [0, 1])"""
```

---

## SignalAggregator 修改

### `__init__` 新增参数

```python
def __init__(
    self,
    factors_dir: Optional[Path] = None,
    min_sharpe_weight: float = 0.0,
    max_factors: int = 20,
    use_registry: bool = True,
    aggregation_method: str = "vote",   # 新增
    meta_dir: Optional[Path] = None,    # 新增
):
```

### `_load_meta()` 新增私有方法

```python
def _load_meta(self, ticker: str) -> Optional[MetaAggregator]:
    """懒加载 meta-model，不存在或加载失败返回 None。"""
    if ticker not in self._meta_cache:
        self._meta_cache[ticker] = MetaAggregator.load(ticker, self._meta_dir)
    return self._meta_cache[ticker]
```

`self._meta_cache: dict[str, Optional[MetaAggregator]]` 在 `__init__` 中初始化为 `{}`。

### `aggregate()` 末尾新增分支

```python
# 在加权投票计算 consensus / confidence 之后
if self._aggregation_method == "stacking" and votes:
    meta = self._load_meta(ticker)   # 懒加载，不存在返回 None
    if meta is not None:
        base_signals = {v["strategy_name"]: v["signal"] for v in votes}
        feat = meta.build_feature_vector(base_signals, data)
        consensus, confidence = meta.predict(feat)
    else:
        logger.debug("%s: meta-model 不存在，fallback 到 vote", ticker)
        # consensus / confidence 已由 vote 逻辑计算好，无需再赋值
```

---

## config.yaml 新增字段

```yaml
stacking:
  aggregation_method: vote    # "vote" | "stacking"
  meta_dir: data/meta
  label_days: 5
  n_splits: 5
```

`PositionAnalyzer` 构造 `SignalAggregator` 时从 config 读取 `stacking.aggregation_method`，其余零修改。

---

## main.py 新增步骤

```python
def _train_meta_model(ticker: str, data: pd.DataFrame, config: dict,
                      factors_dir: Path, meta_dir: Path):
    """
    在基础策略训练完后调用。失败时 warning，不中断主流程。
    """
    try:
        from engine.meta_aggregator import MetaAggregator
        from engine.signal_aggregator import SignalAggregator
        agg = SignalAggregator(factors_dir=factors_dir)
        artifacts = agg._load_factors()
        if not artifacts:
            logger.warning("无 active 因子，跳过 meta-model 训练")
            return
        ma = MetaAggregator(meta_dir=meta_dir)
        metrics = ma.train(ticker, data, artifacts, config)
        ma.save(ticker)
        logger.info("meta-model 训练完成 %s", metrics, extra={"ticker": ticker})
    except Exception as e:
        logger.warning("meta-model 训练失败，不影响基础策略: %s", e)
```

调用位置：`step2_train_*()` 完成、`_save_factor()` 执行后。

---

## Look-Ahead Bias 防护

1. Walk-Forward 切分在特征构造**之前**完成，测试折的数据从不参与训练
2. 基础策略重跑历史信号时使用 `run()`（已有 `signal.shift(1)` 延迟），无额外 look-ahead
3. 标签 `close[t+label_days]` 只用于训练集，推断时不使用未来数据
4. 最终模型用全量历史训练，推断时特征只取 `data.iloc[-1]`（当日最新）

---

## 测试策略

| 测试文件 | 覆盖内容 |
|---------|---------|
| `tests/test_meta_aggregator.py` | `build_feature_vector` 形状；`train()` 返回合理 accuracy；`save/load` round-trip；`predict()` 输出 0/1 + proba ∈ [0,1] |
| `tests/test_signal_aggregator.py` | `aggregation_method="stacking"` 有 meta-model 时用 stacking；meta-model 不存在时 fallback 到 vote |
| `tests/test_integration.py` | train→save→load→predict 端到端，全离线合成数据 |

---

## 边界条件 / 错误处理

| 场景 | 处理方式 |
|------|---------|
| meta-model 文件不存在 | `MetaAggregator.load()` 返回 None，SignalAggregator fallback 到 vote |
| 基础策略数量 < 2（特征不足） | `train()` 记录 warning，不保存 meta-model |
| Walk-Forward 某折样本不足 | 跳过该折，继续其余折；若所有折都失败，不保存 |
| `predict()` 特征维度与训练不一致 | 捕获异常，fallback 到 vote，记录 warning |
| meta-model 训练失败 | `_train_meta_model()` 捕获异常，logger.warning，主流程继续 |
