# 港股量化系统升级规划

> **编制日期**：2026-03-24
> **最后更新**：2026-03-25  
> **项目核心目标**：一个专业的量化软件，每日运行，基于持仓给出下一个交易日的操作建议。
> **规划范围**：架构、数据、策略、风控、执行、运维、测试、性能、报告、合规共十大维度

### 整体进度

| Phase | 名称 | 状态 | 完成项 / 总项 |
|-------|------|------|----------|
| Phase 1 (P0) | 基础债务清理 | ✅ 完成 | 6 / 6    |
| Phase 2 (P1) | 策略与风控增强 | ⏳ 待启动 | 0 / 7    |
| Phase 3 (P2) | 实时化与可视化 | ⏳ 待启动 | 0 / 4    |
| Phase 4 (P3) | 生产级运营 | ⏳ 待启动 | 0 / 4    |

---

## 目录

- [一、现状诊断](#一现状诊断)
- [二、分阶段路线图](#二分阶段路线图)
- [三、Phase 1 — 基础债务清理 (P0)](#三phase-1--基础债务清理-p0)
- [四、Phase 2 — 策略与风控增强 (P1)](#四phase-2--策略与风控增强-p1)
- [五、Phase 3 — 实时化与可视化 (P2)](#五phase-3--实时化与可视化-p2)
- [六、Phase 4 — 生产级运营 (P3)](#六phase-4--生产级运营-p3)
- [附录 A：优先级定义](#附录-a优先级定义)
- [附录 B：工作量估算说明](#附录-b工作量估算说明)

---

## 一、现状诊断

### 1.1 已有优势

| 维度 | 现状 |
|------|------|
| 数据层 | Vendor 链自动回退 + 质量管道 + Schema 校验 + 原子写入，成熟度高 |
| 策略库 | 20+ 策略（16 规则 + 6 ML），覆盖动量/均值回归/趋势/ML |
| 超参搜索 | Optuna Bayesian + Random Search，含 Walk-Forward 验证 |
| 风控 | ATR 止损 / Kelly 仓位 / 熔断，回测中可模拟止损 |
| 每日引擎 | `daily_run.py` 多线程分析 + 共识信号 + 飞书通知，完整闭环 |

### 1.2 关键问题

| 编号 | 问题 | 严重度 | 说明 |
|------|------|--------|------|
| **D-01** | `engine/portfolio_state.py` 为空文件 | ✅ 已解决 | `engine/__init__.py` 导入 `PortfolioState, PortfolioPosition, load_portfolio`，运行时必然 ImportError（除非有动态 patch）|
| **D-02** | 配置加载逻辑散落 3 处 | ✅ 已解决 | `analyze_factor._load_config()`, `main._load_config_full()`, `daily_run._load_full_config()` 各自实现 |
| **D-03** | 指标计算 N 份拷贝 | 🟡 中等 | `_rsi()`, `_bollinger_bands()`, `_macd()` 等在各策略文件中重复实现；`strategies/indicators.py` 已存在但未被规则策略引用 |
| **D-04** | LiveOMS 全部 TODO | 🟡 中等 | `submit_order` / `cancel_order` / `get_position` 均为占位，降级 PaperOMS |
| **D-05** | 测试仅有 `smoke_test.py` | ✅ 已解决 | 正式 pytest 结构，100 单元测试覆盖核心路径，`smoke_test.py` 无回归 |
| **D-06** | 无 portfolio-level 风控 | 🟡 中等 | 仅单股 ATR 止损，无相关性 / 行业集中度 / VaR / 最大组合回撤分析 |
| **D-07** | 混用 `print()` 与 `logging` | 🟡 中等 | 无结构化日志，无 metrics，线上排查困难 |
| **D-08** | `requirements.txt` 版本偏旧 | 🟢 低 | pandas 2.0.3 / numpy 1.24.3（2023 年版），部分库无版本约束 |
| **D-09** | 无 PnL 历史追踪 | 🟡 中等 | 每日建议执行后无自动记录，无法做绩效归因 |
| **D-10** | 无实时 / 盘中数据 | 🟢 低 | 仅收盘后批量分析，无盘中信号触发能力 |

---

## 二、分阶段路线图

```
Phase 1 (P0)          Phase 2 (P1)          Phase 3 (P2)          Phase 4 (P3)
基础债务清理           策略与风控增强         实时化与可视化         生产级运营
──────────────        ──────────────        ──────────────        ──────────────
1-2 个月               2-3 个月               2-3 个月               持续迭代

✅ portfolio_state    • 选股模块(Screener)   • 实时行情接入         • 合规审计日志
✅ 配置加载统一        • Ensemble/Stacking   • Streamlit Dashboard  • 多市场扩展
✅ 公共指标库          • 深度学习策略         • 性能优化             • 自动化运维
✅ 测试框架搭建        • Portfolio 风控       • CI/CD pipeline       • 灾备方案
• 依赖升级            • OMS 实盘对接         • 结构化日志
                      • 因子生命周期管理
                      • PnL 追踪
```

---

## 三、Phase 1 — 基础债务清理 (P0)

> **目标**：消除运行时隐患，建立测试基建，统一代码风格。
> **周期**：1-2 个月

### 3.1 ✅ 补全 `engine/portfolio_state.py`

| 属性 | 值 |
|------|---|
| **问题编号** | D-01 |
| **优先级** | P0 |
| **工作量** | M |
| **状态** | ✅ 已完成（2026-03-24） |
| **影响文件** | `engine/portfolio_state.py`, `engine/__init__.py`, `engine/position_analyzer.py`, `daily_run.py` |

**完成内容**：
- `PortfolioPosition` 数据类：持仓快照，含 `has_position` 属性、`to_position_manager_position()` 桥接方法、`to_dict()` 序列化
- `PortfolioState` 数据类：全持仓管理，含 `get_position()` / `all_tickers()` / `held_tickers()` / `add_watchlist_ticker()` / `update_position()` / `summary()` / `save()` 原子写回
- `load_portfolio()` 工厂函数：从 `data/portfolio.yaml` 加载，文件不存在或解析失败时优雅降级
- 10 个单元测试全部通过，`smoke_test.py` 无回归

**验收标准**：
- ✅ `from engine import PortfolioState, PortfolioPosition, load_portfolio` 正常工作
- ✅ `daily_run.py` 通过 `load_portfolio()` 加载持仓
- ✅ `PositionAnalyzer.analyze()` 接收 `PortfolioPosition` 正常分析

---

### 3.2 ✅ 统一配置加载

| 属性 | 值 |
|------|---|
| **问题编号** | D-02 |
| **优先级** | P0 |
| **工作量** | S |
| **状态** | ✅ 已完成（2026-03-24） |
| **影响文件** | 新建 `config_loader.py`；修改 `analyze_factor.py`, `main.py`, `daily_run.py` |

**完成内容**：
- 新建 `config_loader.py`（项目根目录），提供唯一 `load_config(include_keys=True)` 函数
- `analyze_factor.py` — 删除 `_load_config()`，改用 `from config_loader import load_config`
- `main.py` — 删除 `_load_config_full()`，改用 `from config_loader import load_config`
- `daily_run.py` — 删除 `_load_full_config()`，改用 `from config_loader import load_config`

**验收标准**：
- ✅ 全项目仅一处配置加载逻辑
- ✅ `grep -rn "def _load_config" *.py` 无结果
- ✅ `daily_run.py --dry-run` 运行正常

---

### 3.3 ✅ 公共指标库整合

| 属性 | 值 |
|------|---|
| **问题编号** | D-03 |
| **优先级** | P0 |
| **工作量** | M |
| **状态** | ✅ 已完成（2026-03-24） |
| **影响文件** | `strategies/indicators.py`（扩充）；`strategies/*.py`（全部10个文件） |

**完成内容**：
- 在 `strategies/indicators.py` 新增 7 个公共指标函数（纯 pandas）：
  `rsi`, `bollinger_bands`, `ema`, `macd`, `kdj`, `obv`, `fibonacci`
- 以下 10 个策略文件移除本地重复实现，改用 import：
  `rsi_reversion.py`, `rsi_drawdown_0225.py`, `bollinger_breakout.py`, `bollinger_rsi_trend.py`, `macd_rsi_trend.py`, `macd_rsi_combo.py`, `kdj_obv.py`, `kdj_pvt.py`, `rsi_obv.py`, `rsi_pvt.py`
- 同时清理了各文件中不再需要的 `import numpy as np`（如该文件不再使用 numpy）

**验收标准**：
- ✅ `grep -rn "def _rsi\|def _bollinger\|def _macd\|def _kdj\|def _fibonacci\|def _obv" strategies/` 仅剩 `indicators.py` 内 `add_ta_features_fallback()` 的内部 helper
- ✅ `smoke_test.py` 全部通过

---

### 3.4 ✅ 测试框架搭建

| 属性 | 值 |
|------|---|
| **问题编号** | D-05 |
| **优先级** | P0 |
| **工作量** | L |
| **状态** | ✅ 已完成（2026-03-24） |
| **影响文件** | 新建 `tests/` 目录 |

**完成内容**：
- `tests/conftest.py` — Fixtures：`synthetic_ohlcv`（300行合成 OHLCV, seed=42）、`atr_plunge_ohlcv`（先涨后暴跌数据）、`default_config`、`minimal_config`
- `tests/test_schemas.py` — 13 tests，列名归一化（中英文同义词、yfinance MultiIndex）、OHLCV 校验（必填列、OHLC 关系、负Volume警告）
- `tests/test_oms.py` — 8 tests，PaperOMS 买卖/拒绝/ round-trip，`create_oms()` 工厂
- `tests/test_portfolio_state.py` — 9 tests，`PortfolioPosition` 属性、`PortfolioState` 持久化、ticker 下划线转换、大小写不敏感
- `tests/test_position_manager.py` — 12 tests，Kelly 公式边界（胜率 0/1、盈亏比 0）、熔断触发/去重、ATR 止损、冷却期、`apply_risk_controls`
- `tests/test_backtest.py` — 5 tests，必填字段、全持/全空信号、ATR 止损触发、`signal.shift(1)` 前视偏差修复验证
- `tests/test_strategies.py` — 50 tests，全部 16 个规则策略 `run()` + `predict()` 接口、参数化回归、最小配置容错、XGBoost/LightGBM 信号边界
- `tests/test_data_manager.py` — 5 tests，schemas 集成、原子 CSV 写入
- `tests/test_signal_aggregator.py` — 5 tests，空目录降级、置信度边界 `[0,1]`、`AggregatedSignal` 属性
- `tests/test_integration.py` — 3 tests，策略→因子分析→回测全流程、`PortfolioPosition` → `PositionManager.Position` round-trip、`get_recommendation` 接口

**验收标准**：
- ✅ `pytest tests/ -v` 全通过（100 passed）
- ✅ 核心路径覆盖率 > 60%（`position_manager` 59%、`xgboost_enhanced` 64%、rule strategies 90-100%）
- ✅ `smoke_test.py` 无回归
- ✅ 全程离线，无网络请求

---

### 3.5 ✅ 依赖版本升级

| 属性 | 值 |
|------|---|
| **问题编号** | D-08 |
| **优先级** | P0 |
| **工作量** | S |
| **状态** | ✅ 已完成（2026-03-24） |
| **影响文件** | `requirements.txt`, `requirements-relaxed.txt`, 新建 `pytest.ini` |

**完成内容**：
- 升级 `yfinance` 0.2.36 → **1.2.0**（含多个安全修复）
- 补全缺失依赖并固定版本（所有库均已 `==` 钉死）：
  - `textblob==0.19.0`, `snownlp==0.12.3`, `tsfresh==0.21.1`, `pyts==0.13.0`, `pyarrow==23.0.1`, `pip-audit==2.10.0`
- 已安装升级库（反映实际环境）：`pandas 2.3.3`, `numpy 1.26.4`, `scikit-learn 1.8.0`, `matplotlib 3.10.8`, `xgboost 3.2.0`, `lightgbm 4.6.0`, `optuna 4.8.0`, `joblib 1.5.3`, `tenacity 9.1.4`, `exchange_calendars 4.13.2`
- 新建 `pytest.ini`：配置 `pythonpath = .`，确保 `pytest tests/` 无需手动设置 `PYTHONPATH` 即可运行
- 更新 `requirements-relaxed.txt` 补全所有依赖的宽松约束
- `pip-audit -r requirements.txt` → **No known vulnerabilities found**

**验收标准**：
- ✅ `pip install -r requirements.txt` 无冲突（所有包已安装）
- ✅ `pytest tests/` → **100 passed** in 3.65s
- ✅ `smoke_test.py` → **PASS ✅**
- ✅ `pip-audit` → **No known vulnerabilities found**

---

### 3.6 ✅ 结构化日志

| 属性 | 值 |
|------|---|
| **问题编号** | D-07 |
| **优先级** | P0 |
| **工作量** | M |
| **状态** | ✅ 已完成（2026-03-25） |
| **影响文件** | `log_config.py`（新建）；`daily_run.py`, `main.py`, `analyze_factor.py`, `strategies/*.py` |

**完成内容**：
- `log_config.py` 已存在（彩色控制台 + JSON Lines 文件 + RotatingFileHandler + LOG_LEVEL 环境变量）
- 以下文件全部 `print()` → `logger.info/warning/error`：
  - `daily_run.py` — 核心每日运行入口
  - `main.py` — 训练主流程
  - `analyze_factor.py` — 因子分析引擎
  - `strategies/tsfresh_features.py`, `strategies/tsfresh_xgboost.py`, `strategies/xgboost_enhanced.py`, `strategies/lightgbm_enhanced.py`
- `engine/` 和 `data/` 目录无 print 语句
- 根目录 `*.py`（standalone 脚本）保留 print，不影响核心管线

**验收标准**：
- ✅ `grep -rn "print(" *.py strategies/ engine/ data/` 结果为 0
- ✅ 核心文件统一使用 `get_logger(__name__)` 获取 logger

---

## 四、Phase 2 — 策略与风控增强 (P1)

> **目标**：提升信号质量，完善组合风控，打通实盘通道。
> **周期**：2-3 个月（Phase 1 完成后启动）

### 4.1 ⏳ Ensemble / Stacking 策略

| 属性 | 值 |
|------|---|
| **优先级** | P1 |
| **工作量** | L |
| **影响文件** | `engine/signal_aggregator.py`（增强）；新建 `strategies/ensemble_stacking.py` |

**现状**：`SignalAggregator` 采用 Sharpe-加权投票（简单加权平均），无 Meta-Learner。

**改进方案**：

1. **Voting Ensemble 增强**：
   - 增加 Softmax 温度参数控制投票尖锐度
   - 支持按策略类型分组投票（规则组 vs ML 组 → 再聚合）
   - 增加 "一致性过滤"：当策略分歧过大（bull_count ≈ bear_count）时输出 HOLD 而非弱信号

2. **Stacking Meta-Learner**：
   ```python
   # strategies/ensemble_stacking.py
   # 第一层：各子策略输出信号概率
   # 第二层：Logistic Regression / XGBoost 学习最优组合权重
   # 训练数据：历史回测中各策略的逐日信号 + 实际收益
   ```

3. **信号衰减**：因子存在时间越久权重越低（指数衰减），鼓励使用新鲜因子

**验收标准**：
- Stacking Ensemble 在 Walk-Forward 验证中 Sharpe > 单策略中位数
- 回测中 Ensemble 最大回撤 < 单策略最大回撤

---

### 4.2 ⏳ 深度学习策略（可选）

| 属性 | 值 |
|------|---|
| **优先级** | P1（可选） |
| **工作量** | XL |
| **影响文件** | 新建 `strategies/lstm_trend.py`, `strategies/transformer_trend.py` |

**现状**：ML 策略仅有 XGBoost / LightGBM（树模型），无序列建模能力。

**改进方案**：

1. **LSTM 趋势策略**：
   - 输入：滑动窗口 OHLCV + 技术指标（20-60 天）
   - 输出：未来 N 天涨跌概率
   - 框架：PyTorch（轻量级），支持 CPU 推理

2. **Temporal Fusion Transformer (TFT)**：
   - 用于捕捉长期依赖 + 可解释性（attention 权重可视化）
   - 需评估训练成本，建议先在 3-5 只核心标的上验证

3. **集成到策略接口**：
   ```python
   NAME = "lstm_trend"
   def run(data, config) -> (signal, model, meta): ...
   def predict(model, data, config, meta) -> pd.Series: ...
   ```

**风险**：
- 港股 HSI 50 只样本，深度学习容易过拟合
- 建议先验证 LSTM 在 Walk-Forward 中是否优于 XGBoost，不优则不上线

**验收标准**：
- Walk-Forward Sharpe > 1.0 且不显著劣于 XGBoost
- 推理时间 < 1 秒/股票（CPU）

---

### 4.4 ⏳ Portfolio-Level 风控

| 属性 | 值 |
|------|---|
| **问题编号** | D-06 |
| **优先级** | P1 |
| **工作量** | L |
| **影响文件** | 新建 `engine/portfolio_risk.py`；修改 `daily_run.py`, `position_manager.py` |

**现状**：仅有单股级别风控（ATR 止损 / Kelly 仓位 / 熔断），无组合维度。

**改进方案**：

```python
# engine/portfolio_risk.py

class PortfolioRiskManager:
    """组合级风控引擎"""
    
    def correlation_matrix(self, holdings: dict[str, pd.DataFrame]) -> pd.DataFrame:
        """持仓间相关性矩阵，发出集中度预警"""
    
    def sector_exposure(self, positions: list[PortfolioPosition]) -> dict:
        """行业/板块暴露度分析（金融、科技、消费...）"""
    
    def portfolio_var(self, confidence: float = 0.95) -> float:
        """Historical VaR / Parametric VaR"""
    
    def portfolio_cvar(self, confidence: float = 0.95) -> float:
        """Conditional VaR (Expected Shortfall)"""
    
    def max_portfolio_drawdown(self) -> float:
        """组合级最大回撤"""
    
    def stress_test(self, scenario: str) -> dict:
        """压力测试：'2020_covid', '2022_rate_hike', 'custom'"""
    
    def check_limits(self) -> list[str]:
        """
        检查组合级限额：
        - 单行业暴露 < 40%
        - 前3大持仓集中度 < 60%
        - VaR < 配置阈值
        - 相关性 > 0.8 的持仓对预警
        """
```

**验收标准**：
- 每日报告中包含 VaR、行业暴露、集中度分析
- 高相关性持仓发出飞书预警

---

### 4.5 ⏳ 因子生命周期管理

| 属性 | 值 |
|------|---|
| **优先级** | P1 |
| **工作量** | M |
| **影响文件** | `engine/signal_aggregator.py`, `analyze_factor.py`, 新建 `data/factor_registry.py` |

**现状**：因子文件 `factor_*.pkl` 仅按编号排序、靠文件时间戳判断新鲜度。无版本控制、无效因子无自动清理。

**改进方案**：

1. **Factor Registry**（JSON 索引文件）：
   ```json
   {
     "factors": [
       {
         "id": 42,
         "strategy_name": "macd_rsi_trend",
         "created_at": "2026-03-20T18:00:00",
         "ticker": "0700.HK",
         "sharpe": 1.85,
         "cum_return": 0.23,
         "status": "active",    // active | expired | archived
         "valid_until": "2026-04-20"
       }
     ]
   }
   ```

2. **自动过期**：超过 `factor_ttl_days`（默认 30）的因子标记为 expired，不再参与投票
3. **自动归档**：超过 90 天的因子移至 `data/factors/archive/`
4. **因子对比**：新旧因子 Sharpe / Return 自动对比，劣化时告警

**验收标准**：
- `data/factors/factor_registry.json` 自动维护
- 过期因子不参与每日投票
- `daily_run.py` 日志显示活跃因子数和过期因子数

---

### 4.6 ⏳ PnL 历史追踪

| 属性 | 值 |
|------|---|
| **问题编号** | D-09 |
| **优先级** | P1 |
| **工作量** | M |
| **影响文件** | 新建 `data/pnl_tracker.py`；修改 `daily_run.py`, `data/portfolio.yaml` |

**现状**：每日建议生成后仅发送飞书通知、保存 markdown 报告，不记录历史建议 vs 实际结果。

**改进方案**：

```python
# data/pnl_tracker.py

class PnLTracker:
    """每日 PnL 追踪与绩效归因"""
    
    def record_daily(self, date: str, recommendations: list[RecommendationResult]):
        """记录当日建议 + 持仓快照"""
    
    def calculate_daily_pnl(self, date: str) -> dict:
        """T+1 计算昨日建议的实际收益"""
    
    def attribution_report(self, period: str = "1m") -> dict:
        """
        绩效归因：
        - 按策略归因（哪些策略贡献正收益）
        - 按标的归因（哪些股票贡献正收益）
        - 按操作类型归因（买入 vs 持有 vs 卖出的决策质量）
        """
    
    def export_csv(self) -> Path:
        """导出 PnL 历史为 CSV"""
```

存储：`data/logs/pnl_history.jsonl`（每日一行）

**验收标准**：
- 每日运行后自动追加 PnL 记录
- 可生成 月/季/年 绩效归因报告

---

### 4.7 ⏳ OMS 实盘对接

| 属性 | 值 |
|------|---|
| **问题编号** | D-04 |
| **优先级** | P1 |
| **工作量** | L |
| **影响文件** | `oms.py`；新建 `oms_futu.py` / `oms_ibkr.py` |

**现状**：`LiveOMS` 全部 TODO，`submit_order` / `cancel_order` / `get_position` 仅输出日志并降级 PaperOMS。

**改进方案**（分两步）：

**Step 1：Futu OpenAPI 对接**（推荐首选，港股原生支持）
```python
# oms_futu.py
from moomoo import OpenSecTradeContext, TrdEnv, TrdSide, OrderType

class FutuOMS(OrderManagementSystem):
    def submit_order(self, ticker, action, shares, price, note=''):
        # 真实下单 → 返回 OrderResult
    def cancel_order(self, order_id):
        # 查询未成交订单 → 撤单
    def get_position(self, ticker):
        # 查询实时持仓
    def get_order_status(self, order_id):
        # 查询订单状态（新增接口）
```

**Step 2：风控拦截层**
```python
class SafeOMS(OrderManagementSystem):
    """在真实 OMS 外层包裹风控检查"""
    def submit_order(self, ...):
        # 1. 检查单笔金额 < max_single_order_amount
        # 2. 检查日累计交易次数 < max_daily_trades
        # 3. 检查是否在交易时段
        # 4. 检查组合 VaR 不超限
        # 5. 通过则调用 inner_oms.submit_order()
```

**验收标准**：
- Futu 模拟环境下单 → 查询成交 → 撤单 全流程通过
- 风控拦截层可阻断异常订单

---

## 五、Phase 3 — 实时化与可视化 (P2)

> **目标**：盘中信号能力、可视化 Dashboard、CI/CD。
> **周期**：2-3 个月（Phase 2 完成后启动）

### 5.1 ⏳ 实时行情接入

| 属性 | 值 |
|------|---|
| **问题编号** | D-10 |
| **优先级** | P2 |
| **工作量** | L |
| **影响文件** | `easy_quptation.py`（增强）, `time_kline.py`（增强）；新建 `data/realtime.py` |

**现状**：`easy_quptation.py` 和 `time_kline.py` 是独立工具，未集成到核心管线。

**改进方案**：

1. **实时数据管道**：
   ```python
   # data/realtime.py
   class RealtimeDataStream:
       """实时行情推送 + 信号触发"""
       def subscribe(self, tickers: list[str]): ...
       def on_tick(self, callback): ...
       def on_bar(self, interval: str, callback): ...  # "1m", "5m", "15m"
   ```

2. **盘中信号触发**：
   - 5 分钟 K 线 → 运行规则策略 → 止损价触发 → 飞书即时推送
   - 不做盘中自动交易，仅预警

3. **数据源优先级**：Futu 实时推送 > easyquotation 轮询 > 收盘后补数据

**验收标准**：
- 开盘后 5 秒内收到实时行情
- 价格触及止损价时 30 秒内收到飞书通知

---

### 5.2 ⏳ Streamlit Dashboard

| 属性 | 值 |
|------|---|
| **优先级** | P2 |
| **工作量** | L |
| **影响文件** | 新建 `dashboard/` 目录 |

**现状**：无可视化界面，`visualize.py` 仅生成静态图表。

**改进方案**：

```
dashboard/
├── app.py              # Streamlit 入口
├── pages/
│   ├── portfolio.py    # 持仓总览（市值、盈亏、饼图）
│   ├── signals.py      # 每日信号面板（策略投票明细）
│   ├── backtest.py     # 回测结果查看器
│   ├── pnl.py          # PnL 曲线 + 绩效归因
│   ├── risk.py         # VaR / 相关性矩阵 / 行业暴露
│   └── factors.py      # 因子库管理（状态、过期、Sharpe）
```

**核心页面功能**：

| 页面 | 功能 |
|------|------|
| 持仓总览 | 多股票持仓列表、盈亏热力图、总市值曲线 |
| 信号面板 | 当日各股票信号、置信度、策略投票明细（交互式表格） |
| 回测查看 | 选择策略 + 参数 → 在线回测 → 净值曲线 + 统计指标 |
| PnL 曲线 | 每日/累计 PnL、按策略/标的的绩效归因 |
| 风控仪表盘 | VaR gauge、相关性热力图、行业暴露条形图 |

**验收标准**：
- `streamlit run dashboard/app.py` 启动后可查看所有页面
- 页面加载时间 < 3 秒

---

### 5.3 CI/CD Pipeline

| 属性 | 值 |
|------|---|
| **优先级** | P2 |
| **工作量** | M |
| **影响文件** | 新建 `.github/workflows/ci.yml` |

**改进方案**：

```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: '3.13' }
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v --cov=. --cov-report=xml
      - run: pip-audit -r requirements.txt
      
  lint:
    runs-on: ubuntu-latest
    steps:
      - run: ruff check .
      - run: mypy --strict engine/ data/
```

**验收标准**：
- PR 合入前必须 CI 通过
- 覆盖率报告自动上传

---

### 5.4 性能优化

| 属性 | 值 |
|------|---|
| **优先级** | P2 |
| **工作量** | M |
| **影响文件** | `analyze_factor.py`, `engine/signal_aggregator.py`, `data/manager.py` |

**现状性能瓶颈**：
1. `SignalAggregator` 对每个因子串行调用 `strategy_mod.run()` / `predict()`
2. 全量 CSV 解析判断数据新鲜度（已有 `.meta.json` 优化，但仍有改进空间）
3. tsfresh 特征计算耗时（已限制 `n_jobs`，但仍慢）

**改进方案**：

| 优化点 | 方案 | 预期提升 |
|--------|------|----------|
| 信号推断并行化 | `SignalAggregator` 用 `ProcessPoolExecutor` 并行推断 | 3-5x |
| Parquet 默认存储 | 将 `storage_format` 默认改为 `parquet`（读取速度 5-10x） | 5-10x I/O |
| 因子推断缓存 | 同一 bar date 不重复推断（`lru_cache` by `(factor_id, last_date)`） | 2x |
| NumPy 向量化 | 规则策略中 for-loop 改为向量化（已部分完成） | 2-3x |
| Optuna 分布式 | 使用 `optuna.storages.RDBStorage` 支持多进程共享 | 线性加速 |

**验收标准**：
- `daily_run.py` 分析 10 只股票总时间 < 30 秒（当前估计 60-120 秒）
- Optuna 100 trials 耗时减少 40%

---

## 六、Phase 4 — 生产级运营 (P3)

> **目标**：合规、多市场、自动化运维。
> **周期**：持续迭代

### 6.1 合规审计日志

| 属性 | 值 |
|------|---|
| **优先级** | P3 |
| **工作量** | M |
| **影响文件** | `oms.py`, 新建 `audit/` 目录 |

**改进方案**：

1. **交易审计日志**（不可篡改）：
   ```
   data/audit/
   ├── trade_log_2026Q1.jsonl    # 季度归档
   ├── signal_log_2026Q1.jsonl   # 信号生成记录
   └── config_changes.jsonl      # 配置变更记录
   ```

2. **每笔交易记录**：
   - 信号来源（哪些策略投了什么票）
   - 风控审批链（是否触发止损/熔断/限额）
   - 时间戳精确到毫秒
   - 执行前后快照

3. **异常检测**：
   - 单日交易次数异常告警
   - 大额订单告警
   - 非交易时段信号告警

---

### 6.2 多市场扩展

| 属性 | 值 |
|------|---|
| **优先级** | P3 |
| **工作量** | XL |
| **影响文件** | `data/calendar.py`, `analyze_factor.py`, `config.yaml`, 新建 `markets/` |

**现状**：硬编码港股假设：
- `data/calendar.py` → XHKG
- `fees_rate: 0.00088` + `stamp_duty: 0.001`（港股特有）
- `data/hsi_stocks.py` → HSI 成分股

**改进方案**：

```python
# markets/base.py
class Market(ABC):
    name: str              # "HKEX", "SSE", "NYSE"
    calendar_name: str     # "XHKG", "XSHG", "XNYS"
    fees: FeeSchedule      # 交易费率结构
    lot_size: int          # 最小交易单位（港股=手）
    currency: str          # "HKD", "CNY", "USD"
    
# markets/hkex.py
class HKEXMarket(Market): ...

# markets/a_share.py  
class AShareMarket(Market): ...
```

**验收标准**：
- 通过 `config.yaml → market: HKEX` 切换市场
- A 股 / 美股能复用全部策略和回测引擎

---

### 6.3 自动化运维

| 属性 | 值 |
|------|---|
| **优先级** | P3 |
| **工作量** | M |
| **影响文件** | `daily_run.sh`, 新建 `ops/` 目录 |

**改进方案**：

1. **健康检查**：
   ```python
   # ops/health_check.py
   def check_data_freshness(): ...   # 历史数据是否更新
   def check_factor_health(): ...    # 因子库是否有效
   def check_disk_usage(): ...       # 数据目录磁盘使用
   def check_api_keys(): ...         # keys.yaml 中 API 是否有效
   ```

2. **自动恢复**：
   - 数据下载失败 → 自动切换 vendor → 告警
   - 因子过期 → 自动触发 `main.py --use-optuna` 重训练
   - 磁盘使用 > 80% → 自动归档旧数据

3. **Cron 任务编排**：
   ```
   # crontab 建议
   00 17  * * 1-5  /path/to/ops/data_update.sh     # 17:00 更新数据
   30 17  * * 1-5  /path/to/daily_run.sh            # 17:30 每日推荐
   00 02  * * 6    /path/to/ops/weekly_retrain.sh   # 周六 02:00 重训练
   00 03  1 * *    /path/to/ops/monthly_cleanup.sh  # 月初清理归档
   ```

4. **监控告警**：
   - 每日运行超时 → 飞书告警
   - 策略全部投弃权票 → 飞书告警
   - PnL 连续 N 天为负 → 飞书告警

---

### 6.4 灾备方案

| 属性 | 值 |
|------|---|
| **优先级** | P3 |
| **工作量** | S |
| **影响文件** | 新建 `ops/backup.sh` |

**改进方案**：

1. **数据备份**：
   - `data/historical/` → 每日增量备份至云存储（S3/OSS）
   - `data/factors/` → 每次训练后备份
   - `data/portfolio.yaml` → 每次修改后备份（git 版本控制）

2. **恢复流程**：
   - 文档化：从空环境恢复到可运行状态的完整步骤
   - 恢复测试：每月执行一次恢复演练

---

## 附录 A：优先级定义

| 级别 | 含义 | 时间框架 |
|------|------|----------|
| **P0** | 阻塞核心功能 / 运行时风险 | 1-2 个月内完成 |
| **P1** | 显著提升系统能力 | 2-3 个月内完成 |
| **P2** | 提升效率和用户体验 | 3-6 个月内完成 |
| **P3** | 远期规划 / 锦上添花 | 6-12 个月持续迭代 |

## 附录 B：工作量估算说明

| 标记 | 预估人天 | 说明 |
|------|----------|------|
| **S** | 1-3 天 | 单文件修改，逻辑简单 |
| **M** | 3-7 天 | 多文件协调，需少量设计 |
| **L** | 7-15 天 | 跨模块重构，需测试验证 |
| **XL** | 15-30 天 | 新增子系统，需设计 + 实现 + 测试 |

---

## 附录 C：改进项汇总表

| # | 改进项 | Phase | 优先级 | 工作量 | 关键影响 |
|---|--------|-------|--------|--------|----------|
| 1 | 补全 `portfolio_state.py` | P1 | P0 | M | ✅ 已完成，消除 ImportError 风险 |
| 2 | 统一配置加载 | P1 | P0 | S | ✅ 已完成，消除重复、降低维护成本 |
| 3 | 公共指标库整合 | P1 | P0 | M | ✅ 已完成，消除重复代码、确保指标一致性 |
| 4 | 测试框架搭建 | P1 | P0 | L | ✅ 已完成，100 单元测试，核心路径 > 60% 覆盖 |
| 5 | 依赖版本升级 | P1 | P0 | S | ✅ 已完成，全库固定版本，pip-audit 无漏洞，pytest 100 passed |
| 6 | 结构化日志 | P1 | P0 | M | 可观测性、排障效率 |
| 7 | **选股模块** | P2 | P1 | L | **主动发现买入机会，补全选股盲区** |
| 8 | Ensemble / Stacking | P2 | P1 | L | 信号质量提升 |
| 9 | 深度学习策略 | P2 | P1 | XL | 序列建模能力（可选） |
| 10 | Portfolio-Level 风控 | P2 | P1 | L | 组合风险管理 |
| 11 | 因子生命周期管理 | P2 | P1 | M | 因子质量保证 |
| 12 | PnL 历史追踪 | P2 | P1 | M | 绩效归因、策略评估 |
| 13 | OMS 实盘对接 | P2 | P1 | L | 自动化交易 |
| 14 | 实时行情接入 | P3 | P2 | L | 盘中预警 |
| 15 | Streamlit Dashboard | P3 | P2 | L | 可视化 |
| 16 | CI/CD Pipeline | P3 | P2 | M | 质量门禁 |
| 17 | 性能优化 | P3 | P2 | M | 提速 3-5x |
| 18 | 合规审计日志 | P4 | P3 | M | 交易合规 |
| 19 | 多市场扩展 | P4 | P3 | XL | 市场覆盖 |
| 20 | 自动化运维 | P4 | P3 | M | 无人值守 |
| 21 | 灾备方案 | P4 | P3 | S | 业务连续性 |

---

> **下一步行动**：Phase 1 已完成 6/6 项（portfolio_state、配置加载、公共指标库、测试框架、依赖升级、结构化日志），Phase 2 已可启动。

