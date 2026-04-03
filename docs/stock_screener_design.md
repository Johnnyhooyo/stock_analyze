# 选股模块需求分析与设计方案

> **编制日期**：2026-03-24
> **关联文档**：[升级规划](upgrade_plan.md)
> **结论**：**需要。** 当前系统有明显的"选股盲区"，选股模块是提升核心目标价值的关键一环。

---

## 一、问题诊断：当前系统的选股盲区

### 1.1 现有选股方式

| 方式 | 入口 | 说明 | 局限性 |
|------|------|------|--------|
| **手动指定** | `portfolio.yaml` | 人工填写持仓/观察列表 | 完全依赖主观判断，无数据驱动 |
| **HSI 全扫** | `--watchlist hsi` | 扫描 ~80 只 HSI 成分股 | 等权扫描，无优先级排序；扫描范围固定 |
| **命令行指定** | `--tickers 0700.HK` | 临时分析指定标的 | 需要人先知道要分析什么 |

### 1.2 核心问题

```
当前流程：
  人工选股 → 写入 portfolio.yaml → daily_run.py 分析 → 给出持/买/卖建议
                ↑
            这里是黑箱：靠直觉、新闻、朋友推荐...

理想流程：
  全市场扫描 → 量化选股排序 → 推荐候选池 → daily_run.py 深度分析 → 操作建议
                ↑                   ↑
            选股模块            自动更新 watchlist
```

**根本问题**：系统擅长 *"给定一只股票，告诉你该做什么"*，但不擅长 *"在 80+ 只 HSI 股票中，今天应该重点关注哪几只"*。

对于项目核心目标 *"每日运行，基于持仓给出下一个交易日的操作建议"*：
- **持仓股**：现有 `PositionAnalyzer` 已覆盖（持有/卖出/止损）
- **空仓股/新建仓机会**：目前**完全缺失**，系统无法主动发现买入机会

---

## 二、选股模块的核心价值

### 2.1 解决的问题

| # | 问题 | 选股模块如何解决 |
|---|------|-----------------|
| 1 | 不知道该关注哪些股票 | 全市场量化排序，每日输出 Top-N 候选 |
| 2 | HSI 全扫耗时太长（80股 × 20因子） | 先粗筛 → 再精选，减少深度分析数量 |
| 3 | 发现买入机会太被动 | 主动扫描：突破/超跌反弹/异常放量/趋势启动 |
| 4 | 持仓调整缺乏替代标的 | 卖出后自动推荐替换候选 |
| 5 | 行业轮动/板块机会无感知 | 板块强弱排序 + 行业动量 |

### 2.2 与现有模块的关系

```
┌─────────────────────────────────────────────────────────────┐
│                    每日推荐引擎 (daily_run.py)               │
│                                                             │
│  ┌──────────────┐    ┌───────────────┐    ┌──────────────┐  │
│  │   选股模块    │───▶│ SignalAggregator│───▶│PositionAnalyzer│ │
│  │  (新增)       │    │  (现有)        │    │  (现有)       │  │
│  │              │    │               │    │              │  │
│  │ 全市场 → Top-N│    │ 多策略共识投票  │    │ 风控+建议生成  │  │
│  └──────────────┘    └───────────────┘    └──────────────┘  │
│        ↓                                                    │
│  ┌──────────────┐                                           │
│  │ portfolio.yaml│  ← 自动更新 watchlist                     │
│  │  (观察列表)   │                                           │
│  └──────────────┘                                           │
└─────────────────────────────────────────────────────────────┘
```

选股模块位于管线 **最前端**，充当"漏斗口"：
1. **粗筛**：全市场快速扫描（轻量指标，<1 秒/股票）
2. **精排**：对粗筛通过的 Top-N 标的运行完整 `SignalAggregator`
3. **输出**：推荐候选池 → 自动更新 `portfolio.yaml` 的观察列表

---

## 三、设计方案

### 3.1 模块结构

```
engine/
├── stock_screener.py          # 选股引擎核心
├── screener_factors.py        # 选股因子定义
├── signal_aggregator.py       # (现有) 多策略共识
├── position_analyzer.py       # (现有) 持仓分析
└── portfolio_risk.py          # (规划中) 组合风控
```

### 3.2 选股因子体系

选股因子需要 **轻量、快速**（全市场扫描用），与交易策略（深度分析用）区分：

#### Tier 1：动量与趋势（速度快，信号明确）

| 因子名 | 计算方式 | 选股逻辑 |
|--------|----------|----------|
| **价格动量** | `Close / Close.shift(N) - 1` | N=5/10/20/60 日涨幅排序 |
| **均线多头排列** | MA5 > MA10 > MA20 > MA60 | 全满足 = 趋势确认 |
| **突破信号** | Close > 20日新高 | 突破买入候选 |
| **RSI 超卖反弹** | RSI14 < 30 且 RSI14 > RSI14.shift(1) | 均值回归候选 |
| **MACD 金叉** | MACD Line 上穿 Signal Line | 趋势启动信号 |

#### Tier 2：量价配合（中等计算量）

| 因子名 | 计算方式 | 选股逻辑 |
|--------|----------|----------|
| **放量突破** | Volume > 2×MA20_Volume 且 Close 创新高 | 有资金配合的突破 |
| **缩量回调** | Volume < 0.5×MA20_Volume 且位于上升趋势 | 回调买入点 |
| **OBV 趋势** | OBV 创 20 日新高 | 资金流入确认 |
| **VWAP 偏离** | (Close - VWAP) / VWAP | 偏离过大 = 短期修正机会 |

#### Tier 3：基本面辅助（低频更新）

| 因子名 | 数据来源 | 选股逻辑 |
|--------|----------|----------|
| **市盈率分位** | yfinance `.info['trailingPE']` | PE < 历史 25 分位 = 低估 |
| **股息率** | yfinance `.info['dividendYield']` | 高息股偏好 |
| **市值** | Close × 总股本 | 过滤 < 50 亿港元的小盘股（可选） |

#### Tier 4：另类数据（已有模块可复用）

| 因子名 | 现有模块 | 选股逻辑 |
|--------|----------|----------|
| **情感分析** | `sentiment_analysis.py` | 情感从负转正 = 关注信号 |
| **搜索热度** | `google_trends.py` | 热度突然上升 = 关注信号 |

### 3.3 核心类设计

```python
# engine/stock_screener.py

@dataclass
class ScreenerResult:
    """单只股票的选股评分结果"""
    ticker: str
    composite_score: float        # 综合评分 [0, 100]
    rank: int                     # 排名
    momentum_score: float         # 动量得分
    trend_score: float            # 趋势得分
    volume_score: float           # 量价得分
    valuation_score: float        # 估值得分（可选）
    sentiment_score: float        # 情感得分（可选）
    signals: list[str]            # 触发的选股信号 ["突破20日新高", "放量", ...]
    sector: str                   # 所属板块
    last_close: float
    change_pct_5d: float          # 5日涨跌幅
    change_pct_20d: float         # 20日涨跌幅
    avg_volume_ratio: float       # 近5日成交量 / 20日均量


class StockScreener:
    """
    选股引擎 — 从全市场快速筛选高潜力标的

    设计理念：
      - 轻量级：每只股票 < 0.5 秒，80 只 HSI 全扫 < 40 秒
      - 多维度：动量 + 趋势 + 量价 + 估值 + 情感
      - 可配置：各维度权重通过 config.yaml 调整
      - 与交易策略解耦：选股 ≠ 交易信号，选股是 "值得关注"，交易信号是 "值得操作"

    用法：
        screener = StockScreener(config)
        results = screener.screen(tickers, data_dict)
        top_picks = screener.top_n(results, n=10)
    """

    def __init__(self, config: dict):
        self.config = config
        scr_cfg = config.get("screener", {})
        self.weights = {
            "momentum": scr_cfg.get("weight_momentum", 0.30),
            "trend":    scr_cfg.get("weight_trend",    0.25),
            "volume":   scr_cfg.get("weight_volume",   0.20),
            "valuation":scr_cfg.get("weight_valuation", 0.15),
            "sentiment":scr_cfg.get("weight_sentiment", 0.10),
        }
        self.top_n_count = scr_cfg.get("top_n", 10)
        self.min_score = scr_cfg.get("min_score", 50.0)

    def screen(
        self,
        tickers: list[str],
        data_dict: dict[str, pd.DataFrame],
    ) -> list[ScreenerResult]:
        """
        对给定股票池执行多因子选股评分。

        Args:
            tickers:   待筛选股票列表
            data_dict: {ticker: OHLCV DataFrame} 历史数据字典

        Returns:
            按 composite_score 降序排列的 ScreenerResult 列表
        """
        ...

    def _score_momentum(self, df: pd.DataFrame) -> float:
        """动量评分: 5日/10日/20日/60日涨幅加权"""
        ...

    def _score_trend(self, df: pd.DataFrame) -> float:
        """趋势评分: 均线排列 + RSI 位置 + MACD 方向"""
        ...

    def _score_volume(self, df: pd.DataFrame) -> float:
        """量价评分: 放量突破 / 缩量回调 / OBV 趋势"""
        ...

    def _score_valuation(self, ticker: str) -> float:
        """估值评分: PE分位 + 股息率（低频，可缓存）"""
        ...

    def _score_sentiment(self, ticker: str) -> float:
        """情感评分: 复用 sentiment_analysis.py"""
        ...

    def _detect_signals(self, df: pd.DataFrame) -> list[str]:
        """
        检测离散选股信号（用于人类可读的推荐理由）:
        - "突破20日新高"
        - "MACD金叉"
        - "放量上涨"
        - "RSI超卖反弹"
        - "均线多头排列"
        """
        ...

    def top_n(
        self,
        results: list[ScreenerResult],
        n: int = None,
        exclude_held: bool = True,
        portfolio_state = None,
    ) -> list[ScreenerResult]:
        """
        返回 Top-N 候选，可排除已持仓标的。
        """
        ...

    # ── 板块分析 ──────────────────────────────────────

    def sector_ranking(
        self, results: list[ScreenerResult]
    ) -> list[dict]:
        """
        按板块聚合评分，返回板块强弱排序。
        用途：发现行业轮动机会。
        
        Returns:
            [{"sector": "科技/互联网", "avg_score": 72.5, "top_stock": "0700.HK", ...}, ...]
        """
        ...
```

### 3.4 与 daily_run.py 的集成

```python
# daily_run.py 中新增选股阶段（插入在"并发分析所有股票"之前）

def main():
    ...
    
    # ── [新增] 选股阶段 ──────────────────────────────────────────
    if args.watchlist in ("hsi", "all") or args.enable_screener:
        logger.info("选股模块启动")
        from engine.stock_screener import StockScreener
        
        screener = StockScreener(config)
        
        # 快速加载所有候选股票的历史数据
        candidate_tickers = get_hsi_stocks()  # 或更大范围
        data_dict = {}
        for t in candidate_tickers:
            try:
                df = data_mgr.load(t, period="1y")  # 选股只需 1 年数据
                if df is not None and len(df) > 60:
                    data_dict[t] = df
            except Exception:
                pass
        
        # 执行选股评分
        screen_results = screener.screen(list(data_dict.keys()), data_dict)
        top_picks = screener.top_n(
            screen_results, 
            n=10, 
            exclude_held=True, 
            portfolio_state=portfolio_state,
        )
        
        # 板块分析
        sector_ranking = screener.sector_ranking(screen_results)
        
        # 将 Top-N 加入本次分析列表
        for pick in top_picks:
            if pick.ticker not in tickers:
                tickers.append(pick.ticker)
                portfolio_state.add_watchlist_ticker(pick.ticker)
        
        logger.info("选股完成", extra={"total_scanned": len(screen_results), "top_n": len(top_picks)})
        for i, pick in enumerate(top_picks, 1):
            logger.info(
                f"  {i}. {pick.ticker} 评分={pick.composite_score:.1f}  信号: {', '.join(pick.signals[:3])}"
            )
    
    # ── (现有) 并发分析所有股票 ──────────────────────────────────
    ...
```

### 3.5 每日报告增强

在现有 Markdown 报告和飞书通知中增加选股板块：

```markdown
## 📊 今日选股推荐

> 基于动量(30%) + 趋势(25%) + 量价(20%) + 估值(15%) + 情感(10%) 综合评分

| 排名 | 标的 | 评分 | 5日涨幅 | 20日涨幅 | 量比 | 选股信号 |
|------|------|------|---------|---------|------|----------|
| 1 | 1810.HK 小米 | 85.2 | +6.3% | +15.1% | 1.8x | 突破20日新高, 放量, 均线多头 |
| 2 | 3690.HK 美团 | 78.6 | +4.1% | +8.3% | 1.5x | MACD金叉, OBV新高 |
| 3 | 9988.HK 阿里 | 72.1 | +2.8% | -3.2% | 2.1x | RSI超卖反弹, 放量反弹 |

### 板块强弱排序

| 板块 | 平均评分 | 龙头股 | 趋势 |
|------|---------|--------|------|
| 科技/互联网 | 72.5 | 0700.HK | 📈 上升 |
| 消费 | 65.3 | 2020.HK | 📈 上升 |
| 金融/银行 | 52.1 | 0005.HK | ➡️ 横盘 |
| 地产 | 38.7 | 0016.HK | 📉 下降 |
```

### 3.6 配置项

```yaml
# config.yaml 新增

# ── 选股模块配置 ────────────────────────────────────────────
screener:
  enabled: true                    # 是否启用选股模块
  
  # 选股池范围
  universe: "hsi"                  # "hsi" = HSI成分股, "custom" = 自定义列表
  custom_universe: []              # universe="custom" 时使用的自定义股票列表
  
  # 评分权重（总和应为 1.0）
  weight_momentum: 0.30            # 动量因子权重
  weight_trend: 0.25               # 趋势因子权重
  weight_volume: 0.20              # 量价因子权重
  weight_valuation: 0.15           # 估值因子权重
  weight_sentiment: 0.10           # 情感因子权重
  
  # 输出控制
  top_n: 10                        # 输出 Top-N 候选
  min_score: 50.0                  # 最低评分阈值
  auto_add_to_watchlist: true      # 是否自动将 Top-N 加入观察列表
  
  # 性能控制
  data_period: "1y"                # 选股用历史数据（无需太长）
  enable_valuation: false          # 是否启用估值因子（需额外API调用，较慢）
  enable_sentiment: false          # 是否启用情感因子（每只约3-10秒，全扫很慢）
  
  # 板块定义（用于行业轮动分析）
  sectors:
    科技/互联网: ["0700.HK", "9988.HK", "3690.HK", "9618.HK", "9888.HK", "9961.HK", "1810.HK"]
    金融/银行: ["0005.HK", "0388.HK", "0939.HK", "3988.HK", "3968.HK", "6820.HK"]
    消费: ["0027.HK", "2319.HK", "2269.HK", "2020.HK", "6611.HK", "6837.HK"]
    地产: ["0016.HK", "0017.HK", "1109.HK", "0688.HK", "0823.HK"]
    汽车/制造: ["1211.HK", "0175.HK", "2382.HK", "0285.HK"]
    医药: ["1177.HK", "2266.HK"]
    能源: ["0883.HK", "0857.HK"]
    公用事业: ["0066.HK", "0686.HK", "0002.HK", "0003.HK"]
```

---

## 四、实施计划

### 4.1 分步交付

| 阶段 | 内容 | 测试文件 | 依赖 |
|------|------|---------|------|
| **Step 1** | 基础选股引擎：动量 + 趋势 + 量价评分 | `tests/test_stock_screener.py`（19 tests） | 无 |
| **Step 2** | 集成到 `daily_run.py`，报告增加选股板块 | `tests/test_integration.py`（2 tests） | Step 1 |
| **Step 3** | 板块分析 + 行业轮动 | `tests/test_integration.py`（2 tests） | Step 1 |
| **Step 4** | 估值 + 情感因子（可选） | `tests/test_stock_screener.py`（6 tests） | Step 1 |
| **Step 5** | 选股因子回测验证框架 | `tests/test_screener_backtest.py`（7 tests） | Step 1 |

**总工作量**：L（约 2-3 周）

### 4.2 优先级定位

**P1（Phase 2）** — 与 Ensemble 策略增强、Portfolio 风控同期推进。

理由：
- 不是运行时阻塞问题（P0），但直接影响系统核心价值
- 选股模块为"发现新机会"，是现有"管理已有持仓"能力的自然延伸
- 技术上独立，可并行开发不影响现有流程

### 4.3 关键设计决策

| 决策点 | 选择 | 理由 |
|--------|------|------|
| 选股 vs 交易信号 | 严格分离 | 选股 = "值得关注"，交易 = "值得操作"，避免过度交易 |
| 选股频率 | 每日一次（收盘后） | 与现有 daily_run 节奏一致，不增加运维复杂度 |
| 选股池范围 | 先 HSI，后扩展 | HSI 80+ 只足够起步，数据已有基础 |
| 评分模型 | 线性加权 → 后期可升级 ML | 简单可解释，便于调参和 debug |
| 自动买入 | **否** | 选股推荐 + 人工确认 → 手动加入持仓，降低风险 |

---

## 五、预期收益

| 指标 | 当前 | 有选股模块后 |
|------|------|------------|
| 每日关注范围 | 手动选的 1-5 只 | 全 HSI 80+ 只自动筛选 |
| 发现买入机会 | 被动（看新闻/朋友推荐） | 主动（量化评分排序） |
| 选股耗时 | 人工 30-60 分钟 | 自动 < 60 秒 |
| 行业轮动感知 | 无 | 每日板块强弱排序 |
| 报告完整度 | 仅持仓建议 | 持仓建议 + 新机会推荐 + 板块分析 |

---

## 六、风险与注意事项

1. **过度交易风险**：选股模块每日推荐新标的，可能诱导频繁换股。
   - 缓解：设置冷却期（同一标的连续 N 天出现才推荐），分离"选股推荐"与"交易信号"。

2. **数据量增加**：全市场扫描需要更多 API 调用。
   - 缓解：选股用 1 年数据（vs 交易用 3-5 年），增量更新机制已有。

3. **选股因子过拟合**：权重调优可能对历史数据过拟合。
   - 缓解：Phase 2 Step 5 建立选股因子回测框架，用滚动窗口验证。

4. **HSI 成分股局限**：仅 80 只股票可能遗漏港股通/中小盘机会。
   - 缓解：Phase 4 扩展 universe 至港股通（500+ 只），但需控制 API 调用量。

5. **sector 列表硬编码维护负担**：`config.yaml` 中 `sectors` 的 ticker 列表会随 HSI 成分调整而过期。
   - 缓解：定期同步 `data/hsi_stocks.py` 与 `sectors` 定义；后续改为从 `hsi_stocks.py` 动态读取并通过预定义标签匹配。

