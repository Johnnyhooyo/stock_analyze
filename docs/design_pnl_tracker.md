# PnL 历史追踪设计方案

> **编制日期**：2026-04-03  
> **对应 upgrade_plan.md**：Phase 2 § 4.6  
> **对应问题**：D-09  
> **优先级**：P1  
> **工作量估算**：M（4-6 天）

---

## 一、现状与问题

### 1.1 现状

`daily_run.py` 每日计算的 `total_pnl` 是**当前持仓浮动盈亏**（市值 - 成本），存在于报告 Markdown 和飞书通知中，但：

- 不持久化到任何结构化存储
- 无法回溯历史建议质量
- 无法评估系统建议（买入/持有/卖出）的实际效果
- 无法做策略归因（哪类策略信号最准确）

### 1.2 核心需求

1. **每日自动追加**：`daily_run.py` 每次运行后，将当日建议快照写入 `data/logs/pnl_history.jsonl`
2. **T+1 收益计算**：次日运行时，补充前一日建议在 T+1 的实际涨跌
3. **绩效归因**：按策略、按标的、按操作类型，统计建议准确率
4. **CSV 导出**：供 Excel / Pandas 离线分析

### 1.3 边界说明

- 本模块追踪**系统建议**的历史，**不追踪用户的实际成交**（无 broker API）
- "T+1 收益"= 建议日收盘价 → 次交易日收盘价的变化，代表"如果完全按建议执行"的理论收益
- 对"持有"建议，视为建议正确当信号方向与次日涨跌一致；"卖出"建议正确当次日下跌

---

## 二、数据模型

### 2.1 每日快照记录（`pnl_history.jsonl`，每日每股一行）

```json
{
  "date": "2026-04-03",
  "ticker": "0700.HK",
  "action": "持有",
  "signal": 1,
  "confidence_pct": 0.72,
  "confidence_label": "高",
  "last_close": 395.20,
  "shares": 200,
  "avg_cost": 380.00,
  "unrealized_pnl": 3040.00,
  "unrealized_pnl_pct": 4.00,
  "has_position": true,
  "stop_loss_price": 381.50,
  "atr_stop_triggered": false,
  "kelly_shares": 220,
  "top_strategy": "macd_rsi_trend",
  "top_strategy_sharpe": 1.85,
  "bullish_count": 8,
  "bearish_count": 3,
  "total_strategies": 11,
  "strategy_votes": {
    "macd_rsi_trend": 1,
    "rsi_reversion": 1,
    "bollinger_rsi_trend": 0,
    "xgboost_enhanced": 1
  },
  "t1_close": null,
  "t1_return_pct": null,
  "t1_correct": null,
  "recorded_at": "2026-04-03T18:05:32"
}
```

### 2.2 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `date` | str | 建议日期（交易日） |
| `ticker` | str | 股票代码 |
| `action` | str | 系统建议（买入/持有/卖出/观望/止损卖出/熔断观望） |
| `signal` | int | 共识信号 1=看涨 0=看跌 |
| `confidence_pct` | float | 共识置信度 [0,1] |
| `last_close` | float | 建议日收盘价 |
| `shares` / `avg_cost` | int/float | 持仓状态快照 |
| `unrealized_pnl_pct` | float | 当日浮动盈亏% |
| `strategy_votes` | dict | 每个参与策略的投票 |
| `t1_close` | float\|null | 次交易日收盘价（T+1 填充） |
| `t1_return_pct` | float\|null | 次日收益率（`t1_close/last_close - 1`） |
| `t1_correct` | bool\|null | 建议方向是否正确（T+1 填充） |

### 2.3 正确性判定规则

```python
def _is_correct(action: str, t1_return_pct: float) -> bool:
    if action in ("买入", "持有"):
        return t1_return_pct > 0
    if action in ("卖出", "止损卖出"):
        return t1_return_pct < 0
    # 观望 / 熔断观望：次日若未大跌（> -2%）视为正确
    if action in ("观望", "熔断观望"):
        return t1_return_pct > -0.02
    return False
```

---

## 三、模块设计

### 3.1 `data/pnl_tracker.py`

```python
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from data.calendar import get_next_trading_day
from log_config import get_logger

logger = get_logger(__name__)

PNL_PATH = Path("data/logs/pnl_history.jsonl")


class PnLTracker:
    """
    每日 PnL 快照追踪与绩效归因。

    典型调用流程（daily_run.py）：
      tracker = PnLTracker()
      tracker.record_daily(date, recommendations)   # 当日运行后追加快照
      tracker.fill_t1_returns(date, price_data)     # 次日运行时补充 T+1 数据
    """

    def __init__(self, pnl_path: Path = PNL_PATH):
        self._path = pnl_path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # ── 写入 ──────────────────────────────────────────────────────

    def record_daily(
        self,
        date: str,
        recommendations: list,      # list[RecommendationResult]
    ) -> int:
        """
        追加当日所有股票的建议快照。
        若同一 date × ticker 记录已存在，跳过（幂等）。
        返回实际写入行数。
        """
        existing = self._existing_keys()
        written = 0
        with open(self._path, "a", encoding="utf-8") as f:
            for rec in recommendations:
                key = f"{date}:{rec.ticker}"
                if key in existing:
                    continue
                row = self._rec_to_row(date, rec)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                written += 1
        if written:
            logger.info("PnL 快照已追加", extra={"date": date, "count": written})
        return written

    def fill_t1_returns(
        self,
        prev_date: str,
        price_map: dict[str, float],    # ticker → 当日收盘价（即 T+1 收盘价）
    ) -> int:
        """
        读取 prev_date 的所有记录，填充 t1_close / t1_return_pct / t1_correct。
        使用原地重写（读全量 + 修改 + 原子写回）。
        返回填充行数。
        """
        rows = self._read_all()
        filled = 0
        for row in rows:
            if row["date"] != prev_date:
                continue
            if row.get("t1_close") is not None:
                continue
            ticker = row["ticker"]
            if ticker not in price_map:
                continue
            t1_close = price_map[ticker]
            t1_ret = t1_close / row["last_close"] - 1
            row["t1_close"] = round(t1_close, 4)
            row["t1_return_pct"] = round(t1_ret * 100, 4)
            row["t1_correct"] = _is_correct(row["action"], t1_ret)
            filled += 1
        if filled:
            self._write_all(rows)
            logger.info("T+1 收益已填充", extra={"date": prev_date, "count": filled})
        return filled

    # ── 查询与归因 ────────────────────────────────────────────────

    def load_df(self) -> pd.DataFrame:
        """加载完整历史为 DataFrame。"""
        rows = self._read_all()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def attribution_by_strategy(self, period_days: int = 30) -> pd.DataFrame:
        """
        按策略归因：统计每个策略的信号准确率（仅含 t1_correct 已填充的记录）。

        返回 DataFrame：strategy_name | correct_count | total_count | accuracy_pct | avg_t1_return
        """
        df = self.load_df()
        if df.empty or "strategy_votes" not in df.columns:
            return pd.DataFrame()

        cutoff = pd.Timestamp.now() - pd.Timedelta(days=period_days)
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] >= cutoff].copy()
        df = df[df["t1_correct"].notna()]

        rows = []
        for _, record in df.iterrows():
            votes = record.get("strategy_votes") or {}
            t1_ret = record["t1_return_pct"]
            correct = record["t1_correct"]
            for strat, vote in votes.items():
                # 策略投票方向与 T+1 实际涨跌一致则算该策略正确
                strat_correct = (vote == 1 and t1_ret > 0) or (vote == 0 and t1_ret < 0)
                rows.append({"strategy": strat, "correct": strat_correct, "t1_return": t1_ret})

        if not rows:
            return pd.DataFrame()
        adf = pd.DataFrame(rows)
        result = (
            adf.groupby("strategy")
            .agg(
                correct_count=("correct", "sum"),
                total_count=("correct", "count"),
                avg_t1_return=("t1_return", "mean"),
            )
            .reset_index()
        )
        result["accuracy_pct"] = result["correct_count"] / result["total_count"] * 100
        return result.sort_values("accuracy_pct", ascending=False)

    def attribution_by_ticker(self, period_days: int = 30) -> pd.DataFrame:
        """
        按标的归因：每只股票的建议准确率与平均 T+1 收益。
        """
        df = self.load_df()
        if df.empty:
            return pd.DataFrame()
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=period_days)
        df["date"] = pd.to_datetime(df["date"])
        df = df[df["date"] >= cutoff]
        df = df[df["t1_correct"].notna()]
        result = (
            df.groupby("ticker")
            .agg(
                correct_count=("t1_correct", "sum"),
                total_count=("t1_correct", "count"),
                avg_t1_return_pct=("t1_return_pct", "mean"),
                avg_confidence=("confidence_pct", "mean"),
            )
            .reset_index()
        )
        result["accuracy_pct"] = result["correct_count"] / result["total_count"] * 100
        return result.sort_values("avg_t1_return_pct", ascending=False)

    def attribution_by_action(self, period_days: int = 30) -> pd.DataFrame:
        """
        按操作类型归因：各类建议（买入/持有/卖出等）的准确率。
        """
        df = self.load_df()
        if df.empty:
            return pd.DataFrame()
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=period_days)
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= cutoff) & df["t1_correct"].notna()]
        result = (
            df.groupby("action")
            .agg(
                correct_count=("t1_correct", "sum"),
                total_count=("t1_correct", "count"),
                avg_t1_return_pct=("t1_return_pct", "mean"),
            )
            .reset_index()
        )
        result["accuracy_pct"] = result["correct_count"] / result["total_count"] * 100
        return result

    def summary_report(self, period_days: int = 30) -> dict:
        """
        生成摘要报告字典（供飞书通知 / markdown 报告使用）。
        """
        df = self.load_df()
        if df.empty:
            return {"error": "no data"}
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=period_days)
        df["date"] = pd.to_datetime(df["date"])
        df = df[(df["date"] >= cutoff) & df["t1_correct"].notna()]
        total = len(df)
        if total == 0:
            return {"total": 0}
        correct = df["t1_correct"].sum()
        return {
            "period_days": period_days,
            "total_recommendations": total,
            "correct_count": int(correct),
            "overall_accuracy_pct": round(correct / total * 100, 1),
            "avg_t1_return_pct": round(df["t1_return_pct"].mean(), 3),
            "best_ticker": df.groupby("ticker")["t1_return_pct"].mean().idxmax(),
            "worst_ticker": df.groupby("ticker")["t1_return_pct"].mean().idxmin(),
        }

    def export_csv(self, output_path: Optional[Path] = None) -> Path:
        """导出完整历史为 CSV。"""
        df = self.load_df()
        if output_path is None:
            output_path = Path("data/logs/pnl_history.csv")
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info("PnL 历史已导出", extra={"path": str(output_path)})
        return output_path

    # ── 内部工具 ──────────────────────────────────────────────────

    def _existing_keys(self) -> set[str]:
        keys = set()
        if not self._path.exists():
            return keys
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    keys.add(f"{d['date']}:{d['ticker']}")
                except (json.JSONDecodeError, KeyError):
                    pass
        return keys

    def _read_all(self) -> list[dict]:
        rows = []
        if not self._path.exists():
            return rows
        with open(self._path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        return rows

    def _write_all(self, rows: list[dict]) -> None:
        """原子写回（临时文件 → 替换）"""
        tmp = self._path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp.replace(self._path)

    @staticmethod
    def _rec_to_row(date: str, rec) -> dict:
        """RecommendationResult → JSON 行字典"""
        agg = rec.agg_signal
        votes = {}
        if agg and agg.raw_votes:
            votes = {v["strategy"]: v["signal"] for v in agg.raw_votes if "strategy" in v}

        return {
            "date": date,
            "ticker": rec.ticker,
            "action": rec.action,
            "signal": rec.signal,
            "confidence_pct": round(rec.confidence_pct, 4),
            "confidence_label": rec.confidence_label,
            "last_close": rec.last_close,
            "shares": rec.shares,
            "avg_cost": rec.avg_cost,
            "unrealized_pnl": round(rec.profit_amount, 2) if hasattr(rec, "profit_amount") else None,
            "unrealized_pnl_pct": round(rec.profit_pct, 4) if rec.has_position else None,
            "has_position": rec.has_position,
            "stop_loss_price": getattr(rec, "stop_loss_price", None),
            "atr_stop_triggered": getattr(rec, "atr_stop_triggered", False),
            "kelly_shares": rec.kelly_shares,
            "top_strategy": agg.top_strategy if agg else "",
            "top_strategy_sharpe": round(agg.top_strategy_sharpe, 4) if agg else None,
            "bullish_count": agg.bullish_count if agg else 0,
            "bearish_count": agg.bearish_count if agg else 0,
            "total_strategies": agg.total_strategies if agg else 0,
            "strategy_votes": votes,
            "t1_close": None,
            "t1_return_pct": None,
            "t1_correct": None,
            "recorded_at": datetime.now().isoformat(timespec="seconds"),
        }


def _is_correct(action: str, t1_return: float) -> bool:
    """判断建议方向是否与次日实际涨跌一致。"""
    if action in ("买入", "持有"):
        return t1_return > 0.0
    if action in ("卖出", "止损卖出"):
        return t1_return < 0.0
    if action in ("观望", "熔断观望"):
        return t1_return > -0.02   # 次日未大跌则"正确回避"
    return False
```

---

## 四、集成点

### 4.1 `daily_run.py` — 主流程集成

```python
# daily_run.py

from data.pnl_tracker import PnLTracker
from data.calendar import get_prev_trading_day

def main():
    today = datetime.now().strftime("%Y-%m-%d")
    tracker = PnLTracker()

    # ── Step 1：T+1 补充（上个交易日的快照 + 今日收盘价）──────────
    prev_date = get_prev_trading_day(today)
    if prev_date:
        # price_map: {ticker: last_close}，在数据加载后获得
        tracker.fill_t1_returns(prev_date, price_map)

    # ── 核心分析流程（不变）──────────────────────────────────────
    recommendations = _run_analysis(...)

    # ── Step 2：追加当日快照 ────────────────────────────────────
    tracker.record_daily(today, recommendations)

    # ── Step 3：每周五/月底输出摘要报告 ─────────────────────────
    if _should_generate_report():
        report = tracker.summary_report(period_days=30)
        logger.info("月度绩效摘要", extra=report)
        # 可选：发送飞书通知
```

### 4.2 T+1 价格来源

T+1 `price_map` 来自 `daily_run.py` 数据加载阶段（已下载当日 OHLCV），以 ticker 的 `last_close` 填充：

```python
price_map = {ticker: df["Close"].iloc[-1] for ticker, df in ticker_data_map.items()}
```

---

## 五、文件结构与影响范围

```
新建：
  data/pnl_tracker.py             # PnLTracker 类（≈ 230 行）
  data/logs/pnl_history.jsonl     # 自动生成，不提交 git

修改：
  daily_run.py                    # 添加 record_daily + fill_t1_returns 调用
  data/__init__.py                # 导出 PnLTracker

新建测试：
  tests/test_pnl_tracker.py       # ≈ 15 个单元测试，全离线
```

**`.gitignore` 更新**：
```
data/logs/pnl_history.jsonl
data/logs/pnl_history.csv
```

---

## 六、测试计划

| 测试 | 说明 |
|------|------|
| `test_record_daily_idempotent` | 同 date×ticker 重复 record 不写第二行 |
| `test_fill_t1_returns` | 给定 price_map 正确填充 t1_close/pct/correct |
| `test_is_correct_buy` | 买入信号 + 次日上涨 → correct=True |
| `test_is_correct_sell` | 卖出信号 + 次日下跌 → correct=True |
| `test_is_correct_hold_when_down` | 观望 + 次日跌 2.5% → correct=False |
| `test_attribution_by_strategy` | 含 2 条记录时 accuracy_pct 计算正确 |
| `test_attribution_by_ticker` | 多 ticker 聚合结果正确 |
| `test_attribution_by_action` | 按 action 分组计数正确 |
| `test_summary_report_empty` | 无数据时返回 {"error": "no data"} |
| `test_export_csv` | CSV 文件生成，列名包含 date/ticker/action |
| `test_atomic_write` | _write_all 使用 .tmp 替换，不会损坏原文件 |
| `test_load_df_columns` | DataFrame 包含所有必填列 |

---

## 七、验收标准

| 指标 | 目标 |
|------|------|
| `pnl_history.jsonl` | 每次 `daily_run.py` 运行后自动追加 |
| T+1 数据回填 | 次日运行后前一日记录 `t1_correct` 非 null |
| 归因报告 | `summary_report(30)` 返回准确率与最佳/最差标的 |
| `pytest tests/test_pnl_tracker.py` | 全通过（全离线） |
| 回归测试 | `pytest tests/` 全通过，`smoke_test.py` 通过 |
| 飞书集成（可选） | 每周一附带 7 日绩效摘要 |

---

## 八、实施步骤

```
Step 1  实现 data/pnl_tracker.py（PnLTracker + _is_correct）         [1.5d]
Step 2  单元测试 tests/test_pnl_tracker.py                            [1d]
Step 3  集成 daily_run.py（record_daily + fill_t1_returns）           [1d]
Step 4  确认 RecommendationResult 中 profit_amount 字段存在
        （若不存在，从 shares × (last_close - avg_cost) 计算）        [0.5d]
Step 5  更新 .gitignore / data/__init__.py                            [0.5d]
Step 6  更新 upgrade_plan.md 4.6 节状态为 ✅                           [0.5d]
```

---

## 九、未来扩展（Phase 3+）

| 扩展 | 说明 |
|------|------|
| Streamlit Dashboard | PnL 曲线图、策略归因柱状图 |
| 月度飞书报告 | 月末自动发送完整绩效归因报告 |
| 策略权重自适应 | 归因准确率差的策略在 SignalAggregator 中降权 |
| 已实现收益 vs 建议收益 | 若未来对接 OMS，追踪真实成交 vs 理论建议的差距 |
