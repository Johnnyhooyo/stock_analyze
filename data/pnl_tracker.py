"""
data/pnl_tracker.py — 每日 PnL 快照追踪与绩效归因

负责：
  1. 每日追加建议快照到 data/logs/pnl_history.jsonl
  2. T+1 次日收益回填
  3. 按策略 / 标的 / 操作类型归因分析
  4. CSV 导出

典型调用流程（daily_run.py）：
  tracker = PnLTracker()
  tracker.record_daily(date, recommendations)   # 当日运行后追加快照
  tracker.fill_t1_returns(prev_date, price_map)  # 次日运行时补充 T+1 数据
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from log_config import get_logger

logger = get_logger(__name__)

PNL_PATH = Path("data/logs/pnl_history.jsonl")


def _is_correct(action: str, t1_return: float) -> bool:
    """判断建议方向是否与次日实际涨跌一致。"""
    if action in ("买入", "持有"):
        return t1_return > 0.0
    if action in ("卖出", "止损卖出"):
        return t1_return < 0.0
    if action in ("观望", "熔断观望"):
        return t1_return > -0.02
    return False


class PnLTracker:
    """
    每日 PnL 快照追踪与绩效归因。
    """

    def __init__(self, pnl_path: Path = PNL_PATH):
        self._path = pnl_path
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def record_daily(
        self,
        date: str,
        recommendations: list,
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
        price_map: dict[str, float],
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
            if not row.get("last_close"):
                logger.warning("last_close 为 0，跳过 T+1 回填", extra={"ticker": ticker, "date": prev_date})
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

    def load_df(self) -> pd.DataFrame:
        """加载完整历史为 DataFrame。"""
        rows = self._read_all()
        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def attribution_by_strategy(self, period_days: int = 30) -> pd.DataFrame:
        """
        按策略归因：统计每个策略的信号准确率（仅含 t1_correct 已填充的记录）。

        返回 DataFrame：strategy | correct_count | total_count | accuracy_pct | avg_t1_return
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
            for strat, vote in votes.items():
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
            return {"total_recommendations": 0}
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
            votes = {v["strategy_name"]: v["signal"] for v in agg.raw_votes}

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
            "unrealized_pnl": round(rec.profit, 2) if rec.has_position else None,
            "unrealized_pnl_pct": round(rec.profit_pct, 4) if rec.has_position else None,
            "has_position": rec.has_position,
            "stop_loss_price": rec.stop_price if rec.stop_price > 0 else None,
            "atr_stop_triggered": False,
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