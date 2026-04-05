"""
engine/position_analyzer.py — 单只股票推荐生成

整合：
  - 多策略共识信号（SignalAggregator）
  - ATR 止损 / Kelly 仓位 / 熔断（PositionManager.apply_risk_controls）
  - 情感分析（可选）
  - 历史统计价格区间

输出 RecommendationResult 数据类，供 daily_run.py 汇总并发送飞书通知。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pandas as pd

from log_config import get_logger
from position_manager import PositionManager, calc_atr
from engine.portfolio_state import PortfolioPosition
from engine.signal_aggregator import SignalAggregator, AggregatedSignal

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────
#  推荐结果数据类
# ──────────────────────────────────────────────────────────────────

@dataclass
class RecommendationResult:
    """单只股票的每日操作建议"""

    # 基本信息
    ticker: str
    last_date: str           # 最后交易日 "YYYY-MM-DD"
    last_close: float        # 最新收盘价

    # 操作建议
    action: str = "观望"     # 买入 / 卖出 / 持有 / 观望 / 止损卖出 / 熔断观望
    reason: str = ""         # 操作理由（简洁说明）
    signal: int = 0          # 最终信号 1=看涨, 0=看跌（经风控修正后）

    # 风控
    stop_price: float = 0.0       # 建议止损价
    kelly_shares: int = 0         # Kelly 建议仓位（股数）
    kelly_amount: float = 0.0     # Kelly 建议仓位（金额）
    circuit_breaker: bool = False  # 熔断是否触发
    consecutive_loss_days: int = 0

    # 共识信号详情
    agg_signal: Optional[AggregatedSignal] = None
    confidence_pct: float = 0.0
    confidence_label: str = "—"

    # 持仓盈亏
    has_position: bool = False
    shares: int = 0
    avg_cost: float = 0.0
    market_value: float = 0.0
    profit: float = 0.0
    profit_pct: float = 0.0
    peak_price: float = 0.0

    # 情感分析（可选）
    sentiment: Optional[dict] = None

    # 价格参考区间（历史统计 P10~P90）
    price_lo_1d: float = 0.0
    price_hi_1d: float = 0.0

    # ATR
    atr: float = 0.0

    # 风险标志
    risk_flags: list[str] = field(default_factory=list)

    @property
    def action_emoji(self) -> str:
        emoji_map = {
            "买入": "🟢",
            "持有": "🟡",
            "观望": "⚪",
            "卖出": "🔴",
            "止损卖出": "🔴",
            "熔断观望": "⚠️",
        }
        return emoji_map.get(self.action, "⚪")

    @property
    def summary_line(self) -> str:
        """单行摘要，用于终端输出。"""
        pos_str = f"{self.shares}股@{self.avg_cost:.2f}" if self.has_position else "空仓"
        pnl_str = f"({self.profit_pct:+.1f}%)" if self.has_position else ""
        stop_str = f"止损={self.stop_price:.2f}" if self.stop_price > 0 else ""
        return (
            f"{self.action_emoji} {self.ticker:<12s} "
            f"{self.action:<6s}  收盘={self.last_close:.2f}  "
            f"{pos_str}{pnl_str}  "
            f"{stop_str}  "
            f"置信={self.confidence_label}({self.confidence_pct:.0%})"
        )


# ──────────────────────────────────────────────────────────────────
#  PositionAnalyzer
# ──────────────────────────────────────────────────────────────────

class PositionAnalyzer:
    """
    单只股票推荐生成器。

    用法：
        analyzer = PositionAnalyzer(config, factors_dir)
        result = analyzer.analyze("0700.HK", hist_df, portfolio_pos)
    """

    def __init__(
        self,
        config: dict,
        factors_dir: Optional[Path] = None,
        enable_sentiment: bool = True,
        max_factors: int = 20,
    ):
        """
        Args:
            config:            全局 config.yaml 内容
            factors_dir:       factor_*.pkl 目录（默认 data/factors/）
            enable_sentiment:  是否运行情感分析（耗时，可在 daily_run 中配置）
            max_factors:       最多参与投票的因子数量
        """
        self.config = config
        self.factors_dir = factors_dir or (
            Path(__file__).parent.parent / "data" / "factors"
        )
        self.enable_sentiment = enable_sentiment
        self.risk_cfg = config.get("risk_management", {})
        self.portfolio_value = float(
            self.risk_cfg.get("portfolio_value", 200_000.0)
        )
        self._aggregator = SignalAggregator(
            factors_dir=self.factors_dir,
            max_factors=max_factors,
        )

    # ── 历史统计价格区间 ──────────────────────────────────────────

    @staticmethod
    def _price_range(
        returns: pd.Series,
        last_close: float,
        horizon_days: int = 1,
        lower_pct: float = 0.10,
        upper_pct: float = 0.90,
    ) -> tuple[float, float]:
        """
        基于历史滚动窗口累计收益率分布（P10~P90）计算参考价格区间。
        非价格预测，仅作统计参考。
        """
        rolling = returns.rolling(horizon_days).sum().dropna()
        if len(rolling) >= 20:
            p_lo = float(rolling.quantile(lower_pct))
            p_hi = float(rolling.quantile(upper_pct))
        else:
            mu = float(returns.mean() * horizon_days)
            sig = float(returns.std() * (horizon_days ** 0.5))
            _Z = {0.10: -1.282, 0.90: 1.282}
            p_lo = mu + _Z[lower_pct] * sig if sig > 0 else 0.0
            p_hi = mu + _Z[upper_pct] * sig if sig > 0 else 0.0
        return last_close * (1 + p_lo), last_close * (1 + p_hi)

    # ── 主分析入口 ────────────────────────────────────────────────

    def analyze(
        self,
        ticker: str,
        data: pd.DataFrame,
        portfolio_pos: Optional[PortfolioPosition] = None,
    ) -> RecommendationResult:
        """
        对单只股票进行完整分析，返回 RecommendationResult。

        Args:
            ticker:        股票代码（如 "0700.HK"）
            data:          历史 OHLCV DataFrame（已对齐 Close/High/Low/Volume）
            portfolio_pos: 该股票的当前持仓（None = 空仓）

        Returns:
            RecommendationResult
        """
        ticker = ticker.upper()
        df = data.copy().sort_index()
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            df.index = pd.to_datetime(df.index)

        last_date = str(df.index.max().date())
        last_close = float(df["Close"].iloc[-1])

        # ── 当前持仓状态 ─────────────────────────────────────────
        has_position = portfolio_pos is not None and portfolio_pos.has_position
        shares = portfolio_pos.shares if has_position else 0
        avg_cost = portfolio_pos.avg_cost if has_position else 0.0
        peak_price = portfolio_pos.peak_price if has_position else 0.0

        market_value = shares * last_close if has_position else 0.0
        cost_basis = shares * avg_cost if has_position else 0.0
        profit = market_value - cost_basis
        profit_pct = (profit / cost_basis * 100) if cost_basis > 0 else 0.0

        # ── 风控参数 ─────────────────────────────────────────────
        atr_period = int(self.risk_cfg.get("atr_period", 14))
        atr_val = calc_atr(df, period=atr_period)

        entry_price = avg_cost if has_position else last_close
        if peak_price <= 0:
            peak_price = max(entry_price, last_close)

        # 当日盈亏率（昨收 vs 今收）
        if len(df) >= 2:
            prev_close = float(df["Close"].iloc[-2])
            today_pnl_pct = (last_close - prev_close) / prev_close if prev_close > 0 else 0.0
        else:
            today_pnl_pct = 0.0

        # ── 多策略共识信号 ────────────────────────────────────────
        ticker_config = {**self.config, "ticker": ticker}
        agg = self._aggregator.aggregate(ticker, df, ticker_config)

        # ── 风控层（PositionManager）─────────────────────────────
        pm_position = portfolio_pos.to_position_manager_position(last_close) if has_position else None
        pm = PositionManager(
            position=pm_position,
            portfolio_value=self.portfolio_value,
            risk_config=self.risk_cfg,
            ticker=ticker,
        )
        if has_position and portfolio_pos.trailing_peak is not None:
            pm._trailing._peak = portfolio_pos.trailing_peak

        rec = pm.apply_risk_controls(
            signal=agg.consensus_signal,
            price=last_close,
            atr=atr_val,
            entry_price=entry_price,
            peak_price=peak_price,
            today_pnl_pct=today_pnl_pct,
            capital=self.portfolio_value,
            win_rate=0.0,   # 共识模式不提供单一胜率
            profit_loss_ratio=0.0,
            trade_date=last_date,
        )

        # 风险标志收集
        risk_flags: list[str] = []
        if rec.get("circuit_breaker"):
            risk_flags.append(f"熔断触发（连续亏损 {rec.get('consecutive_loss_days', 0)} 天）")
        if rec.get("action") in ("止损卖出",):
            risk_flags.append(f"ATR 止损触发（止损位 {rec.get('stop_price', 0):.2f}）")
        if agg.confidence_pct < 0.55:
            risk_flags.append("信号置信度偏低（策略分歧大）")

        # ── 情感分析（可选）──────────────────────────────────────
        sentiment_result = None
        if self.enable_sentiment:
            try:
                from sentiment_analysis import analyze_stock_sentiment
                sentiment_result = analyze_stock_sentiment(ticker)
            except Exception as e:
                logger.warning("%s 情感分析失败: %s", ticker, e, extra={"ticker": ticker})

        # ── 历史统计价格区间 ──────────────────────────────────────
        returns = df["Close"].pct_change().dropna()
        price_lo, price_hi = self._price_range(returns, last_close, horizon_days=1)

        # ── 置信度标签 ────────────────────────────────────────────
        conf_pct = agg.confidence_pct
        if conf_pct >= 0.75:
            conf_label = "高"
        elif conf_pct >= 0.55:
            conf_label = "中"
        else:
            conf_label = "低"

        return RecommendationResult(
            ticker=ticker,
            last_date=last_date,
            last_close=last_close,
            action=rec.get("action", "观望"),
            reason=rec.get("reason", ""),
            signal=rec.get("signal", 0),
            stop_price=rec.get("stop_price", 0.0),
            kelly_shares=rec.get("kelly_shares", 0),
            kelly_amount=rec.get("kelly_amount", 0.0),
            circuit_breaker=rec.get("circuit_breaker", False),
            consecutive_loss_days=rec.get("consecutive_loss_days", 0),
            agg_signal=agg,
            confidence_pct=conf_pct,
            confidence_label=conf_label,
            has_position=has_position,
            shares=shares,
            avg_cost=avg_cost,
            market_value=market_value,
            profit=profit,
            profit_pct=profit_pct,
            peak_price=peak_price,
            sentiment=sentiment_result,
            price_lo_1d=price_lo,
            price_hi_1d=price_hi,
            atr=atr_val,
            risk_flags=risk_flags,
        )

