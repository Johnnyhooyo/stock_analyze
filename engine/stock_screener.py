"""
选股引擎 — 全市场量化选股，输出高潜力标的候选池

设计理念：
  - 轻量级：每只股票 < 0.5 秒，支持全量港股扫描
  - 多维度：动量 + 趋势 + 量价（Step 1 范围）
  - 可配置：各维度权重通过 config.yaml 调整
  - 与交易策略解耦：选股 = "值得关注"，交易信号 = "值得操作"

Usage:
    from engine.stock_screener import StockScreener, ScreenerResult
    from engine.portfolio_state import PortfolioState

    screener = StockScreener(config)
    results = screener.screen(tickers, data_dict)
    top_picks = screener.top_n(results, n=10, exclude_held=True,
                                portfolio_state=portfolio_state)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from log_config import get_logger
from .screener_factors import ScreenerFactors, FactorResult

if TYPE_CHECKING:
    from .portfolio_state import PortfolioState

logger = get_logger(__name__)


@dataclass
class ScreenerResult:
    ticker: str
    composite_score: float
    rank: int = 0
    momentum_score: float = 0.0
    trend_score: float = 0.0
    volume_score: float = 0.0
    valuation_score: float = 0.0
    sentiment_score: float = 50.0
    signals: list[str] = field(default_factory=list)
    sector: str = "未知"
    last_close: float = 0.0
    change_pct_5d: float = 0.0
    change_pct_20d: float = 0.0
    avg_volume_ratio: float = 1.0
    rsi_14: float = 50.0
    macd_hist: float = 0.0
    obv_slope: float = 0.0
    liquidity_score: float = 0.0
    median_turnover: float = 0.0
    trading_days: int = 0

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "composite_score": round(self.composite_score, 1),
            "rank": self.rank,
            "momentum_score": round(self.momentum_score, 1),
            "trend_score": round(self.trend_score, 1),
            "volume_score": round(self.volume_score, 1),
            "liquidity_score": round(self.liquidity_score, 1),
            "valuation_score": round(self.valuation_score, 1),
            "sentiment_score": round(self.sentiment_score, 1),
            "signals": self.signals,
            "sector": self.sector,
            "last_close": self.last_close,
            "change_pct_5d": round(self.change_pct_5d, 2),
            "change_pct_20d": round(self.change_pct_20d, 2),
            "avg_volume_ratio": round(self.avg_volume_ratio, 2),
            "median_turnover": round(self.median_turnover, 2),
            "trading_days": self.trading_days,
        }


class StockScreener:
    """
    选股引擎

    Args:
        config: 配置字典，从 config_loader.load_config() 获得
    """

    def __init__(self, config: dict):
        self.config = config
        scr_cfg = config.get("screener", {})
        self.weights = {
            "momentum": scr_cfg.get("weight_momentum", 0.35),
            "trend": scr_cfg.get("weight_trend", 0.35),
            "volume": scr_cfg.get("weight_volume", 0.30),
            "liquidity": scr_cfg.get("weight_liquidity", 0.00),
            "valuation": scr_cfg.get("weight_valuation", 0.00),
            "sentiment": scr_cfg.get("weight_sentiment", 0.00),
        }
        self.enable_valuation = scr_cfg.get("enable_valuation", False)
        self.enable_sentiment = scr_cfg.get("enable_sentiment", False)
        self.top_n_count = scr_cfg.get("top_n", 10)
        self.min_score = scr_cfg.get("min_score", 50.0)
        self.min_price = max(0.0, float(scr_cfg.get("min_price", 0.0)))
        self.liquidity_window = max(1, int(scr_cfg.get("liquidity_window", 20)))
        # min_avg_turnover is retained as a compatibility fallback for old configs.
        self.min_median_turnover = max(
            0.0,
            float(scr_cfg.get(
                "min_median_turnover", scr_cfg.get("min_avg_turnover", 0.0)
            )),
        )
        self.min_trading_days = max(0, int(scr_cfg.get("min_trading_days", 0)))
        self.stability_window = max(
            self.liquidity_window, int(scr_cfg.get("stability_window", 60))
        )
        self.max_consecutive_no_trade_days = max(0, int(
            scr_cfg.get("max_consecutive_no_trade_days", 0)
        ))
        self.max_abs_return_1d = max(
            0.0, float(scr_cfg.get("max_abs_return_1d", 0.0))
        )
        self.max_data_lag_days = max(0, int(scr_cfg.get("max_data_lag_days", 0)))
        self.universe = scr_cfg.get("universe", "hk")
        self.sectors = scr_cfg.get("sectors", {})
        self._factors = ScreenerFactors()

        active_weights = {
            k: v
            for k, v in self.weights.items()
            if (k != "valuation" or self.enable_valuation)
            and (k != "sentiment" or self.enable_sentiment)
        }
        total = sum(active_weights.values())
        assert abs(total - 1.0) < 0.01, (
            f"screener weights must sum to 1.0, got {total:.4f}"
        )

    def screen(
        self,
        tickers: list[str],
        data_dict: dict[str, pd.DataFrame],
    ) -> list[ScreenerResult]:
        """
        对给定股票池执行多因子选股评分。

        Args:
            tickers: 待筛选股票列表
            data_dict: {ticker: OHLCV DataFrame} 历史数据字典

        Returns:
            按 composite_score 降序排列的 ScreenerResult 列表
        """
        results: list[ScreenerResult] = []
        latest_dates = [
            self._as_naive_timestamp(df.index[-1])
            for df in data_dict.values()
            if df is not None and len(df) and isinstance(df.index, pd.DatetimeIndex)
        ]
        valid_latest_dates = [d for d in latest_dates if d is not None]
        if valid_latest_dates:
            date_counts = pd.Series(valid_latest_dates).value_counts()
            most_common = date_counts[date_counts == date_counts.max()].index
            market_date = max(most_common)
        else:
            market_date = None

        for ticker in tickers:
            df = data_dict.get(ticker)
            if df is None or len(df) < 60:
                logger.debug(f"[Screener] {ticker}: 数据不足，跳过")
                continue

            metrics = self._market_metrics(df)
            passed, reason = self._passes_market_filters(metrics, market_date)
            if not passed:
                logger.debug(f"[Screener] {ticker}: {reason}，跳过")
                continue

            try:
                result = self._evaluate_ticker(ticker, df, metrics)
                if result is not None and result.composite_score >= self.min_score:
                    results.append(result)
            except Exception as e:
                logger.warning(f"[Screener] {ticker}: 评估异常 {e}")

        results.sort(key=lambda x: x.composite_score, reverse=True)

        for i, r in enumerate(results, 1):
            r.rank = i

        logger.info(
            f"[Screener] 选股完成: {len(tickers)} 只扫描, "
            f"{len(results)} 只通过评分门槛({self.min_score})"
        )
        return results

    @staticmethod
    def _as_naive_timestamp(value) -> Optional[pd.Timestamp]:
        try:
            ts = pd.Timestamp(value)
            return ts.tz_localize(None) if ts.tzinfo is not None else ts
        except (TypeError, ValueError):
            return None

    def _market_metrics(self, df: pd.DataFrame) -> dict:
        """计算流动性指标；零成交日计入窗口，防止偶发爆量掩盖风险。"""
        if "Close" not in df.columns or "Volume" not in df.columns:
            return {}

        history = df[["Close", "Volume"]].tail(
            max(self.liquidity_window, self.stability_window)
        ).apply(
            pd.to_numeric, errors="coerce"
        )
        recent = history.tail(self.liquidity_window)
        if recent.empty:
            return {}

        valid = (
            np.isfinite(recent["Close"])
            & np.isfinite(recent["Volume"])
            & (recent["Close"] > 0)
            & (recent["Volume"] > 0)
        )
        turnover = (recent["Close"] * recent["Volume"]).where(valid, 0.0)
        positive_turnover = turnover[turnover > 0]
        median_turnover = float(turnover.median())
        q25_turnover = float(turnover.quantile(0.25))
        trading_days = int(valid.sum())

        history_valid = (
            np.isfinite(history["Close"])
            & np.isfinite(history["Volume"])
            & (history["Close"] > 0)
            & (history["Volume"] > 0)
        )
        max_no_trade_run = 0
        current_run = 0
        for is_valid in history_valid.tolist():
            current_run = 0 if is_valid else current_run + 1
            max_no_trade_run = max(max_no_trade_run, current_run)

        last_close = float(recent["Close"].iloc[-1])
        prev_close = float(recent["Close"].iloc[-2]) if len(recent) >= 2 else np.nan
        return_1d = (
            last_close / prev_close - 1.0
            if np.isfinite(last_close) and np.isfinite(prev_close) and prev_close > 0
            else np.nan
        )

        # 100万到1亿港元映射为0到100分，并奖励持续有成交、成交额稳定。
        turnover_score = float(np.clip(
            (np.log10(max(median_turnover, 1_000_000)) - 6) / 2 * 100,
            0,
            100,
        ))
        activity_score = trading_days / max(len(recent), 1) * 100
        consistency_score = (
            float(np.clip(q25_turnover / median_turnover * 100, 0, 100))
            if median_turnover > 0 else 0.0
        )
        liquidity_score = (
            turnover_score * 0.50
            + activity_score * 0.30
            + consistency_score * 0.20
        )

        return {
            "last_date": self._as_naive_timestamp(recent.index[-1]),
            "last_close": last_close,
            "median_turnover": median_turnover,
            "avg_turnover": float(positive_turnover.mean()) if len(positive_turnover) else 0.0,
            "trading_days": trading_days,
            "max_no_trade_run": max_no_trade_run,
            "return_1d": float(return_1d),
            "liquidity_score": float(np.clip(liquidity_score, 0, 100)),
        }

    def _passes_market_filters(
        self, metrics: dict, market_date: Optional[pd.Timestamp] = None
    ) -> tuple[bool, str]:
        """在因子评分前检查价格、成交连续性和异常单日波动。"""
        if not metrics:
            return False, "缺少 Close/Volume 列或无近期行情"

        last_date = metrics.get("last_date")
        if (
            self.max_data_lag_days > 0
            and market_date is not None
            and last_date is not None
        ):
            lag_days = (market_date.normalize() - last_date.normalize()).days
            if lag_days > self.max_data_lag_days:
                return False, f"行情落后市场最新日期 {lag_days} 天"

        last_close = metrics["last_close"]
        if not np.isfinite(last_close) or last_close <= 0:
            return False, "最新收盘价无效"
        if last_close < self.min_price:
            return False, f"最新收盘价 {last_close:.3f} 低于 {self.min_price:.3f}"

        trading_days = int(metrics["trading_days"])
        if trading_days < self.min_trading_days:
            return False, f"近{self.liquidity_window}日有效成交仅 {trading_days} 天"

        median_turnover = metrics["median_turnover"]
        if not np.isfinite(median_turnover) or median_turnover < self.min_median_turnover:
            return False, (
                f"近{self.liquidity_window}日成交额中位数 {median_turnover:.0f} "
                f"低于 {self.min_median_turnover:.0f}"
            )

        max_no_trade_run = int(metrics["max_no_trade_run"])
        if (
            self.max_consecutive_no_trade_days > 0
            and max_no_trade_run > self.max_consecutive_no_trade_days
        ):
            return False, f"近期最长连续无成交 {max_no_trade_run} 天"

        return_1d = metrics["return_1d"]
        if (
            self.max_abs_return_1d > 0
            and np.isfinite(return_1d)
            and abs(return_1d) > self.max_abs_return_1d
        ):
            return False, f"最新单日涨跌 {return_1d:+.1%} 超过异常波动阈值"

        return True, ""

    def _evaluate_ticker(
        self, ticker: str, df: pd.DataFrame, metrics: Optional[dict] = None
    ) -> Optional[ScreenerResult]:
        metrics = metrics or self._market_metrics(df)
        factor_result: FactorResult = self._factors.calc_all(
            df,
            enable_valuation=self.enable_valuation,
            enable_sentiment=self.enable_sentiment,
            ticker=ticker,
        )

        momentum = factor_result.momentum_score
        trend = factor_result.trend_score
        volume = factor_result.volume_score
        liquidity = metrics.get("liquidity_score", 0.0)
        valuation = factor_result.valuation_score
        sentiment = factor_result.sentiment_score

        composite = (
            momentum * self.weights["momentum"]
            + trend * self.weights["trend"]
            + volume * self.weights["volume"]
            + liquidity * self.weights["liquidity"]
            + valuation * self.weights["valuation"]
            + sentiment * self.weights["sentiment"]
        )
        composite = max(0.0, min(100.0, composite))

        last_close = float(df["Close"].iloc[-1])

        sector = self._get_sector(ticker)

        return ScreenerResult(
            ticker=ticker,
            composite_score=composite,
            momentum_score=momentum,
            trend_score=trend,
            volume_score=volume,
            liquidity_score=liquidity,
            valuation_score=valuation,
            sentiment_score=sentiment,
            signals=factor_result.signals,
            sector=sector,
            last_close=last_close,
            change_pct_5d=factor_result.change_pct_5d,
            change_pct_20d=factor_result.change_pct_20d,
            avg_volume_ratio=factor_result.avg_volume_ratio,
            rsi_14=factor_result.rsi_14,
            macd_hist=factor_result.macd_hist,
            obv_slope=factor_result.obv_slope,
            median_turnover=metrics.get("median_turnover", 0.0),
            trading_days=int(metrics.get("trading_days", 0)),
        )

    def _get_sector(self, ticker: str) -> str:
        for sector, members in self.sectors.items():
            if ticker in members:
                return sector
        return "其他"

    def top_n(
        self,
        results: list[ScreenerResult],
        n: Optional[int] = None,
        exclude_held: bool = True,
        portfolio_state: Optional["PortfolioState"] = None,
    ) -> list[ScreenerResult]:
        """
        返回 Top-N 候选，可排除已持仓标的。

        Args:
            results: screen() 返回的结果列表
            n: 返回数量，默认 top_n_count
            exclude_held: 是否排除已持仓标的
            portfolio_state: PortfolioState 实例，用于判断已持仓
        """
        if n is None:
            n = self.top_n_count

        candidates = list(results)

        if exclude_held and portfolio_state is not None:
            held = set(portfolio_state.held_tickers())
            candidates = [r for r in candidates if r.ticker not in held]

        return candidates[:n]

    def sector_ranking(
        self, results: list[ScreenerResult]
    ) -> list[dict]:
        """
        按板块聚合评分，返回板块强弱排序。

        Returns:
            [{"sector": str, "avg_score": float, "count": int,
              "top_stock": str, "top_score": float}, ...]
        """
        from collections import defaultdict

        sector_data: dict[str, list[ScreenerResult]] = defaultdict(list)
        for r in results:
            sector_data[r.sector].append(r)

        rankings = []
        for sector, items in sector_data.items():
            avg = np.mean([r.composite_score for r in items])
            top = max(items, key=lambda r: r.composite_score)
            rankings.append({
                "sector": sector,
                "avg_score": round(float(avg), 1),
                "count": len(items),
                "top_stock": top.ticker,
                "top_score": round(top.composite_score, 1),
            })

        rankings.sort(key=lambda x: x["avg_score"], reverse=True)
        return rankings
