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

    def to_dict(self) -> dict:
        return {
            "ticker": self.ticker,
            "composite_score": round(self.composite_score, 1),
            "rank": self.rank,
            "momentum_score": round(self.momentum_score, 1),
            "trend_score": round(self.trend_score, 1),
            "volume_score": round(self.volume_score, 1),
            "valuation_score": round(self.valuation_score, 1),
            "sentiment_score": round(self.sentiment_score, 1),
            "signals": self.signals,
            "sector": self.sector,
            "last_close": self.last_close,
            "change_pct_5d": round(self.change_pct_5d, 2),
            "change_pct_20d": round(self.change_pct_20d, 2),
            "avg_volume_ratio": round(self.avg_volume_ratio, 2),
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
            "valuation": scr_cfg.get("weight_valuation", 0.00),
            "sentiment": scr_cfg.get("weight_sentiment", 0.00),
        }
        self.enable_valuation = scr_cfg.get("enable_valuation", False)
        self.enable_sentiment = scr_cfg.get("enable_sentiment", False)
        self.top_n_count = scr_cfg.get("top_n", 10)
        self.min_score = scr_cfg.get("min_score", 50.0)
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

        for ticker in tickers:
            df = data_dict.get(ticker)
            if df is None or len(df) < 60:
                logger.debug(f"[Screener] {ticker}: 数据不足，跳过")
                continue

            try:
                result = self._evaluate_ticker(ticker, df)
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

    def _evaluate_ticker(self, ticker: str, df: pd.DataFrame) -> Optional[ScreenerResult]:
        factor_result: FactorResult = self._factors.calc_all(
            df,
            enable_valuation=self.enable_valuation,
            enable_sentiment=self.enable_sentiment,
            ticker=ticker,
        )

        momentum = factor_result.momentum_score
        trend = factor_result.trend_score
        volume = factor_result.volume_score
        valuation = factor_result.valuation_score
        sentiment = factor_result.sentiment_score

        composite = (
            momentum * self.weights["momentum"]
            + trend * self.weights["trend"]
            + volume * self.weights["volume"]
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
