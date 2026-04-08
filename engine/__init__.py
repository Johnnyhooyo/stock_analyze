"""
engine/ — 每日推荐引擎核心包

子模块：
  portfolio_state   — 多持仓状态加载 / 保存（data/portfolio.yaml）
  signal_aggregator — 多策略共识信号聚合
  position_analyzer — 单只股票推荐生成（信号 + 风控 + 建议）
  stock_screener    — 选股引擎（Phase 2 新增）
"""

from .portfolio_state import PortfolioState, PortfolioPosition, load_portfolio
from .signal_aggregator import SignalAggregator, AggregatedSignal
from .position_analyzer import PositionAnalyzer, RecommendationResult
from .stock_screener import StockScreener, ScreenerResult
from .screener_factors import ScreenerFactors, FactorResult
from .meta_aggregator import MetaAggregator

__all__ = [
    "PortfolioState",
    "PortfolioPosition",
    "load_portfolio",
    "SignalAggregator",
    "AggregatedSignal",
    "PositionAnalyzer",
    "RecommendationResult",
    "StockScreener",
    "ScreenerResult",
    "ScreenerFactors",
    "FactorResult",
    "MetaAggregator",
]

