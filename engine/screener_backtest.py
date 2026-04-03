"""
选股因子回测验证框架 — 评估选股评分的历史盈利能力

功能：
  - 对历史数据进行滚动选股回测
  - 每次调仓日，用选股引擎对所有标的评分，取 Top-N 做多
  - 计算命中率、平均收益、夏普比率、最大回撤等指标

Usage:
    from engine.screener_backtest import ScreenerBacktester
    bt = ScreenerBacktester(config)
    result = bt.backtest(data_dict)  # data_dict: {ticker: OHLCV DataFrame}
    print(f"命中率: {result.hit_rate:.1%}")
    print(f"平均收益: {result.avg_return_pct:.2f}%")
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from log_config import get_logger
from .stock_screener import StockScreener, ScreenerResult

logger = get_logger(__name__)


@dataclass
class ScreenerBacktestResult:
    hit_rate: float
    avg_return_pct: float
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    n_picks: int
    n_periods: int
    equity_curve: list[float]

    def summary(self) -> str:
        return (
            f"选股回测结果:\n"
            f"  命中率       : {self.hit_rate:.1%}\n"
            f"  平均收益     : {self.avg_return_pct:.2f}%\n"
            f"  累计收益     : {self.total_return_pct:.2f}%\n"
            f"  夏普比率     : {self.sharpe_ratio:.2f}\n"
            f"  最大回撤     : {self.max_drawdown_pct:.2f}%\n"
            f"  总选股次数   : {self.n_picks}\n"
            f"  调仓次数     : {self.n_periods}"
        )


class ScreenerBacktester:

    def __init__(self, config: dict):
        scr_cfg = config.get("screener", {})
        self.top_n = scr_cfg.get("top_n", 10)
        self.rebalance_days = scr_cfg.get("rebalance_days", 20)
        self.min_hold_days = scr_cfg.get("min_hold_days", 5)
        self.min_data_days = 60
        self.min_picks_per_period = 1
        self._screener = StockScreener(config)

    def backtest(
        self,
        data_dict: dict[str, pd.DataFrame],
    ) -> ScreenerBacktestResult:
        all_dates = self._common_dates(data_dict)
        if len(all_dates) < self.min_data_days:
            logger.warning(
                f"[ScreenerBT] 数据不足 ({len(all_dates)} 天)，跳过回测"
            )
            return ScreenerBacktestResult(
                hit_rate=0.0,
                avg_return_pct=0.0,
                total_return_pct=0.0,
                sharpe_ratio=0.0,
                max_drawdown_pct=0.0,
                n_picks=0,
                n_periods=0,
                equity_curve=[],
            )

        period_returns: list[float] = []
        equity_curve: list[float] = [1.0]
        n_picks_total = 0

        for rebal_date in self._rebalance_dates(all_dates):
            picks = self._run_screener_at_date(data_dict, all_dates, rebal_date)
            if not picks:
                continue

            n_picks_total += len(picks)
            ret = self._measure_picks_return(data_dict, picks, rebal_date)
            period_returns.append(ret)

            equity_curve.append(equity_curve[-1] * (1 + ret / 100))

        if not period_returns:
            return ScreenerBacktestResult(
                hit_rate=0.0,
                avg_return_pct=0.0,
                total_return_pct=0.0,
                sharpe_ratio=0.0,
                max_drawdown_pct=0.0,
                n_picks=0,
                n_periods=0,
                equity_curve=equity_curve[1:],
            )

        period_returns_arr = np.array(period_returns)
        hits = period_returns_arr > 0
        hit_rate = float(np.mean(hits)) if len(hits) > 0 else 0.0
        avg_return = float(np.mean(period_returns_arr))
        total_return = float(np.prod(1 + period_returns_arr / 100) - 1) * 100
        sharpe = self._sharpe_ratio(period_returns_arr)

        equity_arr = np.array(equity_curve)
        peak = np.maximum.accumulate(equity_arr)
        drawdown = (equity_arr - peak) / peak * 100
        max_drawdown = float(np.min(drawdown)) if len(drawdown) > 0 else 0.0

        return ScreenerBacktestResult(
            hit_rate=hit_rate,
            avg_return_pct=avg_return,
            total_return_pct=total_return,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_drawdown,
            n_picks=n_picks_total,
            n_periods=len(period_returns),
            equity_curve=equity_curve[1:],
        )

    def _common_dates(self, data_dict: dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
        date_sets = []
        for df in data_dict.values():
            if df is not None and len(df) > 0 and isinstance(df.index, pd.DatetimeIndex):
                date_sets.append(set(df.index))
        if not date_sets:
            return pd.DatetimeIndex([])
        common = date_sets[0]
        for s in date_sets[1:]:
            common = common.intersection(s)
        return pd.DatetimeIndex(sorted(common))

    def _rebalance_dates(self, dates: pd.DatetimeIndex) -> list[pd.Timestamp]:
        if len(dates) < self.min_data_days:
            return []
        step = self.rebalance_days
        return list(dates[::step])

    def _run_screener_at_date(
        self,
        data_dict: dict[str, pd.DataFrame],
        all_dates: pd.DatetimeIndex,
        rebal_date: pd.Timestamp,
    ) -> list[ScreenerResult]:
        idx = all_dates.get_indexer([rebal_date], method="pad")[0]
        lookback_end = max(0, idx - 250)
        sliced: dict[str, pd.DataFrame] = {}
        for ticker, df in data_dict.items():
            if len(df) <= lookback_end:
                continue
            sliced[ticker] = df.iloc[lookback_end : idx + 1].copy()
        if len(sliced) < 3:
            return []
        tickers = list(sliced.keys())
        results = self._screener.screen(tickers, sliced)
        return self._screener.top_n(results, n=self.top_n)

    def _measure_picks_return(
        self,
        data_dict: dict[str, pd.DataFrame],
        picks: list[ScreenerResult],
        rebal_date: pd.Timestamp,
    ) -> float:
        all_dates = self._common_dates(data_dict)
        idx = all_dates.get_indexer([rebal_date], method="pad")[0]
        start_idx = min(idx + 1, len(all_dates) - 1)
        end_idx = min(start_idx + self.rebalance_days, len(all_dates) - 1)
        if start_idx >= end_idx:
            return 0.0

        returns = []
        for pick in picks:
            df = data_dict.get(pick.ticker)
            if df is None or len(df) <= end_idx:
                continue
            entry_price = float(df.iloc[start_idx]["Close"])
            exit_price = float(df.iloc[end_idx]["Close"])
            if entry_price <= 0:
                continue
            ret = (exit_price - entry_price) / entry_price * 100
            returns.append(ret)

        if not returns:
            return 0.0
        return float(np.mean(returns))

    def _sharpe_ratio(self, returns: np.ndarray, risk_free: float = 0.0) -> float:
        if len(returns) < 2 or np.std(returns) == 0:
            return 0.0
        excess = returns - risk_free
        return float(np.mean(excess) / np.std(excess) * np.sqrt(252 / self.rebalance_days))
