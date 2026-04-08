"""
engine/portfolio_risk.py — 组合级风控检查器

每日在个股分析完成后，基于所有持仓做组合层面的风控检查：

    1. 总仓位上限     : 持仓市值 / 总资产 > max_position_ratio (默认 0.80)  → 买入前阻断
    2. 行业集中度限制 : 同板块持仓 / 总资产 > max_sector_ratio (默认 0.40)   → 买入前阻断
    3. 持仓相关性监控 : 20 日滚动相关系数 > max_correlation (默认 0.80)       → 警告
    4. 组合级止损     : 总亏损 < loss_threshold (默认 -0.10)                  → 触发去杠杆
    5. VaR 预警       : 95% 历史 VaR < var_threshold (默认 -0.05)             → 警告

日常用法 (daily_run.py)::

    from engine.portfolio_risk import PortfolioRiskChecker
    checker = PortfolioRiskChecker(config)
    risk = checker.check(results, portfolio_state.portfolio_value)
    if risk.position_breach or risk.should_deleverage:
        for r in results:
            if r.action == "买入":
                r.action, r.signal = "观望", 0
                r.reason = "[组合风控] " + (risk.flags[0] if risk.flags else "仓位超限")
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from log_config import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════
#  结果数据类
# ══════════════════════════════════════════════════════════════════

@dataclass
class PortfolioRiskResult:
    """组合风控检查结果汇总。"""

    # ── 1. 总仓位 ────────────────────────────────────────────────
    position_ratio: float = 0.0          # 当前持仓市值 / 总资产
    position_breach: bool = False        # 是否超过上限

    # ── 2. 行业集中度 ────────────────────────────────────────────
    sector_concentration: dict[str, float] = field(default_factory=dict)  # sector → ratio
    sector_breaches: dict[str, bool]       = field(default_factory=dict)  # sector → bool

    # ── 3. 相关性 ────────────────────────────────────────────────
    high_corr_pairs: list[tuple[str, str, float]] = field(default_factory=list)

    # ── 4. 组合止损 ──────────────────────────────────────────────
    total_pnl_pct: float = 0.0           # 全仓位加权亏损（负值表示亏损）
    should_deleverage: bool = False      # True → 触发全面减仓

    # ── 5. VaR ───────────────────────────────────────────────────
    var_95: float = 0.0                  # 95% VaR（负值，e.g. -0.03 = 单日最大损失 3%）
    var_breach: bool = False             # True → 超过阈值

    # ── 综合标志 ─────────────────────────────────────────────────
    flags: list[str] = field(default_factory=list)

    @property
    def has_warnings(self) -> bool:
        return bool(self.flags)

    def to_dict(self) -> dict:
        return {
            "position_ratio":       round(self.position_ratio, 4),
            "position_breach":      self.position_breach,
            "sector_concentration": {k: round(v, 4) for k, v in self.sector_concentration.items()},
            "sector_breaches":      self.sector_breaches,
            "high_corr_pairs":      self.high_corr_pairs,
            "total_pnl_pct":        round(self.total_pnl_pct, 4),
            "should_deleverage":    self.should_deleverage,
            "var_95":               round(self.var_95, 4),
            "var_breach":           self.var_breach,
            "flags":                self.flags,
        }


# ══════════════════════════════════════════════════════════════════
#  检查器
# ══════════════════════════════════════════════════════════════════

class PortfolioRiskChecker:
    """
    组合级风控检查器（无状态，每日 check() 一次）。

    config 结构 (config.yaml → portfolio_risk)::

        portfolio_risk:
          max_position_ratio: 0.80
          max_sector_ratio:   0.40
          max_correlation:    0.80
          correlation_window: 20
          loss_threshold:    -0.10   # 负值
          var_threshold:     -0.05   # 负值
    """

    def __init__(self, config: dict):
        rc = config.get("portfolio_risk", {})
        self._max_position_ratio = float(rc.get("max_position_ratio", 0.80))
        self._max_sector_ratio   = float(rc.get("max_sector_ratio",   0.40))
        self._max_corr           = float(rc.get("max_correlation",    0.80))
        self._corr_window        = int(rc.get("correlation_window",   20))
        self._loss_threshold     = float(rc.get("loss_threshold",    -0.10))
        self._var_threshold      = float(rc.get("var_threshold",     -0.05))
        # 板块映射来自 screener 配置 (sector_name → [ticker, ...])
        self._sectors: dict[str, list[str]] = config.get("screener", {}).get("sectors", {})

    # ── 公开接口 ──────────────────────────────────────────────────

    def check(
        self,
        results: list,
        portfolio_value: float,
        price_data: Optional[dict[str, pd.DataFrame]] = None,
    ) -> PortfolioRiskResult:
        """
        执行全部组合风控检查。

        Args:
            results        : list[RecommendationResult]
            portfolio_value: 总资产（港元）
            price_data     : {ticker: OHLCV DataFrame}，传入时启用相关性 & VaR 检查

        Returns:
            PortfolioRiskResult
        """
        res = PortfolioRiskResult()
        if not results or portfolio_value <= 0:
            return res

        held = [r for r in results if r.has_position and r.market_value > 0]

        self._check_position_ratio(res, held, portfolio_value)
        self._check_sector_concentration(res, held, portfolio_value)
        self._check_portfolio_stoploss(res, held)
        if price_data:
            self._check_correlation(res, held, price_data)
            self._check_var(res, held, price_data, portfolio_value)

        self._build_flags(res)

        if res.flags:
            logger.warning(
                "组合风控告警 [%d 项]",
                len(res.flags),
                extra={"flags": res.flags},
            )
        return res

    # ── 私有检查方法 ──────────────────────────────────────────────

    def _check_position_ratio(
        self, res: PortfolioRiskResult, held: list, portfolio_value: float
    ) -> None:
        total_mv = sum(r.market_value for r in held)
        res.position_ratio = total_mv / portfolio_value
        res.position_breach = res.position_ratio > self._max_position_ratio

    def _check_sector_concentration(
        self, res: PortfolioRiskResult, held: list, portfolio_value: float
    ) -> None:
        # 构建 ticker → sector 映射（未知板块统一为"未知"）
        ticker_sector: dict[str, str] = {}
        for sector, tickers in self._sectors.items():
            for t in tickers:
                ticker_sector[t] = sector

        sector_mv: dict[str, float] = defaultdict(float)
        for r in held:
            s = ticker_sector.get(r.ticker, "未知")
            sector_mv[s] += r.market_value

        res.sector_concentration = {
            s: mv / portfolio_value for s, mv in sector_mv.items()
        }
        res.sector_breaches = {
            s: ratio > self._max_sector_ratio
            for s, ratio in res.sector_concentration.items()
        }

    def _check_correlation(
        self,
        res: PortfolioRiskResult,
        held: list,
        price_data: dict[str, pd.DataFrame],
    ) -> None:
        """计算持仓股票间近 N 日收益率相关系数，高相关对加入告警。"""
        returns: dict[str, pd.Series] = {}
        for r in held:
            df = price_data.get(r.ticker)
            if df is None or df.empty or "Close" not in df.columns:
                continue
            rets = df["Close"].pct_change().dropna()
            tail = rets.tail(self._corr_window)
            if len(tail) >= max(5, self._corr_window // 2):
                returns[r.ticker] = tail

        tickers = list(returns.keys())
        pairs: list[tuple[str, str, float]] = []
        for i in range(len(tickers)):
            for j in range(i + 1, len(tickers)):
                t1, t2 = tickers[i], tickers[j]
                s1, s2 = returns[t1], returns[t2]
                # 对齐索引后计算
                s1, s2 = s1.align(s2, join="inner")
                if len(s1) < 5:
                    continue
                corr = float(s1.corr(s2))
                if not np.isnan(corr) and abs(corr) > self._max_corr:
                    pairs.append((t1, t2, round(corr, 3)))

        res.high_corr_pairs = sorted(pairs, key=lambda x: -abs(x[2]))

    def _check_portfolio_stoploss(
        self, res: PortfolioRiskResult, held: list
    ) -> None:
        """基于持仓成本与当前市值计算组合总盈亏率。"""
        total_cost = sum(r.avg_cost * r.shares for r in held)
        total_mv   = sum(r.market_value for r in held)
        if total_cost > 0:
            res.total_pnl_pct = (total_mv - total_cost) / total_cost
        else:
            res.total_pnl_pct = 0.0
        res.should_deleverage = res.total_pnl_pct < self._loss_threshold

    def _check_var(
        self,
        res: PortfolioRiskResult,
        held: list,
        price_data: dict[str, pd.DataFrame],
        portfolio_value: float,
    ) -> None:
        """
        计算组合 95% 历史 VaR（按持仓市值加权各股收益率后取 5th 百分位）。
        VaR 以负值表示（e.g. -0.03 = 单日最大损失 3%）。
        """
        total_mv = sum(r.market_value for r in held)
        if total_mv <= 0:
            return

        weighted_returns: Optional[pd.Series] = None
        for r in held:
            df = price_data.get(r.ticker)
            if df is None or df.empty or "Close" not in df.columns:
                continue
            rets = df["Close"].pct_change().dropna().tail(252)
            if len(rets) < 20:
                continue
            weight = r.market_value / total_mv
            if weighted_returns is None:
                weighted_returns = rets * weight
            else:
                weighted_returns = weighted_returns.add(rets * weight, fill_value=0.0)

        if weighted_returns is None or len(weighted_returns) < 20:
            return

        res.var_95    = float(np.percentile(weighted_returns.values, 5))
        res.var_breach = res.var_95 < self._var_threshold

    # ── 标志生成 ──────────────────────────────────────────────────

    def _build_flags(self, res: PortfolioRiskResult) -> None:
        flags: list[str] = []

        if res.position_breach:
            flags.append(
                f"总仓位 {res.position_ratio:.1%} 超过上限 {self._max_position_ratio:.0%}"
            )

        for sector, breached in res.sector_breaches.items():
            if breached and sector != "未知":
                ratio = res.sector_concentration.get(sector, 0.0)
                flags.append(
                    f"板块集中度 [{sector}] {ratio:.1%} 超过上限 {self._max_sector_ratio:.0%}"
                )

        if res.should_deleverage:
            flags.append(
                f"组合总亏损 {res.total_pnl_pct:.1%} 触发减仓阈值 {self._loss_threshold:.0%}"
            )

        for t1, t2, corr in res.high_corr_pairs:
            flags.append(f"高相关持仓 {t1}↔{t2} corr={corr:+.2f}")

        if res.var_breach:
            flags.append(
                f"95% VaR={res.var_95:.2%} 超过阈值 {self._var_threshold:.0%}"
            )

        res.flags = flags
