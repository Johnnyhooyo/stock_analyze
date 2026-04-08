"""
tests/test_portfolio_risk.py — PortfolioRiskChecker 单元测试

全程离线，不依赖网络或磁盘因子。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import pytest

from engine.portfolio_risk import PortfolioRiskChecker, PortfolioRiskResult


# ── Stub RecommendationResult ────────────────────────────────────

@dataclass
class _Rec:
    """最小 RecommendationResult stub（只含 risk checker 需要的字段）。"""
    ticker: str
    has_position: bool = False
    shares: int = 0
    avg_cost: float = 0.0
    market_value: float = 0.0
    last_close: float = 0.0
    action: str = "观望"
    signal: int = 0
    reason: str = ""
    risk_flags: list = field(default_factory=list)


# ── Fixtures ─────────────────────────────────────────────────────

def _make_config(
    max_position_ratio: float = 0.80,
    max_sector_ratio:   float = 0.40,
    max_correlation:    float = 0.80,
    loss_threshold:     float = -0.10,
    var_threshold:      float = -0.05,
    sectors: dict       = None,
) -> dict:
    return {
        "portfolio_risk": {
            "max_position_ratio": max_position_ratio,
            "max_sector_ratio":   max_sector_ratio,
            "max_correlation":    max_correlation,
            "correlation_window": 20,
            "loss_threshold":     loss_threshold,
            "var_threshold":      var_threshold,
        },
        "screener": {"sectors": sectors or {}},
    }


def _make_ohlcv(returns: list[float]) -> pd.DataFrame:
    """从收益率序列合成 OHLCV DataFrame。"""
    dates = pd.bdate_range("2024-01-01", periods=len(returns) + 1)
    closes = np.cumprod([100.0] + [1 + r for r in returns])
    return pd.DataFrame({
        "Open":  closes[:-1],
        "High":  closes[1:] * 1.002,
        "Low":   closes[1:] * 0.998,
        "Close": closes[1:],
        "Volume": np.ones(len(returns)) * 1_000_000,
    }, index=dates[1:])


# ══════════════════════════════════════════════════════════════════

class TestPositionRatio:
    def test_no_breach_below_limit(self):
        results = [_Rec("0700.HK", has_position=True, market_value=70_000)]
        cfg = _make_config(max_position_ratio=0.80)
        risk = PortfolioRiskChecker(cfg).check(results, portfolio_value=100_000)
        assert abs(risk.position_ratio - 0.70) < 1e-9
        assert not risk.position_breach

    def test_breach_above_limit(self):
        results = [_Rec("0700.HK", has_position=True, market_value=85_000)]
        cfg = _make_config(max_position_ratio=0.80)
        risk = PortfolioRiskChecker(cfg).check(results, portfolio_value=100_000)
        assert risk.position_breach
        assert any("总仓位" in f for f in risk.flags)

    def test_flat_portfolio_no_breach(self):
        results = [_Rec("0700.HK", has_position=False, market_value=0)]
        risk = PortfolioRiskChecker(_make_config()).check(results, portfolio_value=100_000)
        assert risk.position_ratio == 0.0
        assert not risk.position_breach


class TestSectorConcentration:
    def test_sector_breach(self):
        results = [
            _Rec("0700.HK", has_position=True, market_value=50_000),
            _Rec("0005.HK", has_position=True, market_value=10_000),
        ]
        cfg = _make_config(
            max_sector_ratio=0.40,
            sectors={"科技": ["0700.HK"], "金融": ["0005.HK"]},
        )
        risk = PortfolioRiskChecker(cfg).check(results, portfolio_value=100_000)
        assert risk.sector_breaches.get("科技") is True
        assert risk.sector_breaches.get("金融") is False
        assert any("科技" in f for f in risk.flags)

    def test_no_sector_breach(self):
        results = [
            _Rec("0700.HK", has_position=True, market_value=30_000),
            _Rec("0005.HK", has_position=True, market_value=30_000),
        ]
        cfg = _make_config(
            max_sector_ratio=0.40,
            sectors={"科技": ["0700.HK"], "金融": ["0005.HK"]},
        )
        risk = PortfolioRiskChecker(cfg).check(results, portfolio_value=100_000)
        assert not any(risk.sector_breaches.values())

    def test_unknown_sector_not_flagged(self):
        """未配置板块的股票归入"未知"，不触发板块告警（只检查已知板块）。"""
        results = [_Rec("9999.HK", has_position=True, market_value=90_000)]
        cfg = _make_config(max_sector_ratio=0.40, sectors={})
        risk = PortfolioRiskChecker(cfg).check(results, portfolio_value=100_000)
        assert not any(f for f in risk.flags if "板块" in f)


class TestPortfolioStoploss:
    def test_deleverage_triggered(self):
        # 成本 100_000，市值 88_000 → 亏损 12% > 阈值 10%
        results = [_Rec("0700.HK", has_position=True,
                        shares=100, avg_cost=1000.0, market_value=88_000)]
        cfg = _make_config(loss_threshold=-0.10)
        risk = PortfolioRiskChecker(cfg).check(results, portfolio_value=200_000)
        assert risk.should_deleverage
        assert any("减仓" in f for f in risk.flags)

    def test_deleverage_not_triggered(self):
        results = [_Rec("0700.HK", has_position=True,
                        shares=100, avg_cost=1000.0, market_value=95_000)]
        cfg = _make_config(loss_threshold=-0.10)
        risk = PortfolioRiskChecker(cfg).check(results, portfolio_value=200_000)
        assert not risk.should_deleverage

    def test_pnl_pct_accurate(self):
        # 成本 50_000，市值 55_000 → +10%
        results = [_Rec("0700.HK", has_position=True,
                        shares=100, avg_cost=500.0, market_value=55_000)]
        risk = PortfolioRiskChecker(_make_config()).check(results, portfolio_value=100_000)
        assert abs(risk.total_pnl_pct - 0.10) < 1e-9


class TestCorrelation:
    def _make_corr_results(self):
        return [
            _Rec("A", has_position=True, market_value=50_000),
            _Rec("B", has_position=True, market_value=50_000),
        ]

    def test_high_corr_detected(self):
        rets = np.random.RandomState(42).normal(0.001, 0.02, 30)
        price_data = {
            "A": _make_ohlcv(list(rets)),
            "B": _make_ohlcv(list(rets * 0.98 + 0.0001)),  # 近完全正相关
        }
        cfg = _make_config(max_correlation=0.80)
        risk = PortfolioRiskChecker(cfg).check(
            self._make_corr_results(), 100_000, price_data=price_data
        )
        assert len(risk.high_corr_pairs) == 1
        assert risk.high_corr_pairs[0][2] > 0.80

    def test_low_corr_not_flagged(self):
        rng = np.random.RandomState(0)
        price_data = {
            "A": _make_ohlcv(list(rng.normal(0.001, 0.02, 30))),
            "B": _make_ohlcv(list(rng.normal(-0.001, 0.02, 30))),
        }
        cfg = _make_config(max_correlation=0.80)
        risk = PortfolioRiskChecker(cfg).check(
            self._make_corr_results(), 100_000, price_data=price_data
        )
        assert len(risk.high_corr_pairs) == 0

    def test_corr_skipped_without_price_data(self):
        risk = PortfolioRiskChecker(_make_config()).check(
            self._make_corr_results(), 100_000, price_data=None
        )
        assert risk.high_corr_pairs == []


class TestVaR:
    def test_var_breach(self):
        # 每天 -3% 收益率 → VaR 远超阈值
        bad_rets = [-0.03] * 60
        price_data = {"0700.HK": _make_ohlcv(bad_rets)}
        results = [_Rec("0700.HK", has_position=True,
                        shares=100, avg_cost=100.0, market_value=100_000)]
        cfg = _make_config(var_threshold=-0.02)
        risk = PortfolioRiskChecker(cfg).check(results, 100_000, price_data=price_data)
        assert risk.var_95 < -0.02
        assert risk.var_breach

    def test_var_no_breach(self):
        # 稳健正收益
        good_rets = [0.001] * 60
        price_data = {"0700.HK": _make_ohlcv(good_rets)}
        results = [_Rec("0700.HK", has_position=True,
                        shares=100, avg_cost=100.0, market_value=100_000)]
        cfg = _make_config(var_threshold=-0.05)
        risk = PortfolioRiskChecker(cfg).check(results, 100_000, price_data=price_data)
        assert not risk.var_breach

    def test_var_skipped_without_price_data(self):
        results = [_Rec("0700.HK", has_position=True, market_value=100_000)]
        risk = PortfolioRiskChecker(_make_config()).check(results, 100_000)
        assert risk.var_95 == 0.0
        assert not risk.var_breach


class TestEdgeCases:
    def test_empty_results(self):
        risk = PortfolioRiskChecker(_make_config()).check([], 100_000)
        assert not risk.has_warnings
        assert risk.position_ratio == 0.0

    def test_zero_portfolio_value(self):
        results = [_Rec("0700.HK", has_position=True, market_value=50_000)]
        risk = PortfolioRiskChecker(_make_config()).check(results, 0.0)
        assert not risk.has_warnings

    def test_to_dict_serializable(self):
        results = [_Rec("0700.HK", has_position=True,
                        shares=100, avg_cost=1000.0, market_value=85_000)]
        risk = PortfolioRiskChecker(_make_config(loss_threshold=-0.10)).check(
            results, 100_000
        )
        d = risk.to_dict()
        assert isinstance(d, dict)
        assert "flags" in d
        assert "should_deleverage" in d
