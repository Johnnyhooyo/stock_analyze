"""
tests/conftest.py — pytest fixtures for stock_analyze test suite
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ── Synthetic OHLCV ──────────────────────────────────────────────────────

@pytest.fixture
def synthetic_ohlcv() -> pd.DataFrame:
    """300-row synthetic OHLCV DataFrame (seed=42)."""
    np.random.seed(42)
    n = 300
    dates = pd.bdate_range(end=datetime.today(), periods=n)
    close = np.cumprod(1 + np.random.normal(0, 0.01, n)) * 100
    return pd.DataFrame(
        {
            "Open":   close * (1 + np.random.normal(0, 0.002, n)),
            "High":   close * (1 + np.abs(np.random.normal(0, 0.005, n))),
            "Low":    close * (1 - np.abs(np.random.normal(0, 0.005, n))),
            "Close":  close,
            "Volume": np.random.randint(500_000, 5_000_000, n).astype(float),
        },
        index=dates,
    )


@pytest.fixture
def atr_plunge_ohlcv() -> pd.DataFrame:
    """OHLCV that first rises then plunges — designed to trigger ATR stop-loss."""
    n = 120
    dates = pd.bdate_range(end=datetime.today(), periods=n)
    prices_up   = np.linspace(100, 130, 60)
    prices_down = np.linspace(129, 85, 60)
    closes  = np.concatenate([prices_up, prices_down])
    highs   = closes * 1.003
    lows    = closes * 0.997
    return pd.DataFrame(
        {
            "Open":   closes * 1.001,
            "High":   highs,
            "Low":    lows,
            "Close":  closes,
            "Volume": np.ones(n) * 1_000_000,
        },
        index=dates,
    )


# ── Config fixtures ──────────────────────────────────────────────────────

@pytest.fixture
def default_config() -> dict:
    """Default config dict matching config.yaml structure."""
    return {
        "initial_capital": 100_000.0,
        "invest_fraction": 0.95,
        "lookback_months": 3,
        "slippage": 0.001,
        "fees_rate": 0.00088,
        "stamp_duty": 0.001,
        "risk_management": {
            "use_atr_stop": True,
            "atr_period": 14,
            "atr_multiplier": 2.0,
            "trailing_stop": True,
            "simulate_in_backtest": True,
            "cooldown_bars": 0,
            "use_kelly": False,
            "kelly_fraction": 0.5,
            "max_position_pct": 0.25,
            "portfolio_value": 200_000.0,
            "daily_loss_limit": 0.05,
            "max_consecutive_loss_days": 3,
        },
    }


@pytest.fixture
def minimal_config() -> dict:
    """Empty dict — tests fallback to hard-coded defaults."""
    return {}
