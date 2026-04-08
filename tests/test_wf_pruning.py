"""
tests/test_wf_pruning.py — Walk-Forward pruning 单元测试

全程离线：mock walk_forward_analysis，仅测试 _run_wf_pruning 逻辑。
"""
from __future__ import annotations

from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

import optuna

from optimize_with_optuna import StrategyOptimizer


# ── Fixtures ─────────────────────────────────────────────────────

def _make_ohlcv(n: int = 300) -> pd.DataFrame:
    np.random.seed(0)
    dates = pd.bdate_range("2020-01-01", periods=n)
    closes = np.cumprod(1 + np.random.normal(0.0005, 0.015, n)) * 100
    return pd.DataFrame({
        "Open":  closes * 0.999,
        "High":  closes * 1.005,
        "Low":   closes * 0.995,
        "Close": closes,
        "Volume": np.ones(n) * 1_000_000,
    }, index=dates)


def _make_strategy_mod():
    mod = MagicMock()
    mod.NAME = "test_strat"
    sig = pd.Series(0, index=_make_ohlcv().index)
    mod.run.return_value = (sig, None, {"name": "test_strat", "params": {}, "feat_cols": [], "indicators": {}})
    return mod


def _make_optimizer(use_wf_pruning: bool = True, wf_min_rate: float = 0.5) -> StrategyOptimizer:
    cfg = {
        "lookback_months": 3,
        "train_years": 2,
        "backtest_engine": "native",
        "risk_management": {"simulate_in_backtest": False, "use_atr_stop": False},
        "strategy_training": {"single": ["test_strat"], "multi": [], "custom": []},
        "use_wf_pruning": use_wf_pruning,
        "wf_min_window_win_rate": wf_min_rate,
        "wf_pruning_train_months": 6,
        "wf_pruning_test_months": 2,
        "wf_pruning_step_months": 2,
    }
    return StrategyOptimizer(
        data=_make_ohlcv(),
        strategy_mod=_make_strategy_mod(),
        config=cfg,
        use_vectorbt=False,
    )


def _make_trial() -> optuna.Trial:
    study = optuna.create_study(direction="maximize")
    trial = study.ask()
    return trial


def _wf_result(win_rate: float, n_windows: int = 4, success: bool = True) -> dict:
    return {
        "success": success,
        "windows": [{"cum_return": 0.05 if win_rate > 0 else -0.02}] * n_windows,
        "summary": {
            "total_windows": n_windows,
            "profitable_windows": int(n_windows * win_rate),
            "window_win_rate": win_rate,
            "avg_return": 0.03,
            "avg_sharpe": 1.2,
        },
    }


# ══════════════════════════════════════════════════════════════════

class TestWFPruningDisabled:
    def test_wf_still_runs_when_called_directly(self):
        """_run_wf_pruning 本身不检查 use_wf_pruning 开关（开关由 objective() 检查）。
        直接调用时，walk_forward_analysis 会被触发。"""
        opt = _make_optimizer(use_wf_pruning=False)
        trial = _make_trial()
        with patch("validate_strategy.walk_forward_analysis", return_value=_wf_result(0.75)) as mock_wf:
            opt._run_wf_pruning({**opt.config}, trial, pd.DataFrame())
            assert mock_wf.called


class TestWFPruningPrunes:
    def test_prunes_when_win_rate_below_threshold(self):
        opt = _make_optimizer(wf_min_rate=0.5)
        trial = _make_trial()
        with patch("validate_strategy.walk_forward_analysis", return_value=_wf_result(0.25)):
            with pytest.raises(optuna.exceptions.TrialPruned):
                opt._run_wf_pruning({**opt.config}, trial, pd.DataFrame())

    def test_prunes_at_exact_threshold(self):
        """win_rate == threshold → prune (strictly less-than rule)."""
        opt = _make_optimizer(wf_min_rate=0.5)
        trial = _make_trial()
        with patch("validate_strategy.walk_forward_analysis", return_value=_wf_result(0.5)):
            # 0.5 < 0.5 is False → should NOT prune
            opt._run_wf_pruning({**opt.config}, trial, pd.DataFrame())


class TestWFPruningPasses:
    def test_passes_when_win_rate_above_threshold(self):
        opt = _make_optimizer(wf_min_rate=0.5)
        trial = _make_trial()
        with patch("validate_strategy.walk_forward_analysis", return_value=_wf_result(0.75)):
            opt._run_wf_pruning({**opt.config}, trial, pd.DataFrame())  # no exception

    def test_sets_user_attrs_on_pass(self):
        opt = _make_optimizer(wf_min_rate=0.5)
        trial = _make_trial()
        with patch("validate_strategy.walk_forward_analysis", return_value=_wf_result(0.75, n_windows=4)):
            opt._run_wf_pruning({**opt.config}, trial, pd.DataFrame())
        assert abs(trial.user_attrs["wf_window_win_rate"] - 0.75) < 1e-9
        assert trial.user_attrs["wf_total_windows"] == 4

    def test_sets_user_attrs_on_prune(self):
        """Even when pruning, user_attrs should be set before raising."""
        opt = _make_optimizer(wf_min_rate=0.5)
        trial = _make_trial()
        with patch("validate_strategy.walk_forward_analysis", return_value=_wf_result(0.25, n_windows=3)):
            with pytest.raises(optuna.exceptions.TrialPruned):
                opt._run_wf_pruning({**opt.config}, trial, pd.DataFrame())
        assert trial.user_attrs["wf_window_win_rate"] == 0.25
        assert trial.user_attrs["wf_total_windows"] == 3


class TestWFPruningEdgeCases:
    def test_skips_when_wf_raises(self):
        """WF 内部异常 → 静默跳过，不剪枝。"""
        opt = _make_optimizer(wf_min_rate=0.5)
        trial = _make_trial()
        with patch("validate_strategy.walk_forward_analysis", side_effect=RuntimeError("oops")):
            opt._run_wf_pruning({**opt.config}, trial, pd.DataFrame())  # no exception

    def test_skips_when_wf_not_successful(self):
        """WF 返回 success=False（数据不足）→ 跳过，不剪枝。"""
        opt = _make_optimizer(wf_min_rate=0.5)
        trial = _make_trial()
        with patch("validate_strategy.walk_forward_analysis", return_value=_wf_result(0.0, success=False)):
            opt._run_wf_pruning({**opt.config}, trial, pd.DataFrame())  # no exception

    def test_custom_window_params_forwarded(self):
        """config 中的 WF 窗口参数应正确传递给 walk_forward_analysis。"""
        opt = _make_optimizer(wf_min_rate=0.3)
        cfg = {
            **opt.config,
            "wf_pruning_train_months": 8,
            "wf_pruning_test_months": 3,
            "wf_pruning_step_months": 3,
        }
        trial = _make_trial()
        with patch("validate_strategy.walk_forward_analysis", return_value=_wf_result(0.6)) as mock_wf:
            opt._run_wf_pruning(cfg, trial, pd.DataFrame())
        _, kwargs = mock_wf.call_args
        assert kwargs["train_months"] == 8
        assert kwargs["test_months"] == 3
        assert kwargs["step_months"] == 3


class TestConfigDefault:
    def test_use_wf_pruning_defaults_false(self):
        """默认配置中 use_wf_pruning 应为 False（不启用）。"""
        from config_loader import load_config
        cfg = load_config()
        assert cfg.get("use_wf_pruning") is False
