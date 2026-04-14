"""
tests/test_factor_registry.py — 因子注册表单元测试（全离线）
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import pytest

from data.factor_registry import (
    FactorRegistry,
    FactorRecord,
    TTL_DAYS,
    ARCHIVE_AFTER_DAYS,
    _get_training_type,
    _migrate_existing_factors,
)


@pytest.fixture
def tmp_registry(tmp_path):
    """返回以 tmp_path 为 registry_path 的 FactorRegistry（无数据）"""
    reg_path = tmp_path / "factor_registry.json"
    return FactorRegistry(registry_path=reg_path)


@pytest.fixture
def tmp_factors_dir(tmp_path):
    """返回临时 factors 目录（含 archive 子目录）"""
    fd = tmp_path / "factors"
    fd.mkdir()
    (fd / "archive").mkdir()
    return fd


def _make_artifact(
    run_id: int,
    strategy_name: str = "macd_rsi_trend",
    ticker: str = "0700.HK",
    sharpe: float = 1.5,
    saved_at: str = None,
) -> dict:
    """创建符合 _save_factor 输出格式的模拟因子 artifact。"""
    return {
        "model": {},
        "meta": {"name": strategy_name},
        "config": {"ticker": ticker},
        "run_id": run_id,
        "cum_return": 0.20,
        "annualized_return": 0.15,
        "sharpe_ratio": sharpe,
        "max_drawdown": -0.12,
        "volatility": 0.10,
        "win_rate": 0.55,
        "profit_loss_ratio": 1.5,
        "calmar_ratio": 1.0,
        "sortino_ratio": 1.2,
        "total_trades": 18,
        "validated": "double",
        "holdout": {},
        "wf_summary": {},
        "val_period": "",
        "saved_at": saved_at or datetime.now().isoformat(),
    }


class TestFactorRecord:
    def test_active_when_within_ttl(self):
        d = {
            "id": 1,
            "filename": "factor_0001.pkl",
            "subdir": "0700_HK",
            "strategy_name": "macd_rsi_trend",
            "ticker": "0700.HK",
            "training_type": "single",
            "sharpe_ratio": 1.5,
            "cum_return": 0.2,
            "max_drawdown": -0.1,
            "total_trades": 10,
            "created_at": datetime.now().isoformat(),
            "valid_until": (datetime.now() + timedelta(days=30)).isoformat(),
            "status": "active",
            "archived_at": None,
            "notes": "",
        }
        rec = FactorRecord(d)
        assert rec.is_active() is True
        assert rec.status == "active"

    def test_not_active_when_expired(self):
        d = {
            "id": 1,
            "filename": "factor_0001.pkl",
            "subdir": None,
            "strategy_name": "xgboost_enhanced",
            "ticker": None,
            "training_type": "multi",
            "sharpe_ratio": 1.5,
            "cum_return": 0.2,
            "max_drawdown": -0.1,
            "total_trades": 10,
            "created_at": datetime.now().isoformat(),
            "valid_until": (datetime.now() - timedelta(days=1)).isoformat(),
            "status": "active",
            "archived_at": None,
            "notes": "",
        }
        rec = FactorRecord(d)
        assert rec.is_active() is False

    def test_not_active_when_status_expired(self):
        d = {
            "id": 1,
            "filename": "factor_0001.pkl",
            "subdir": None,
            "strategy_name": "macd_rsi_trend",
            "ticker": "0700.HK",
            "training_type": "single",
            "sharpe_ratio": 1.5,
            "cum_return": 0.2,
            "max_drawdown": -0.1,
            "total_trades": 10,
            "created_at": datetime.now().isoformat(),
            "valid_until": (datetime.now() + timedelta(days=30)).isoformat(),
            "status": "expired",
            "archived_at": None,
            "notes": "",
        }
        rec = FactorRecord(d)
        assert rec.is_active() is False
        assert rec.status == "expired"


class TestRegister:
    def test_register_new_factor(self, tmp_registry, tmp_factors_dir):
        rec = tmp_registry.register(
            factor_id=1,
            filename="factor_0001.pkl",
            subdir="0700_HK",
            strategy_name="macd_rsi_trend",
            ticker="0700.HK",
            training_type="single",
            sharpe_ratio=1.5,
            cum_return=0.2,
            max_drawdown=-0.12,
            total_trades=18,
        )
        assert rec.id == 1
        assert rec.status == "active"
        assert rec.training_type == "single"
        assert rec.sharpe_ratio == 1.5

        assert tmp_registry._path.exists()
        data = json.loads(tmp_registry._path.read_text())
        assert len(data["factors"]) == 1
        assert data["factors"][0]["status"] == "active"

    def test_register_multi_strategy_ticker_none(self, tmp_registry):
        rec = tmp_registry.register(
            factor_id=10,
            filename="factor_0010.pkl",
            subdir=None,
            strategy_name="xgboost_enhanced",
            ticker=None,
            training_type="multi",
            sharpe_ratio=2.0,
            cum_return=0.3,
            max_drawdown=-0.09,
            total_trades=22,
        )
        assert rec.ticker is None
        assert rec.training_type == "multi"

    def test_ttl_single_30_days(self, tmp_registry):
        rec = tmp_registry.register(
            factor_id=1,
            filename="factor_0001.pkl",
            subdir=None,
            strategy_name="rsi_divergence",
            ticker="0700.HK",
            training_type="single",
            sharpe_ratio=1.0,
            cum_return=0.1,
            max_drawdown=-0.1,
            total_trades=5,
        )
        diff = (rec.valid_until - datetime.now()).days
        assert 29 <= diff <= 30

    def test_ttl_multi_60_days(self, tmp_registry):
        rec = tmp_registry.register(
            factor_id=1,
            filename="factor_0001.pkl",
            subdir=None,
            strategy_name="xgboost_enhanced",
            ticker=None,
            training_type="multi",
            sharpe_ratio=1.0,
            cum_return=0.1,
            max_drawdown=-0.1,
            total_trades=5,
        )
        diff = (rec.valid_until - datetime.now()).days
        assert 59 <= diff <= 60

    def test_sharpe_degradation_warning(self, tmp_registry):
        tmp_registry.register(
            factor_id=1,
            filename="factor_0001.pkl",
            subdir=None,
            strategy_name="macd_rsi_trend",
            ticker="0700.HK",
            training_type="single",
            sharpe_ratio=2.0,
            cum_return=0.3,
            max_drawdown=-0.1,
            total_trades=10,
        )
        rec_new = tmp_registry.register(
            factor_id=2,
            filename="factor_0002.pkl",
            subdir=None,
            strategy_name="macd_rsi_trend",
            ticker="0700.HK",
            training_type="single",
            sharpe_ratio=1.5,
            cum_return=0.2,
            max_drawdown=-0.1,
            total_trades=10,
        )
        assert "劣化" in rec_new.notes
        assert "1.50" in rec_new.notes
        assert "2.00" in rec_new.notes

    def test_no_warning_when_sharpe_slightly_lower(self, tmp_registry):
        tmp_registry.register(
            factor_id=1,
            filename="factor_0001.pkl",
            subdir=None,
            strategy_name="macd_rsi_trend",
            ticker="0700.HK",
            training_type="single",
            sharpe_ratio=2.0,
            cum_return=0.3,
            max_drawdown=-0.1,
            total_trades=10,
        )
        rec_new = tmp_registry.register(
            factor_id=2,
            filename="factor_0002.pkl",
            subdir=None,
            strategy_name="macd_rsi_trend",
            ticker="0700.HK",
            training_type="single",
            sharpe_ratio=1.9,
            cum_return=0.25,
            max_drawdown=-0.1,
            total_trades=10,
        )
        assert rec_new.notes == ""


class TestActiveRecords:
    def test_active_records_by_ticker(self, tmp_registry):
        tmp_registry.register(
            factor_id=1, filename="factor_0001.pkl", subdir="0700_HK",
            strategy_name="macd_rsi_trend", ticker="0700.HK",
            training_type="single", sharpe_ratio=1.5, cum_return=0.2,
            max_drawdown=-0.1, total_trades=10,
        )
        tmp_registry.register(
            factor_id=2, filename="factor_0002.pkl", subdir="0005_HK",
            strategy_name="bollinger_rsi_trend", ticker="0005.HK",
            training_type="single", sharpe_ratio=1.3, cum_return=0.15,
            max_drawdown=-0.1, total_trades=8,
        )
        active = tmp_registry.active_records(ticker="0700.HK")
        assert len(active) == 1
        assert active[0].ticker == "0700.HK"

    def test_multi_factor_visible_to_all_tickers(self, tmp_registry):
        tmp_registry.register(
            factor_id=1, filename="factor_0001.pkl", subdir=None,
            strategy_name="xgboost_enhanced", ticker=None,
            training_type="multi", sharpe_ratio=2.0, cum_return=0.3,
            max_drawdown=-0.09, total_trades=22,
        )
        active_0700 = tmp_registry.active_records(ticker="0700.HK")
        active_0005 = tmp_registry.active_records(ticker="0005.HK")
        assert len(active_0700) == 1
        assert len(active_0005) == 1

    def test_active_records_excludes_expired(self, tmp_registry):
        d = {
            "version": 1,
            "last_updated": datetime.now().isoformat(),
            "factors": [
                {
                    "id": 1,
                    "filename": "factor_0001.pkl",
                    "subdir": None,
                    "strategy_name": "macd_rsi_trend",
                    "ticker": "0700.HK",
                    "training_type": "single",
                    "sharpe_ratio": 1.5,
                    "cum_return": 0.2,
                    "max_drawdown": -0.1,
                    "total_trades": 10,
                    "created_at": (datetime.now() - timedelta(days=60)).isoformat(),
                    "valid_until": (datetime.now() - timedelta(days=30)).isoformat(),
                    "status": "expired",
                    "archived_at": None,
                    "notes": "",
                }
            ],
        }
        tmp_registry._data = d
        tmp_registry._save()
        active = tmp_registry.active_records(ticker="0700.HK")
        assert len(active) == 0


class TestExpireStale:
    def test_expire_stale_marks_active_expired(self, tmp_registry):
        past = (datetime.now() - timedelta(days=1)).isoformat()
        d = {
            "version": 1,
            "last_updated": datetime.now().isoformat(),
            "factors": [
                {
                    "id": 1,
                    "filename": "factor_0001.pkl",
                    "subdir": None,
                    "strategy_name": "macd_rsi_trend",
                    "ticker": "0700.HK",
                    "training_type": "single",
                    "sharpe_ratio": 1.5,
                    "cum_return": 0.2,
                    "max_drawdown": -0.1,
                    "total_trades": 10,
                    "created_at": past,
                    "valid_until": past,
                    "status": "active",
                    "archived_at": None,
                    "notes": "",
                }
            ],
        }
        tmp_registry._data = d
        tmp_registry._save()
        count = tmp_registry.expire_stale()
        assert count == 1
        assert tmp_registry._data["factors"][0]["status"] == "expired"

    def test_expire_stale_none_expired(self, tmp_registry):
        future = (datetime.now() + timedelta(days=30)).isoformat()
        d = {
            "version": 1,
            "last_updated": datetime.now().isoformat(),
            "factors": [
                {
                    "id": 1,
                    "filename": "factor_0001.pkl",
                    "subdir": None,
                    "strategy_name": "macd_rsi_trend",
                    "ticker": "0700.HK",
                    "training_type": "single",
                    "sharpe_ratio": 1.5,
                    "cum_return": 0.2,
                    "max_drawdown": -0.1,
                    "total_trades": 10,
                    "created_at": datetime.now().isoformat(),
                    "valid_until": future,
                    "status": "active",
                    "archived_at": None,
                    "notes": "",
                }
            ],
        }
        tmp_registry._data = d
        count = tmp_registry.expire_stale()
        assert count == 0


class TestArchiveOld:
    def test_archive_moves_file(self, tmp_registry, tmp_factors_dir):
        past = (datetime.now() - timedelta(days=100)).isoformat()
        ticker_sub = tmp_factors_dir / "0700_HK"
        ticker_sub.mkdir()
        pkl_path = ticker_sub / "factor_0001.pkl"
        pkl_path.write_bytes(b"fake pkl content")
        assert pkl_path.exists()

        d = {
            "version": 1,
            "last_updated": datetime.now().isoformat(),
            "factors": [
                {
                    "id": 1,
                    "filename": "factor_0001.pkl",
                    "subdir": "0700_HK",
                    "strategy_name": "macd_rsi_trend",
                    "ticker": "0700.HK",
                    "training_type": "single",
                    "sharpe_ratio": 1.5,
                    "cum_return": 0.2,
                    "max_drawdown": -0.1,
                    "total_trades": 10,
                    "created_at": past,
                    "valid_until": past,
                    "status": "expired",
                    "archived_at": None,
                    "notes": "",
                }
            ],
        }
        tmp_registry._data = d
        tmp_registry._save()
        count = tmp_registry.archive_old(factors_dir=tmp_factors_dir)
        assert count == 1
        assert tmp_registry._data["factors"][0]["status"] == "archived"
        assert tmp_registry._data["factors"][0]["archived_at"] is not None
        assert not pkl_path.exists()
        assert (tmp_factors_dir / "archive" / "factor_0001.pkl").exists()

    def test_archive_not_yet_90_days(self, tmp_registry, tmp_factors_dir):
        past = (datetime.now() - timedelta(days=60)).isoformat()
        d = {
            "version": 1,
            "last_updated": datetime.now().isoformat(),
            "factors": [
                {
                    "id": 1,
                    "filename": "factor_0001.pkl",
                    "subdir": None,
                    "strategy_name": "macd_rsi_trend",
                    "ticker": "0700.HK",
                    "training_type": "single",
                    "sharpe_ratio": 1.5,
                    "cum_return": 0.2,
                    "max_drawdown": -0.1,
                    "total_trades": 10,
                    "created_at": past,
                    "valid_until": past,
                    "status": "expired",
                    "archived_at": None,
                    "notes": "",
                }
            ],
        }
        tmp_registry._data = d
        tmp_registry._save()
        count = tmp_registry.archive_old(factors_dir=tmp_factors_dir)
        assert count == 0
        assert tmp_registry._data["factors"][0]["status"] == "expired"


class TestSummary:
    def test_summary_counts(self, tmp_registry):
        d = {
            "version": 1,
            "last_updated": datetime.now().isoformat(),
            "factors": [
                {
                    "id": 1,
                    "filename": "factor_0001.pkl",
                    "subdir": None,
                    "strategy_name": "macd_rsi_trend",
                    "ticker": "0700.HK",
                    "training_type": "single",
                    "sharpe_ratio": 1.5,
                    "cum_return": 0.2,
                    "max_drawdown": -0.1,
                    "total_trades": 10,
                    "created_at": datetime.now().isoformat(),
                    "valid_until": (datetime.now() + timedelta(days=30)).isoformat(),
                    "status": "active",
                    "archived_at": None,
                    "notes": "",
                },
                {
                    "id": 2,
                    "filename": "factor_0002.pkl",
                    "subdir": None,
                    "strategy_name": "rsi_divergence",
                    "ticker": "0700.HK",
                    "training_type": "single",
                    "sharpe_ratio": 1.2,
                    "cum_return": 0.15,
                    "max_drawdown": -0.1,
                    "total_trades": 8,
                    "created_at": (datetime.now() - timedelta(days=60)).isoformat(),
                    "valid_until": (datetime.now() - timedelta(days=30)).isoformat(),
                    "status": "expired",
                    "archived_at": None,
                    "notes": "",
                },
                {
                    "id": 3,
                    "filename": "factor_0003.pkl",
                    "subdir": None,
                    "strategy_name": "bollinger_rsi_trend",
                    "ticker": "0700.HK",
                    "training_type": "single",
                    "sharpe_ratio": 1.0,
                    "cum_return": 0.1,
                    "max_drawdown": -0.1,
                    "total_trades": 5,
                    "created_at": (datetime.now() - timedelta(days=150)).isoformat(),
                    "valid_until": (datetime.now() - timedelta(days=120)).isoformat(),
                    "status": "archived",
                    "archived_at": (datetime.now() - timedelta(days=30)).isoformat(),
                    "notes": "",
                },
            ],
        }
        tmp_registry._data = d
        summary = tmp_registry.summary()
        assert summary["active"] == 1
        assert summary["expired"] == 1
        assert summary["archived"] == 1


class TestAtomicSave:
    def test_atomic_save_no_corruption(self, tmp_registry, tmp_path):
        tmp_registry.register(
            factor_id=1,
            filename="factor_0001.pkl",
            subdir=None,
            strategy_name="macd_rsi_trend",
            ticker="0700.HK",
            training_type="single",
            sharpe_ratio=1.5,
            cum_return=0.2,
            max_drawdown=-0.1,
            total_trades=10,
        )
        tmp = tmp_path / "factor_registry.json"
        tmp.write_text("{ invalid json }")
        tmp.replace(tmp_registry._path)
        new_reg = FactorRegistry(registry_path=tmp_registry._path)
        assert new_reg._data["factors"] == []


class TestGetTrainingType:
    def test_xgboost_is_multi(self):
        assert _get_training_type("xgboost_enhanced") == "multi"
        assert _get_training_type("xgboost_enhanced_tsfresh") == "multi"

    def test_lightgbm_is_multi(self):
        assert _get_training_type("lightgbm_enhanced") == "multi"

    def test_linear_family_is_single(self):
        # ridge/linear/random_forest 的 run() 不处理 ticker 分组，
        # 强制归为 single，让它们在目标股票自身数据上训练。
        assert _get_training_type("ridge_regression") == "single"
        assert _get_training_type("linear_regression") == "single"
        assert _get_training_type("random_forest") == "single"

    def test_rule_strategy_is_single(self):
        assert _get_training_type("macd_rsi_trend") == "single"
        assert _get_training_type("bollinger_rsi_trend") == "single"
        assert _get_training_type("rsi_divergence") == "single"


class TestMigrationScript:
    def test_migrate_existing_factors(self, tmp_registry, tmp_factors_dir):
        joblib.dump(
            _make_artifact(5, "macd_rsi_trend", "0700.HK", 1.5),
            tmp_factors_dir / "factor_0005.pkl",
        )
        ticker_sub = tmp_factors_dir / "0700_HK"
        ticker_sub.mkdir()
        joblib.dump(
            _make_artifact(12, "bollinger_rsi_trend", "0700.HK", 1.3),
            ticker_sub / "factor_0012.pkl",
        )
        n = _migrate_existing_factors(tmp_factors_dir, registry_path=tmp_registry._path)
        assert n == 2
        data = json.loads(tmp_registry._path.read_text())
        assert len(data["factors"]) == 2

    def test_migrate_skips_already_registered(self, tmp_registry, tmp_factors_dir):
        joblib.dump(
            _make_artifact(5, "macd_rsi_trend", "0700.HK", 1.5),
            tmp_factors_dir / "factor_0005.pkl",
        )
        tmp_registry.register(
            factor_id=5,
            filename="factor_0005.pkl",
            subdir=None,
            strategy_name="macd_rsi_trend",
            ticker="0700.HK",
            training_type="single",
            sharpe_ratio=1.5,
            cum_return=0.2,
            max_drawdown=-0.1,
            total_trades=10,
        )
        n = _migrate_existing_factors(tmp_factors_dir, registry_path=tmp_registry._path)
        assert n == 0
