"""
data/factor_registry.py — 因子生命周期注册表

职责：
  - 维护 data/factors/factor_registry.json（因子索引）
  - 自动 TTL（规则策略 30 天，ML 策略 60 天）
  - 过期 > 90 天自动归档（移动 pkl 到 archive/）
  - Sharpe 对比（新因子劣化时记录警告）
  - 投票过滤（只加载 status=active 的因子）

用法：
  from data.factor_registry import FactorRegistry

  registry = FactorRegistry()
  registry.register(...)          # 注册新因子
  registry.expire_stale()        # 将超期因子标记为 expired
  registry.archive_old()          # 归档过期超过 90 天的因子
  active = registry.active_records(ticker="0700.HK")  # 过滤可用因子
"""

from __future__ import annotations

import argparse
import joblib
import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from log_config import get_logger

logger = get_logger(__name__)

REGISTRY_PATH = Path("data/factors/factor_registry.json")

TTL_DAYS = {
    "single": 30,
    "multi": 60,
    "custom": 45,
}
ARCHIVE_AFTER_DAYS = 90


class FactorRecord:
    """单条因子记录（纯数据，无副作用）"""

    def __init__(self, data: dict):
        self._d = data

    @property
    def id(self) -> int:
        return self._d["id"]

    @property
    def filename(self) -> str:
        return self._d["filename"]

    @property
    def subdir(self) -> Optional[str]:
        return self._d.get("subdir")

    @property
    def strategy_name(self) -> str:
        return self._d["strategy_name"]

    @property
    def ticker(self) -> Optional[str]:
        return self._d.get("ticker")

    @property
    def training_type(self) -> str:
        return self._d.get("training_type", "single")

    @property
    def sharpe_ratio(self) -> float:
        return self._d.get("sharpe_ratio", 0.0)

    @property
    def cum_return(self) -> float:
        return self._d.get("cum_return", 0.0)

    @property
    def max_drawdown(self) -> float:
        return self._d.get("max_drawdown", 0.0)

    @property
    def total_trades(self) -> int:
        return self._d.get("total_trades", 0)

    @property
    def status(self) -> str:
        return self._d["status"]

    @property
    def valid_until(self) -> datetime:
        return datetime.fromisoformat(self._d["valid_until"])

    @property
    def created_at(self) -> datetime:
        return datetime.fromisoformat(self._d["created_at"])

    @property
    def archived_at(self) -> Optional[datetime]:
        val = self._d.get("archived_at")
        return datetime.fromisoformat(val) if val else None

    @property
    def notes(self) -> str:
        return self._d.get("notes", "")

    def is_active(self) -> bool:
        return self.status == "active" and datetime.now() <= self.valid_until

    def to_dict(self) -> dict:
        return dict(self._d)


class FactorRegistry:
    """
    因子生命周期注册表。

    职责：
      - 注册新因子
      - 每日 TTL 检查（active → expired）
      - 归档过期超过 ARCHIVE_AFTER_DAYS 的因子（移动 pkl 文件）
      - 按 ticker / strategy / status 查询
    """

    def __init__(self, registry_path: Path = REGISTRY_PATH):
        self._path = registry_path
        self._data = self._load()

    def _load(self) -> dict:
        if not self._path.exists():
            return {"version": 1, "last_updated": "", "factors": []}
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {"version": 1, "last_updated": "", "factors": []}

    def _save(self) -> None:
        """原子写入，防止中断损坏文件。"""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        self._data["last_updated"] = datetime.now().isoformat(timespec="seconds")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)
        tmp.replace(self._path)

    def all_records(self) -> list[FactorRecord]:
        return [FactorRecord(d) for d in self._data["factors"]]

    def active_records(self, ticker: Optional[str] = None) -> list[FactorRecord]:
        """返回 status=active 且未过 TTL 的记录（可按 ticker 过滤）"""
        now = datetime.now()
        result = []
        for d in self._data["factors"]:
            if d["status"] != "active":
                continue
            if datetime.fromisoformat(d["valid_until"]) < now:
                continue
            if ticker is not None and d.get("ticker") not in (ticker, None):
                continue
            result.append(FactorRecord(d))
        return result

    def register(
        self,
        factor_id: int,
        filename: str,
        subdir: Optional[str],
        strategy_name: str,
        ticker: Optional[str],
        training_type: str,
        sharpe_ratio: float,
        cum_return: float,
        max_drawdown: float,
        total_trades: int,
    ) -> FactorRecord:
        """
        注册新因子。若同 ticker × 策略已有 active 因子，
        自动对比 Sharpe 并记录警告。
        """
        now = datetime.now()
        ttl = TTL_DAYS.get(training_type, 30)
        valid_until = (now + timedelta(days=ttl)).strftime("%Y-%m-%dT%H:%M:%S")

        new_record = {
            "id": factor_id,
            "filename": filename,
            "subdir": subdir,
            "strategy_name": strategy_name,
            "ticker": ticker,
            "training_type": training_type,
            "sharpe_ratio": sharpe_ratio,
            "cum_return": cum_return,
            "max_drawdown": max_drawdown,
            "total_trades": total_trades,
            "created_at": now.isoformat(timespec="seconds"),
            "valid_until": valid_until,
            "status": "active",
            "archived_at": None,
            "notes": "",
        }

        existing = [
            d for d in self._data["factors"]
            if d["status"] == "active"
            and d["strategy_name"] == strategy_name
            and d.get("ticker") == ticker
        ]
        if existing:
            best_old = max(d["sharpe_ratio"] for d in existing)
            if sharpe_ratio < best_old - 0.2:
                new_record["notes"] = (
                    f"Sharpe 劣化：{sharpe_ratio:.2f} vs 历史最优 {best_old:.2f}"
                )
                logger.warning(
                    "因子 Sharpe 劣化",
                    extra={
                        "strategy": strategy_name,
                        "ticker": ticker,
                        "new_sharpe": sharpe_ratio,
                        "old_best_sharpe": best_old,
                    },
                )

        self._data["factors"].append(new_record)
        self._save()
        logger.info(
            "已注册因子",
            extra={"factor_id": factor_id, "strategy": strategy_name,
                   "ticker": ticker, "valid_until": valid_until},
        )
        return FactorRecord(new_record)

    def expire_stale(self) -> int:
        """将超过 valid_until 的 active 因子标记为 expired。返回过期数量。"""
        now = datetime.now()
        count = 0
        for d in self._data["factors"]:
            if d["status"] == "active" and datetime.fromisoformat(d["valid_until"]) < now:
                d["status"] = "expired"
                count += 1
        if count:
            self._save()
            logger.info("因子 TTL 过期", extra={"expired_count": count})
        return count

    def archive_old(self, factors_dir: Path = Path("data/factors")) -> int:
        """
        将 expired 超过 ARCHIVE_AFTER_DAYS 的因子 pkl 移至 archive/。
        返回归档数量。
        """
        archive_dir = factors_dir / "archive"
        archive_dir.mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        count = 0
        for d in self._data["factors"]:
            if d["status"] != "expired":
                continue
            expired_on = datetime.fromisoformat(d["valid_until"])
            if (now - expired_on).days < ARCHIVE_AFTER_DAYS:
                continue
            subdir = d.get("subdir")
            src = factors_dir / subdir / d["filename"] if subdir else factors_dir / d["filename"]
            if src.exists():
                dst = archive_dir / d["filename"]
                shutil.move(str(src), str(dst))
            d["status"] = "archived"
            d["archived_at"] = now.isoformat(timespec="seconds")
            count += 1
        if count:
            self._save()
            logger.info("因子已归档", extra={"archived_count": count})
        return count

    def summary(self) -> dict:
        """返回各状态因子数量，供每日日志使用。"""
        counts: dict[str, int] = {"active": 0, "expired": 0, "archived": 0}
        for d in self._data["factors"]:
            counts[d.get("status", "active")] = counts.get(d.get("status", "active"), 0) + 1
        return counts


def _get_training_type(strategy_name: str) -> str:
    """
    根据策略名推断训练类型。

    注意：此函数与 optimize_with_optuna._get_training_type 逻辑完全相同，
    须同步更新。若新增策略类型（如 rnn_trend），两处均需修改。
    """
    if "xgboost" in strategy_name or "lightgbm" in strategy_name or \
       "ridge" in strategy_name or "linear" in strategy_name or "forest" in strategy_name:
        return "multi"
    return "single"


def _migrate_existing_factors(
    factors_dir: Path = Path("data/factors"),
    registry_path: Path = REGISTRY_PATH,
) -> int:
    """
    一次性迁移脚本：扫描 data/factors/ 下所有 pkl（含子目录），
    读取因子元数据并注册到 factor_registry.json。
    返回迁移的因子数量。
    """
    if not factors_dir.exists():
        logger.warning("factors_dir 不存在，跳过迁移: %s", factors_dir)
        return 0

    registry = FactorRegistry(registry_path=registry_path)
    count = 0

    patterns = [
        factors_dir.glob("factor_*.pkl"),
    ]
    for sd in factors_dir.iterdir():
        if sd.is_dir() and not sd.name.startswith("."):
            patterns.append(sd.glob("factor_*.pkl"))

    seen_ids = set(r.id for r in registry.all_records())

    for pattern in patterns:
        for pkl_path in pattern:
            try:
                art = joblib.load(pkl_path)
            except Exception as e:
                logger.warning("加载因子失败，跳过: %s (%s)", pkl_path, e)
                continue

            try:
                run_id = int(pkl_path.stem.split("_")[1])
            except (IndexError, ValueError):
                logger.warning("因子文件名无法解析 run_id，跳过: %s", pkl_path)
                continue

            if run_id in seen_ids:
                continue

            meta = art.get("meta", {})
            strategy_name = meta.get("name", "unknown")
            training_type = _get_training_type(strategy_name)
            ticker = art.get("config", {}).get("ticker")
            if ticker is None and training_type == "multi":
                ticker = None
            elif ticker is None:
                subdir = pkl_path.parent.relative_to(factors_dir) if pkl_path.parent != factors_dir else None
                if subdir and subdir.name.endswith("_HK"):
                    ticker = subdir.name.replace("_HK", ".HK").replace("_", "")

            ttl = TTL_DAYS.get(training_type, 30)
            saved_at_str = art.get("saved_at", "")
            if saved_at_str:
                try:
                    saved_at = datetime.fromisoformat(saved_at_str)
                    valid_until = (saved_at + timedelta(days=ttl)).strftime("%Y-%m-%dT%H:%M:%S")
                except ValueError:
                    valid_until = (datetime.now() + timedelta(days=ttl)).strftime("%Y-%m-%dT%H:%M:%S")
            else:
                valid_until = (datetime.now() + timedelta(days=ttl)).strftime("%Y-%m-%dT%H:%M:%S")

            subdir = None
            if pkl_path.parent != factors_dir:
                subdir = pkl_path.parent.relative_to(factors_dir)

            new_record = {
                "id": run_id,
                "filename": pkl_path.name,
                "subdir": str(subdir) if subdir else None,
                "strategy_name": strategy_name,
                "ticker": ticker,
                "training_type": training_type,
                "sharpe_ratio": art.get("sharpe_ratio", 0.0),
                "cum_return": art.get("cum_return", 0.0),
                "max_drawdown": art.get("max_drawdown", 0.0),
                "total_trades": art.get("total_trades", 0),
                "created_at": saved_at_str or datetime.now().isoformat(timespec="seconds"),
                "valid_until": valid_until,
                "status": "active",
                "archived_at": None,
                "notes": "从存量因子迁移",
            }

            if datetime.fromisoformat(valid_until) < datetime.now():
                new_record["status"] = "expired"

            registry._data["factors"].append(new_record)
            seen_ids.add(run_id)
            count += 1

    if count:
        registry._save()
        logger.info("迁移完成", extra={"migrated_count": count})

    return count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="因子注册表工具")
    parser.add_argument("--migrate", action="store_true", help="迁移存量因子到注册表")
    args = parser.parse_args()

    if args.migrate:
        factors_dir = Path(__file__).parent.parent / "data" / "factors"
        print(f"开始迁移存量因子: {factors_dir}")
        n = _migrate_existing_factors(factors_dir)
        print(f"迁移完成，共 {n} 个因子")
    else:
        parser.print_help()
