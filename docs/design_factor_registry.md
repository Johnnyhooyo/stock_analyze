# 因子生命周期管理设计方案

> **编制日期**：2026-04-03  
> **对应 upgrade_plan.md**：Phase 2 § 4.5  
> **优先级**：P1  
> **工作量估算**：M（5-7 天）

---

## 一、现状与问题

### 1.1 现状

| 项目 | 现状 |
|------|------|
| 存储格式 | `data/factors/factor_XXXX.pkl`，按自增 ID 排序 |
| 新鲜度判断 | `engine/signal_aggregator.py` 对每只 ticker 子目录中的所有 pkl 文件全量加载，无 TTL 过滤 |
| pkl 内容 | `saved_at`（ISO 字符串）、`sharpe_ratio`、`cum_return`、`meta["name"]`（策略名）等字段 |
| 过期因子 | 永久积累在目录中，随时间增多会拖慢加载速度 |
| 跨股因子（ML） | `config["ticker"]` 为 `None`，因子存放在全局目录，非 ticker 子目录 |

### 1.2 问题

1. **无 TTL**：训练于 2 个月前的规则因子仍参与今日投票，市场制度变化后可能持续给出错误信号
2. **无状态跟踪**：不知道哪些因子"有效"、哪些已"退化"
3. **磁盘积累**：factor_*.pkl 文件随运行次数线性增长，无清理机制
4. **无对比**：新因子注册时无法与旧因子 Sharpe 自动对比

---

## 二、设计目标

1. **因子索引**：维护 `data/factors/factor_registry.json`，记录每个因子的元数据和状态
2. **自动 TTL**：规则策略因子默认 30 天过期，ML 策略因子默认 60 天
3. **自动归档**：过期 > 90 天的因子 pkl 移入 `data/factors/archive/`
4. **投票过滤**：`SignalAggregator` 只加载 `status = "active"` 的因子
5. **Sharpe 对比**：注册新因子时，若同 ticker × 同策略已有 active 因子，自动对比 Sharpe；劣化时记录警告
6. **每日日志**：`daily_run.py` 运行时显示 `活跃因子 N 个 / 过期 M 个 / 归档 K 个`

---

## 三、数据模型

### 3.1 `factor_registry.json`

```json
{
  "version": 1,
  "last_updated": "2026-04-03T18:00:00",
  "factors": [
    {
      "id": 42,
      "filename": "factor_0042.pkl",
      "subdir": "0700_HK",
      "strategy_name": "macd_rsi_trend",
      "ticker": "0700.HK",
      "training_type": "single",
      "sharpe_ratio": 1.85,
      "cum_return": 0.23,
      "max_drawdown": -0.12,
      "total_trades": 18,
      "created_at": "2026-03-20T18:00:00",
      "valid_until": "2026-04-19",
      "status": "active",
      "archived_at": null,
      "notes": ""
    },
    {
      "id": 7,
      "filename": "factor_0007.pkl",
      "subdir": null,
      "strategy_name": "xgboost_enhanced",
      "ticker": null,
      "training_type": "multi",
      "sharpe_ratio": 2.10,
      "cum_return": 0.31,
      "max_drawdown": -0.09,
      "total_trades": 22,
      "created_at": "2026-03-15T06:00:00",
      "valid_until": "2026-05-14",
      "status": "active",
      "archived_at": null,
      "notes": ""
    }
  ]
}
```

### 3.2 状态机

```
        注册
         │
         ▼
      ◆ active ─────────── TTL 到期 ──────────► ◆ expired
         │                                            │
         │                                    超过 90 天
         │                                            │
         └──────────── 手动归档 ───────────────────► ◆ archived
```

| 状态 | 描述 | 参与每日投票 |
|------|------|------------|
| `active` | 在 TTL 内，正常使用 | ✅ 是 |
| `expired` | 超过 TTL，等待归档 | ❌ 否 |
| `archived` | pkl 已移至 archive/，不再可用 | ❌ 否 |

---

## 四、模块设计

### 4.1 `data/factor_registry.py`

```python
from __future__ import annotations

import json
import shutil
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from log_config import get_logger

logger = get_logger(__name__)

REGISTRY_PATH = Path("data/factors/factor_registry.json")
ARCHIVE_DIR   = Path("data/factors/archive")

# TTL（天）按训练类型区分
TTL_DAYS = {
    "single": 30,   # 规则策略
    "multi":  60,   # ML 策略
    "custom": 45,
}
ARCHIVE_AFTER_DAYS = 90  # 过期后多久移入 archive


class FactorRecord:
    """单条因子记录（纯数据，无副作用）"""

    def __init__(self, data: dict):
        self._d = data

    # ── 属性快捷访问 ──────────────────────────────
    @property
    def id(self) -> int:             return self._d["id"]
    @property
    def filename(self) -> str:       return self._d["filename"]
    @property
    def subdir(self) -> Optional[str]: return self._d.get("subdir")
    @property
    def strategy_name(self) -> str:  return self._d["strategy_name"]
    @property
    def ticker(self) -> Optional[str]: return self._d.get("ticker")
    @property
    def training_type(self) -> str:  return self._d.get("training_type", "single")
    @property
    def sharpe_ratio(self) -> float: return self._d.get("sharpe_ratio", 0.0)
    @property
    def status(self) -> str:         return self._d["status"]
    @property
    def valid_until(self) -> datetime:
        return datetime.fromisoformat(self._d["valid_until"])

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

    # ── 读取 ──────────────────────────────────────

    def _load(self) -> dict:
        if not self._path.exists():
            return {"version": 1, "last_updated": "", "factors": []}
        with open(self._path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self) -> None:
        """原子写入，防止中断损坏文件。"""
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
                # None ticker = multi 策略，对所有 ticker 可用
                continue
            result.append(FactorRecord(d))
        return result

    # ── 注册 ──────────────────────────────────────

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
        ttl = TTL_DAYS.get(training_type, 30)
        valid_until = (datetime.now() + timedelta(days=ttl)).strftime("%Y-%m-%dT%H:%M:%S")

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
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "valid_until": valid_until,
            "status": "active",
            "archived_at": None,
            "notes": "",
        }

        # Sharpe 对比：同 ticker + 同策略的 active 因子
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
                    f"⚠️ Sharpe 劣化：{sharpe_ratio:.2f} vs 历史最优 {best_old:.2f}"
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

    # ── 生命周期维护 ──────────────────────────────

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
        ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
        now = datetime.now()
        count = 0
        for d in self._data["factors"]:
            if d["status"] != "expired":
                continue
            expired_on = datetime.fromisoformat(d["valid_until"])
            if (now - expired_on).days < ARCHIVE_AFTER_DAYS:
                continue
            # 确定 pkl 路径
            subdir = d.get("subdir")
            src = factors_dir / subdir / d["filename"] if subdir else factors_dir / d["filename"]
            if src.exists():
                dst = ARCHIVE_DIR / d["filename"]
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
```

### 4.2 集成点 A：`analyze_factor.py` — 因子保存时注册

在 `save_factor()` 函数（或等效位置）保存 pkl 后，立即调用 `FactorRegistry.register()`：

```python
# analyze_factor.py
from data.factor_registry import FactorRegistry

def _save_factor(factor_data: dict, factors_dir: Path, run_id: int) -> Path:
    filename = f"factor_{run_id:04d}.pkl"
    # ... 现有 pickle 保存逻辑 ...

    # 注册到因子注册表
    registry = FactorRegistry()
    registry.register(
        factor_id=run_id,
        filename=filename,
        subdir=subdir,                      # ticker 子目录名，无则 None
        strategy_name=factor_data["meta"]["name"],
        ticker=factor_data["config"].get("ticker"),
        training_type=_get_training_type(factor_data["meta"]["name"]),
        sharpe_ratio=factor_data["sharpe_ratio"],
        cum_return=factor_data["cum_return"],
        max_drawdown=factor_data["max_drawdown"],
        total_trades=factor_data["total_trades"],
    )
    return path
```

### 4.3 集成点 B：`engine/signal_aggregator.py` — 加载时过滤

```python
# engine/signal_aggregator.py
from data.factor_registry import FactorRegistry

class SignalAggregator:
    def __init__(self, factors_dir: str = "data/factors", use_registry: bool = True):
        self._use_registry = use_registry
        if use_registry:
            self._registry = FactorRegistry()

    def _load_factors(self, ticker: str, factors_dir: Path) -> list[dict]:
        if self._use_registry:
            active = self._registry.active_records(ticker=ticker)
            active_filenames = {r.filename for r in active}
            # 只加载 registry 中 active 的 pkl
            return [self._load_pkl(p) for p in ... if p.name in active_filenames]
        else:
            # 降级：按旧逻辑加载全部
            return [self._load_pkl(p) for p in ...]
```

### 4.4 集成点 C：`daily_run.py` — 每日维护 + 日志

```python
# daily_run.py — 在每日运行入口（主函数最开头）

from data.factor_registry import FactorRegistry

def _maintain_factor_registry() -> None:
    registry = FactorRegistry()
    expired = registry.expire_stale()
    archived = registry.archive_old()
    summary = registry.summary()
    logger.info(
        "因子注册表状态",
        extra={
            "active": summary["active"],
            "newly_expired": expired,
            "newly_archived": archived,
            "total_archived": summary["archived"],
        },
    )
```

---

## 五、迁移策略（存量因子）

对于当前已存在的 `factor_*.pkl` 文件（无注册表记录），提供一次性迁移脚本：

```bash
python3 -m data.factor_registry --migrate
```

迁移逻辑：
1. 扫描 `data/factors/` 下所有 pkl（含子目录）
2. 读取每个 pkl 的 `saved_at`、`sharpe_ratio`、`meta["name"]` 等字段
3. 按 `saved_at` + TTL 计算 `valid_until`（若已超期则 status=expired）
4. 写入 `factor_registry.json`

---

## 六、文件结构与影响范围

```
新建：
  data/factor_registry.py         # FactorRegistry 类（≈ 200 行）
  data/factors/archive/           # 归档目录（自动创建）
  data/factors/factor_registry.json  # 自动生成，不提交 git

修改：
  analyze_factor.py               # _save_factor() 调用 registry.register()
  engine/signal_aggregator.py     # _load_factors() 按 registry 过滤
  daily_run.py                    # 入口调用 _maintain_factor_registry()
  data/__init__.py                # 导出 FactorRegistry

新建测试：
  tests/test_factor_registry.py   # ≈ 15 个单元测试，全离线
```

**`.gitignore` 更新**：
```
data/factors/factor_registry.json
data/factors/archive/
```

---

## 七、测试计划

| 测试 | 说明 |
|------|------|
| `test_register_new_factor` | 注册后 JSON 文件更新，status=active |
| `test_active_records_by_ticker` | 按 ticker 过滤，multi 因子对所有 ticker 可见 |
| `test_ttl_expiry_single` | single 因子 30 天后 expire_stale() 标记为 expired |
| `test_ttl_expiry_multi` | multi 因子 60 天后才过期 |
| `test_archive_moves_file` | expired + 90天后 archive_old() 移动 pkl |
| `test_sharpe_degradation_warning` | 新因子 Sharpe < 旧因子 - 0.2 时 notes 含 ⚠️ |
| `test_active_records_excludes_expired` | expired 因子不出现在 active_records() 中 |
| `test_migration_script` | --migrate 对存量 pkl 正确生成注册表 |
| `test_summary` | summary() 返回正确的三类计数 |
| `test_atomic_save` | 写入中断不损坏 JSON（使用 .tmp 原子替换） |

---

## 八、验收标准

| 指标 | 目标 |
|------|------|
| `data/factors/factor_registry.json` | 每次 `main.py` 训练后自动更新 |
| 过期因子 | 不参与每日投票（`SignalAggregator` 过滤） |
| `daily_run.py` 日志 | 显示活跃/过期/归档因子数 |
| 归档文件 | pkl 移入 `data/factors/archive/`，不影响已归档前的运行 |
| `pytest tests/test_factor_registry.py` | 全通过（全离线） |
| 回归测试 | `pytest tests/` 100+ passed，`smoke_test.py` 通过 |

---

## 九、实施步骤

```
Step 1  实现 data/factor_registry.py（FactorRegistry + FactorRecord）   [1.5d]
Step 2  单元测试 tests/test_factor_registry.py                           [1d]
Step 3  集成点 A：analyze_factor.py 注册因子                             [0.5d]
Step 4  集成点 B：signal_aggregator.py 按 registry 过滤                  [0.5d]
Step 5  集成点 C：daily_run.py 每日维护 + 日志                           [0.5d]
Step 6  迁移脚本 data/factor_registry.py --migrate                       [0.5d]
Step 7  更新 .gitignore / data/__init__.py                               [0.5d]
Step 8  更新 upgrade_plan.md 4.5 节状态为 ✅                              [0.5d]
```
