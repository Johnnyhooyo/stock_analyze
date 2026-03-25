"""
engine/portfolio_state.py — 持仓状态管理

负责：
  - 从 data/portfolio.yaml 加载多股票持仓状态
  - 提供 PortfolioPosition / PortfolioState 数据类
  - 支持每日收盘后更新峰值/连续亏损天数并回写 YAML

与 position_manager.py 的关系：
  - PortfolioPosition  持有"静态"持仓快照（从 YAML 读入）
  - Position (position_manager.py) 持有"运行时"持仓（含当前价、每日盈亏）
  - to_position_manager_position() 完成两者的转换
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

from log_config import get_logger

logger = get_logger(__name__)

# portfolio.yaml 默认路径
_DEFAULT_PORTFOLIO_PATH = Path(__file__).parent.parent / "data" / "portfolio.yaml"


# ──────────────────────────────────────────────────────────────────
#  PortfolioPosition — 单只股票的持仓快照
# ──────────────────────────────────────────────────────────────────

@dataclass
class PortfolioPosition:
    """
    单只股票的持仓状态（从 portfolio.yaml 读入的静态快照）。

    字段说明：
        ticker                — 股票代码，如 "0700.HK"
        shares                — 持股数量（0 = 空仓/观察）
        avg_cost              — 平均持仓成本（港元/股）
        peak_price            — 持仓期间最高价（ATR 移动止损基准，0 = 用入场价代替）
        consecutive_loss_days — 连续亏损天数（系统自动维护）
        trailing_peak         — TrailingStop 记录的移动峰值（系统自动维护，None = 未初始化）
    """
    ticker: str
    shares: int = 0
    avg_cost: float = 0.0
    peak_price: float = 0.0
    consecutive_loss_days: int = 0
    trailing_peak: Optional[float] = None

    # ── 派生属性 ──────────────────────────────────────────────────

    @property
    def has_position(self) -> bool:
        """是否实际持仓（shares > 0 且 avg_cost > 0）"""
        return self.shares > 0 and self.avg_cost > 0.0

    # ── 与 PositionManager 桥接 ───────────────────────────────────

    def to_position_manager_position(self, current_price: float):
        """
        转换为 position_manager.Position 实例，供 PositionManager 使用。

        Args:
            current_price: 最新收盘价（用于计算当日市值 / 盈亏）

        Returns:
            position_manager.Position
        """
        from position_manager import Position  # 延迟导入，避免循环依赖

        return Position(
            shares=self.shares,
            avg_cost=self.avg_cost,
            current_price=current_price,
        )

    # ── 序列化（写回 YAML）────────────────────────────────────────

    def to_dict(self) -> dict:
        """转换为可写入 portfolio.yaml 的字典"""
        return {
            "ticker": self.ticker,
            "shares": self.shares,
            "avg_cost": self.avg_cost,
            "peak_price": self.peak_price,
            "consecutive_loss_days": self.consecutive_loss_days,
            "trailing_peak": self.trailing_peak,
        }

    def __repr__(self) -> str:
        status = f"{self.shares}股@{self.avg_cost:.2f}" if self.has_position else "空仓"
        return f"PortfolioPosition({self.ticker}, {status})"


# ──────────────────────────────────────────────────────────────────
#  PortfolioState — 全持仓状态管理
# ──────────────────────────────────────────────────────────────────

@dataclass
class PortfolioState:
    """
    全持仓状态（portfolio.yaml 的内存表示）。

    字段说明：
        portfolio_value — 总资产（港元），用于 Kelly 仓位计算
        positions       — {ticker: PortfolioPosition} 持仓字典
        path            — portfolio.yaml 文件路径（保存时用）
    """
    portfolio_value: float = 200_000.0
    positions: dict[str, PortfolioPosition] = field(default_factory=dict)
    path: Path = field(default_factory=lambda: _DEFAULT_PORTFOLIO_PATH)

    # ── 持仓查询 ──────────────────────────────────────────────────

    def get_position(self, ticker: str) -> Optional[PortfolioPosition]:
        """
        获取指定股票的持仓。
        ticker 大小写不敏感（统一转大写后查找）。
        不存在时返回 None（表示空仓，不在观察列表）。
        """
        return self.positions.get(ticker.upper())

    def all_tickers(self) -> list[str]:
        """返回所有持仓/观察列表中的股票代码（保持插入顺序）"""
        return list(self.positions.keys())

    def held_tickers(self) -> list[str]:
        """只返回实际持仓（shares > 0）的股票代码"""
        return [t for t, p in self.positions.items() if p.has_position]

    # ── 持仓写入 ──────────────────────────────────────────────────

    def add_watchlist_ticker(self, ticker: str) -> None:
        """
        将股票加入观察列表（若已存在则不变）。
        创建一个 shares=0 的空仓 PortfolioPosition。
        """
        ticker = ticker.upper()
        if ticker not in self.positions:
            self.positions[ticker] = PortfolioPosition(ticker=ticker)

    def update_position(
        self,
        ticker: str,
        *,
        peak_price: Optional[float] = None,
        consecutive_loss_days: Optional[int] = None,
        trailing_peak: Optional[float] = None,
        shares: Optional[int] = None,
        avg_cost: Optional[float] = None,
    ) -> None:
        """
        更新指定股票的持仓字段（只更新传入的非 None 字段）。

        典型用法（daily_run.py 收盘后更新）：
            portfolio_state.update_position(
                ticker,
                peak_price=r.peak_price,
                consecutive_loss_days=r.consecutive_loss_days,
            )
        """
        ticker = ticker.upper()
        pos = self.positions.get(ticker)
        if pos is None:
            # 如果不存在，先创建再更新
            pos = PortfolioPosition(ticker=ticker)
            self.positions[ticker] = pos

        if peak_price is not None:
            pos.peak_price = float(peak_price)
        if consecutive_loss_days is not None:
            pos.consecutive_loss_days = int(consecutive_loss_days)
        if trailing_peak is not None:
            pos.trailing_peak = float(trailing_peak)
        if shares is not None:
            pos.shares = int(shares)
        if avg_cost is not None:
            pos.avg_cost = float(avg_cost)

    # ── 汇总信息 ──────────────────────────────────────────────────

    def summary(self) -> str:
        """生成终端可读的持仓摘要（供 daily_run.py 打印）"""
        lines = [
            f"  💼 投资组合  总资产={self.portfolio_value:,.0f} 港元  "
            f"标的数={len(self.positions)}  持仓={len(self.held_tickers())}",
        ]
        for ticker, pos in self.positions.items():
            if pos.has_position:
                lines.append(
                    f"     ├ {ticker:<12s}  {pos.shares}股@{pos.avg_cost:.2f}  "
                    f"峰值={pos.peak_price:.2f}  "
                    f"连续亏损={pos.consecutive_loss_days}天"
                )
            else:
                lines.append(f"     ├ {ticker:<12s}  空仓/观察")
        return "\n".join(lines)

    # ── 持久化 ────────────────────────────────────────────────────

    def save(self, path: Optional[Path] = None) -> None:
        """
        将当前状态原子写回 portfolio.yaml。
        使用临时文件 + rename 保证写入原子性，防止写到一半崩溃导致文件损坏。

        Args:
            path: 目标路径（默认使用初始化时的 self.path）
        """
        target = path or self.path
        target.parent.mkdir(parents=True, exist_ok=True)

        # 构造 YAML 内容
        positions_yaml: dict = {}
        for ticker, pos in self.positions.items():
            # YAML 键：用下划线代替点，如 "0700.HK" → "0700_HK"
            key = ticker.replace(".", "_")
            positions_yaml[key] = pos.to_dict()

        data = {
            "portfolio_value": self.portfolio_value,
            "positions": positions_yaml,
        }

        # 在注释头部添加说明（直接构造字符串，PyYAML 不支持注释）
        header = (
            "# portfolio.yaml — 每日推荐引擎持仓状态文件\n"
            "#\n"
            "# 使用说明：\n"
            "#   1. 在 positions 下添加你的持仓股票（或观察列表）\n"
            "#   2. shares=0 表示空仓（但仍会生成信号供参考）\n"
            "#   3. peak_price=0 表示使用入场价作为 ATR 止损基准\n"
            "#   4. consecutive_loss_days / trailing_peak 由系统自动维护，勿手动修改\n"
            "#\n"
        )

        tmp = target.with_suffix(".yaml.tmp")
        try:
            yaml_body = yaml.dump(data, allow_unicode=True, default_flow_style=False)
            tmp.write_text(header + yaml_body, encoding="utf-8")
            shutil.move(str(tmp), str(target))
        except Exception:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            raise

    def __repr__(self) -> str:
        return (
            f"PortfolioState("
            f"value={self.portfolio_value:,.0f}, "
            f"positions={list(self.positions.keys())})"
        )


# ──────────────────────────────────────────────────────────────────
#  load_portfolio — 工厂函数
# ──────────────────────────────────────────────────────────────────

def load_portfolio(path: Optional[Path] = None) -> PortfolioState:
    """
    从 portfolio.yaml 加载持仓状态，返回 PortfolioState。

    文件不存在时返回一个空的 PortfolioState（不报错）。
    YAML 解析失败时同样返回空状态并打印警告。

    Args:
        path: portfolio.yaml 路径（默认 data/portfolio.yaml）

    Returns:
        PortfolioState
    """
    target = path or _DEFAULT_PORTFOLIO_PATH

    if not target.exists():
        logger.warning("portfolio.yaml 不存在（%s），返回空持仓状态", target)
        return PortfolioState(path=target)

    try:
        with open(target, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning("portfolio.yaml 解析失败：%s，返回空持仓状态", e)
        return PortfolioState(path=target)

    portfolio_value = float(raw.get("portfolio_value", 200_000.0))
    positions: dict[str, PortfolioPosition] = {}

    for _key, entry in (raw.get("positions") or {}).items():
        if not isinstance(entry, dict):
            continue

        # ticker 字段优先，否则从键名还原（"0700_HK" → "0700.HK"）
        ticker = entry.get("ticker") or _key.replace("_", ".", 1)
        ticker = ticker.upper()

        pos = PortfolioPosition(
            ticker=ticker,
            shares=int(entry.get("shares", 0)),
            avg_cost=float(entry.get("avg_cost", 0.0)),
            peak_price=float(entry.get("peak_price", 0.0)),
            consecutive_loss_days=int(entry.get("consecutive_loss_days", 0)),
            trailing_peak=(
                float(entry["trailing_peak"])
                if entry.get("trailing_peak") is not None
                else None
            ),
        )
        positions[ticker] = pos

    return PortfolioState(
        portfolio_value=portfolio_value,
        positions=positions,
        path=target,
    )

