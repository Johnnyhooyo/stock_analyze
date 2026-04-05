"""
持仓管理模块
用于管理股票持仓、分析盈亏、提供交易建议，并提供统一风控层。
"""

import json
import math
import os
import pandas as pd
from dataclasses import dataclass
from typing import Optional


# ──────────────────────────────────────────────────────────────────
#  持仓数据
# ──────────────────────────────────────────────────────────────────

@dataclass
class Position:
    """持仓数据"""
    shares: int           # 持股数量
    avg_cost: float       # 平均成本
    current_price: Optional[float] = None  # 当前价格（None 表示未初始化）
    entry_atr: float = 0.0   # 入场时的 ATR 值（供止损用）
    daily_pnl: float = 0.0   # 当日盈亏金额

    def _ensure_price(self):
        """确保 current_price 已初始化"""
        if self.current_price is None:
            raise ValueError(
                "current_price 未初始化，请先设置 position.current_price = last_close"
            )

    @property
    def market_value(self) -> float:
        self._ensure_price()
        return self.shares * self.current_price

    @property
    def cost_basis(self) -> float:
        return self.shares * self.avg_cost

    @property
    def profit(self) -> float:
        self._ensure_price()
        return self.market_value - self.cost_basis

    @property
    def profit_pct(self) -> float:
        if self.cost_basis == 0:
            return 0.0
        return (self.profit / self.cost_basis) * 100


# ──────────────────────────────────────────────────────────────────
#  移动止损辅助类（可供策略层复用）
# ──────────────────────────────────────────────────────────────────

class TrailingStop:
    """
    ATR 移动止损跟踪器。
    策略层可实例化此类，取代手写的 peak_price 逻辑。
    """

    def __init__(self, multiplier: float = 2.0):
        self.multiplier = multiplier
        self._peak: Optional[float] = None

    def reset(self):
        self._peak = None

    def update(self, close: float, atr: float) -> bool:
        """
        更新最高价，返回是否触发止损。
        Args:
            close: 当日收盘价
            atr:   当日 ATR 值
        Returns:
            True 表示触发止损，应平仓
        """
        if self._peak is None or close > self._peak:
            self._peak = close
        stop_price = self._peak - self.multiplier * atr
        return close < stop_price

    @property
    def peak(self) -> Optional[float]:
        return self._peak


# ──────────────────────────────────────────────────────────────────
#  风控状态持久化（连续亏损天数等跨运行状态）
# ──────────────────────────────────────────────────────────────────

_STATE_DIR = os.path.join(os.path.dirname(__file__), "data", "logs")
# 全局状态文件（向后兼容，当 ticker=None 时使用）
_STATE_FILE = os.path.join(_STATE_DIR, "risk_state.json")

_DEFAULT_STATE = {"consecutive_loss_days": 0, "last_trade_date": "", "trailing_peak": None}


def _state_path(ticker: Optional[str] = None) -> str:
    """返回对应 ticker 的风控状态文件路径（per-ticker 隔离，防止并发覆盖）。"""
    if ticker:
        safe = ticker.replace(".", "_").upper()
        return os.path.join(_STATE_DIR, f"risk_state_{safe}.json")
    return _STATE_FILE


def _load_risk_state(ticker: Optional[str] = None) -> dict:
    """加载风控状态文件（不存在则返回默认值）。ticker 指定时使用 per-ticker 文件。"""
    path = _state_path(ticker)
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return dict(_DEFAULT_STATE)


def _save_risk_state(state: dict, ticker: Optional[str] = None):
    """持久化风控状态（原子写入，防止并发写入损坏文件）。"""
    import tempfile
    path = _state_path(ticker)
    try:
        os.makedirs(_STATE_DIR, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(suffix=".json.tmp", dir=_STATE_DIR)
        try:
            os.close(fd)
            with open(tmp_path, "w") as f:
                json.dump(state, f, indent=2)
            os.replace(tmp_path, path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
    except Exception:
        pass


# ──────────────────────────────────────────────────────────────────
#  PositionManager
# ──────────────────────────────────────────────────────────────────

class PositionManager:
    """持仓管理器（含统一风控层）"""

    def __init__(
        self,
        position: Optional[Position] = None,
        portfolio_value: float = 100_000.0,
        daily_loss_limit: float = 0.05,
        max_consecutive_loss_days: int = 3,
        risk_config: Optional[dict] = None,
        ticker: Optional[str] = None,
    ):
        self.position = position
        self.portfolio_value = portfolio_value
        self.daily_loss_limit = daily_loss_limit
        self.max_consecutive_loss_days = max_consecutive_loss_days

        # 解析 risk_management 配置块
        rc = risk_config or {}
        self.use_atr_stop: bool = rc.get("use_atr_stop", True)
        self.atr_period: int = int(rc.get("atr_period", 14))
        self.atr_multiplier: float = float(rc.get("atr_multiplier", 2.0))
        self.use_kelly: bool = rc.get("use_kelly", False)
        self.max_position_pct: float = float(rc.get("max_position_pct", 0.25))
        self.daily_loss_limit = float(rc.get("daily_loss_limit", daily_loss_limit))
        self.max_consecutive_loss_days = int(
            rc.get("max_consecutive_loss_days", max_consecutive_loss_days)
        )
        self.trailing_stop: bool = rc.get("trailing_stop", True)
        self._ticker = ticker  # 用于 per-ticker 风控状态隔离

        # 移动止损跟踪器
        self._trailing = TrailingStop(multiplier=self.atr_multiplier)

        # fix #10: 恢复持久化的 trailing peak
        saved_state = _load_risk_state(ticker=self._ticker)
        if saved_state.get("trailing_peak") is not None:
            self._trailing._peak = saved_state["trailing_peak"]

    # ── 基本操作 ──────────────────────────────────────────────────

    def set_position(self, shares: int, avg_cost: float, current_price: float):
        self.position = Position(
            shares=shares,
            avg_cost=avg_cost,
            current_price=current_price,
        )

    # ── ATR 动态止损 ──────────────────────────────────────────────

    def set_atr_stop(self, atr_period: int = 14, multiplier: float = 2.0):
        """设置 ATR 止损参数"""
        self.atr_period = atr_period
        self.atr_multiplier = multiplier
        self._trailing = TrailingStop(multiplier=multiplier)

    def check_atr_stop(
        self,
        close_price: float,
        entry_price: float,
        peak_price: float,
        atr: float,
    ) -> bool:
        """
        检查是否触发 ATR 止损。
        触发条件: close < max(peak_price - multiplier * atr, entry_price - multiplier * atr)
        使用 entry_price 作为止损价下限，防止止损位低于入场成本保护线。
        Returns:
            True 表示应退出持仓
        """
        trailing_stop = peak_price - self.atr_multiplier * atr
        entry_stop = entry_price - self.atr_multiplier * atr
        stop_price = max(trailing_stop, entry_stop)
        return close_price < stop_price

    # ── Kelly 仓位公式 ────────────────────────────────────────────

    def calculate_kelly_size(
        self,
        win_rate: float,
        profit_loss_ratio: float,
        capital: Optional[float] = None,
        current_price: Optional[float] = None,
        kelly_fraction: float = 0.5,
    ) -> int:
        """
        半 Kelly 仓位计算。
        Kelly = (p * B - q) / B，其中 B = profit_loss_ratio，p = win_rate，q = 1-p。
        最终股数受 max_position_pct 上限约束。

        Args:
            win_rate:          胜率（0~1）
            profit_loss_ratio: 平均盈利 / 平均亏损比
            capital:           可用资金（默认使用 portfolio_value）
            current_price:     当前价格（默认使用 position.current_price）
            kelly_fraction:    Kelly 缩放系数（默认 0.5，即半 Kelly）
        Returns:
            建议买入股数（整数，已受上限约束）
        """
        capital = capital or self.portfolio_value
        price = current_price or (self.position.current_price if self.position else 1.0)

        if profit_loss_ratio <= 0 or win_rate <= 0 or price <= 0:
            return 0

        q = 1.0 - win_rate
        B = profit_loss_ratio
        kelly_f = (win_rate * B - q) / B
        kelly_f = max(0.0, kelly_f) * kelly_fraction  # 半 Kelly，不允许负值

        # 上限约束
        max_f = self.max_position_pct
        kelly_f = min(kelly_f, max_f)

        shares = int(kelly_f * capital / price)
        return self.validate_position_size(shares, price, capital)

    # ── 最大仓位校验 ──────────────────────────────────────────────

    def validate_position_size(
        self, proposed_shares: int, price: float, capital: Optional[float] = None
    ) -> int:
        """返回实际可建仓数量，不超过资本的 max_position_pct"""
        capital = capital or self.portfolio_value
        if price <= 0:
            return 0
        max_shares = int(self.max_position_pct * capital / price)
        return min(proposed_shares, max_shares)

    # ── 熔断机制 ──────────────────────────────────────────────────

    def check_circuit_breaker(
        self,
        today_pnl_pct: float,
        trade_date: str = "",
    ) -> dict:
        """
        每日收盘后调用，更新并检查熔断状态。
        触发条件：单日亏损 > daily_loss_limit 或连续亏损天数 > max_consecutive_loss_days。

        Args:
            today_pnl_pct:  当日收益率（负数为亏损，如 -0.03 = -3%）
            trade_date:     交易日字符串（用于去重，如 "2026-03-20"）
        Returns:
            {'tripped': bool, 'reason': str, 'action': str,
             'consecutive_loss_days': int}
        """
        state = _load_risk_state(ticker=self._ticker)

        # 同一天不重复计数，但用最新 pnl 重新评估是否触发熔断
        if trade_date and state.get("last_trade_date") == trade_date:
            tripped = (
                today_pnl_pct < -self.daily_loss_limit
                or state["consecutive_loss_days"] >= self.max_consecutive_loss_days
            )
            if tripped:
                reason = (
                    f"单日亏损 {today_pnl_pct:.2%} 超过限制 {self.daily_loss_limit:.2%}"
                    if today_pnl_pct < -self.daily_loss_limit
                    else f"已连续亏损 {state['consecutive_loss_days']} 天"
                )
            else:
                reason = "（今日已统计，风控正常）"
            return {
                "tripped": tripped,
                "reason": reason,
                "action": "观望" if tripped else "正常",
                "consecutive_loss_days": state["consecutive_loss_days"],
            }

        # 更新连续亏损天数
        if today_pnl_pct < 0:
            state["consecutive_loss_days"] += 1
        else:
            state["consecutive_loss_days"] = 0
        state["last_trade_date"] = trade_date
        # fix #10: 持久化 trailing peak
        state["trailing_peak"] = self._trailing._peak
        _save_risk_state(state, ticker=self._ticker)

        loss_days = state["consecutive_loss_days"]

        # 判断触发条件
        if today_pnl_pct < -self.daily_loss_limit:
            return {
                "tripped": True,
                "reason": f"单日亏损 {today_pnl_pct:.2%} 超过限制 {self.daily_loss_limit:.2%}",
                "action": "暂停交易，观望",
                "consecutive_loss_days": loss_days,
            }
        if loss_days >= self.max_consecutive_loss_days:
            return {
                "tripped": True,
                "reason": f"已连续亏损 {loss_days} 天（阈值 {self.max_consecutive_loss_days} 天）",
                "action": "暂停交易，观望",
                "consecutive_loss_days": loss_days,
            }

        return {
            "tripped": False,
            "reason": "风控正常",
            "action": "正常",
            "consecutive_loss_days": loss_days,
        }

    # ── 统一风控入口 ──────────────────────────────────────────────

    def apply_risk_controls(
        self,
        signal: int,
        price: float,
        atr: float,
        entry_price: float,
        peak_price: float,
        today_pnl_pct: float,
        capital: Optional[float] = None,
        win_rate: float = 0.0,
        profit_loss_ratio: float = 0.0,
        trade_date: str = "",
        oms=None,           # 可选：OrderManagementSystem 实例，非 None 时自动提交订单
        ticker: str = '',   # OMS 下单所需股票代码
    ) -> dict:
        """
        统一风控检查，依次执行：
          1. ATR 止损检查
          2. 熔断检查
          3. Kelly 仓位建议（若 use_kelly=True）
          4. 仓位上限校验
          5. 原始信号透传

        Returns:
            dict，包含最终 action / signal / stop_price /
            kelly_shares / circuit_breaker 等字段
        """
        capital = capital or self.portfolio_value
        stop_price = peak_price - self.atr_multiplier * atr if atr > 0 else 0.0
        result = {
            "signal":           signal,
            "action":           "",
            "reason":           "",
            "stop_price":       round(stop_price, 4),
            "kelly_shares":     0,
            "kelly_amount":     0.0,
            "circuit_breaker":  False,
            "consecutive_loss_days": 0,
        }

        # 1. ATR 止损
        if self.use_atr_stop and self.position and self.position.shares > 0:
            if self.check_atr_stop(price, entry_price, peak_price, atr):
                result.update({
                    "signal":  0,
                    "action":  "止损卖出",
                    "reason":  f"价格 {price:.2f} 跌破 ATR 止损位 {stop_price:.2f}（峰值 {peak_price:.2f} - {self.atr_multiplier}×ATR {atr:.2f}）",
                })
                return result

        # 2. 熔断检查
        cb = self.check_circuit_breaker(today_pnl_pct, trade_date)
        result["circuit_breaker"] = cb["tripped"]
        result["consecutive_loss_days"] = cb["consecutive_loss_days"]
        if cb["tripped"]:
            result.update({
                "signal": 0,
                "action": "熔断观望",
                "reason": cb["reason"],
            })
            return result

        # 3. Kelly 仓位建议
        kelly_shares = 0
        if self.use_kelly and win_rate > 0 and profit_loss_ratio > 0:
            kelly_shares = self.calculate_kelly_size(
                win_rate, profit_loss_ratio, capital, price
            )
            result["kelly_shares"] = kelly_shares
            result["kelly_amount"] = round(kelly_shares * price, 2)
        else:
            # 即使不用 Kelly，也做仓位上限校验（使用 max_position_pct 计算建议仓位）
            if self.position and price > 0:
                proposed = int(self.max_position_pct * capital / price)
                result["kelly_shares"] = self.validate_position_size(
                    proposed, price, capital
                )
                result["kelly_amount"] = round(result["kelly_shares"] * price, 2)

        # 4. 原始信号透传
        result["signal"] = signal
        if signal == 1 and (not self.position or self.position.shares == 0):
            shares_str = f"，建议 {kelly_shares} 股" if kelly_shares > 0 else ""
            result.update({
                "action": "买入",
                "reason": f"策略看涨信号{shares_str}（止损位 {stop_price:.2f}）",
            })
        elif signal == 0 and self.position and self.position.shares > 0:
            result.update({
                "action": "卖出",
                "reason": "策略看跌信号",
            })
        elif signal == 1 and self.position and self.position.shares > 0:
            result.update({
                "action": "持有",
                "reason": f"策略看涨，继续持有（止损位 {stop_price:.2f}）",
            })
        else:
            result.update({
                "action": "观望",
                "reason": "策略看跌，维持空仓",
            })

        # ── 可选：通过 OMS 提交订单 ──────────────────────────────
        if oms is not None and result["action"] in ("买入", "卖出", "止损卖出"):
            try:
                shares_to_trade = result.get("kelly_shares", 0)
                if shares_to_trade <= 0 and self.position:
                    shares_to_trade = self.position.shares
                if shares_to_trade > 0:
                    oms_action = "卖出" if result["action"] in ("卖出", "止损卖出") else "买入"
                    oms.submit_order(
                        ticker=ticker or "UNKNOWN",
                        action=oms_action,
                        shares=shares_to_trade,
                        price=price,
                        note=result.get("reason", ""),
                    )
            except Exception as _e:
                import logging as _logging
                _logging.getLogger(__name__).warning(f"OMS 下单失败: {_e}")

        return result

    # ── 原 get_recommendation（向后兼容） ─────────────────────────

    def get_recommendation(self, signal: int, predicted_return: float) -> dict:
        """
        根据信号和预测收益率获取交易建议（保持向后兼容）。
        若已有持仓和 ATR 数据，优先调用 apply_risk_controls。
        """
        if self.position is None:
            return {
                "action": "无法判断",
                "reason": "未设置持仓数据",
                "shares": 0,
                "amount": 0,
                "stop_price": 0.0,
                "kelly_shares": 0,
                "circuit_breaker": False,
            }

        shares        = self.position.shares
        avg_cost      = self.position.avg_cost
        current_price = self.position.current_price
        profit        = self.position.profit
        profit_pct    = self.position.profit_pct

        if signal == 1 and shares == 0:
            action = "买入"
            reason = f"策略看涨信号，预计上涨 {predicted_return*100:.2f}%"
        elif signal == 0 and shares > 0:
            action = "卖出"
            reason = f"策略看跌信号，预计下跌 {abs(predicted_return)*100:.2f}%"
        elif signal == 1 and shares > 0:
            action = "持有"
            reason = "策略看涨信号，继续持有"
        else:
            action = "观望"
            reason = "策略看跌信号，维持空仓"

        return {
            "action":           action,
            "reason":           reason,
            "shares":           shares,
            "avg_cost":         avg_cost,
            "current_price":    current_price,
            "profit":           profit,
            "profit_pct":       profit_pct,
            "signal":           signal,
            "predicted_return": predicted_return,
            "stop_price":       0.0,
            "kelly_shares":     0,
            "circuit_breaker":  False,
        }

    # ── 报告生成 ──────────────────────────────────────────────────

    def generate_report(self, recommendations: list) -> str:
        """生成持仓报告（含风控状态）"""
        if self.position is None:
            return "## 持仓状态\n\n未设置持仓数据\n"

        p   = self.position
        rec = recommendations[0] if recommendations else {}

        stop_price  = rec.get("stop_price", 0.0)
        kelly_sh    = rec.get("kelly_shares", 0)
        kelly_amt   = rec.get("kelly_amount", 0.0)
        cb_tripped  = rec.get("circuit_breaker", False)
        loss_days   = rec.get("consecutive_loss_days", 0)

        cb_str      = f"⚠️ 已触发（连续亏损 {loss_days} 天）" if cb_tripped else f"✅ 正常（连续亏损 {loss_days} 天）"
        stop_str    = f"{stop_price:.2f}" if stop_price > 0 else "—"
        kelly_str   = f"{kelly_sh} 股（≈ {kelly_amt:.0f} 元）" if kelly_sh > 0 else "—"

        md = f"""## 持仓状态

| 项目 | 值 |
|------|-----|
| 持股数量 | {p.shares} 股 |
| 平均成本 | {p.avg_cost:.2f} 元 |
| 当前价格 | {p.current_price:.2f} 元 |
| 市值 | {p.market_value:.2f} 元 |
| 持仓成本 | {p.cost_basis:.2f} 元 |
| 盈亏金额 | {p.profit:+.2f} 元 |
| 盈亏比例 | {p.profit_pct:+.2f}% |

## 风控状态

| 项目 | 值 |
|------|-----|
| 建议止损价 | {stop_str} |
| Kelly 建议股数 | {kelly_str} |
| 熔断状态 | {cb_str} |

## 交易建议

| 操作 | 原因 |
|------|------|
| **{rec.get('action', 'N/A')}** | {rec.get('reason', 'N/A')} |

"""
        return md


# ──────────────────────────────────────────────────────────────────
#  辅助函数
# ──────────────────────────────────────────────────────────────────

def load_position_from_config(config: dict) -> Optional[Position]:
    """从配置加载持仓。
    ⚠️ current_price 初始化为 None，调用方**必须**在使用 market_value / profit 前赋值为最新收盘价：
        position.current_price = last_close
    """
    shares   = config.get("position_shares", 0)
    avg_cost = config.get("position_avg_cost", 0)
    if shares > 0 and avg_cost > 0:
        return Position(
            shares=shares,
            avg_cost=avg_cost,
            current_price=None,  # 未初始化；调用方必须更新为最新价格后再使用
        )
    return None


def calc_atr(data: pd.DataFrame, period: int = 14) -> float:
    """
    从历史 K 线数据计算最新 ATR 值（不依赖 ta 库）。
    Args:
        data:   包含 high / low / close 列的 DataFrame
        period: ATR 周期（默认 14）
    Returns:
        最新一期 ATR 值；数据不足时返回 0.0
    """
    if len(data) < period + 1:
        return 0.0
    # 兼容大小写列名（如 High/high, Low/low, Close/close）
    col_map = {}
    for col in data.columns:
        if col.lower() in ("high", "low", "close"):
            col_map[col.lower()] = col
    missing = {"high", "low", "close"} - set(col_map.keys())
    if missing:
        return 0.0
    df = data[[col_map["high"], col_map["low"], col_map["close"]]].copy().tail(period + 10)
    df.columns = ["high", "low", "close"]
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"]  - df["close"].shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_series = tr.rolling(period).mean()
    val = atr_series.iloc[-1]
    return float(val) if not math.isnan(val) else 0.0


def calc_atr_series(data: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    从历史 K 线数据计算完整 ATR Series（不依赖 ta 库）。
    Args:
        data:   包含 High / Low / Close 列的 DataFrame
        period: ATR 周期（默认 14）
    Returns:
        与 data 索引对齐的 ATR Series；数据不足的早期值为 NaN
    """
    col_map = {}
    for col in data.columns:
        if col.lower() in ("high", "low", "close"):
            col_map[col.lower()] = col
    missing = {"high", "low", "close"} - set(col_map.keys())
    if missing:
        return pd.Series(dtype=float, index=data.index)
    df = data[[col_map["high"], col_map["low"], col_map["close"]]].copy()
    df.columns = ["high", "low", "close"]
    tr1 = df["high"] - df["low"]
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"]  - df["close"].shift(1)).abs()
    tr  = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def simulate_atr_stoploss(
    data: pd.DataFrame,
    signal: pd.Series,
    atr_period: int = 14,
    atr_multiplier: float = 2.0,
    trailing: bool = True,
    cooldown_bars: int = 0,
) -> pd.Series:
    """
    回测 ATR 止损模拟（纯函数，无副作用）。

    逐 bar 扫描策略信号，当持仓期间收盘价跌破 ATR 止损位时将信号强制置 0（平仓），
    并可选进入冷却期（模拟熔断暂停）。

    Args:
        data:            含 High / Low / Close 的 OHLCV DataFrame
        signal:          原始策略信号 Series (1=多头持仓, 0=空仓)，索引须与 data 对齐
        atr_period:      ATR 计算周期（默认 14）
        atr_multiplier:  止损价 = max(峰值, 入场价) - multiplier × ATR（默认 2.0）
        trailing:        True=移动止损（跟踪峰值）；False=固定入场价止损
        cooldown_bars:   止损触发后冷却 bar 数（0=无冷却）

    Returns:
        修正后的信号 Series（与输入索引完全一致），止损触发时为 0
    """
    # 预计算 ATR
    atr_series = calc_atr_series(data, atr_period)

    # 对齐索引
    sig = pd.Series(signal, dtype=float).reindex(data.index).fillna(0)
    close = data[next(c for c in data.columns if c.lower() == "close")]

    out = sig.copy()

    in_position   = False
    entry_price   = 0.0
    peak_price    = 0.0
    cooldown_left = 0

    for i, idx in enumerate(data.index):
        raw_sig  = int(sig.iloc[i])
        c        = float(close.iloc[i])
        atr_val  = float(atr_series.iloc[i]) if not pd.isna(atr_series.iloc[i]) else 0.0

        # 冷却期内强制空仓
        if cooldown_left > 0:
            out.iloc[i] = 0
            cooldown_left -= 1
            in_position = False
            continue

        if not in_position:
            if raw_sig == 1:
                # 新开仓
                in_position = True
                entry_price = c
                peak_price  = c
                out.iloc[i] = 1
            else:
                out.iloc[i] = 0
        else:
            # 持仓中
            if raw_sig == 0:
                # 策略自行平仓
                in_position = False
                out.iloc[i] = 0
            else:
                # 更新峰值
                if trailing:
                    peak_price = max(peak_price, c)

                # 计算止损价：取移动止损 vs 入场止损的较大值（更保守）
                if atr_val > 0:
                    trailing_stop = peak_price - atr_multiplier * atr_val
                    entry_stop    = entry_price - atr_multiplier * atr_val
                    stop_price    = max(trailing_stop, entry_stop) if trailing else entry_stop
                    if c < stop_price:
                        # 触发止损，强制平仓
                        out.iloc[i]   = 0
                        in_position   = False
                        cooldown_left = cooldown_bars
                        continue

                out.iloc[i] = 1

    return out.astype(int)


# ──────────────────────────────────────────────────────────────────
#  单元测试（python3 position_manager.py）
# ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("PositionManager 风控单元测试")
    print("=" * 60)

    rc = {
        "use_atr_stop":             True,
        "atr_multiplier":           2.0,
        "use_kelly":                True,
        "max_position_pct":         0.25,
        "daily_loss_limit":         0.05,
        "max_consecutive_loss_days": 3,
    }
    pm = PositionManager(portfolio_value=100_000, risk_config=rc)
    pm.set_position(shares=200, avg_cost=60.0, current_price=65.0)

    # ── ATR 止损测试 ──
    print("\n[1] ATR 止损")
    atr = 1.5
    peak = 68.0
    close_ok  = 65.5   # 68 - 2*1.5 = 65 → 65.5 未触发
    close_bad = 64.5   # 64.5 < 65 → 触发
    print(f"  close={close_ok}  → 触发止损: {pm.check_atr_stop(close_ok,  60, peak, atr)}")   # False
    print(f"  close={close_bad} → 触发止损: {pm.check_atr_stop(close_bad, 60, peak, atr)}")   # True

    # ── Kelly 仓位测试 ──
    print("\n[2] Kelly 仓位")
    shares = pm.calculate_kelly_size(win_rate=0.55, profit_loss_ratio=1.8,
                                     capital=100_000, current_price=65.0)
    print(f"  胜率55%，盈亏比1.8 → Kelly 建议股数: {shares}")

    # ── 熔断测试 ──
    print("\n[3] 熔断机制")
    cb1 = pm.check_circuit_breaker(-0.06, trade_date="2099-01-01")  # 单日超限
    cb2 = pm.check_circuit_breaker(-0.03, trade_date="2099-01-02")  # 第1天亏损
    cb3 = pm.check_circuit_breaker(-0.03, trade_date="2099-01-03")  # 第2天
    cb4 = pm.check_circuit_breaker(-0.03, trade_date="2099-01-04")  # 第3天 → 触发
    print(f"  单日-6%   → tripped={cb1['tripped']}  reason={cb1['reason']}")
    print(f"  连续第1天 → tripped={cb2['tripped']}  days={cb2['consecutive_loss_days']}")
    print(f"  连续第2天 → tripped={cb3['tripped']}  days={cb3['consecutive_loss_days']}")
    print(f"  连续第3天 → tripped={cb4['tripped']}  days={cb4['consecutive_loss_days']}")

    # ── 统一风控入口测试 ──
    print("\n[4] apply_risk_controls（止损触发）")
    res = pm.apply_risk_controls(
        signal=1, price=64.5, atr=atr,
        entry_price=60.0, peak_price=peak,
        today_pnl_pct=-0.01, capital=100_000,
        win_rate=0.55, profit_loss_ratio=1.8,
        trade_date="2099-02-01",
    )
    print(f"  action={res['action']}  reason={res['reason']}")

    print("\n[5] apply_risk_controls（正常买入信号）")
    pm2 = PositionManager(portfolio_value=100_000, risk_config=rc)
    pm2.set_position(shares=0, avg_cost=0, current_price=65.0)
    res2 = pm2.apply_risk_controls(
        signal=1, price=65.0, atr=atr,
        entry_price=65.0, peak_price=65.0,
        today_pnl_pct=0.01, capital=100_000,
        win_rate=0.55, profit_loss_ratio=1.8,
        trade_date="2099-02-01",
    )
    print(f"  action={res2['action']}  kelly_shares={res2['kelly_shares']}  stop_price={res2['stop_price']}")

    # ── simulate_atr_stoploss 测试 ──
    print("\n[6] simulate_atr_stoploss（先涨后暴跌触发止损）")
    import numpy as np
    dates = pd.date_range('2025-01-01', periods=60, freq='B')
    # 价格：先从100涨到120（前30天），再暴跌到95（后30天）
    prices_up   = np.linspace(100, 120, 30)
    prices_down = np.linspace(119, 95, 30)
    closes      = np.concatenate([prices_up, prices_down])
    highs       = closes * 1.005
    lows        = closes * 0.995
    sim_data    = pd.DataFrame({'Close': closes, 'High': highs, 'Low': lows}, index=dates)
    # 全程持仓信号（持仓中不主动平仓）
    raw_signal  = pd.Series(1, index=dates)
    modified    = simulate_atr_stoploss(sim_data, raw_signal, atr_period=5, atr_multiplier=2.0)
    stop_bars   = (modified == 0).sum()
    first_stop  = modified[modified == 0].index[0].date() if stop_bars > 0 else None
    print(f"  止损触发 bar 数: {stop_bars}（期望 > 0）")
    print(f"  首次止损日期:    {first_stop}（期望在下跌阶段）")
    assert stop_bars > 0, "❌ 未触发止损！"
    assert first_stop >= dates[30].date(), "❌ 止损触发时间异常（应在下跌阶段）"
    print("  ✅ 止损触发正确")

    print("\n[7] simulate_atr_stoploss（冷却期测试）")
    modified_cd = simulate_atr_stoploss(sim_data, raw_signal, atr_period=5,
                                         atr_multiplier=2.0, cooldown_bars=5)
    # 冷却期后不应立即重新开仓
    if first_stop:
        stop_idx = list(dates).index(pd.Timestamp(first_stop))
        cooldown_end = stop_idx + 5
        if cooldown_end < len(dates):
            for ci in range(stop_idx, min(cooldown_end + 1, len(dates))):
                assert modified_cd.iloc[ci] == 0, f"❌ 冷却期 bar {ci} 信号未为 0！"
    print("  ✅ 冷却期信号正确")

    print("\n✅ 测试完成")
