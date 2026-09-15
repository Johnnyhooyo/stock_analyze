"""纸面交易执行器：把每日建议转成可持久的模拟持仓和资产记录。"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from log_config import get_logger
from oms import PaperOMS
from engine.hk_fees import affordable_hk_shares, calculate_hk_stock_fees
from engine.portfolio_state import holding_days

logger = get_logger(__name__)

_ROOT = Path(__file__).parent.parent


@dataclass
class ExecutedTrade:
    trade_date: str
    ticker: str
    action: str
    shares: int
    price: float
    gross_amount: float
    fee: float
    realized_pnl: float = 0.0
    reason: str = ""
    avg_cost: float = 0.0
    buy_fee: float = 0.0
    fee_breakdown: dict = field(default_factory=dict)
    entry_date: str = ""
    holding_days: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class PaperTradingEngine:
    """先卖后买的组合级纸面交易执行器。"""

    def __init__(self, config: dict, orders_file: Path | None = None, assets_file: Path | None = None):
        self.config = config
        cfg = config.get("paper_trading", {})
        risk = config.get("risk_management", {})
        portfolio_risk = config.get("portfolio_risk", {})
        self.enabled = bool(cfg.get("enabled", False))
        self.min_confidence = float(cfg.get("min_confidence", 0.55))
        self.lot_size = max(1, int(cfg.get("lot_size", 1)))
        self.board_lot_cfg = config.get("board_lots", {})
        self.max_new_positions = max(0, int(cfg.get("max_new_positions_per_day", 3)))
        self.reserve_cash_pct = min(max(float(cfg.get("reserve_cash_pct", 0.05)), 0.0), 1.0)
        self.max_position_pct = min(max(float(risk.get("max_position_pct", 0.25)), 0.0), 1.0)
        self.max_total_position_pct = min(
            max(float(portfolio_risk.get("max_position_ratio", 0.80)), 0.0), 1.0
        )
        self.slippage_rate = max(0.0, float(config.get("slippage", 0.0)))
        self.assets_file = assets_file or (_ROOT / "data" / "logs" / "asset_history.jsonl")
        self.oms = PaperOMS(orders_file=orders_file)

    def _round_lot(self, shares: int) -> int:
        return max(0, shares // self.lot_size * self.lot_size)

    def _board_lot(self, ticker: str) -> int | None:
        if not self.board_lot_cfg.get("enabled", False):
            return self.lot_size
        overrides = {
            str(key).upper(): int(value)
            for key, value in self.board_lot_cfg.get("overrides", {}).items()
        }
        if ticker.upper() in overrides:
            return max(1, overrides[ticker.upper()])
        from data.hk_board_lots import get_board_lot

        lot_size = get_board_lot(ticker)
        if lot_size is None and not self.board_lot_cfg.get("required", False):
            return self.lot_size
        return lot_size

    @staticmethod
    def _is_flat(portfolio_state, ticker: str) -> bool:
        pos = portfolio_state.get_position(ticker)
        return pos is None or not pos.has_position

    def _append_asset_snapshot(self, snapshot: dict) -> None:
        self.assets_file.parent.mkdir(parents=True, exist_ok=True)
        rows: list[dict] = []
        if self.assets_file.exists():
            for line in self.assets_file.read_text(encoding="utf-8").splitlines():
                try:
                    row = json.loads(line)
                    if row.get("trade_date") != snapshot["trade_date"]:
                        rows.append(row)
                except (TypeError, ValueError):
                    continue
        rows.append(snapshot)
        tmp = self.assets_file.with_suffix(self.assets_file.suffix + ".tmp")
        tmp.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        tmp.replace(self.assets_file)

    def execute(self, results: list, portfolio_state, trade_date: str) -> list[ExecutedTrade]:
        if not self.enabled:
            return []

        by_ticker = {r.ticker.upper(): r for r in results}
        prices = {ticker: float(r.last_close) for ticker, r in by_ticker.items() if r.last_close > 0}
        portfolio_state.mark_to_market(prices, trade_date)
        trades: list[ExecutedTrade] = []

        # 卖出优先：止损或看跌时一次性卖出全部实际持仓。
        for r in results:
            if r.action not in ("卖出", "止损卖出"):
                continue
            pos = portfolio_state.get_position(r.ticker)
            if pos is None or not pos.has_position or r.last_close <= 0:
                continue
            shares = pos.shares
            avg_cost = pos.avg_cost
            allocated_buy_fee = pos.buy_fees
            entry_date = pos.entry_date
            days_held = holding_days(entry_date, trade_date)
            fill_price = float(r.last_close) * (1.0 - self.slippage_rate)
            gross = shares * fill_price
            fee_detail = calculate_hk_stock_fees(gross, r.ticker, self.config)
            fee = fee_detail.total
            order = self.oms.submit_order(r.ticker, "卖出", shares, fill_price, note=r.reason)
            if order.status != "submitted":
                logger.warning("纸面卖出被拒绝: %s %s", r.ticker, order.message)
                continue
            realized = portfolio_state.sell(r.ticker, shares, fill_price, fee)
            trades.append(ExecutedTrade(
                trade_date, r.ticker, "卖出", shares, fill_price, gross,
                fee, realized, r.reason, avg_cost, allocated_buy_fee,
                fee_detail.to_dict(), entry_date, days_held,
            ))
            r.has_position = False
            r.shares = 0
            r.avg_cost = 0.0
            r.market_value = 0.0
            r.profit = 0.0
            r.profit_pct = 0.0

        portfolio_state.mark_to_market(prices, trade_date)
        total_assets = float(portfolio_state.portfolio_value)
        min_cash = total_assets * self.reserve_cash_pct
        held_market_value = total_assets - float(portfolio_state.cash)
        portfolio_room = max(0.0, total_assets * self.max_total_position_pct - held_market_value)

        candidates = sorted(
            (
                r for r in results
                if r.action == "买入"
                and r.last_close > 0
                and r.confidence_pct >= self.min_confidence
                and self._is_flat(portfolio_state, r.ticker)
            ),
            key=lambda r: (-r.confidence_pct, r.ticker),
        )
        if self.max_new_positions:
            candidates = candidates[:self.max_new_positions]
        else:
            candidates = []

        bought_tickers: set[str] = set()
        for r in candidates:
            available_cash = max(0.0, float(portfolio_state.cash) - min_cash)
            budget = min(total_assets * self.max_position_pct, portfolio_room, available_cash)
            fill_price = float(r.last_close) * (1.0 + self.slippage_rate)
            lot_size = self._board_lot(r.ticker)
            if lot_size is None:
                r.action = "观望"
                r.signal = 0
                r.reason = "纸面交易：缺少港交所每手股数，拒绝按1股模拟买入"
                continue
            shares = affordable_hk_shares(
                budget, fill_price, r.ticker, self.config, lot_size
            )
            if shares <= 0:
                r.action = "观望"
                r.signal = 0
                r.reason = "纸面交易：可用资金或组合仓位不足"
                continue
            gross = shares * fill_price
            fee_detail = calculate_hk_stock_fees(gross, r.ticker, self.config)
            fee = fee_detail.total
            order = self.oms.submit_order(r.ticker, "买入", shares, fill_price, note=r.reason)
            if order.status != "submitted":
                logger.warning("纸面买入被拒绝: %s %s", r.ticker, order.message)
                r.action = "观望"
                r.signal = 0
                r.reason = f"纸面交易：买入订单被拒绝（{order.message}）"
                continue
            portfolio_state.buy(
                r.ticker, shares, fill_price, fee, trade_date=trade_date
            )
            pos = portfolio_state.get_position(r.ticker)
            trades.append(ExecutedTrade(
                trade_date, r.ticker, "买入", shares, fill_price, gross,
                fee, 0.0, r.reason, fill_price, fee, fee_detail.to_dict(),
                pos.entry_date, holding_days(pos.entry_date, trade_date),
            ))
            bought_tickers.add(r.ticker.upper())
            portfolio_room = max(0.0, portfolio_room - gross - fee)
            r.has_position = True
            r.shares = pos.shares
            r.avg_cost = pos.avg_cost
            r.buy_fees = pos.buy_fees
            r.entry_date = pos.entry_date
            r.holding_days = holding_days(pos.entry_date, trade_date)
            r.market_value = pos.shares * float(r.last_close)
            r.profit = r.market_value - pos.shares * pos.avg_cost
            r.profit_pct = r.profit / (pos.shares * pos.avg_cost) * 100 if pos.avg_cost > 0 else 0.0
            r.peak_price = max(r.peak_price, fill_price)
            r.kelly_shares = shares
            r.kelly_amount = gross + fee

        # 没有真正成交的买入建议不应在报告中继续标为“买入”。
        for r in results:
            if r.action != "买入" or r.ticker.upper() in bought_tickers:
                continue
            r.action = "观望"
            r.signal = 0
            if r.confidence_pct < self.min_confidence:
                r.reason = (
                    f"纸面交易：置信度 {r.confidence_pct:.0%} "
                    f"低于买入阈值 {self.min_confidence:.0%}"
                )
            else:
                r.reason = "纸面交易：达到当日新建仓上限或资金/仓位不足"

        portfolio_state.mark_to_market(prices, trade_date)
        market_value = float(portfolio_state.portfolio_value) - float(portfolio_state.cash)
        unrealized_pnl = sum(
            pos.shares * (prices.get(ticker, pos.avg_cost) - pos.avg_cost)
            for ticker, pos in portfolio_state.positions.items()
            if pos.has_position
        )
        initial = float(portfolio_state.initial_capital or portfolio_state.portfolio_value)
        snapshot = {
            "trade_date": trade_date,
            "cash": round(float(portfolio_state.cash), 2),
            "market_value": round(market_value, 2),
            "total_assets": round(float(portfolio_state.portfolio_value), 2),
            "realized_pnl": round(float(portfolio_state.realized_pnl), 2),
            "unrealized_pnl": round(unrealized_pnl, 2),
            "total_return_pct": round((portfolio_state.portfolio_value / initial - 1.0) * 100, 4) if initial > 0 else 0.0,
            "held_count": len(portfolio_state.held_tickers()),
            "trades": [t.to_dict() for t in trades],
        }
        self._append_asset_snapshot(snapshot)
        return trades
