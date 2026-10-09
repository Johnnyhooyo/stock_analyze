"""将收盘后的纸面信号延至下一条可用 K 线开盘价结算。"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import date
from pathlib import Path

import pandas as pd

from engine.paper_trader import ExecutedTrade, PaperTradingEngine
from engine.position_analyzer import RecommendationResult
from log_config import get_logger

logger = get_logger(__name__)
_PENDING_FILE = Path(__file__).parent.parent / "data" / "logs" / "pending_paper_signals.json"
_TRADE_ACTIONS = {"买入", "卖出", "止损卖出"}


def _read_pending(path: Path) -> list[dict]:
    if not path.exists():
        return []
    content = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(content, list):
        raise ValueError(f"待结算纸面信号格式错误: {path}")
    return content


def _write_pending(path: Path, pending: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(pending, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def queue_paper_signals(results: list, path: Path = _PENDING_FILE) -> int:
    """只保存可交易建议；同一标的未结算时不重复排队。"""
    pending = _read_pending(path)
    waiting = {str(item["ticker"]).upper() for item in pending}
    added = 0
    for result in results:
        ticker = result.ticker.upper()
        if result.action not in _TRADE_ACTIONS or ticker in waiting:
            continue
        pending.append({
            "ticker": ticker,
            "action": result.action,
            "reason": result.reason,
            "confidence_pct": float(result.confidence_pct),
            "signal_date": result.last_date,
        })
        waiting.add(ticker)
        added += 1
    if added:
        _write_pending(path, pending)
    return added


def cancel_pending_buys(path: Path = _PENDING_FILE) -> int:
    """组合止损触发后撤销尚未结算的纸面买入信号。"""
    pending = _read_pending(path)
    remaining = [item for item in pending if item.get("action") != "买入"]
    canceled = len(pending) - len(remaining)
    if canceled:
        _write_pending(path, remaining)
    return canceled


def _first_next_open(data: pd.DataFrame, signal_date: str) -> tuple[str, float] | None:
    if data is None or data.empty or "Open" not in data.columns:
        return None
    dates = pd.to_datetime(data.index).date
    future = data.loc[dates > date.fromisoformat(signal_date)].sort_index()
    if future.empty:
        return None
    first = future.iloc[0]
    opening = float(first["Open"])
    if not pd.notna(opening) or opening <= 0:
        return None
    return str(pd.Timestamp(future.index[0]).date()), opening


def _valuation_prices(data_mgr, portfolio_state, period: str, fill_date: str) -> dict[str, float]:
    """按结算日开盘价估值其他持仓，避免用成本价高估可买额度。"""
    prices: dict[str, float] = {}
    target = date.fromisoformat(fill_date)
    for ticker in portfolio_state.held_tickers():
        try:
            data = data_mgr.load(ticker, period=period)
            dates = pd.to_datetime(data.index).date
            on_date = data.loc[dates == target]
            if not on_date.empty and "Open" in on_date.columns:
                price = float(on_date.iloc[0]["Open"])
            else:
                previous = data.loc[dates < target]
                price = float(previous.iloc[-1]["Close"]) if not previous.empty else 0.0
            if pd.notna(price) and price > 0:
                prices[ticker] = price
        except (FileNotFoundError, KeyError, IndexError, ValueError) as exc:
            logger.warning("纸面估值缺少行情: %s %s", ticker, exc)
    return prices


def settle_paper_signals(
    config: dict,
    portfolio_state,
    data_mgr,
    period: str,
    path: Path = _PENDING_FILE,
    trader: PaperTradingEngine | None = None,
) -> list[ExecutedTrade]:
    """下一根 K 线出现后，按该 K 线开盘价结算此前保存的信号。"""
    pending = _read_pending(path)
    if not pending:
        return []
    ready: dict[str, list[tuple[dict, float]]] = defaultdict(list)
    remaining: list[dict] = []
    for item in pending:
        try:
            data = data_mgr.load(item["ticker"], period=period)
            next_open = _first_next_open(data, item["signal_date"])
        except (FileNotFoundError, KeyError, ValueError) as exc:
            logger.warning("纸面信号等待有效行情: %s %s", item["ticker"], exc)
            next_open = None
        if next_open is None:
            remaining.append(item)
        else:
            fill_date, opening = next_open
            ready[fill_date].append((item, opening))

    if not ready:
        return []
    trader = trader or PaperTradingEngine(config)
    trades: list[ExecutedTrade] = []
    for fill_date in sorted(ready):
        results = [
            RecommendationResult(
                ticker=item["ticker"],
                last_date=item["signal_date"],
                last_close=opening,
                action=item["action"],
                reason=item["reason"],
                confidence_pct=item["confidence_pct"],
            )
            for item, opening in ready[fill_date]
        ]
        prices = _valuation_prices(data_mgr, portfolio_state, period, fill_date)
        prices.update({r.ticker: r.last_close for r in results})
        trades.extend(trader.execute(results, portfolio_state, fill_date, valuation_prices=prices))
        portfolio_state.save()
    _write_pending(path, remaining)
    return trades
