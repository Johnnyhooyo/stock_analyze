#!/usr/bin/env python3
"""
daily_run.py — 每日量化推荐引擎入口

职责：
  1. 数据刷新（下载最新行情）
  2. 因子新鲜度检查（可选重新训练）
  3. 多股票信号生成 + 持仓风控分析
  4. 汇总每日操作建议
  5. 发送飞书通知 + 保存报告

典型用法：
  python3 daily_run.py                          # 使用 portfolio.yaml 中的观察列表
  python3 daily_run.py --tickers 0700.HK 0005.HK  # 指定股票
  python3 daily_run.py --watchlist hk           # 分析全部港股
  python3 daily_run.py --skip-notify            # 不发送飞书通知
  python3 daily_run.py --retrain                # 强制重新训练因子
  python3 daily_run.py --dry-run                # 打印建议但不保存状态

Cron 参考（每个交易日 18:30 HKT 运行）：
  30 18 * * 1-5 cd /path/to/stock_analyze && python3 daily_run.py >> logs/daily.log 2>&1
"""

from __future__ import annotations

import argparse
import sys
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from config_loader import load_config
from log_config import get_logger
from data.pnl_tracker import PnLTracker

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════
#  因子新鲜度检查
# ══════════════════════════════════════════════════════════════════

def _check_factor_freshness(factors_dir: Path, min_age_days: int = 7) -> bool:
    """
    检查 factors_dir 及其所有直接子目录中，最新因子是否在 min_age_days 天内。
    只要有任一目录的因子还新鲜就返回 True。
    """
    import joblib as _jl

    def _newest_in(d: Path):
        cands = sorted(
            d.glob("factor_*.pkl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        return cands[0] if cands else None

    dirs = [factors_dir]
    if factors_dir.is_dir():
        dirs += [
            sd for sd in factors_dir.iterdir()
            if sd.is_dir() and not sd.name.startswith(".")
        ]

    newest = None
    newest_mtime = 0
    for d in dirs:
        p = _newest_in(d)
        if p is None:
            continue
        mtime = p.stat().st_mtime
        if mtime > newest_mtime:
            newest_mtime = mtime
            newest = p

    if newest is None:
        return False

    try:
        art = _jl.load(newest)
        saved_at_str = art.get("saved_at", "")
        if saved_at_str:
            age_days = (datetime.now() - datetime.fromisoformat(saved_at_str)).days
            return age_days <= min_age_days
    except Exception:
        pass
    age_days = (datetime.now() - datetime.fromtimestamp(newest.stat().st_mtime)).days
    return age_days <= min_age_days


# ══════════════════════════════════════════════════════════════════
#  单股票分析（线程安全）
# ══════════════════════════════════════════════════════════════════

def _analyze_one_ticker(
    ticker: str,
    config: dict,
    analyzer,       # PositionAnalyzer 实例（共享只读）
    portfolio_state,  # PortfolioState 实例
    data_mgr,       # DataManager 实例
    period: str = "5y",
) -> Optional[object]:
    """
    下载 / 加载单只股票数据，运行 PositionAnalyzer，返回 RecommendationResult。
    线程安全（只读共享状态）。
    """

    try:
        # 加载历史数据
        try:
            df = data_mgr.load(ticker, period=period)
        except FileNotFoundError:
            df = None
        if df is None or df.empty:
            df, _ = data_mgr.download(ticker, period=period)
        if df is None or df.empty:
            logger.warning("%s: 无法获取历史数据，跳过", ticker, extra={"ticker": ticker})
            return None

        # 获取持仓状态
        pos = portfolio_state.get_position(ticker)

        # 运行分析
        result = analyzer.analyze(ticker, df, pos)
        return result

    except Exception as e:
        logger.error("%s: 分析失败 — %s", ticker, e, extra={"ticker": ticker}, exc_info=("--debug" in sys.argv))
        return None


# ══════════════════════════════════════════════════════════════════
#  汇总报告生成
# ══════════════════════════════════════════════════════════════════

def _build_daily_report(
    results: list,
    portfolio_value: float,
    run_date: str,
    market_is_open: bool,
    config: dict = None,
    screener_results: list = None,
    sector_ranking: list = None,
) -> dict:
    """
    将所有 RecommendationResult 汇总为 daily_report 字典，
    供飞书通知和 Markdown 存储使用。

    Args:
        results: PositionAnalyzer.analyze() 返回的 RecommendationResult 列表
        portfolio_value: 组合总资产
        run_date: 运行日期
        market_is_open: 今日是否为港股交易日
        config: 配置字典（用于选股权重）
        screener_results: StockScreener.screen() 返回的 ScreenerResult 列表
        sector_ranking: StockScreener.sector_ranking() 返回的板块排名列表
    """
    recommendations = []
    total_market_value = 0.0
    total_cost_basis = 0.0
    held_count = 0
    buy_signals: list[str] = []
    sell_signals: list[str] = []

    for r in results:
        if r is None:
            continue
        agg = r.agg_signal
        rec_entry = {
            "ticker": r.ticker,
            "last_date": r.last_date,
            "last_close": r.last_close,
            "action": r.action,
            "reason": r.reason,
            "signal": r.signal,
            "action_emoji": r.action_emoji,
            "has_position": r.has_position,
            "shares": r.shares,
            "avg_cost": r.avg_cost,
            "market_value": r.market_value,
            "profit": r.profit,
            "profit_pct": r.profit_pct,
            "stop_price": r.stop_price,
            "kelly_shares": r.kelly_shares,
            "kelly_amount": r.kelly_amount,
            "circuit_breaker": r.circuit_breaker,
            "consecutive_loss_days": r.consecutive_loss_days,
            "confidence_pct": r.confidence_pct,
            "confidence_label": r.confidence_label,
            "bullish_count": agg.bullish_count if agg else 0,
            "bearish_count": agg.bearish_count if agg else 0,
            "total_strategies": agg.total_strategies if agg else 0,
            "top_strategy": agg.top_strategy if agg else "",
            "ml_signal": agg.ml_signal if agg else None,
            "rule_signal": agg.rule_signal if agg else None,
            "sentiment": r.sentiment,
            "price_lo_1d": r.price_lo_1d,
            "price_hi_1d": r.price_hi_1d,
            "atr": r.atr,
            "risk_flags": r.risk_flags,
        }
        recommendations.append(rec_entry)

        if r.has_position:
            total_market_value += r.market_value
            total_cost_basis += r.avg_cost * r.shares
            held_count += 1

        if r.action in ("买入",):
            buy_signals.append(r.ticker)
        elif r.action in ("卖出", "止损卖出"):
            sell_signals.append(r.ticker)

    total_pnl = total_market_value - total_cost_basis
    total_pnl_pct = (total_pnl / total_cost_basis * 100) if total_cost_basis > 0 else 0.0
    cash_value = portfolio_value - total_market_value
    cash_pct = (cash_value / portfolio_value * 100) if portfolio_value > 0 else 100.0

    return {
        "run_date": run_date,
        "market_is_open": market_is_open,
        "portfolio_value": portfolio_value,
        "total_market_value": total_market_value,
        "total_cost_basis": total_cost_basis,
        "total_pnl": total_pnl,
        "total_pnl_pct": total_pnl_pct,
        "cash_value": cash_value,
        "cash_pct": cash_pct,
        "held_count": held_count,
        "total_tickers": len(recommendations),
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "recommendations": recommendations,
        "screener_results": [
            r.to_dict() if hasattr(r, "to_dict") else r for r in (screener_results or [])
        ],
        "screener_weights": {
            "momentum": config.get("screener", {}).get("weight_momentum", 0.35),
            "trend": config.get("screener", {}).get("weight_trend", 0.35),
            "volume": config.get("screener", {}).get("weight_volume", 0.30),
        },
        "sector_ranking": sector_ranking or [],
    }


# ══════════════════════════════════════════════════════════════════
#  Markdown 报告
# ══════════════════════════════════════════════════════════════════

def _build_markdown_report(daily_report: dict) -> str:
    """生成每日操作建议的 Markdown 报告。"""
    run_date = daily_report["run_date"]
    pv = daily_report["portfolio_value"]
    mv = daily_report["total_market_value"]
    pnl = daily_report["total_pnl"]
    pnl_pct = daily_report["total_pnl_pct"]
    cash = daily_report["cash_value"]
    cash_pct = daily_report["cash_pct"]
    buy_sigs = daily_report["buy_signals"]
    sell_sigs = daily_report["sell_signals"]
    market_str = "✅ 交易日" if daily_report["market_is_open"] else "❌ 非交易日"

    lines: list[str] = [
        f"# 每日量化操作建议  —  {run_date}",
        "",
        f"> 市场状态: {market_str}   |   生成时间: {datetime.now().strftime('%H:%M:%S')}",
        "",
        "## 投资组合概况",
        "",
        "| 项目 | 值 |",
        "|------|-----|",
        f"| 总资产 | {pv:,.0f} 港元 |",
        f"| 持仓市值 | {mv:,.2f} 港元 |",
        f"| 可用现金 | {cash:,.2f} 港元（{cash_pct:.1f}%） |",
        f"| 持仓盈亏 | {pnl:+,.2f} 港元（{pnl_pct:+.2f}%） |",
        f"| 持仓股票数 | {daily_report['held_count']} 只 |",
        "",
    ]

    if buy_sigs:
        lines.append(f"🟢 **今日买入信号**: {', '.join(buy_sigs)}")
    if sell_sigs:
        lines.append(f"🔴 **今日卖出信号**: {', '.join(sell_sigs)}")
    if buy_sigs or sell_sigs:
        lines.append("")

    lines.extend([
        "## 各股操作建议",
        "",
        "> **策略共识**：多个量化策略投票，看涨策略数 vs 看跌策略数；**置信度** = 看涨策略占总参与策略数的比例。  ",
        "> **止损位**：基于 ATR（平均真实波幅）动态计算的建议离场价，触及时建议卖出控损。  ",
        "> **Kelly 建议仓位**：凯利公式根据历史胜率与盈亏比推算的最优买入股数，以最大化长期收益。  ",
        "",
        "| 标的 | 操作 | 收盘价 | 持仓 | 盈亏 | 止损位 | 策略共识 | 置信度 |",
        "|------|------|--------|------|------|--------|---------|--------|",
    ])

    for r in daily_report["recommendations"]:
        pos_str = f"{r['shares']}股@{r['avg_cost']:.2f}" if r["has_position"] else "空仓"
        pnl_str = f"{r['profit_pct']:+.1f}%" if r["has_position"] else "—"
        stop_str = f"{r['stop_price']:.2f}" if r["stop_price"] > 0 else "—"
        consensus_str = f"{r['bullish_count']}↑/{r['bearish_count']}↓"
        lines.append(
            f"| {r['ticker']} "
            f"| {r['action_emoji']} {r['action']} "
            f"| {r['last_close']:.2f} "
            f"| {pos_str} "
            f"| {pnl_str} "
            f"| {stop_str} "
            f"| {consensus_str} "
            f"| {r['confidence_label']}({r['confidence_pct']:.0%}) |"
        )

    lines.append("")
    lines.append("## 详细建议")
    lines.append("")

    for r in daily_report["recommendations"]:
        emoji = r["action_emoji"]
        lines.extend([
            f"### {emoji} {r['ticker']}  —  {r['action']}",
            "",
            f"- **最后交易日**: {r['last_date']}",
            f"- **收盘价**: {r['last_close']:.2f} 港元",
            f"- **操作建议**: {r['action']}",
            f"- **操作理由**: {r['reason']}",
        ])

        if r["has_position"]:
            lines.extend([
                f"- **持仓**: {r['shares']} 股 @ {r['avg_cost']:.2f}",
                f"- **市值**: {r['market_value']:.2f} 港元",
                f"- **盈亏**: {r['profit']:+.2f} 港元（{r['profit_pct']:+.2f}%）",
            ])

        if r["stop_price"] > 0:
            lines.append(f"- **止损价**: {r['stop_price']:.2f} 港元")
        if r["kelly_shares"] > 0:
            lines.append(f"- **Kelly 建议仓位**: {r['kelly_shares']} 股（≈{r['kelly_amount']:.0f} 港元）")
        if r["circuit_breaker"]:
            lines.append(f"- ⚠️ **熔断触发**（连续亏损 {r['consecutive_loss_days']} 天，已暂停交易信号）")

        # 共识信号
        lines.extend([
            "",
            "**多策略共识**",
            "> 看涨/看跌数量：多个量化策略同时运行的投票结果；置信度越高，共识越强。",
            "",
            f"| 看涨策略 | 看跌策略 | 总参与 | 置信度 |",
            f"|---------|---------|-------|--------|",
            f"| {r['bullish_count']} | {r['bearish_count']} | {r['total_strategies']} | {r['confidence_pct']:.0%} |",
        ])

        if r["top_strategy"]:
            lines.append(f"- **权重最高策略**: {r['top_strategy']}")
        if r["ml_signal"] is not None:
            ml_str = "做多" if r["ml_signal"] == 1 else "做空"
            lines.append(f"- **ML 信号**: {ml_str}")
        if r["rule_signal"] is not None:
            rule_str = "做多" if r["rule_signal"] == 1 else "做空"
            lines.append(f"- **规则策略共识**: {rule_str}")

        # 参考区间
        if r["price_lo_1d"] > 0 and r["price_hi_1d"] > 0:
            lines.append(
                f"- **统计参考区间（P10~P90，1日）**: [{r['price_lo_1d']:.2f}, {r['price_hi_1d']:.2f}]"
            )
            lines.append("> ⚠️ **P10~P90**：历史相同持仓期收益率分布的第10至90百分位对应价格区间，**非价格预测**，仅作参考。")

        # 情感分析
        sent = r.get("sentiment")
        if sent and isinstance(sent, dict):
            sent_emoji = (
                "🟢" if sent.get("sentiment") == "positive"
                else "🔴" if sent.get("sentiment") == "negative"
                else "⚪"
            )
            lines.extend([
                "",
                f"**情感分析**: {sent_emoji} {sent.get('sentiment', 'neutral')}  "
                f"（极性分数 {sent.get('polarity', 0):.3f}，"
                f"正面{sent.get('positive_count', 0)}/负面{sent.get('negative_count', 0)}/中性{sent.get('neutral_count', 0)} 条）",
                "> 极性分数范围 -1（极负面）~ +1（极正面），基于新闻/社交媒体文本情绪分析。",
            ])

        # 风险标志
        if r["risk_flags"]:
            lines.extend(["", "**⚠️ 风险提示**"])
            for flag in r["risk_flags"]:
                lines.append(f"- {flag}")

        lines.append("")

    # ── 选股推荐板块 ───────────────────────────────────────────
    screener_results = daily_report.get("screener_results", [])
    if screener_results:
        sw = daily_report.get("screener_weights", {})
        w_mom = sw.get("momentum", 0.35)
        w_trend = sw.get("trend", 0.35)
        w_vol = sw.get("volume", 0.30)

        lines.extend([
            "---",
            "",
            "## 今日选股推荐",
            "",
            f"> 基于量化多因子评分（动量{int(w_mom*100)}% + 趋势{int(w_trend*100)}% + 量价{int(w_vol*100)}%）",
            "",
            "| 排名 | 标的 | 综合评分 | 动量 | 趋势 | 量价 | 5日涨幅 | 20日涨幅 | 选股信号 |",
            "|------|------|---------|------|------|------|---------|---------|----------|",
        ])

        for r in screener_results:
            sig_str = ", ".join(r.get("signals", [])[:3])
            lines.append(
                f"| {r.get('rank', '-')} "
                f"| {r.get('ticker', '')} "
                f"| **{r.get('composite_score', 0):.1f}** "
                f"| {r.get('momentum_score', 0):.0f} "
                f"| {r.get('trend_score', 0):.0f} "
                f"| {r.get('volume_score', 0):.0f} "
                f"| {r.get('change_pct_5d', 0):+.1f}% "
                f"| {r.get('change_pct_20d', 0):+.1f}% "
                f"| {sig_str} |"
            )

        lines.append("")
        lines.append(
            "> ⚠️ 选股推荐仅表示量化模型认为该标的值得关注，不构成买入建议。 "
            "请结合自身持仓、风险偏好和市场环境综合判断。"
        )
        lines.append("")

    # ── 板块强弱排名 ─────────────────────────────────────────
    sector_ranking = daily_report.get("sector_ranking", [])
    if sector_ranking:
        lines.extend([
            "",
            "### 板块强弱排序",
            "",
            "| 板块 | 平均评分 | 股票数 | 龙头股 | 龙头评分 | 趋势 |",
            "|------|---------|-------|--------|---------|------|",
        ])
        trend_emoji = {"上升": "📈", "横盘": "➡️", "下降": "📉"}
        for s in sector_ranking:
            trend = "上升" if s.get("avg_score", 0) >= 65 else ("横盘" if s.get("avg_score", 0) >= 50 else "下降")
            emoji = trend_emoji.get(trend, "➡️")
            lines.append(
                f"| {s.get('sector', '其他')} "
                f"| **{s.get('avg_score', 0):.1f}** "
                f"| {s.get('count', 0)} "
                f"| {s.get('top_stock', '-')} "
                f"| {s.get('top_score', 0):.1f} "
                f"| {emoji} {trend} |"
            )
        lines.append("")

    lines.extend([
        "---",
        "",
        "⚠️ 以上建议由量化模型自动生成，仅供参考，不构成投资建议。",
        "   策略基于历史数据，实际市场走势可能存在较大差异。",
        "",
        "*本报告由每日量化推荐引擎自动生成*",
    ])

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="每日量化推荐引擎 — 基于持仓给出下一交易日操作建议"
    )
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="指定分析的股票代码列表，如 0700.HK 0005.HK",
    )
    parser.add_argument(
        "--watchlist", choices=["hk", "hsi", "portfolio", "all"], default="portfolio",
        help="使用预设观察列表（hk/hsi=全量港股，portfolio=portfolio.yaml，all=两者合并）",
    )
    parser.add_argument(
        "--retrain", action="store_true",
        help="强制重新训练因子（即使因子仍新鲜）",
    )
    parser.add_argument(
        "--min-factor-age-days", type=int, default=None,
        help="因子过期天数阈值（超过则重训，默认读 config.yaml → daily_run.min_factor_age_days）",
    )
    parser.add_argument(
        "--skip-notify", action="store_true",
        help="不发送飞书通知",
    )
    parser.add_argument(
        "--skip-sentiment", action="store_true",
        help="跳过情感分析（更快）",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="不保存持仓状态更新",
    )
    parser.add_argument(
        "--max-workers", type=int, default=None,
        help="并发分析线程数（默认读 config.yaml → daily_run.max_workers）",
    )
    parser.add_argument(
        "--period", type=str, default="5y",
        help="历史数据周期（默认 5y）",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="打印详细错误堆栈",
    )
    parser.add_argument(
        "--enable-screener", action="store_true",
        help="强制启用选股模块（覆盖 config.yaml enable_screener=false）",
    )
    parser.add_argument(
        "--no-screener", action="store_true",
        help="禁用选股模块（覆盖 config.yaml enable_screener=true）",
    )
    parser.add_argument(
        "--screener-top-n", type=int, default=None,
        help="选股模块输出 Top-N 候选（默认读 config.yaml screener.top_n）",
    )
    args = parser.parse_args()

    run_date = datetime.now().strftime("%Y-%m-%d")

    logger.info("每日量化推荐引擎启动", extra={"run_date": run_date})

    # ── 加载配置 ─────────────────────────────────────────────────
    config = load_config()
    daily_cfg = config.get("daily_run", {})
    feishu_webhook = config.get("feishu_webhook")

    min_factor_age_days = (
        args.min_factor_age_days
        or daily_cfg.get("min_factor_age_days", 7)
    )
    max_workers = args.max_workers or daily_cfg.get("max_workers", 4)

    # ── 因子注册表每日维护 ─────────────────────────────────────────
    factors_dir = Path(__file__).parent / "data" / "factors"
    try:
        from data.factor_registry import FactorRegistry
        registry = FactorRegistry()
        expired = registry.expire_stale()
        archived = registry.archive_old(factors_dir=factors_dir)
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
    except Exception as e:
        logger.warning("因子注册表维护失败（非阻塞）: %s", e)

    # ── 检查港股交易日 ────────────────────────────────────────────
    from data.calendar import is_trading_day as _is_hk_trading_day
    from datetime import date as _date
    today = _date.today()
    market_is_open = _is_hk_trading_day(today)
    if not market_is_open:
        logger.info("非港股交易日，继续生成昨日收盘后建议", extra={"today": str(today)})

    # ── 决定目标股票列表 ──────────────────────────────────────────
    from engine.portfolio_state import load_portfolio
    from data.hk_stocks import get_all_hk_stocks

    portfolio_state = load_portfolio()
    logger.info("持仓状态已加载", extra={"portfolio_summary": portfolio_state.summary()})

    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
        # 确保这些 ticker 在 portfolio 中注册（空仓观察）
        for t in tickers:
            portfolio_state.add_watchlist_ticker(t)
    elif args.watchlist in ("hk", "hsi"):
        tickers = get_all_hk_stocks()
    elif args.watchlist == "all":
        all_hk = get_all_hk_stocks()
        port = portfolio_state.all_tickers()
        tickers = list(dict.fromkeys(all_hk + port))
    else:
        # portfolio（默认）
        tickers = portfolio_state.all_tickers()
        if not tickers:
            # 如 portfolio.yaml 为空，降级用 config.yaml 中的 ticker
            default_ticker = config.get("ticker", "0700.HK").upper()
            tickers = [default_ticker]
            portfolio_state.add_watchlist_ticker(default_ticker)

    logger.info("分析目标", extra={
        "ticker_count": len(tickers),
        "tickers": tickers
    })

    # ── 因子新鲜度检查 / 重训 ────────────────────────────────────
    factors_dir = Path(__file__).parent / "data" / "factors"
    need_retrain = args.retrain

    if not need_retrain and daily_cfg.get("retrain_on_stale", True):
        is_fresh = _check_factor_freshness(factors_dir, min_age_days=min_factor_age_days)
        if not is_fresh:
            logger.warning("因子已超过阈值天数未更新", extra={
                "min_factor_age_days": min_factor_age_days,
                "action": "使用现有因子继续分析"
            })
            need_retrain = False  # daily_run 不自动重训，提示即可

    if need_retrain:
        logger.info("强制重新训练因子（--retrain）— daily_run.py 不执行重训，退出")
        sys.exit(0)

    # 检查是否存在因子文件
    factor_files = list(factors_dir.glob("factor_*.pkl"))
    if not factor_files and factors_dir.is_dir():
        for _sub in factors_dir.iterdir():
            if _sub.is_dir() and not _sub.name.startswith("."):
                factor_files.extend(_sub.glob("factor_*.pkl"))
    if not factor_files:
        logger.error(
            "data/factors/ 及其子目录均无因子文件，"
            "请先运行 main.py 或 main.py --portfolio 进行策略训练"
        )
        sys.exit(1)

    logger.info("因子文件已加载", extra={
        "total_factors": len(factor_files),
        "factors_used": min(len(factor_files), 20)
    })

    # ── 数据管理器 ───────────────────────────────────────────────
    from data.manager import DataManager
    data_mgr = DataManager()

    # ── 增量更新港股数据（若需要） ───────────────────────────────
    if args.watchlist in ("hk", "hsi", "all") or len(tickers) > 5:
        logger.info("开始增量更新股票数据", extra={"ticker_count": len(tickers)})
        try:
            hk_result = data_mgr.download_hk_incremental(
                period=args.period,
                stocks=tickers,
            )
            updated = hk_result.get("updated", 0)
            skipped = hk_result.get("skipped", 0)
            failed = hk_result.get("failed", [])
            logger.info("数据更新完成", extra={
                "updated": updated,
                "skipped": skipped,
                "failed_count": len(failed),
                "failed_tickers": failed
            })
        except Exception as e:
            logger.warning("批量数据更新失败", extra={"error": str(e)})

    # ── [Step 2] 选股模块 ─────────────────────────────────────────
    screener_results = []
    sector_ranking = []
    enable_screener = (
        not args.no_screener
        and (args.enable_screener or daily_cfg.get("enable_screener", True))
    )
    if enable_screener:
        from engine.stock_screener import StockScreener
        from data.hk_stocks import get_all_hk_stocks

        logger.info("选股模块启动", extra={"universe": config.get("screener", {}).get("universe", "hk")})

        screener = StockScreener(config)
        screener_top_n = args.screener_top_n or screener.top_n_count

        candidate_tickers = get_all_hk_stocks()
        data_dict = {}
        for t in candidate_tickers:
            try:
                df = data_mgr.load(t, period=config.get("period", "5y"))
                if df is not None and len(df) > 60:
                    data_dict[t] = df
            except Exception:
                pass

        screen_all = screener.screen(list(data_dict.keys()), data_dict)
        top_picks = screener.top_n(
            screen_all,
            n=screener_top_n,
            exclude_held=True,
            portfolio_state=portfolio_state,
        )

        logger.info("选股扫描完成", extra={
            "total_scanned": len(screen_all),
            "top_n": len(top_picks),
        })

        orig_tickers = set(tickers)
        for pick in top_picks:
            logger.info("选股候选", extra={
                "ticker": pick.ticker,
                "score": round(pick.composite_score, 1),
                "signals": pick.signals,
            })
            if pick.ticker not in tickers:
                tickers.append(pick.ticker)
                portfolio_state.add_watchlist_ticker(pick.ticker)
                logger.info("加入分析列表", extra={"ticker": pick.ticker})

        # 持久化新增 watchlist 到 portfolio.yaml
        if not args.dry_run and daily_cfg.get("persist_screener_picks", True):
            new_tickers = [p.ticker for p in top_picks if p.ticker not in orig_tickers]
            portfolio_state.save()
            logger.info("选股 watchlist 已持久化", extra={"new_tickers": new_tickers})

        screener_results = top_picks
        sector_ranking = screener.sector_ranking(screen_all)
        logger.info("选股模块完成", extra={
            "total_tickers_now": len(tickers),
            "screener_candidates": len(top_picks),
            "sectors_identified": len(sector_ranking),
        })

    # ── 初始化分析器 ──────────────────────────────────────────────
    from engine.position_analyzer import PositionAnalyzer

    analyzer = PositionAnalyzer(
        config=config,
        factors_dir=factors_dir,
        enable_sentiment=(not args.skip_sentiment),
        max_factors=20,
    )

    # ── 并发分析所有股票 ──────────────────────────────────────────
    logger.info("开始股票分析", extra={"max_workers": max_workers, "ticker_count": len(tickers)})

    results = []

    def _analyze_with_log(ticker: str):
        res = _analyze_one_ticker(
            ticker=ticker,
            config=config,
            analyzer=analyzer,
            portfolio_state=portfolio_state,
            data_mgr=data_mgr,
            period=args.period,
        )
        if res:
            logger.debug("单只股票分析完成", extra={
                "ticker": res.ticker,
                "action": res.action,
                "summary": res.summary_line
            })
        return res

    if max_workers == 1 or len(tickers) == 1:
        for t in tickers:
            results.append(_analyze_with_log(t))
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_analyze_with_log, t): t for t in tickers}
            for fut in as_completed(futures):
                try:
                    results.append(fut.result())
                except Exception as e:
                    ticker = futures[fut]
                    logger.warning("线程异常", extra={"ticker": ticker, "error": str(e)})
                    results.append(None)

    results = [r for r in results if r is not None]
    logger.info("股票分析完成", extra={
        "completed": len(results),
        "total": len(tickers)
    })

    # ── 汇总报告 ──────────────────────────────────────────────────
    daily_report = _build_daily_report(
        results=results,
        portfolio_value=portfolio_state.portfolio_value,
        run_date=run_date,
        market_is_open=market_is_open,
        config=config,
        screener_results=screener_results,
        sector_ranking=sector_ranking,
    )

    # 记录汇总信息
    logger.info("每日操作建议汇总", extra={
        "portfolio_value": daily_report['portfolio_value'],
        "total_market_value": daily_report['total_market_value'],
        "total_pnl": daily_report['total_pnl'],
        "total_pnl_pct": f"{daily_report['total_pnl_pct']:+.2f}%",
        "buy_signals": daily_report['buy_signals'],
        "sell_signals": daily_report['sell_signals'],
    })

    # ── PnL 追踪：T+1 回填 + 当日快照 ───────────────────────────
    if not args.dry_run and market_is_open:
        from data.calendar import prev_trading_day
        tracker = PnLTracker()
        price_map = {r["ticker"]: r["last_close"] for r in daily_report["recommendations"]}
        prev_date = prev_trading_day(datetime.now().date())
        if price_map:
            tracker.fill_t1_returns(prev_date.strftime("%Y-%m-%d"), price_map)
        tracker.record_daily(run_date, results)

    for r in results:
        pos_str = f"{r.shares}股@{r.avg_cost:.0f}" if r.has_position else "空仓"
        pnl_str = f"{r.profit_pct:+.1f}%" if r.has_position else "—"
        stop_str = f"{r.stop_price:.2f}" if r.stop_price > 0 else "—"
        logger.info("操作建议", extra={
            "ticker": r.ticker,
            "action": r.action,
            "action_emoji": r.action_emoji,
            "last_close": r.last_close,
            "position": pos_str,
            "pnl_pct": pnl_str,
            "stop_price": stop_str,
            "confidence_label": r.confidence_label,
            "confidence_pct": f"{r.confidence_pct:.0%}",
        })

    # ── 保存 Markdown 报告 ────────────────────────────────────────
    report_dir = Path(__file__).parent / "data" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_filename = f"daily_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report_path = report_dir / report_filename

    md_content = _build_markdown_report(daily_report)
    if not args.dry_run:
        report_path.write_text(md_content, encoding="utf-8")
        logger.info("报告已保存", extra={"report_path": str(report_path)})
    else:
        logger.info("dry-run模式，报告未写入磁盘", extra={"report_path": str(report_path)})

    # ── 更新持仓状态（peak_price / consecutive_loss_days） ──────
    if not args.dry_run:
        for r in results:
            if r.has_position:
                # 每日收盘后更新移动止损峰值和连续亏损天数
                portfolio_state.update_position(
                    r.ticker,
                    peak_price=r.peak_price,
                    consecutive_loss_days=r.consecutive_loss_days,
                )
        portfolio_state.save()
        logger.info("持仓状态已更新", extra={"path": str(portfolio_state.path)})

    # ── 飞书通知 ──────────────────────────────────────────────────
    if not args.skip_notify and feishu_webhook:
        try:
            from feishu_notify import send_daily_advisory
            ok = send_daily_advisory(feishu_webhook, daily_report)
            if ok:
                logger.info("飞书通知已发送")
            else:
                logger.warning("飞书通知发送失败")
        except Exception as e:
            logger.warning("飞书通知异常", extra={"error": str(e)})
            if args.debug:
                traceback.print_exc()
    elif not feishu_webhook:
        logger.info("飞书Webhook未配置，跳过通知")
    elif args.skip_notify:
        logger.info("已指定--skip-notify，跳过飞书通知")

    logger.info("每日分析完成", extra={"run_date": run_date})


if __name__ == "__main__":
    main()

