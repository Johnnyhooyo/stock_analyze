"""
飞书群聊机器人通知模块
用于发送分析报告到飞书群聊
"""

import requests

from log_config import get_logger

logger = get_logger(__name__)


def send_feishu_message(webhook_url: str, message: str, msg_type: str = "text") -> bool:
    """
    发送消息到飞书群聊
    """
    if not webhook_url:
        logger.warning("飞书 Webhook 未配置，跳过通知")
        return False

    try:
        if msg_type == "markdown":
            payload = {
                "msg_type": "interactive",
                "card": {
                    "header": {
                        "title": {
                            "tag": "plain_text",
                            "content": "腾讯股票分析报告"
                        },
                        "template": "blue"
                    },
                    "elements": [
                        {
                            "tag": "markdown",
                            "content": message
                        }
                    ]
                }
            }
        else:
            payload = {
                "msg_type": msg_type,
                "content": {
                    "text": message
                }
            }

        response = requests.post(webhook_url, json=payload, timeout=10)
        if response.status_code == 200:
            result = response.json()
            if result.get("code") == 0:
                return True
            else:
                logger.warning("飞书发送失败: %s", result.get('msg', '未知错误'))
                return False
        else:
            logger.warning("飞书发送失败: HTTP %s", response.status_code)
            return False
    except Exception as e:
        logger.error("飞书发送异常: %s", e, exc_info=True)
        return False


def send_full_report_to_feishu(
    webhook_url: str,
    report_data: dict
) -> bool:
    """
    发送完整分析报告到飞书（4阶段评估 + 持仓 + 信号 合并为一条消息）
    """
    if not webhook_url:
        logger.warning("飞书 Webhook 未配置，跳过通知")
        return False

    import math as _m

    def _f(v, fmt='.2%'):
        if v is None or (isinstance(v, float) and _m.isnan(v)):
            return '—'
        try:
            return f"{v:{fmt}}"
        except Exception:
            return '—'

    def _badge(ok):
        return '✅' if ok else '❌'

    # ── 基本信息 ──────────────────────────────────────────────────
    ticker       = report_data.get('ticker', '0700.HK').upper()
    price        = report_data.get('current_price', 0)
    last_date    = report_data.get('last_date', '')
    strategy     = report_data.get('strategy', '')
    is_ml        = report_data.get('is_ml', False)
    signal       = report_data.get('signal', '空仓')
    conf_label   = report_data.get('confidence_label', '')
    conf_emoji   = report_data.get('confidence_emoji', '⚪')
    signal_emoji = '🟢' if signal == '做多' else '🔴'

    # ── 阶段2 验证窗口指标（来自因子文件）─────────────────────────
    train_period = report_data.get('train_period', '—')
    val_period   = report_data.get('val_period', '—')
    val_ret      = report_data.get('cum_return', float('nan'))
    val_sharpe   = report_data.get('sharpe', float('nan'))
    val_dd       = report_data.get('max_drawdown', float('nan'))
    val_trades   = report_data.get('total_trades', 0)
    val_wr       = report_data.get('win_rate', float('nan'))
    ann_ret      = report_data.get('annualized_return', float('nan'))
    calmar       = report_data.get('calmar_ratio', float('nan'))
    volatility   = report_data.get('volatility', float('nan'))
    daily_vol    = report_data.get('avg_volatility', float('nan'))
    val_meets    = report_data.get('meets_threshold', None)


    # ── 持仓 & 操作建议 ───────────────────────────────────────────
    pos          = report_data.get('position', {})
    rec          = report_data.get('recommendation', {})
    shares       = pos.get('shares', 0)
    avg_cost     = pos.get('avg_cost', 0)
    cur_price    = pos.get('current_price', price)
    profit       = pos.get('profit', 0)
    profit_pct   = pos.get('profit_pct', 0)
    stop_price   = pos.get('stop_price', 0) or rec.get('stop_price', 0)
    kelly_shares = pos.get('kelly_shares', 0) or rec.get('kelly_shares', 0)
    kelly_amount = pos.get('kelly_amount', 0) or rec.get('kelly_amount', 0)
    cb           = pos.get('circuit_breaker', False) or rec.get('circuit_breaker', False)
    action       = rec.get('action', '观望')
    reason       = rec.get('reason', '')

    # ── 情感 & 预测 ───────────────────────────────────────────────
    sentiment    = report_data.get('sentiment', {})
    sent_emoji   = ('🟢' if sentiment.get('sentiment') == 'positive'
                    else '🔴' if sentiment.get('sentiment') == 'negative' else '⚪')
    predictions  = report_data.get('predictions', [])

    # ══════════════════════════════════════════════════════════════
    # 构建飞书 Markdown 消息
    # ══════════════════════════════════════════════════════════════
    lines = []

    # ── 标题栏 ────────────────────────────────────────────────────
    action_emoji = '🟢' if action in ('买入', '加仓') else '🔴' if action in ('卖出', '减仓') else '⚪'
    lines += [
        f"## 📊 {ticker}  {last_date}  {price:.2f} HKD",
        "",
        f"**策略** {strategy} {'(ML)' if is_ml else '(规则)'}　"
        f"**信号** {signal_emoji} {signal}　"
        f"**置信** {conf_emoji} {conf_label}",
        "",
    ]

    # ── 持仓状态 & 操作建议 ────────────────────────────────────────
    if shares > 0:
        pnl_emoji = '🟢' if profit >= 0 else '🔴'
        stop_str  = f"{stop_price:.2f}" if stop_price > 0 else "—"
        kelly_str = f"{kelly_shares}股 ≈{kelly_amount:.0f}元" if kelly_shares > 0 else "—"
        cb_str    = "⚠️ 熔断触发" if cb else "✅ 正常"
        lines += [
            "### 💼 持仓状态",
            "",
            "| 持股 | 成本 | 现价 | 盈亏 | 止损 | Kelly | 熔断 |",
            "|------|------|------|------|------|-------|------|",
            f"| {shares}股 | {avg_cost:.2f} | {cur_price:.2f} "
            f"| {pnl_emoji}{profit:+.0f}({profit_pct:+.1f}%) "
            f"| {stop_str} | {kelly_str} | {cb_str} |",
            "",
            f"### {action_emoji} 操作建议：{action}",
            f"> {reason}",
            "",
        ]
    elif rec:
        lines += [
            f"### {action_emoji} 操作建议：{action}",
            f"> {reason}",
            "",
        ]

    # ── 策略评估表 ────────────────────────────────────────────────
    lines += [
        "### 📋 策略评估",
        "",
        "| 阶段 | 时间范围 | 累计收益 | 夏普 | 最大回撤 | 交易次 | 胜率 | 达标 |",
        "|------|---------|---------|------|---------|-------|------|------|",
        f"| 🔵 ①训练 | {train_period} | — | — | — | — | — | — |",
        f"| 🟡 ②验证 | {val_period} | {_f(val_ret)} | {_f(val_sharpe,'.2f')} "
        f"| {_f(val_dd)} | {val_trades} | {_f(val_wr)} "
        f"| {_badge(val_meets) if val_meets is not None else '—'} |",
        "",
        f"**补充** 年化 {_f(ann_ret)} | 波动率 {_f(volatility)} | 近60日波动 {_f(daily_vol)} | 卡玛 {_f(calmar,'.2f')}",
        "",
    ]

    # ── 未来信号预测 ──────────────────────────────────────────────
    if predictions:
        lines += ["### 🔮 未来信号", ""]
        for p in predictions:
            s_emoji = '🟢' if p.get('signal', 0) == 1 else '🔴'
            lo, hi = p.get('price_lo', 0), p.get('price_hi', 0)
            lines.append(
                f"- **{p['date']}** {s_emoji} {p.get('signal_str','—')}  "
                f"区间 [{lo:.2f}, {hi:.2f}]  {p.get('confidence','')}"
            )
        lines.append("")

    # ── 情感分析（精简）──────────────────────────────────────────
    if sentiment:
        news_items = sentiment.get('latest_news', [])[:2]
        news_str = "  ".join(
            f"{'🟢' if n.get('sentiment')=='positive' else '🔴' if n.get('sentiment')=='negative' else '⚪'} {n.get('title','')[:20]}…"
            for n in news_items
        )
        lines += [
            f"### 📰 市场情感 {sent_emoji} {sentiment.get('sentiment','neutral')} "
            f"(极性 {sentiment.get('polarity', 0):.3f} | "
            f"正{sentiment.get('positive_count',0)} 负{sentiment.get('negative_count',0)} 中{sentiment.get('neutral_count',0)})",
        ]
        if news_str:
            lines.append(news_str)
        lines.append("")

    lines += ["---", "⚠️ 量化模型自动生成，仅供参考，不构成投资建议"]

    message = "\n".join(lines)

    # 超长时拆分（4000字限制）
    MAX_LEN = 4000
    if len(message) <= MAX_LEN:
        return send_feishu_message(webhook_url, message, msg_type="markdown")
    split_marker = "### 🔮 未来信号"
    split_pos = message.find(split_marker)
    if split_pos == -1:
        split_pos = MAX_LEN
    ok1 = send_feishu_message(webhook_url, message[:split_pos].rstrip(), msg_type="markdown")
    ok2 = send_feishu_message(webhook_url, message[split_pos:], msg_type="markdown")
    return ok1 and ok2


def send_simple_report_to_feishu(
    webhook_url: str,
    ticker: str,
    current_price: float,
    signal: str,
    predictions: list,
    recommendation: dict
) -> bool:
    """
    发送简化版分析报告到飞书（兼容旧接口）
    """
    report_data = {
        'ticker': ticker,
        'current_price': current_price,
        'signal': signal,
        'predictions': predictions,
        'recommendation': recommendation,
    }
    return send_full_report_to_feishu(webhook_url, report_data)


def send_daily_advisory(webhook_url: str, daily_report: dict) -> bool:
    """
    发送每日操作建议到飞书（daily_run.py 专用接口）。

    Args:
        webhook_url:  飞书 Webhook 地址
        daily_report: daily_run._build_daily_report() 返回的字典

    Returns:
        bool: 是否发送成功
    """
    if not webhook_url:
        logger.warning("飞书 Webhook 未配置，跳过通知")
        return False

    run_date = daily_report.get("run_date", "")
    pv = daily_report.get("portfolio_value", 0)
    mv = daily_report.get("total_market_value", 0)
    pnl = daily_report.get("total_pnl", 0)
    pnl_pct = daily_report.get("total_pnl_pct", 0)
    cash = daily_report.get("cash_value", 0)
    cash_pct = daily_report.get("cash_pct", 100)
    buy_sigs = daily_report.get("buy_signals", [])
    sell_sigs = daily_report.get("sell_signals", [])
    recs = daily_report.get("recommendations", [])
    market_is_open = daily_report.get("market_is_open", True)

    market_str = "✅ 交易日" if market_is_open else "⛔ 非交易日"
    pnl_emoji = "🟢" if pnl >= 0 else "🔴"

    # ── 构建 Markdown 内容 ──────────────────────────────────────
    lines = [
        f"## 📊 每日量化操作建议  {run_date}",
        "",
        f"市场状态: {market_str}",
        "",
        "### 💼 投资组合概况",
        "",
        f"| 总资产 | 持仓市值 | 可用现金 | 持仓盈亏 |",
        f"|--------|---------|---------|---------|",
        f"| {pv:,.0f} | {mv:,.2f} | {cash:,.2f}({cash_pct:.1f}%) | {pnl_emoji}{pnl:+,.2f}({pnl_pct:+.2f}%) |",
        "",
    ]

    if buy_sigs:
        lines.append(f"🟢 **今日买入信号**: {', '.join(buy_sigs)}")
    if sell_sigs:
        lines.append(f"🔴 **今日卖出信号**: {', '.join(sell_sigs)}")
    if buy_sigs or sell_sigs:
        lines.append("")

    lines.extend([
        "### 📋 操作建议明细",
        "",
        "| 标的 | 建议 | 收盘价 | 持仓 | 盈亏 | 止损 | 置信 |",
        "|------|------|--------|------|------|------|------|",
    ])

    for r in recs:
        pos_str = f"{r['shares']}股@{r['avg_cost']:.2f}" if r["has_position"] else "空仓"
        pnl_str = f"{r['profit_pct']:+.1f}%" if r["has_position"] else "—"
        stop_str = f"{r['stop_price']:.2f}" if r["stop_price"] > 0 else "—"
        conf_str = f"{r['confidence_label']}({r['confidence_pct']:.0%})"
        lines.append(
            f"| {r['ticker']} "
            f"| {r['action_emoji']} {r['action']} "
            f"| {r['last_close']:.2f} "
            f"| {pos_str} "
            f"| {pnl_str} "
            f"| {stop_str} "
            f"| {conf_str} |"
        )

    lines.append("")
    lines.append("---")
    lines.append("⚠️ 以上建议由量化模型自动生成，仅供参考，不构成投资建议。")

    # 逐条详细推送（只推送有操作信号的股票，减少消息长度）
    action_recs = [r for r in recs if r["action"] not in ("观望",)]
    if action_recs:
        lines.extend(["", "### 🎯 今日操作详情", ""])
        for r in action_recs:
            cb_str = f"⚠️ 熔断触发" if r["circuit_breaker"] else ""
            kelly_str = (
                f"建议仓位 {r['kelly_shares']} 股（≈{r['kelly_amount']:.0f}港元）"
                if r["kelly_shares"] > 0 else ""
            )
            lines.extend([
                f"**{r['action_emoji']} {r['ticker']}  {r['action']}**",
                f"- 收盘价: {r['last_close']:.2f}  止损: {r['stop_price']:.2f if r['stop_price'] > 0 else '—'}",
                f"- 原因: {r['reason']}",
            ])
            if kelly_str:
                lines.append(f"- {kelly_str}")
            if cb_str:
                lines.append(f"- {cb_str}")
            if r["risk_flags"]:
                for flag in r["risk_flags"]:
                    lines.append(f"- ⚠️ {flag}")
            lines.append("")

    message = "\n".join(lines)

    # 发送（超过 4096 字时拆分为两条）
    MAX_LEN = 4000
    if len(message) <= MAX_LEN:
        return send_feishu_message(webhook_url, message, msg_type="markdown")
    else:
        # 分两段发送：摘要 + 详情
        summary_end = message.find("### 🎯 今日操作详情")
        if summary_end == -1:
            summary_end = MAX_LEN
        ok1 = send_feishu_message(webhook_url, message[:summary_end], msg_type="markdown")
        ok2 = send_feishu_message(webhook_url, message[summary_end:], msg_type="markdown")
        return ok1 and ok2


if __name__ == "__main__":
    import yaml
    from pathlib import Path

    keys_path = Path(__file__).parent / 'keys.yaml'
    with open(keys_path) as f:
        keys = yaml.safe_load(f)

    webhook = keys.get('feishu_webhook')

    result = send_feishu_message(webhook, "测试消息 - 飞书机器人连接正常", msg_type="text")
    print(f"发送结果: {'成功' if result else '失败'}")
