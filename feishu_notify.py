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
    发送完整分析报告到飞书

    Args:
        webhook_url: 飞书 Webhook
        report_data: 包含所有报告数据的字典

    Returns:
        bool: 是否发送成功
    """
    if not webhook_url:
        logger.warning("飞书 Webhook 未配置，跳过通知")
        return False

    # 提取数据
    ticker = report_data.get('ticker', '0700.hk')
    current_price = report_data.get('current_price', 0)
    last_date = report_data.get('last_date', '')
    strategy = report_data.get('strategy', '')
    params = report_data.get('params', {})
    is_ml = report_data.get('is_ml', False)

    # 信号
    signal = report_data.get('signal', '震荡')
    signal_emoji = "🟢" if signal == "上涨" else "🔴"

    # 收益指标
    cum_return = report_data.get('cum_return', 0)
    sharpe = report_data.get('sharpe', 0)
    ann_return = report_data.get('annualized_return', 0)
    max_dd = report_data.get('max_drawdown', 0)
    volatility = report_data.get('volatility', 0)

    # 交易统计
    total_trades = report_data.get('total_trades', 0)
    win_rate = report_data.get('win_rate', 0)
    calmar = report_data.get('calmar_ratio', 0)

    # 预测
    predictions = report_data.get('predictions', [])
    avg_volatility = report_data.get('avg_volatility', 0)

    # 持仓
    position = report_data.get('position', {})
    recommendation = report_data.get('recommendation', {})

    # 验证结果
    validation = report_data.get('validation', {})

    # 情感分析
    sentiment = report_data.get('sentiment', {})

    # 构建消息
    lines = []

    # 标题
    lines.append(f"## 📊 {ticker} 股票分析报告")
    lines.append("")

    # 基本信息
    lines.append("### 📈 基本信息")
    lines.append(f"- **分析日期**: {last_date}")
    lines.append(f"- **当前价格**: {current_price:.2f} HKD")
    lines.append(f"- **策略**: {strategy}")
    lines.append(f"- **策略类型**: {'ML模型' if is_ml else '规则策略'}")
    lines.append(f"- **信号**: {signal_emoji} {signal}")
    lines.append("")

    # 情感分析
    if sentiment:
        sent_emoji = "🟢" if sentiment.get('sentiment') == 'positive' else "🔴" if sentiment.get('sentiment') == 'negative' else "⚪"
        lines.append("### 📰 情感分析")
        lines.append(f"- **情感倾向**: {sent_emoji} {sentiment.get('sentiment', 'neutral')}")
        lines.append(f"- **情感分数**: {sentiment.get('polarity', 0):.3f}")
        lines.append(f"- **正面新闻**: {sentiment.get('positive_count', 0)} 篇")
        lines.append(f"- **负面新闻**: {sentiment.get('negative_count', 0)} 篇")
        lines.append(f"- **中性新闻**: {sentiment.get('neutral_count', 0)} 篇")
        if sentiment.get('latest_news'):
            lines.append("**最新新闻:**")
            for n in sentiment['latest_news'][:3]:
                n_emoji = "🟢" if n.get('sentiment') == 'positive' else "🔴" if n.get('sentiment') == 'negative' else "⚪"
                lines.append(f"- {n_emoji} {n.get('title', '')}")
        lines.append("")

    # 收益指标
    import math as _math
    _ann_return_str = f"{ann_return:+.2%}" if (ann_return is not None and _math.isfinite(ann_return)) else "N/A"
    lines.append("### 💰 收益指标")
    lines.append(f"- **累计收益**: {cum_return:+.2%}")
    lines.append(f"- **年化收益**: {_ann_return_str}")
    lines.append(f"- **夏普比率**: {sharpe:.2f}")
    lines.append(f"- **卡玛比率**: {calmar:.2f}")
    lines.append("")

    # 风险指标
    lines.append("### ⚠️ 风险指标")
    lines.append(f"- **最大回撤**: {max_dd:.2%}")
    lines.append(f"- **年化波动率**: {volatility:.2%}")
    lines.append(f"- **近60日波动率**: {avg_volatility:.2%}")
    lines.append("")

    # 交易统计
    lines.append("### 📊 交易统计")
    lines.append(f"- **交易次数**: {total_trades}")
    lines.append(f"- **胜率**: {win_rate:.2%}")
    lines.append("")

    # 预测结果
    if predictions:
        lines.append("### 🔮 预测结果")
        for p in predictions:
            # Support both old-style keys and new-style keys
            if 'direction' in p:
                emoji = "🟢" if p['direction'] == "上涨" else "🔴"
                price_str = f"{p['price']:.2f} " if 'price' in p else ""
                ret_str = f"{p['return']:+.2%} " if 'return' in p else ""
                range_str = f"(区间: [{p['low']:.2f}, {p['high']:.2f}])" if 'low' in p else ""
                lines.append(f"- {p['date']}: {price_str}{emoji} {ret_str}{range_str}")
            else:
                signal_str = p.get('signal_str', '—')
                emoji = "🟢" if p.get('signal', 0) == 1 else "🔴"
                pred_ret = p.get('pred_ret_raw', '—')
                lo = p.get('price_lo', 0)
                hi = p.get('price_hi', 0)
                confidence = p.get('confidence', '')
                lines.append(f"- {p['date']}: {signal_str} {emoji} 模型输出: {pred_ret}  区间: [{lo:.2f}, {hi:.2f}]  {confidence}")
        lines.append("")

    # 持仓状态
    if position and position.get('shares', 0) > 0:
        lines.append("### 💼 持仓状态")
        lines.append(f"- **持股数量**: {position.get('shares', 0)} 股")
        lines.append(f"- **平均成本**: {position.get('avg_cost', 0):.2f} HKD")
        lines.append(f"- **当前价格**: {position.get('current_price', 0):.2f} HKD")
        lines.append(f"- **盈亏金额**: {position.get('profit', 0):+.2f} HKD")
        lines.append(f"- **盈亏比例**: {position.get('profit_pct', 0):+.2%}")
        lines.append("")

    # 交易建议
    if recommendation:
        action = recommendation.get('action', 'N/A')
        reason = recommendation.get('reason', '')
        lines.append("### 🎯 交易建议")
        lines.append(f"- **操作**: {action}")
        lines.append(f"- **原因**: {reason}")
        lines.append(f"- **预测收益率**: {recommendation.get('predicted_return', 0):+.2%}")
        lines.append("")

    # 验证结果
    if validation:
        oos = validation.get('out_of_sample', {})
        wf = validation.get('walk_forward', {})

        lines.append("### ✅ 策略验证")

        if oos:
            lines.append("**样本外测试**:")
            lines.append(f"- 策略收益: {oos.get('cum_return', 0):+.2%}")
            lines.append(f"- 买入持有: {oos.get('bh_return', 0):+.2%}")
            lines.append(f"- 超额收益: {oos.get('excess_return', 0):+.2%}")
            lines.append(f"- 夏普比率: {oos.get('sharpe', 0):.2f}")
            lines.append(f"- 最大回撤: {oos.get('max_drawdown', 0):.2%}")
            lines.append("")

        if wf:
            lines.append("**Walk-Forward 分析**:")
            lines.append(f"- 窗口胜率: {wf.get('window_win_rate', 0):.2%}")
            lines.append(f"- 交易胜率: {wf.get('trade_win_rate', 0):.2%}")
            lines.append(f"- 平均收益: {wf.get('avg_return', 0):.2%}")
            lines.append(f"- 平均夏普: {wf.get('avg_sharpe', 0):.2f}")
            lines.append("")

    # 风险提示
    lines.append("---")
    lines.append("⚠️ 以上仅供参考，不构成投资建议")

    message = "\n".join(lines)

    return send_feishu_message(webhook_url, message, msg_type="markdown")


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
