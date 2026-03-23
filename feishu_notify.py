"""
飞书群聊机器人通知模块
用于发送分析报告到飞书群聊
"""

import requests
from typing import Optional, Dict, Any


def send_feishu_message(webhook_url: str, message: str, msg_type: str = "text") -> bool:
    """
    发送消息到飞书群聊
    """
    if not webhook_url:
        print("  ⚠️ 飞书 Webhook 未配置，跳过通知")
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
                print(f"  ⚠️ 飞书发送失败: {result.get('msg', '未知错误')}")
                return False
        else:
            print(f"  ⚠️ 飞书发送失败: HTTP {response.status_code}")
            return False
    except Exception as e:
        print(f"  ⚠️ 飞书发送异常: {e}")
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
        print("  ⚠️ 飞书 Webhook 未配置，跳过通知")
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


if __name__ == "__main__":
    import yaml
    from pathlib import Path

    keys_path = Path(__file__).parent / 'keys.yaml'
    with open(keys_path) as f:
        keys = yaml.safe_load(f)

    webhook = keys.get('feishu_webhook')

    result = send_feishu_message(webhook, "测试消息 - 飞书机器人连接正常", msg_type="text")
    print(f"发送结果: {'成功' if result else '失败'}")
