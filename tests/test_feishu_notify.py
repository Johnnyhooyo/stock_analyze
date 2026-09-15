from unittest.mock import Mock, patch

from feishu_notify import send_daily_advisory


def test_daily_advisory_uses_feishu_compatible_markdown():
    report = {
        "run_date": "2026-09-08", "portfolio_value": 199_969,
        "total_market_value": 159_971.05, "cash_value": 39_997.62,
        "cash_pct": 20, "total_pnl": -31.34, "total_pnl_pct": -0.02,
        "market_is_open": True, "buy_signals": ["0087.HK"], "sell_signals": ["0428.HK"],
        "executed_trades": [{
            "ticker": "0428.HK", "action": "卖出", "shares": 100,
            "avg_cost": 1.00, "price": 1.20, "fee": 0.50,
            "realized_pnl": 19.50,
        }],
        "recommendations": [{
            "ticker": "0005.HK", "action": "持有", "action_emoji": "🟡",
            "last_close": 167.60, "has_position": True, "shares": 299,
            "avg_cost": 167.01, "profit": 176.41, "profit_pct": 0.4,
            "entry_date": "2026-09-01", "holding_days": 8,
            "stop_price": 162.14,
            "confidence_label": "中", "confidence_pct": 0.67,
            "circuit_breaker": False, "kelly_shares": 0, "kelly_amount": 0,
            "reason": "继续持有", "risk_flags": [],
        }],
    }
    response = Mock(status_code=200)
    response.json.return_value = {"code": 0}

    with patch("feishu_notify.requests.post", return_value=response) as post:
        assert send_daily_advisory("https://example.invalid/hook", report)

    content = post.call_args.kwargs["json"]["card"]["elements"][0]["content"]
    assert "**📊 每日量化操作建议  2026-09-08**" in content
    assert "**📌 持仓与当日卖出**" in content
    assert "**0005.HK × 299股**  🟢" in content
    assert "买入价 167.01  ·  当前价 167.60  ·  卖出价 —" in content
    assert "建仓 2026-09-01  ·  持仓第8天" in content
    assert "收益金额 +176.41  ·  收益率 +0.35%" in content
    assert "买入成本/股 0.1884  ·  卖出成本/股（预估）0.1918" in content
    assert "真实收益金额" in content
    assert "真实收益率" in content
    assert "**0428.HK × 100股**  当日已卖出 🟢" in content
    assert "买入价 1.00  ·  当前价 1.20  ·  卖出价 1.20" in content
    assert "收益金额 +20.00  ·  收益率 +20.00%" in content
    assert "买入成本/股 0.0000  ·  卖出成本/股 0.0050" in content
    assert "真实收益金额 +19.50  ·  真实收益率 +19.50%" in content
    assert "**0005.HK**  🟡 持有" in content
    assert not any(line.startswith("#") for line in content.splitlines())
    assert not any(line.startswith("|") for line in content.splitlines())
