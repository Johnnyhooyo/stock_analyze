from unittest.mock import Mock, patch

from feishu_notify import send_daily_advisory


def test_daily_advisory_uses_feishu_compatible_markdown():
    report = {
        "run_date": "2026-09-08", "portfolio_value": 199_969,
        "total_market_value": 159_971.05, "cash_value": 39_997.62,
        "cash_pct": 20, "total_pnl": -31.34, "total_pnl_pct": -0.02,
        "market_is_open": True, "buy_signals": ["0087.HK"], "sell_signals": [],
        "recommendations": [{
            "ticker": "0005.HK", "action": "持有", "action_emoji": "🟡",
            "last_close": 167.60, "has_position": True, "shares": 299,
            "avg_cost": 167.01, "profit_pct": 0.4, "stop_price": 162.14,
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
    assert "**0005.HK**  🟡 持有" in content
    assert not any(line.startswith("#") for line in content.splitlines())
    assert not any(line.startswith("|") for line in content.splitlines())
