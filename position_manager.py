"""
持仓管理模块
用于管理股票持仓、分析盈亏、提供交易建议
"""

import pandas as pd
from dataclasses import dataclass
from typing import Optional


@dataclass
class Position:
    """持仓数据"""
    shares: int          # 持股数量
    avg_cost: float      # 平均成本
    current_price: float # 当前价格

    @property
    def market_value(self) -> float:
        """市值"""
        return self.shares * self.current_price

    @property
    def cost_basis(self) -> float:
        """持仓成本"""
        return self.shares * self.avg_cost

    @property
    def profit(self) -> float:
        """盈亏金额"""
        return self.market_value - self.cost_basis

    @property
    def profit_pct(self) -> float:
        """盈亏比例"""
        if self.cost_basis == 0:
            return 0.0
        return (self.profit / self.cost_basis) * 100


class PositionManager:
    """持仓管理器"""

    def __init__(self, position: Optional[Position] = None):
        self.position = position

    def set_position(self, shares: int, avg_cost: float, current_price: float):
        """设置持仓"""
        self.position = Position(
            shares=shares,
            avg_cost=avg_cost,
            current_price=current_price
        )

    def get_recommendation(self, signal: int, predicted_return: float) -> dict:
        """
        根据信号和预测收益率获取交易建议

        Args:
            signal: 1=持仓, 0=空仓
            predicted_return: 预测收益率（如 -0.0168 = -1.68%）

        Returns:
            dict: 包含建议、操作、数量等信息
        """
        if self.position is None:
            return {
                'action': '无法判断',
                'reason': '未设置持仓数据',
                'shares': 0,
                'amount': 0
            }

        shares = self.position.shares
        avg_cost = self.position.avg_cost
        current_price = self.position.current_price
        profit = self.position.profit
        profit_pct = self.position.profit_pct

        # 交易建议逻辑
        if signal == 1 and shares == 0:
            # 持仓信号 + 当前空仓 → 建议买入
            action = "买入"
            reason = f"策略看涨信号，预计上涨 {predicted_return*100:.2f}%"
        elif signal == 0 and shares > 0:
            # 空仓信号 + 当前持仓 → 建议卖出
            action = "卖出"
            reason = f"策略看跌信号，预计下跌 {abs(predicted_return)*100:.2f}%"
        elif signal == 1 and shares > 0:
            # 持仓信号 + 已有持仓 → 建议持有
            action = "持有"
            reason = f"策略看涨信号，继续持有"
        else:
            # 空仓信号 + 已有空仓
            action = "观望"
            reason = "策略看跌信号，维持空仓"

        return {
            'action': action,
            'reason': reason,
            'shares': shares,
            'avg_cost': avg_cost,
            'current_price': current_price,
            'profit': profit,
            'profit_pct': profit_pct,
            'signal': signal,
            'predicted_return': predicted_return
        }

    def generate_report(self, recommendations: list) -> str:
        """生成持仓报告"""
        if self.position is None:
            return "## 持仓状态\n\n未设置持仓数据\n"

        p = self.position
        rec = recommendations[0] if recommendations else {}

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

## 交易建议

| 操作 | 原因 |
|------|------|
| **{rec.get('action', 'N/A')}** | {rec.get('reason', 'N/A')} |

"""
        return md


def load_position_from_config(config: dict) -> Optional[Position]:
    """从配置加载持仓"""
    shares = config.get('position_shares', 0)
    avg_cost = config.get('position_avg_cost', 0)

    if shares > 0 and avg_cost > 0:
        return Position(
            shares=shares,
            avg_cost=avg_cost,
            current_price=0  # 稍后会更新为最新价格
        )
    return None


if __name__ == "__main__":
    # 测试
    pm = PositionManager()
    pm.set_position(shares=200, avg_cost=530.0, current_price=546.5)

    # 模拟信号：下跌
    rec = pm.get_recommendation(signal=0, predicted_return=-0.0168)
    print(f"建议: {rec['action']}")
    print(f"原因: {rec['reason']}")
    print(f"盈亏: {rec['profit']:.2f} ({rec['profit_pct']:.2f}%)")
