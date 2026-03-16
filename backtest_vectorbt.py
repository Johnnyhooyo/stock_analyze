"""
Vectorbt 回测模块
基于 Vectorbt 框架的回测功能，保留原有回测逻辑
"""

import vectorbt as vbt
import pandas as pd
import numpy as np
from typing import Dict, Any


def backtest_vectorbt(
    data: pd.DataFrame,
    signal: pd.Series,
    config: dict
) -> dict:
    """
    使用 Vectorbt 进行回测

    Args:
        data: 价格数据（需要 Close 列）
        signal: 信号序列 (1=持仓, 0=空仓)
        config: 配置

    Returns:
        dict: 回测结果
    """
    # 获取配置
    initial_capital = float(config.get('initial_capital', 100000.0))
    fees = float(config.get('fees', 0.001))  # 手续费

    # 准备数据
    close = data['Close'].copy()

    # 确保 signal 是数值类型并对齐
    signal = pd.Series(signal).astype(float)
    close, signal = close.align(signal, join='inner')

    # 生成买入和卖出信号
    # 买入点：当前持仓=1，前一天持仓=0
    entries = (signal == 1) & (signal.shift(1).fillna(0) == 0)
    # 卖出点：当前持仓=0，前一天持仓=1
    exits = (signal == 0) & (signal.shift(1).fillna(0) == 1)

    # 运行回测
    portfolio = vbt.Portfolio.from_signals(
        close,
        entries=entries,
        exits=exits,
        init_cash=initial_capital,
        fees=fees,
        slippage=0.001,  # 滑点
    )

    # 获取统计信息
    stats = portfolio.stats()

    # Vectorbt stats keys 使用不同的命名格式
    def get_stat(stats, *keys):
        for k in keys:
            if k in stats.index:
                return stats[k]
        return 0

    # 转换为兼容格式
    result = {
        # 收益指标
        'cum_return': get_stat(stats, 'Total Return [%]') / 100,  # 转换为小数
        'annualized_return': get_stat(stats, 'Annualized Return [%]') / 100 if 'Annualized Return [%]' in stats.index else 0,
        'sharpe_ratio': get_stat(stats, 'Sharpe Ratio'),
        'sortino_ratio': get_stat(stats, 'Sortino Ratio'),
        'calmar_ratio': get_stat(stats, 'Calmar Ratio'),

        # 风险指标
        'max_drawdown': -get_stat(stats, 'Max Drawdown [%]') / 100,  # 取负值表示回撤
        'volatility': get_stat(stats, 'Annualized Volatility [%]') / 100 if 'Annualized Volatility [%]' in stats.index else 0,

        # 交易统计
        'total_trades': get_stat(stats, 'Total Trades'),
        'win_rate': get_stat(stats, 'Win Rate [%]') / 100 if 'Win Rate [%]' in stats.index else 0,

        # 额外指标（Vectorbt特有）
        'profit_factor': get_stat(stats, 'Profit Factor'),
        'expectancy': get_stat(stats, 'Expectancy'),
        'avg_trade': get_stat(stats, 'Avg Trade [%]') / 100 if 'Avg Trade [%]' in stats.index else 0,
        'max_trade': get_stat(stats, 'Best Trade [%]') / 100 if 'Best Trade [%]' in stats.index else 0,
        'min_trade': get_stat(stats, 'Worst Trade [%]') / 100 if 'Worst Trade [%]' in stats.index else 0,

        # 完整统计（用于展示）
        'vb_stats': stats.to_dict() if hasattr(stats, 'to_dict') else {},

        # 组合信息
        'portfolio': portfolio,
    }

    return result


def run_param_scan_vectorbt(
    data: pd.DataFrame,
    strategy_mod,
    config: dict,
    param_grid: dict
) -> list:
    """
    使用 Vectorbt 进行参数扫描

    Args:
        data: 价格数据
        strategy_mod: 策略模块
        config: 配置
        param_grid: 参数网格，如 {'rsi_period': [7, 14, 21], 'rsi_overbought': [65, 70, 75]}

    Returns:
        list: 参数扫描结果列表
    """
    from itertools import product

    # 生成参数组合
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))

    results = []

    for params in combinations:
        # 构建参数配置
        trial_config = config.copy()
        for i, name in enumerate(param_names):
            trial_config[name] = params[i]

        try:
            # 运行策略
            signal, _, meta = strategy_mod.run(data.copy(), trial_config)

            # 回测
            bt_result = backtest_vectorbt(data, signal, trial_config)

            results.append({
                'params': dict(zip(param_names, params)),
                'cum_return': bt_result['cum_return'],
                'sharpe_ratio': bt_result['sharpe_ratio'],
                'max_drawdown': bt_result['max_drawdown'],
                'total_trades': bt_result['total_trades'],
                'win_rate': bt_result['win_rate'],
                'sortino_ratio': bt_result['sortino_ratio'],
                'calmar_ratio': bt_result['calmar_ratio'],
            })
        except Exception as e:
            continue

    return results


if __name__ == "__main__":
    # 测试
    import sys
    sys.path.insert(0, '.')

    # 模拟数据
    dates = pd.date_range('2024-01-01', periods=252)
    np.random.seed(42)
    close = pd.Series(100 * np.cumprod(1 + np.random.randn(252) * 0.02), index=dates)
    data = pd.DataFrame({'Close': close})

    # 模拟信号（简单：金叉买入，死叉卖出）
    signal = pd.Series(0, index=dates)
    for i in range(10, len(dates)):
        if i % 20 < 10:
            signal.iloc[i] = 1
        else:
            signal.iloc[i] = 0

    config = {'initial_capital': 100000, 'fees': 0.001}

    result = backtest_vectorbt(data, signal, config)

    print("Vectorbt 回测结果:")
    print(f"  累计收益: {result['cum_return']:.2%}")
    print(f"  夏普比率: {result['sharpe_ratio']:.2f}")
    print(f"  最大回撤: {result['max_drawdown']:.2%}")
    print(f"  交易次数: {result['total_trades']}")
    print(f"  胜率: {result['win_rate']:.2%}")
