"""
Vectorbt 回测模块
基于 Vectorbt 框架的回测功能，保留原有回测逻辑
"""

import warnings
import vectorbt as vbt
import pandas as pd
import numpy as np
from typing import Dict, Any

# 抑制 Vectorbt 的某些警告
warnings.filterwarnings('ignore', module='vectorbt')


def backtest_vectorbt(
    data: pd.DataFrame,
    signal: pd.Series,
    config: dict
) -> dict:
    """使用 Vectorbt 进行回测"""
    print("[Vectorbt] 使用 Vectorbt 框架进行回测...")

    # 获取配置
    initial_capital = float(config.get('initial_capital', 100000.0))

    # 港股费率设置（默认值）
    # 买入：佣金 ~0.08% + 征费0.005% + 交易费0.0027% ≈ 0.088%
    # 卖出：买入费率 + 印花税0.1% ≈ 0.188%
    fees_rate = config.get('fees_rate', 0.00088)  # 买入费率
    stamp_duty = config.get('stamp_duty', 0.001)  # 印花税（仅卖出）

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

    # 港股往返费率：买入~0.088% + 卖出~0.188%（含印花税）
    # 使用平均费率简化计算
    avg_fees = (fees_rate + fees_rate + stamp_duty) / 2  # ~0.0013 (0.13%)

    # 运行回测
    portfolio = vbt.Portfolio.from_signals(
        close,
        entries=entries,
        exits=exits,
        init_cash=initial_capital,
        fees=avg_fees,
        slippage=0.001,  # 滑点
        freq='1D',  # 设置频率，避免夏普等指标警告
    )

    # 获取统计信息
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            stats = portfolio.stats()
    except Exception as e:
        # 如果 stats 失败，返回默认值
        return {
            'cum_return': 0,
            'annualized_return': 0,
            'sharpe_ratio': 0,
            'sortino_ratio': 0,
            'calmar_ratio': 0,
            'max_drawdown': 0,
            'volatility': 0,
            'win_rate': 0,
            'profit_loss_ratio': 0,
            'total_trades': 0,
            'buy_cnt': 0,
            'sell_cnt': 0,
            'detail': pd.DataFrame(),
        }

    # 创建 detail DataFrame 用于绘图
    detail = data.copy()

    # 添加交易信号和持仓
    detail['signal'] = signal.reindex(detail.index).fillna(0).astype(int)

    # 计算持仓
    detail['position'] = detail['signal']

    # 计算交易点
    detail['trade'] = 0
    prev_pos = detail['position'].shift(1).fillna(0)
    detail['trade'] = np.where(
        (detail['position'] == 1) & (prev_pos == 0), 1,  # 买入
        np.where((detail['position'] == 0) & (prev_pos == 1), -1, 0)  # 卖出
    )

    # 计算每日收益和策略收益
    detail['daily_return'] = detail['Close'].pct_change(fill_method=None)
    detail['strategy'] = detail['position'].shift(1) * detail['daily_return']
    detail['strategy'] = detail['strategy'].fillna(0)

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

        # 详细数据（用于绘图）
        'detail': detail,
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

    config = {'initial_capital': 100000, 'fees_rate': 0.00088, 'stamp_duty': 0.001}

    result = backtest_vectorbt(data, signal, config)

    print("Vectorbt 回测结果:")
    print(f"  累计收益: {result['cum_return']:.2%}")
    print(f"  夏普比率: {result['sharpe_ratio']:.2f}")
    print(f"  最大回撤: {result['max_drawdown']:.2%}")
    print(f"  交易次数: {result['total_trades']}")
    print(f"  胜率: {result['win_rate']:.2%}")
