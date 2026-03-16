"""
策略验证模块 - 补充训练后的测试环节
包括：
1. 样本外测试 (Out-of-Sample Testing)
2. Walk-Forward 分析
3. 交易成本模拟
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import joblib


def out_of_sample_test(
    data: pd.DataFrame,
    strategy_mod,
    params: dict,
    config: dict,
    train_months: int = 12,
    test_months: int = 3
) -> dict:
    """
    样本外测试：将数据划分为训练期和测试期
    - 训练期：用历史数据优化参数
    - 测试期：用优化后的参数回测（模拟真实情况）

    Args:
        data: 历史数据
        strategy_mod: 策略模块
        params: 最优参数
        config: 配置
        train_months: 训练期月数
        test_months: 测试期月数

    Returns:
        测试结果字典
    """
    from analyze_factor import backtest as bt, VECTORBT_AVAILABLE
    try:
        from backtest_vectorbt import backtest_vectorbt as bt_vbt
    except ImportError:
        bt_vbt = None

    df = data.copy().sort_index()

    # 计算训练期和测试期的分割点
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)

    dates = df.index
    total_months = (dates[-1].year - dates[0].year) * 12 + dates[-1].month - dates[0].month

    if total_months < train_months + test_months:
        return {
            'success': False,
            'message': f'数据不足: 总共 {total_months} 个月，需要 {train_months + test_months} 个月'
        }

    # 训练期：取最后 test_months 个月之前的数据
    train_end = dates[-1] - pd.DateOffset(months=test_months)
    train_df = df[df.index <= train_end]

    # 测试期：取最后 test_months 个月的数据
    test_df = df[df.index > train_end]

    if len(test_df) < 20:
        return {
            'success': False,
            'message': f'测试期数据不足: {len(test_df)} 天'
        }

    # 用最优参数在测试期运行策略
    trial_cfg = config.copy()
    trial_cfg.update(params)

    try:
        sig, _, meta = strategy_mod.run(test_df, trial_cfg)
    except Exception as e:
        return {
            'success': False,
            'message': f'策略运行失败: {e}'
        }

    # 回测测试期（根据配置选择引擎）
    backtest_engine = config.get('backtest_engine', 'native')
    if backtest_engine == 'vectorbt' and bt_vbt is not None:
        test_result = bt_vbt(test_df, sig, config)
    else:
        test_result = bt(test_df, sig, config)

    # 计算相对于买入持有策略的超额收益
    buy_hold_return = (test_df['Close'].iloc[-1] / test_df['Close'].iloc[0]) - 1
    strategy_return = test_result['cum_return']
    excess_return = strategy_return - buy_hold_return

    return {
        'success': True,
        'train_period': f"{train_df.index[0].date()} ~ {train_df.index[-1].date()}",
        'test_period': f"{test_df.index[0].date()} ~ {test_df.index[-1].date()}",
        'train_days': len(train_df),
        'test_days': len(test_df),
        'cum_return': strategy_return,
        'buy_hold_return': buy_hold_return,
        'excess_return': excess_return,
        'sharpe_ratio': test_result.get('sharpe_ratio', 0),
        'max_drawdown': test_result.get('max_drawdown', 0),
        'win_rate': test_result.get('win_rate', 0),
        'total_trades': test_result.get('total_trades', 0),
    }


def walk_forward_analysis(
    data: pd.DataFrame,
    strategy_mod,
    config: dict,
    train_months: int = 12,
    test_months: int = 3,
    step_months: int = 3
) -> dict:
    """
    Walk-Forward 分析：滚动窗口验证策略稳定性

    Args:
        data: 历史数据
        strategy_mod: 策略模块
        config: 配置
        train_months: 每次训练期月数
        test_months: 每次测试期月数
        step_months: 滚动步长

    Returns:
        分析结果
    """
    from analyze_factor import backtest as bt, VECTORBT_AVAILABLE
    try:
        from backtest_vectorbt import backtest_vectorbt as bt_vbt
    except ImportError:
        bt_vbt = None

    df = data.copy().sort_index()

    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)

    dates = df.index
    total_months = (dates[-1].year - dates[0].year) * 12 + dates[-1].month - dates[0].month

    if total_months < train_months + test_months:
        return {
            'success': False,
            'message': f'数据不足'
        }

    # 计算滚动窗口
    results = []
    current_end = dates[-1]

    while True:
        # 测试期结束日期
        test_end = current_end
        test_start = test_end - pd.DateOffset(months=test_months) + pd.DateOffset(days=1)

        # 训练期结束日期
        train_end = test_start - pd.DateOffset(days=1)
        train_start = train_end - pd.DateOffset(months=train_months)

        if train_start < dates[0]:
            break

        # 获取训练数据和测试数据
        train_df = df[(df.index >= train_start) & (df.index <= train_end)]
        test_df = df[(df.index >= test_start) & (df.index <= test_end)]

        if len(train_df) < 60 or len(test_df) < 20:
            print(f"  ⚠️ 跳过窗口 {test_start.date()} ~ {test_end.date()}: 数据不足")
            current_end = test_start - pd.DateOffset(months=step_months)
            continue

        # 运行策略：合并训练集和测试集，让指标能正确预热
        try:
            # 合并数据：训练集 + 测试集
            combined_df = pd.concat([train_df, test_df])
            sig, _, _ = strategy_mod.run(combined_df, config)

            # 只取测试集部分的信号
            test_sig = sig.loc[test_start:test_end]

            # 根据配置选择回测引擎
            backtest_engine = config.get('backtest_engine', 'native')
            if backtest_engine == 'vectorbt' and bt_vbt is not None:
                test_result = bt_vbt(test_df, test_sig, config)
            else:
                test_result = bt(test_df, test_sig, config)

            # 检查是否有效（必须有交易）
            if test_result.get('total_trades', 0) > 0:
                results.append({
                    'train_period': f"{train_start.date()} ~ {train_end.date()}",
                    'test_period': f"{test_start.date()} ~ {test_end.date()}",
                    'cum_return': test_result.get('cum_return', 0),
                    'sharpe_ratio': test_result.get('sharpe_ratio', 0),
                    'max_drawdown': test_result.get('max_drawdown', 0),
                    'total_trades': test_result.get('total_trades', 0),
                    'win_rate': test_result.get('win_rate', 0),  # 交易胜率
                })
            else:
                print(f"  ⚠️ 跳过窗口 {test_start.date()} ~ {test_end.date()}: 无交易信号")
        except Exception as e:
            print(f"  ⚠️ 跳过窗口 {test_start.date()} ~ {test_end.date()}: {e}")

        current_end = test_start - pd.DateOffset(months=step_months)

        if current_end < dates[0] + pd.DateOffset(months=train_months):
            break

    if not results:
        return {
            'success': False,
            'message': '没有有效的滚动测试结果'
        }

    # 汇总统计
    returns = [r['cum_return'] for r in results]
    sharpes = [r['sharpe_ratio'] for r in results]
    trade_win_rates = [r.get('win_rate', 0) for r in results if r.get('win_rate', 0) > 0]
    total_trades_all = sum(r.get('total_trades', 0) for r in results)

    # 窗口胜率：盈利窗口数/总窗口数
    window_win_rate = sum(1 for r in returns if r > 0) / len(returns) if returns else 0

    # 交易胜率：汇总所有窗口的交易
    all_trade_returns = []
    for r in results:
        # 简单处理：每个窗口的交易胜率作为参考
        pass

    return {
        'success': True,
        'windows': results,
        'summary': {
            'total_windows': len(results),
            'profitable_windows': sum(1 for r in returns if r > 0),
            'avg_return': np.mean(returns),
            'return_std': np.std(returns),
            'avg_sharpe': np.mean([s for s in sharpes if not np.isnan(s)]),
            'window_win_rate': window_win_rate,  # 窗口胜率
            'trade_win_rate': np.mean(trade_win_rates) if trade_win_rates else 0,  # 交易胜率（各窗口平均）
            'total_trades': total_trades_all,
        }
    }


def backtest_with_costs(
    data: pd.DataFrame,
    signal: pd.Series,
    config: dict,
    commission_rate: float = 0.001,  # 佣金费率 0.1%
    slippage: float = 0.001,          # 滑点 0.1%
) -> dict:
    """
    考虑交易成本的回测

    Args:
        data: 价格数据
        signal: 交易信号
        config: 配置
        commission_rate: 佣金费率
        slippage: 滑点

    Returns:
        回测结果（包含成本调整后的收益）
    """
    from analyze_factor import backtest, VECTORBT_AVAILABLE
    try:
        from backtest_vectorbt import backtest_vectorbt as bt_vbt
    except ImportError:
        bt_vbt = None

    # 原始回测（根据配置选择引擎）
    backtest_engine = config.get('backtest_engine', 'native')
    if backtest_engine == 'vectorbt' and bt_vbt is not None:
        result = bt_vbt(data, signal, config)
    else:
        result = backtest(data, signal, config)

    # 估算交易成本
    total_trades = result.get('total_trades', 0)
    avg_trade_value = 100000  # 假设平均每次交易 10 万

    total_commission = total_trades * avg_trade_value * commission_rate
    total_slippage = total_trades * avg_trade_value * slippage
    total_costs = total_commission + total_slippage

    # 成本调整后的收益
    adjusted_return = result['cum_return'] * 100000 - total_costs
    adjusted_return_pct = adjusted_return / 100000

    result['total_costs'] = total_costs
    result['adjusted_cum_return'] = adjusted_return_pct
    result['commission_rate'] = commission_rate
    result['slippage'] = slippage

    return result


def generate_test_report(
    data: pd.DataFrame,
    strategy_mod,
    params: dict,
    config: dict,
    output_dir: Path = None
) -> str:
    """
    生成完整的策略验证报告

    Returns:
        markdown 格式报告
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / 'data' / 'reports'

    # 1. 样本外测试
    oos_result = out_of_sample_test(
        data, strategy_mod, params, config,
        train_months=12, test_months=3
    )

    # 2. Walk-Forward 分析
    wf_result = walk_forward_analysis(
        data, strategy_mod, config,
        train_months=12, test_months=3, step_months=3
    )

    # 生成报告
    md = f"""# 策略验证报告

生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. 样本外测试 (Out-of-Sample Test)

"""
    if oos_result.get('success'):
        md += f"""
| 指标 | 值 |
|------|-----|
| 训练期 | {oos_result['train_period']} |
| 测试期 | {oos_result['test_period']} |
| 策略收益 | {oos_result['cum_return']:.2%} |
| 买入持有收益 | {oos_result['buy_hold_return']:.2%} |
| 超额收益 | {oos_result['excess_return']:+.2%} |
| 夏普比率 | {oos_result['sharpe_ratio']:.4f} |
| 最大回撤 | {oos_result['max_drawdown']:.2%} |
| 交易次数 | {oos_result['total_trades']} |
"""
    else:
        md += f"\n样本外测试失败: {oos_result.get('message', '未知错误')}\n"

    md += f"""

## 2. Walk-Forward 分析

"""
    if wf_result.get('success'):
        summary = wf_result['summary']
        md += f"""
### 汇总统计

| 指标 | 值 |
|------|-----|
| 总窗口数 | {summary['total_windows']} |
| 盈利窗口数 | {summary['profitable_windows']} |
| 窗口胜率 | {summary['window_win_rate']:.2%} |
| 交易胜率 | {summary['trade_win_rate']:.2%} |
| 平均收益 | {summary['avg_return']:.2%} |
| 收益标准差 | {summary['return_std']:.2%} |
| 平均夏普率 | {summary['avg_sharpe']:.4f} |

### 各窗口详情

| 训练期 | 测试期 | 收益 | 夏普率 | 回撤 | 交易次数 | 交易胜率 |
|--------|--------|------|--------|------|----------|----------|
"""
        for w in wf_result['windows']:
            trade_win = w.get('win_rate', 0)
            md += f"| {w['train_period']} | {w['test_period']} | {w['cum_return']:.2%} | {w['sharpe_ratio']:.4f} | {w['max_drawdown']:.2%} | {w['total_trades']} | {trade_win:.2%} |\n"
    else:
        md += f"\nWalk-Forward 分析失败: {wf_result.get('message', '未知错误')}\n"

    md += """

## 3. 结论

"""
    if oos_result.get('success') and wf_result.get('success'):
        oos_excess = oos_result['excess_return']
        wf_summary = wf_result['summary']
        window_win_rate = wf_summary.get('window_win_rate', 0)
        trade_win_rate = wf_summary.get('trade_win_rate', 0)

        if oos_excess > 0 and window_win_rate > 0.5:
            conclusion = "策略表现良好，具有正向超额收益且 Walk-Forward 胜率较高。"
        elif oos_excess > 0:
            conclusion = "策略具有一定的超额收益，但 Walk-Forward 胜率偏低，需进一步优化。"
        else:
            conclusion = "策略在样本外测试中未能跑赢基准，建议重新审视策略逻辑。"

        md += f"- **样本外超额收益**: {oos_excess:+.2%}\n"
        md += f"- **窗口胜率**: {window_win_rate:.2%}\n"
        md += f"- **交易胜率**: {trade_win_rate:.2%}\n\n"
        md += f"**结论**: {conclusion}\n"

    # 保存报告
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report_path.write_text(md, encoding='utf-8')

    # 返回数据和报告
    validation_data = {
        'out_of_sample': oos_result if oos_result.get('success') else {},
        'walk_forward': wf_result.get('summary', {}) if wf_result.get('success') else {},
    }

    return md, report_path, validation_data
