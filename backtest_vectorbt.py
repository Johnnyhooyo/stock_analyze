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

    # 港股费率设置（从 config 读取，与原生引擎保持一致）
    # 买入：佣金 ~0.08% + 征费0.005% + 交易费0.0027% ≈ 0.088%
    # 卖出：买入费率 + 印花税0.1% ≈ 0.188%
    fees_rate = float(config.get('fees_rate', 0.00088))   # 买入费率
    stamp_duty = float(config.get('stamp_duty', 0.001))   # 印花税（仅卖出）
    slippage = float(config.get('slippage', 0.001))        # 滑点：与原生引擎统一，从 config 读取
    invest_fraction = float(config.get('invest_fraction', 0.95))  # 仓位比例，与原生引擎保持一致

    # 港股往返费率：买入~0.088% + 卖出~0.188%（含印花税）
    # 精确做法：买卖分别设置，此处用单向平均值作为 vectorbt 的 fees 参数
    avg_fees = (fees_rate + fees_rate + stamp_duty) / 2  # ~0.0013 (0.13%)

    # 准备数据
    close = data['Close'].copy()

    # 确保 signal 是数值类型并对齐
    signal = pd.Series(signal).astype(float)
    close, signal = close.align(signal, join='inner')

    # ── ATR 止损回测模拟（Issue #9 修复） ──────────────────────────
    risk_cfg = config.get('risk_management', {})
    if risk_cfg.get('simulate_in_backtest', True) and risk_cfg.get('use_atr_stop', True):
        try:
            from position_manager import simulate_atr_stoploss
            # 对齐 data 到 signal/close 的共同索引后再模拟
            data_aligned = data.reindex(signal.index)
            signal = simulate_atr_stoploss(
                data_aligned, signal,
                atr_period=int(risk_cfg.get('atr_period', 14)),
                atr_multiplier=float(risk_cfg.get('atr_multiplier', 2.0)),
                trailing=bool(risk_cfg.get('trailing_stop', True)),
                cooldown_bars=int(risk_cfg.get('cooldown_bars', 0)),
            ).astype(float)
        except Exception as _e:
            import warnings as _w
            _w.warn(f"[backtest_vectorbt] ATR 止损模拟失败（已跳过）: {_e}")
    # ────────────────────────────────────────────────────────────────

    # ⚠️ 前视偏差修复：信号在第 T 天收盘后生成，最早在第 T+1 天开盘执行。
    # 将信号整体后移 1 个交易日，确保不会在生成信号的同一根 K 线上成交。
    signal_shifted = signal.shift(1).fillna(0)

    # 生成买入和卖出信号
    # 买入点：当前持仓=1，前一天持仓=0
    entries = (signal_shifted == 1) & (signal_shifted.shift(1).fillna(0) == 0)
    # 卖出点：当前持仓=0，前一天持仓=1
    exits = (signal_shifted == 0) & (signal_shifted.shift(1).fillna(0) == 1)

    # 运行回测
    portfolio = vbt.Portfolio.from_signals(
        close,
        entries=entries,
        exits=exits,
        init_cash=initial_capital,
        fees=avg_fees,
        slippage=slippage,
        size=invest_fraction,       # 每次使用 invest_fraction 比例的现金
        size_type='percent',        # 按百分比仓位
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

    # 添加交易信号和持仓（使用后移的信号，与前视偏差修复保持一致）
    detail['signal'] = signal_shifted.reindex(detail.index).fillna(0).astype(int)

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
                v = stats[k]
                # 确保返回 Python 标量，防止 Series/ndarray 泄漏
                if hasattr(v, 'item'):
                    return v.item()
                return float(v) if not isinstance(v, (int, float)) else v
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

    # ⚠️ 前视偏差修复：预先切分 train/val，扫描回测只在验证集上进行
    df = data.copy().sort_index()
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)
    lookback_months = int(config.get('lookback_months', 3))
    train_years     = int(config.get('train_years', 2))
    val_start       = df.index.max() - pd.DateOffset(months=lookback_months)
    tr_start        = val_start - pd.DateOffset(years=train_years)
    train_df        = df.loc[(df.index >= tr_start) & (df.index < val_start)]
    val_df          = df.loc[df.index >= val_start]
    if train_df.empty:
        train_df = df.loc[df.index < val_start]
    if train_df.empty or val_df.empty:
        print("  [param_scan] 数据不足，无法切分 train/val，回退全量扫描")
        train_df = df
        val_df   = df

    # 生成参数组合
    param_names  = list(param_grid.keys())
    param_values = list(param_grid.values())
    combinations = list(product(*param_values))

    results = []

    for params in combinations:
        trial_config = config.copy()
        for i, name in enumerate(param_names):
            trial_config[name] = params[i]

        try:
            # 在训练集上 fit 模型
            _, model, meta = strategy_mod.run(train_df.copy(), trial_config)

            # 在验证集上生成信号（用 predict() 或 no_internal_split 模式）
            if model is not None and hasattr(strategy_mod, 'predict'):
                signal = strategy_mod.predict(model, val_df, trial_config, meta)
                signal = signal.reindex(val_df.index).fillna(0)
            else:
                vcfg = trial_config.copy()
                vcfg['no_internal_split'] = True
                signal, _, _ = strategy_mod.run(val_df.copy(), vcfg)

            # 仅在验证集上回测
            bt_result = backtest_vectorbt(val_df, signal, trial_config)

            results.append({
                'params': dict(zip(param_names, params)),
                'cum_return':   bt_result['cum_return'],
                'sharpe_ratio': bt_result['sharpe_ratio'],
                'max_drawdown': bt_result['max_drawdown'],
                'total_trades': bt_result['total_trades'],
                'win_rate':     bt_result['win_rate'],
                'sortino_ratio': bt_result['sortino_ratio'],
                'calmar_ratio': bt_result['calmar_ratio'],
            })
        except Exception:
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
