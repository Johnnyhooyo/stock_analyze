"""
Optuna 超参数优化模块
替代随机搜索，使用贝叶斯优化更高效地寻找最优参数

用法:
    from optimize_with_optuna import optimize_strategy
    best_params = optimize_strategy(data, strategy_mod, config, n_trials=100)
"""

import optuna
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Callable
import warnings

# 导入回测引擎
try:
    from backtest_vectorbt import backtest_vectorbt
except ImportError:
    backtest_vectorbt = None

try:
    from analyze_factor import backtest as backtest_native
except ImportError:
    backtest_native = None


# ═══════════════════════════════════════════════════════════════════
# 参数空间定义
# ═══════════════════════════════════════════════════════════════════

# 通用参数空间
COMMON_PARAMS = {
    'test_days': (3, 20),
    'drawdown_pct': [0.01, 0.015, 0.02, 0.03, 0.05],
}

# 策略特定参数空间
STRATEGY_PARAMS = {
    'macd_rsi_trend': {
        'macd_fast': (5, 20),
        'macd_slow': (15, 40),
        'macd_signal': (5, 15),
        'rsi_period': (7, 28),
        'rsi_oversold': (20, 45),
        'rsi_overbought': (55, 80),
        'trend_period': (3, 15),
    },
    'bollinger_rsi_trend': {
        'bb_period': (10, 30),
        'bb_std': (1.5, 3.0),
        'rsi_period': (7, 21),
        'rsi_oversold': (20, 40),
        'rsi_overbought': (60, 80),
        'trend_period': (3, 15),
    },
    'rsi_reversion': {
        'rsi_period': (7, 28),
        'rsi_oversold': (15, 35),
        'rsi_overbought': (65, 85),
        'exit_rsi': (45, 70),
    },
    'ma_crossover': {
        'ma_fast': (3, 15),
        'ma_slow': (20, 60),
        'trend_ma': (10, 50),
    },
    'kdj_obv': {
        'kdj_period': (5, 14),
        'kdj_k': (10, 30),
        'kdj_d': (10, 30),
        'obv_ma_period': (10, 40),
    },
    'kdj_pvt': {
        'kdj_period': (5, 14),
        'kdj_k': (10, 30),
        'kdj_d': (10, 30),
        'pvt_ma_period': (10, 40),
    },
    'rsi_obv': {
        'rsi_period': (7, 28),
        'rsi_oversold': (15, 35),
        'rsi_overbought': (65, 85),
        'obv_ma_period': (10, 40),
        'fib_period': (20, 120),
    },
    'rsi_pvt': {
        'rsi_period': (7, 28),
        'rsi_oversold': (15, 35),
        'rsi_overbought': (65, 85),
        'pvt_ma_period': (10, 40),
        'fib_period': (20, 120),
    },
    'bollinger_breakout': {
        'bb_period': (10, 30),
        'bb_std': (1.5, 3.0),
        'atr_period': (10, 28),
        'atr_multiplier': (1.5, 4.0),
    },
    'atr_breakout': {
        'atr_period': (10, 28),
        'atr_multiplier': (1.5, 4.0),
        'trend_ma': (10, 50),
    },
    'stochastic_oscillator': {
        'k_period': (10, 21),
        'd_period': (3, 10),
        'oversold': (10, 25),
        'overbought': (75, 90),
    },
    'vwap_momentum': {
        'vwap_period': (10, 30),
        'momentum_period': (5, 20),
        'volume_ma_period': (10, 40),
    },
    'volume_price_trend': {
        'vpt_ma_period': (10, 40),
        'volume_ma_period': (10, 40),
    },
    'rsi_divergence': {
        'rsi_period': (7, 21),
        'rsi_oversold': (20, 40),
        'rsi_overbought': (60, 80),
        'lookback': (5, 20),
    },
    'macd_rsi_combo': {
        'macd_fast': (5, 20),
        'macd_slow': (15, 40),
        'macd_signal': (5, 15),
        'rsi_period': (7, 21),
        'rsi_oversold': (20, 40),
        'rsi_overbought': (60, 80),
    },
    'xgboost_enhanced': {
        'test_days': (3, 15),
        'xgb_n_estimators': (50, 200),
        'xgb_max_depth': (3, 8),
        'xgb_learning_rate': (0.01, 0.3),
        'label_period': (1, 5),
    },
    'lightgbm_enhanced': {
        'test_days': (3, 15),
        'lgbm_n_estimators': (50, 200),
        'lgbm_max_depth': (3, 8),
        'lgbm_learning_rate': (0.01, 0.3),
        'lgbm_num_leaves': (15, 63),
        'label_period': (1, 5),
    },
}


def _get_param_space(strategy_name: str) -> dict:
    """获取策略的参数空间"""
    space = COMMON_PARAMS.copy()
    if strategy_name in STRATEGY_PARAMS:
        space.update(STRATEGY_PARAMS[strategy_name])
    return space


def _suggest_params(trial: optuna.Trial, param_space: dict) -> dict:
    """根据参数空间建议参数"""
    params = {}
    for name, bounds in param_space.items():
        if isinstance(bounds, (list, tuple)) and len(bounds) == 2:
            # 连续区间
            if isinstance(bounds[0], float):
                params[name] = trial.suggest_float(name, bounds[0], bounds[1])
            else:
                params[name] = trial.suggest_int(name, bounds[0], bounds[1])
        elif isinstance(bounds, list):
            # 离散选择
            params[name] = trial.suggest_categorical(name, bounds)
    return params


# ═══════════════════════════════════════════════════════════════════
# 优化目标函数
# ═══════════════════════════════════════════════════════════════════

class StrategyOptimizer:
    """策略优化器"""

    def __init__(
        self,
        data: pd.DataFrame,
        strategy_mod,
        config: dict,
        metric: str = 'sharpe_ratio',
        direction: str = 'maximize',
        use_vectorbt: bool = True,
    ):
        self.data = data
        self.strategy_mod = strategy_mod
        self.config = config
        self.metric = metric
        self.direction = direction
        self.use_vectorbt = use_vectorbt and backtest_vectorbt is not None

        # 获取策略参数空间
        strategy_name = getattr(strategy_mod, 'NAME', 'unknown')
        self.strategy_name = strategy_name
        self.param_space = _get_param_space(strategy_name)

        # 记录最佳结果
        self.best_value = float('-inf') if direction == 'maximize' else float('inf')
        self.best_params = None
        self.all_results = []

    def objective(self, trial: optuna.Trial) -> float:
        """Optuna 目标函数"""
        # 1. 采样参数
        params = _suggest_params(trial, self.param_space)
        trial_cfg = self.config.copy()
        trial_cfg.update(params)

        # 2. 运行策略
        try:
            signal, _, meta = self.strategy_mod.run(self.data.copy(), trial_cfg)
        except Exception as e:
            raise optuna.TrialPruned(f"策略运行失败: {e}")

        # 3. 回测
        try:
            if self.use_vectorbt:
                result = backtest_vectorbt(self.data, signal, trial_cfg)
            else:
                result = backtest_native(self.data, signal, trial_cfg)
        except Exception as e:
            raise optuna.TrialPruned(f"回测失败: {e}")

        # 4. 获取配置中的阈值
        min_return = float(self.config.get('min_return', 0.10))
        min_sharpe = float(self.config.get('min_sharpe_ratio', 1.0))
        max_dd = float(self.config.get('max_drawdown', -0.15))
        min_trades = int(self.config.get('min_total_trades', 5))

        # 5. 多维度验证（防止过拟合）
        cum_return = result.get('cum_return', 0)
        sharpe = result.get('sharpe_ratio', 0)
        max_drawdown = result.get('max_drawdown', 0)
        total_trades = result.get('total_trades', 0)

        # 检查各项指标，不满足则剪枝
        if total_trades < min_trades:
            raise optuna.TrialPruned(f"交易次数不足({total_trades}<{min_trades})")
        if cum_return < min_return:
            raise optuna.TrialPruned(f"收益不达标({cum_return:.2%}<{min_return:.2%})")
        if sharpe < min_sharpe:
            raise optuna.TrialPruned(f"夏普率不足({sharpe:.2f}<{min_sharpe:.2f})")
        if max_drawdown < max_dd:
            raise optuna.TrialPruned(f"回撤过大({max_drawdown:.2%}<{max_dd:.2%})")

        # 6. 获取优化指标
        value = result.get(self.metric, 0)
        if value is None or np.isnan(value):
            value = 0

        # 记录结果
        self.all_results.append({
            'params': params,
            'value': value,
            'cum_return': result.get('cum_return', 0),
            'sharpe_ratio': result.get('sharpe_ratio', 0),
            'max_drawdown': result.get('max_drawdown', 0),
            'win_rate': result.get('win_rate', 0),
            'total_trades': total_trades,
        })

        # 更新最佳（此时已通过所有阈值检查）
        if self.direction == 'maximize':
            if value > self.best_value:
                self.best_value = value
                self.best_params = params.copy()
        else:
            if value < self.best_value:
                self.best_value = value
                self.best_params = params.copy()

        # 注意：这里不启用自动剪枝，因为单步报告会触发默认的 median 剪枝
        # 如果需要剪枝，可以在创建 study 时配置 pruner

        return value


# ═══════════════════════════════════════════════════════════════════
# 主优化函数
# ═══════════════════════════════════════════════════════════════════

def optimize_strategy(
    data: pd.DataFrame,
    strategy_mod,
    config: dict,
    n_trials: int = 100,
    metric: str = 'sharpe_ratio',
    direction: str = 'maximize',
    timeout: Optional[int] = None,
    n_jobs: int = -1,  # 默认使用全部 CPU 核心并行
    use_vectorbt: bool = True,
    verbose: bool = True,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
) -> Dict[str, Any]:
    """
    使用 Optuna 优化策略参数

    Args:
        data: 历史数据
        strategy_mod: 策略模块
        config: 基础配置
        n_trials: 搜索次数
        metric: 优化指标 (sharpe_ratio, cum_return, max_drawdown, sortino_ratio)
        direction: 优化方向 (maximize / minimize)
        timeout: 超时秒数
        n_jobs: 并行数
        use_vectorbt: 是否使用 Vectorbt 回测
        verbose: 是否打印进度
        study_name: 研究名称（用于持久化）
        storage: 存储路径（用于持久化）

    Returns:
        优化结果字典
    """
    strategy_name = getattr(strategy_mod, 'NAME', 'unknown')

    if verbose:
        print(f"\n{'='*56}")
        print(f"  Optuna 参数优化: {strategy_name}")
        print(f"{'='*56}")
        print(f"  目标: {direction} {metric}")
        print(f"  搜索次数: {n_trials}")
        if timeout:
            print(f"  超时: {timeout}秒")
        print(f"  回测引擎: {'Vectorbt' if use_vectorbt else 'Native'}")

    # 创建优化器
    optimizer = StrategyOptimizer(
        data=data,
        strategy_mod=strategy_mod,
        config=config,
        metric=metric,
        direction=direction,
        use_vectorbt=use_vectorbt,
    )

    # 创建 study
    study = optuna.create_study(
        direction=direction,
        study_name=study_name or strategy_name,
        storage=storage,
        load_if_exists=True,
    )

    # 添加参数定义
    def objective_wrapper(trial):
        return optimizer.objective(trial)

    # 运行优化
    study.optimize(
        objective_wrapper,
        n_trials=n_trials,
        timeout=timeout,
        n_jobs=n_jobs,
        show_progress_bar=verbose and n_jobs == 1,
    )

    # 获取结果
    try:
        best_params = study.best_params if study.best_trial and study.best_trial.state == optuna.trial.TrialState.COMPLETE else {}
        best_value = study.best_value if study.best_trial and study.best_trial.state == optuna.trial.TrialState.COMPLETE else None
    except ValueError:
        best_params = {}
        best_value = None

    if verbose:
        print(f"\n  优化完成!")
        if best_value is not None:
            print(f"  最佳 {metric}: {best_value:.4f}")
            print(f"  最佳参数: {best_params}")
        else:
            completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
            print(f"  未找到有效结果 (有效试验数: {completed})")

    return {
        'strategy_name': strategy_name,
        'best_params': best_params,
        'best_value': best_value,
        'n_trials': len(study.trials),
        'all_results': optimizer.all_results,
        'study': study,
    }


def optimize_all_strategies(
    data: pd.DataFrame,
    strategy_mods: list,
    config: dict,
    n_trials: int = 50,
    metric: str = 'sharpe_ratio',
    direction: str = 'maximize',
    use_vectorbt: bool = True,
    verbose: bool = True,
) -> list:
    """
    批量优化多个策略

    Args:
        data: 历史数据
        strategy_mods: 策略模块列表
        config: 基础配置
        n_trials: 每个策略的搜索次数
        metric: 优化指标
        direction: 优化方向
        use_vectorbt: 是否使用 Vectorbt
        verbose: 是否打印进度

    Returns:
        优化结果列表
    """
    results = []

    for mod in strategy_mods:
        strategy_name = getattr(mod, 'NAME', 'unknown')
        if verbose:
            print(f"\n{'━'*56}")
            print(f"  优化策略: {strategy_name}")
            print(f"{'━'*56}")

        result = optimize_strategy(
            data=data,
            strategy_mod=mod,
            config=config,
            n_trials=n_trials,
            metric=metric,
            direction=direction,
            use_vectorbt=use_vectorbt,
            verbose=verbose,
        )
        results.append(result)

    # 按最佳值排序
    results.sort(key=lambda x: x['best_value'], reverse=(direction == 'maximize'))

    if verbose:
        print(f"\n{'═'*56}")
        print("  优化结果排行榜")
        print(f"{'═'*56}")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r['strategy_name']:<22} {metric}={r['best_value']:.4f}")

    return results


# ═══════════════════════════════════════════════════════════════════
# 多目标优化
# ═══════════════════════════════════════════════════════════════════

def optimize_multiobjective(
    data: pd.DataFrame,
    strategy_mod,
    config: dict,
    n_trials: int = 100,
    metrics: list = None,
    use_vectorbt: bool = True,
    verbose: bool = True,
) -> optuna.Study:
    """
    多目标优化（帕累托前沿）

    Args:
        data: 历史数据
        strategy_mod: 策略模块
        config: 基础配置
        n_trials: 搜索次数
        metrics: 优化指标列表，默认 ['cum_return', 'max_drawdown']
        use_vectorbt: 是否使用 Vectorbt

    Returns:
        Optuna Study 对象
    """
    if metrics is None:
        metrics = ['cum_return', 'max_drawdown']

    strategy_name = getattr(strategy_mod, 'NAME', 'unknown')

    if verbose:
        print(f"\n{'='*56}")
        print(f"  多目标优化: {strategy_name}")
        print(f"  目标: {metrics}")
        print(f"{'='*56}")

    # 创建多目标 study
    study = optuna.create_study(
        directions=['maximize' if m != 'max_drawdown' else 'minimize' for m in metrics],
        study_name=f"{strategy_name}_multi",
    )

    optimizer = StrategyOptimizer(
        data=data,
        strategy_mod=strategy_mod,
        config=config,
        metric=metrics[0],
        direction='maximize',
        use_vectorbt=use_vectorbt,
    )

    def objective(trial):
        # 单次trial可能返回多个值
        params = _suggest_params(trial, optimizer.param_space)
        trial_cfg = config.copy()
        trial_cfg.update(params)

        try:
            signal, _, _ = strategy_mod.run(data.copy(), trial_cfg)
            if use_vectorbt:
                result = backtest_vectorbt(data, signal, trial_cfg)
            else:
                result = backtest_native(data, signal, trial_cfg)
        except Exception:
            raise optuna.TrialPruned()

        # 返回多个目标
        values = []
        for m in metrics:
            v = result.get(m, 0)
            if np.isnan(v):
                v = 0 if m == 'max_drawdown' else -999
            values.append(v)

        return values

    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

    if verbose:
        print(f"\n  帕累托最优解数量: {len(study.best_trials)}")

    return study


# ═══════════════════════════════════════════════════════════════════
# 便捷函数
# ═══════════════════════════════════════════════════════════════════

def quick_optimize(
    data: pd.DataFrame,
    strategy_name: str,
    config: dict,
    n_trials: int = 50,
    **kwargs
) -> Dict[str, Any]:
    """
    快速优化 - 自动导入策略模块

    Args:
        data: 历史数据
        strategy_name: 策略名称 (如 'macd_rsi_trend')
        config: 基础配置
        n_trials: 搜索次数
        **kwargs: 其他参数

    Returns:
        优化结果
    """
    # 动态导入策略模块
    try:
        from importlib import import_module
        mod = import_module(f'strategies.{strategy_name}')
    except ImportError:
        raise ValueError(f"找不到策略: strategies/{strategy_name}.py")

    return optimize_strategy(
        data=data,
        strategy_mod=mod,
        config=config,
        n_trials=n_trials,
        **kwargs
    )
