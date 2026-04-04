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
from typing import Dict, Any, Optional

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
    'rnn_trend': {
        'rnn_hidden_size': [32, 64, 128],
        'rnn_num_layers': [1, 2],
        'rnn_dropout': (0.1, 0.5),
        'rnn_window': (15, 60),
        'rnn_label_period': (3, 10),
        'rnn_label_threshold': (0.01, 0.05),
        'rnn_epochs': [30, 50, 80],
        'rnn_lr': (0.0005, 0.005),
        'rnn_cell_type': ['gru', 'lstm'],
    },
    'xgboost_enhanced': {
        'test_days': (3, 15),
        'xgb_n_estimators': (50, 200),
        'xgb_max_depth': (3, 8),
        'xgb_learning_rate': (0.01, 0.3),
        'label_period': (1, 5),
        # 修复项4：正则化参数
        'xgb_subsample': (0.6, 1.0),
        'xgb_colsample_bytree': (0.6, 1.0),
        'xgb_reg_alpha': (0.0, 1.0),
        'xgb_reg_lambda': (0.0, 1.0),
        'xgb_min_child_weight': (1, 10),
    },
    'xgboost_enhanced_tsfresh': {
        'test_days': (3, 15),
        'xgb_n_estimators': (50, 200),
        'xgb_max_depth': (3, 8),
        'xgb_learning_rate': (0.01, 0.3),
        'label_period': (1, 5),
        # 修复项4：正则化参数
        'xgb_subsample': (0.6, 1.0),
        'xgb_colsample_bytree': (0.6, 1.0),
        'xgb_reg_alpha': (0.0, 1.0),
        'xgb_reg_lambda': (0.0, 1.0),
        'xgb_min_child_weight': (1, 10),
    },
    'xgboost_enhanced_ta_tsfresh': {
        'test_days': (3, 15),
        'xgb_n_estimators': (50, 200),
        'xgb_max_depth': (3, 8),
        'xgb_learning_rate': (0.01, 0.3),
        'label_period': (1, 5),
        # 修复项4：正则化参数
        'xgb_subsample': (0.6, 1.0),
        'xgb_colsample_bytree': (0.6, 1.0),
        'xgb_reg_alpha': (0.0, 1.0),
        'xgb_reg_lambda': (0.0, 1.0),
        'xgb_min_child_weight': (1, 10),
    },
    'lightgbm_enhanced': {
        'test_days': (3, 15),
        'lgbm_n_estimators': (50, 200),
        'lgbm_max_depth': (3, 8),
        'lgbm_learning_rate': (0.01, 0.3),
        'lgbm_num_leaves': (15, 63),
        'label_period': (1, 5),
        # 修复项4：LightGBM 正则化参数（使用原生参数名）
        'lgb_feature_fraction': (0.6, 1.0),
        'lgb_bagging_fraction': (0.6, 1.0),
        'lgb_reg_alpha': (0.0, 1.0),
        'lgb_reg_lambda': (0.0, 1.0),
        'lgb_min_child_samples': (10, 50),
    },
    'lightgbm_enhanced_tsfresh': {
        'test_days': (3, 15),
        'lgbm_n_estimators': (50, 200),
        'lgbm_max_depth': (3, 8),
        'lgbm_learning_rate': (0.01, 0.3),
        'lgbm_num_leaves': (15, 63),
        'label_period': (1, 5),
        # 修复项4：LightGBM 正则化参数（使用原生参数名）
        'lgb_feature_fraction': (0.6, 1.0),
        'lgb_bagging_fraction': (0.6, 1.0),
        'lgb_reg_alpha': (0.0, 1.0),
        'lgb_reg_lambda': (0.0, 1.0),
        'lgb_min_child_samples': (10, 50),
    },
    'lightgbm_enhanced_ta_tsfresh': {
        'test_days': (3, 15),
        'lgbm_n_estimators': (50, 200),
        'lgbm_max_depth': (3, 8),
        'lgbm_learning_rate': (0.01, 0.3),
        'lgbm_num_leaves': (15, 63),
        'label_period': (1, 5),
        # 修复项4：LightGBM 正则化参数（使用原生参数名）
        'lgb_feature_fraction': (0.6, 1.0),
        'lgb_bagging_fraction': (0.6, 1.0),
        'lgb_reg_alpha': (0.0, 1.0),
        'lgb_reg_lambda': (0.0, 1.0),
        'lgb_min_child_samples': (10, 50),
    },
}


def _get_param_space(strategy_name: str, strategy_mod=None) -> dict:
    """获取策略的参数空间，优先读取策略模块中的 PARAM_SPACE 属性"""
    space = COMMON_PARAMS.copy()
    # 优先从策略模块本身读取（单一维护来源）
    if strategy_mod is not None and hasattr(strategy_mod, 'PARAM_SPACE'):
        space.update(strategy_mod.PARAM_SPACE)
    elif strategy_name in STRATEGY_PARAMS:
        space.update(STRATEGY_PARAMS[strategy_name])
    return space


def _suggest_params(trial: optuna.Trial, param_space: dict) -> dict:
    """根据参数空间建议参数"""
    params = {}
    for name, bounds in param_space.items():
        if isinstance(bounds, tuple) and len(bounds) == 2:
            # 连续区间 (tuple)
            if isinstance(bounds[0], float):
                params[name] = trial.suggest_float(name, bounds[0], bounds[1])
            else:
                params[name] = trial.suggest_int(name, bounds[0], bounds[1])
        elif isinstance(bounds, list):
            # 离散选择 (list)
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

        # 获取策略参数空间（优先读取策略模块的 PARAM_SPACE）
        strategy_name = getattr(strategy_mod, 'NAME', 'unknown')
        self.strategy_name = strategy_name
        self.param_space = _get_param_space(strategy_name, strategy_mod)

        # 判断是否为多股票策略
        self.train_type = self._get_training_type(strategy_name)

        # 多股票训练数据（按需加载）
        self.multi_stock_data = None
        if self.train_type == 'multi':
            self.multi_stock_data = self._load_multi_stock_data()

        # 记录最佳结果
        # 注意：当 n_jobs=1（默认串行）时无需加锁；若启用 n_jobs>1 需外部同步
        self.best_value = float('-inf') if direction == 'maximize' else float('inf')
        self.best_params = None
        self.all_results = []

    def _get_training_type(self, strategy_name: str) -> str:
        """获取策略的训练类型"""
        train_config = self.config.get('strategy_training', {})
        if strategy_name in train_config.get('multi', []):
            return 'multi'
        if strategy_name in train_config.get('single', []):
            return 'single'
        if strategy_name in train_config.get('custom', []):
            return 'custom'
        # 默认: xgboost/lightgbm 相关的是 multi，其他是 single
        if 'xgboost' in strategy_name or 'lightgbm' in strategy_name or \
           'ridge' in strategy_name or 'linear' in strategy_name or 'forest' in strategy_name:
            return 'multi'
        return 'single'

    def _load_multi_stock_data(self) -> pd.DataFrame:
        """加载多股票训练数据"""
        try:
            from train_multi_stock import load_all_hk_data
            multi_data = load_all_hk_data(period='5y', min_days=300)
            if not multi_data.empty and 'Close' in multi_data.columns:
                print(f"    [多股票优化] 已加载多股票数据 ({len(multi_data)} 条记录)")
                return multi_data
        except Exception as e:
            print(f"    [多股票优化] 加载多股票数据失败: {e}")
        return pd.DataFrame()

    def _merge_ml_strategy_config(self, cfg: dict) -> dict:
        """
        合并 ML 策略专用配置
        与 run_trial() in analyze_factor.py 保持一致的合并逻辑
        """
        ml_strategies_config = cfg.get('ml_strategies', {})
        strategy_name = self.strategy_name

        if strategy_name not in ml_strategies_config:
            return cfg

        merged_cfg = cfg.copy()
        strategy_base = ml_strategies_config[strategy_name].copy()

        # runtime config 覆盖策略专用配置中的同名参数
        for key in list(strategy_base.keys()):
            if key in cfg:
                strategy_base[key] = cfg[key]

        merged_cfg.update(strategy_base)
        return merged_cfg

    def _run_multi_stock(self, trial_cfg: dict):
        """
        多股票训练模式:
        1. 在多股票数据上训练模型
        2. 在目标股票数据上进行预测
        """
        import pandas as pd

        lookback_months = int(trial_cfg.get('lookback_months', 3))
        train_years = int(trial_cfg.get('train_years', 2))

        # 准备多股票训练数据
        df_train = self.multi_stock_data.copy()
        df_train = df_train.sort_index()
        df_train['returns'] = df_train['Close'].pct_change(fill_method=None)
        df_train = df_train.dropna(subset=['returns'])

        # 目标股票验证数据
        df_target = self.data.copy().sort_index()
        if not pd.api.types.is_datetime64_any_dtype(df_target.index):
            try:
                df_target.index = pd.to_datetime(df_target.index)
            except Exception:
                raise optuna.TrialPruned("目标股票数据日期格式错误")

        df_target['returns'] = df_target['Close'].pct_change(fill_method=None)
        df_target = df_target.dropna(subset=['returns'])

        # 验证时间范围（与目标股票验证开始时间对齐）
        target_end_date = df_target.index.max()
        val_start = target_end_date - pd.DateOffset(months=lookback_months)
        val_df = df_target.loc[df_target.index >= val_start]

        if val_df.empty:
            raise optuna.TrialPruned("验证数据为空")

        # 多股票训练数据截止时间与目标股票验证开始时间对齐
        train_end = val_start
        train_start = train_end - pd.DateOffset(years=train_years)
        train_df = df_train.loc[(df_train.index >= train_start) & (df_train.index < train_end)]

        if train_df.empty:
            # 如果对齐后没有数据，使用全部多股票数据
            train_df = df_train.loc[df_train.index < train_end]

        if train_df.empty:
            raise optuna.TrialPruned("多股票训练数据为空")

        # 步骤1: 在多股票训练数据上训练模型
        train_signal, model, meta = self.strategy_mod.run(train_df, trial_cfg)

        # 步骤2: 在目标股票验证数据上生成信号
        # ⚠️ 前视偏差修复：使用已训练模型独立推断，不在 val_df 上重新 fit
        if model is not None and hasattr(self.strategy_mod, 'predict'):
            val_signal = self.strategy_mod.predict(model, val_df, trial_cfg, meta)
            val_signal = val_signal.reindex(val_df.index).fillna(0)
        else:
            val_cfg = trial_cfg.copy()
            val_cfg['no_internal_split'] = True
            val_signal, _, _ = self.strategy_mod.run(val_df, val_cfg)

        return val_signal, model, meta, val_df

    def objective(self, trial: optuna.Trial) -> float:
        """Optuna 目标函数"""
        # 1. 采样参数
        params = _suggest_params(trial, self.param_space)
        trial_cfg = self.config.copy()
        trial_cfg.update(params)

        # 2. 合并 ML 策略专用配置
        trial_cfg = self._merge_ml_strategy_config(trial_cfg)

        # 3. 根据训练类型运行策略
        try:
            if self.train_type == 'multi' and not self.multi_stock_data.empty:
                # 多股票训练模式
                signal, _, meta, val_df = self._run_multi_stock(trial_cfg)
                backtest_data = val_df
            else:
                # ⚠️ 前视偏差修复：单股票也需训练/验证分离，不能全量 in-sample
                lookback_months = int(trial_cfg.get('lookback_months', 3))
                train_years = int(trial_cfg.get('train_years', 2))
                df_s = self.data.copy().sort_index()
                if not pd.api.types.is_datetime64_any_dtype(df_s.index):
                    df_s.index = pd.to_datetime(df_s.index)
                end_date  = df_s.index.max()
                val_start = end_date - pd.DateOffset(months=lookback_months)
                tr_start  = val_start - pd.DateOffset(years=train_years)
                train_df_s = df_s.loc[(df_s.index >= tr_start) & (df_s.index < val_start)]
                val_df_s   = df_s.loc[df_s.index >= val_start]
                if train_df_s.empty:
                    train_df_s = df_s.loc[df_s.index < val_start]
                if train_df_s.empty or val_df_s.empty:
                    raise optuna.TrialPruned("单股票训练/验证数据不足")
                _, model_s, meta = self.strategy_mod.run(train_df_s, trial_cfg)
                if model_s is not None and hasattr(self.strategy_mod, 'predict'):
                    signal = self.strategy_mod.predict(model_s, val_df_s, trial_cfg, meta)
                    signal = signal.reindex(val_df_s.index).fillna(0)
                else:
                    vcfg = trial_cfg.copy()
                    vcfg['no_internal_split'] = True
                    signal, _, _ = self.strategy_mod.run(val_df_s, vcfg)
                backtest_data = val_df_s
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e:
            import traceback
            traceback.print_exc()
            raise optuna.TrialPruned(f"策略运行失败: {e}")

        # ── 信号安全检查：确保 signal 是一维 int Series ──
        if isinstance(signal, pd.DataFrame):
            signal = signal.iloc[:, 0]
        signal = pd.Series(signal).astype(int)

        # 3. 回测（使用对应的数据进行回测）
        try:
            if self.use_vectorbt:
                result = backtest_vectorbt(backtest_data, signal, trial_cfg)
            else:
                result = backtest_native(backtest_data, signal, trial_cfg)
        except Exception as e:
            raise optuna.TrialPruned(f"回测失败: {e}")

        # 4. 获取配置中的阈值
        min_return = float(self.config.get('min_return', 0.10))
        min_sharpe = float(self.config.get('min_sharpe_ratio', 1.0))
        max_dd = float(self.config.get('max_drawdown', -0.15))
        min_trades = int(self.config.get('min_total_trades', 5))

        # 5. 多维度验证（防止过拟合）
        # ── 强制转 float，防止 vectorbt 返回 Series/array 导致布尔歧义 ──
        cum_return = float(result.get('cum_return', 0))
        sharpe = float(result.get('sharpe_ratio', 0))
        max_drawdown = float(result.get('max_drawdown', 0))
        total_trades = int(result.get('total_trades', 0))

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
        value = float(result.get(self.metric, 0))
        if value is None or np.isnan(value):
            value = 0.0

        # 记录结果并更新最佳
        self.all_results.append({
            'params': params,
            'value': value,
            'cum_return': result.get('cum_return', 0),
            'sharpe_ratio': result.get('sharpe_ratio', 0),
            'max_drawdown': result.get('max_drawdown', 0),
            'win_rate': result.get('win_rate', 0),
            'total_trades': total_trades,
        })
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
        """
        ⚠️ 前视偏差修复：不再用全量数据训练+回测（in-sample）。
        复用 StrategyOptimizer 的 train/val 切分逻辑，确保回测发生在验证集上。
        """
        params = _suggest_params(trial, optimizer.param_space)
        trial_cfg = config.copy()
        trial_cfg.update(params)

        # fix #8: 合并 ML 策略专用配置（与单目标 objective 保持一致）
        trial_cfg = optimizer._merge_ml_strategy_config(trial_cfg)

        try:
            # 复用已修复的单/多股票切分逻辑
            if optimizer.train_type == 'multi' and optimizer.multi_stock_data is not None \
                    and not optimizer.multi_stock_data.empty:
                signal, _, _, backtest_data = optimizer._run_multi_stock(trial_cfg)
            else:
                # 单股票：train/val 切分
                lookback_months = int(trial_cfg.get('lookback_months', 3))
                train_years = int(trial_cfg.get('train_years', 2))
                df_s = data.copy().sort_index()
                if not pd.api.types.is_datetime64_any_dtype(df_s.index):
                    df_s.index = pd.to_datetime(df_s.index)
                val_start  = df_s.index.max() - pd.DateOffset(months=lookback_months)
                tr_start   = val_start - pd.DateOffset(years=train_years)
                train_df_s = df_s.loc[(df_s.index >= tr_start) & (df_s.index < val_start)]
                val_df_s   = df_s.loc[df_s.index >= val_start]
                if train_df_s.empty:
                    train_df_s = df_s.loc[df_s.index < val_start]
                if train_df_s.empty or val_df_s.empty:
                    raise optuna.TrialPruned("数据不足")
                _, model_s, meta_s = strategy_mod.run(train_df_s, trial_cfg)
                if model_s is not None and hasattr(strategy_mod, 'predict'):
                    signal = strategy_mod.predict(model_s, val_df_s, trial_cfg, meta_s)
                    signal = signal.reindex(val_df_s.index).fillna(0)
                else:
                    vcfg = trial_cfg.copy()
                    vcfg['no_internal_split'] = True
                    signal, _, _ = strategy_mod.run(val_df_s, vcfg)
                backtest_data = val_df_s

            if use_vectorbt:
                result = backtest_vectorbt(backtest_data, signal, trial_cfg)
            else:
                result = backtest_native(backtest_data, signal, trial_cfg)
        except optuna.exceptions.TrialPruned:
            raise
        except Exception:
            raise optuna.TrialPruned()

        # 返回多个目标值
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
