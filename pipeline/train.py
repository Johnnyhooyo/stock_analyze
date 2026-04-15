"""
pipeline/train.py — 多策略超参搜索

包含:
  step2_train()         — 分派到 native 或 optuna
  step2_train_native()  — 随机搜索
  step2_train_optuna()  — Bayesian 搜索（Optuna）
"""

from __future__ import annotations

import gc

import pandas as pd
from pathlib import Path

from log_config import get_logger
from config_loader import load_config
from analyze_factor import (
    _discover_strategies,
    run_search,
    backtest,
    _select_best_with_holdout,
)
from pipeline.select import _save_factor, _latest_factor_path

logger = get_logger(__name__)

try:
    from optimize_with_optuna import (
        optimize_strategy,
        get_training_type,
        get_optuna_trials,
    )
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optimize_strategy = None
    get_training_type = None
    get_optuna_trials = None

try:
    from backtest_vectorbt import backtest_vectorbt as backtest_vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    backtest_vbt = None


def step2_train(
    hist_data: pd.DataFrame,
    use_optuna: bool = False,
    strategy_type: str = None,
    factors_dir_override: Path = None,
    ticker: str = None,
) -> tuple:
    if use_optuna and OPTUNA_AVAILABLE:
        return step2_train_optuna(
            hist_data,
            strategy_type=strategy_type,
            factors_dir_override=factors_dir_override,
            ticker=ticker,
        )
    else:
        return step2_train_native(
            hist_data,
            strategy_type=strategy_type,
            factors_dir_override=factors_dir_override,
        )


def step2_train_native(
    hist_data: pd.DataFrame,
    strategy_type: str = None,
    factors_dir_override: Path = None,
) -> tuple:
    """对每个策略执行随机超参搜索（原生方法）。"""
    logger.info("步骤2/3: 多策略超参搜索开始", extra={"search_method": "random_search"})

    cfg = load_config()

    best_result, sorted_results, test_df = run_search(hist_data, cfg, strategy_type=strategy_type)

    strategy_mods = _discover_strategies(strategy_type=strategy_type)
    if sorted_results:
        best_result = _select_best_with_holdout(
            sorted_results, test_df, cfg, strategy_mods, full_data=hist_data
        )

    factors_dir = (
        factors_dir_override
        if factors_dir_override is not None
        else Path(__file__).parent.parent / 'data' / 'factors'
    )
    factor_path = None

    if best_result is not None:
        try:
            factor_path = _save_factor(best_result, factors_dir)
            best_result['factor_path'] = factor_path
            badge = {'double': '🏅 双验证通过', 'double_no_wf': '🥈 双验证（WF不足）',
                     'val_only': '⚠️  仅验证集达标'}.get(best_result.get('validated', ''), '❓')
            logger.info("因子已保存", extra={
                "factor_file": Path(factor_path).name,
                "strategy_name": best_result['strategy_name'],
                "cum_return": f"{best_result['cum_return']:.2%}",
                "sharpe_ratio": best_result.get('sharpe_ratio', float('nan')),
                "validated": best_result.get('validated', '?'),
                "badge": badge
            })
        except Exception as e:
            logger.warning("因子保存失败", extra={"error": str(e)})

    return factor_path, best_result, sorted_results


def step2_train_optuna(
    hist_data: pd.DataFrame,
    strategy_type: str = None,
    factors_dir_override: Path = None,
    ticker: str = None,
) -> tuple:
    """使用 Optuna 贝叶斯优化进行超参搜索(每策略 trial 数由 get_optuna_trials 决定)。"""
    logger.info("步骤2/3: 多策略超参搜索开始", extra={
        "search_method": "optuna",
        "trial_budget": "per-strategy via get_optuna_trials",
        "strategy_type_filter": strategy_type
    })

    if not OPTUNA_AVAILABLE:
        logger.warning("Optuna未安装，回退到随机搜索")
        return step2_train_native(hist_data, strategy_type=strategy_type,
                                  factors_dir_override=factors_dir_override)

    cfg = load_config()
    strategy_mods = _discover_strategies(strategy_type=strategy_type)

    if not strategy_mods:
        logger.critical("未发现任何策略模块")
        return None, None, []

    logger.info("发现策略模块", extra={
        "count": len(strategy_mods),
        "names": [m.NAME for m in strategy_mods]
    })

    backtest_config = {
        'initial_capital': cfg.get('initial_capital', 100000),
        'fees_rate': cfg.get('fees_rate', 0.00088),
        'stamp_duty': cfg.get('stamp_duty', 0.001),
        'test_days': cfg.get('test_days', 5),
        'drawdown_pct': cfg.get('drawdown_pct', 0.02),
        'min_return': cfg.get('min_return', 0.10),
        'min_sharpe_ratio': cfg.get('min_sharpe_ratio', 1.0),
        'max_drawdown': cfg.get('max_drawdown', -0.15),
        'min_total_trades': cfg.get('min_total_trades', 5),
        'strategy_training': cfg.get('strategy_training', {}),
        'lookback_months': cfg.get('lookback_months', 3),
        'train_years': cfg.get('train_years', 5),
        'ml_strategies': cfg.get('ml_strategies', {}),
        'test_months': cfg.get('test_months', 6),
        'early_stop_threshold': cfg.get('early_stop_threshold', 0.03),
        'wf_min_window_win_rate': cfg.get('wf_min_window_win_rate', 0.5),
    }

    use_vectorbt = cfg.get('backtest_engine', 'native') == 'vectorbt'

    # 三段切割
    _df_sorted = hist_data.copy().sort_index()
    if not pd.api.types.is_datetime64_any_dtype(_df_sorted.index):
        _df_sorted.index = pd.to_datetime(_df_sorted.index)
    _test_months_opt = int(backtest_config.get('test_months', 6))
    _data_end        = _df_sorted.index.max()
    _test_start_opt  = _data_end - pd.DateOffset(months=_test_months_opt)
    _test_df_opt     = _df_sorted.loc[_df_sorted.index >= _test_start_opt]
    _search_data_opt = _df_sorted.loc[_df_sorted.index < _test_start_opt]
    _min_months = int(backtest_config.get('lookback_months', 3)) + 6
    _actual_months = ((_search_data_opt.index.max() - _search_data_opt.index.min()).days // 30
                      if not _search_data_opt.empty else 0)
    if _search_data_opt.empty or _actual_months < _min_months:
        _search_data_opt = _df_sorted
        _test_df_opt = pd.DataFrame()
    else:
        logger.info("三段切割完成", extra={
            "search_end": str(_test_start_opt.date()),
            "holdout_records": len(_test_df_opt)
        })
    optuna_data = _search_data_opt

    # 预加载多股票数据:只要本批策略中存在 multi 类型,就在这里加载一次,
    # 后续每个策略复用同一份内存,避免 StrategyOptimizer.__init__ 重复读
    # ~80 个 CSV。
    multi_stock_data = None
    needs_multi = any(
        get_training_type(m.NAME, backtest_config) == 'multi'
        for m in strategy_mods
    )
    if needs_multi:
        try:
            from train_multi_stock import load_all_hk_data
            multi_stock_data = load_all_hk_data(period='5y', min_days=300)
            logger.info("多股票数据预加载完成", extra={
                "rows": len(multi_stock_data),
                "tickers": int(multi_stock_data['ticker'].nunique())
                if not multi_stock_data.empty and 'ticker' in multi_stock_data.columns
                else 0,
            })
        except Exception as e:
            logger.warning("多股票数据预加载失败,回退到策略内自加载", extra={"error": str(e)})
            multi_stock_data = None

    # DEBUG: 诊断 hist_data 和 optuna_data 的 Close 列
    _hd_close_nan = int(hist_data['Close'].isna().sum()) if 'Close' in hist_data.columns else -1
    _od_close_nan = int(optuna_data['Close'].isna().sum()) if 'Close' in optuna_data.columns else -1
    logger.warning(
        "[DEBUG step2_train_optuna] ticker=%s | hist_data: rows=%d, cols=%s, Close_NaN=%d | "
        "optuna_data: rows=%d, cols=%s, Close_NaN=%d",
        ticker or cfg.get('ticker', '?'),
        len(hist_data), list(hist_data.columns), _hd_close_nan,
        len(optuna_data), list(optuna_data.columns), _od_close_nan,
        extra={
            "hist_data_index_dtype": str(hist_data.index.dtype),
            "hist_data_close_head5": str(hist_data['Close'].head(5).tolist()) if 'Close' in hist_data.columns else "N/A",
        }
    )

    all_results = []
    best_of_all = None
    best_value = float('-inf')

    for mod in strategy_mods:
        n_trials = get_optuna_trials(mod.NAME)
        logger.info("开始优化策略", extra={
            "strategy_module": mod.NAME,
            "n_trials": n_trials,
        })
        result = optimize_strategy(
            data=optuna_data,
            strategy_mod=mod,
            config=backtest_config,
            n_trials=n_trials,
            metric='sharpe_ratio',
            direction='maximize',
            use_vectorbt=use_vectorbt,
            verbose=True,
            ticker=ticker,
            multi_stock_data=multi_stock_data,
        )
        if result['best_value'] is not None and result['best_value'] > best_value:
            best_value = result['best_value']
            # 只保留下游真正需要的标量字段,避免长期持有 study/optimizer 引用
            best_of_all = {
                'strategy_name': result.get('strategy_name', mod.NAME),
                'best_value': result['best_value'],
                'best_params': result.get('best_params', {}),
            }
        for r in result.get('all_results', []):
            r['strategy_name'] = mod.NAME
            all_results.append(r)

        # 释放本轮策略相关的内存:Optuna study、optimizer.all_results
        # 引用、ML 策略的进程级特征缓存
        result.pop('study', None)
        del result
        try:
            from strategies.rnn_trend import clear_rnn_feature_cache
            clear_rnn_feature_cache()
        except Exception:
            pass
        try:
            from train_multi_stock import clear_multi_feature_cache
            clear_multi_feature_cache()
        except Exception:
            pass
        collected = gc.collect()
        logger.debug("策略迭代结束已清理内存", extra={
            "strategy_module": mod.NAME,
            "gc_collected": collected,
        })

    # 所有策略跑完,释放预加载的多股票数据
    if multi_stock_data is not None:
        del multi_stock_data
        gc.collect()

    sorted_results = sorted(all_results, key=lambda x: x.get('value', float('-inf')), reverse=True)

    # 排行榜
    if sorted_results:
        logger.info("优化结果排行榜Top10", extra={"total_results": len(sorted_results)})
        for i, r in enumerate(sorted_results[:10], 1):
            logger.info("排名", extra={
                "rank": i,
                "strategy_name": r.get('strategy_name', 'unknown'),
                "sharpe": r.get('value', 0),
                "params": r.get('params', {})
            })

    factors_dir = (
        factors_dir_override
        if factors_dir_override is not None
        else Path(__file__).parent.parent / 'data' / 'factors'
    )

    # 转换为兼容格式
    compat_results = []
    for r in sorted_results:
        compat_results.append({
            'strategy_name':  r.get('strategy_name', ''),
            'params':         r.get('params', {}),
            'cum_return':     r.get('cum_return', 0),
            'sharpe_ratio':   r.get('value', float('nan')),
            'max_drawdown':   r.get('max_drawdown', float('nan')),
            'win_rate':       r.get('win_rate', 0),
            'total_trades':   r.get('total_trades', 0),
            'meets_threshold': r.get('value', 0) > 0,
            'model':          r.get('model'),
            'meta':           r.get('meta', {'name': r.get('strategy_name', '')}),
            'config':         backtest_config,
            'detail':         r.get('detail', pd.DataFrame()),
        })

    factor_path = None
    if compat_results:
        best_result = _select_best_with_holdout(
            compat_results, _test_df_opt, backtest_config, strategy_mods,
            full_data=hist_data
        )
    elif best_of_all:
        best_result = {
            'strategy_name': best_of_all['strategy_name'],
            'config': backtest_config.copy(),
            'meta': {'name': best_of_all['strategy_name'], 'params': best_of_all.get('best_params', {})},
            'cum_return': 0,
            'sharpe_ratio': best_of_all['best_value'],
            'validated': 'val_only',
        }
    else:
        best_result = None

    if best_result is not None:
        try:
            factor_path = _save_factor(best_result, factors_dir)
            best_result['factor_path'] = factor_path
            badge = {'double': '🏅', 'double_no_wf': '🥈', 'val_only': '⚠️'}.get(
                best_result.get('validated', ''), '❓')
            logger.info("因子已保存", extra={
                "factor_file": Path(factor_path).name,
                "strategy_name": best_result['strategy_name'],
                "sharpe_ratio": best_result.get('sharpe_ratio', float('nan')),
                "badge": badge,
                "validated": best_result.get('validated', '?')
            })
        except Exception as e:
            logger.warning("因子保存失败", extra={"error": str(e)})

    return factor_path, best_result, sorted_results
