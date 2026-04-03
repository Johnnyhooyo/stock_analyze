"""
main.py — 腾讯股票分析一体化流程

步骤 1 : 数据就绪检查
        - 历史日线数据（本地不存在则下载）
步骤 2 : 多策略 × 100 次超参搜索
        - 复用 analyze_factor.__main__ 中的逻辑
        - 每个策略最多 max_tries 次随机参数组合
        - 保存所有满足阈值的因子，并输出排行榜
步骤 3 : 预测
        - 基于最优模型预测未来 n 个交易日（日线）
"""

import argparse
import sys
import numpy as np
import pandas as pd
import joblib
import yaml
from pathlib import Path
from datetime import datetime, timedelta

# ──────────────────────────────────────────────────────────────────
#  本地模块
# ──────────────────────────────────────────────────────────────────
from log_config import get_logger
from data.manager import DataManager
from data.calendar import prev_trading_day as _prev_hk_trading_day, is_trading_day as _is_hk_trading_day
from analyze_factor import (
    _discover_strategies,
    run_search, backtest,
    _select_best_with_holdout,
    run_factor_analysis,
)
from config_loader import load_config
try:
    from backtest_vectorbt import backtest_vectorbt as backtest_vbt
except ImportError:
    backtest_vbt = None
from position_manager import PositionManager, load_position_from_config, calc_atr
from feishu_notify import send_full_report_to_feishu
from sentiment_analysis import analyze_stock_sentiment, get_sentiment_signal

try:
    from optimize_with_optuna import optimize_strategy, optimize_all_strategies
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optimize_strategy = None
    optimize_all_strategies = None

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════════════
#  辅助函数
# ══════════════════════════════════════════════════════════════════

def _last_trading_day(ref: datetime | None = None) -> datetime:
    """
    返回 ref（默认今天）之前最近一个港股交易日（正确识别周末和公众假期）。
    结果为当天 16:10（收市时间），表示该交易日数据应已完整。
    """
    d = (ref or datetime.now()).date()
    d = _prev_hk_trading_day(d)   # 使用港股节假日辅助函数
    return datetime(d.year, d.month, d.day, 16, 10)


def _hist_data_is_stale(hist_file_path: str) -> bool:
    """
    判断历史数据文件是否过期，正确处理港股节假日。
    - 18点前：当天数据不可用，最新应该是上一个港股交易日
    - 18点后：当天数据可能可用，最新应该是今天（如为交易日）
    """
    from datetime import date as _date, time as _time

    try:
        # 使用 index_col=0 读取，不依赖硬编码的 'date' 列名
        df = pd.read_csv(hist_file_path, index_col=0, parse_dates=True)
        if df.empty:
            return True
        latest_date = df.index.max()
        if pd.isna(latest_date):
            return True
        # 移除时区
        if hasattr(latest_date, 'tzinfo') and latest_date.tzinfo is not None:
            latest_date = latest_date.tz_convert(None)

        now = datetime.now()
        today = _date.today()
        cutoff_time = _time(18, 0)

        if now.time() < cutoff_time:
            # 18点前：需要上一个港股交易日
            target_date = _prev_hk_trading_day(today)
        else:
            # 18点后：如今天是交易日则需要今天，否则最近一个交易日
            if _is_hk_trading_day(today):
                target_date = today
            else:
                target_date = _prev_hk_trading_day(today)

        return latest_date.date() < target_date
    except Exception:
        return True


# ── factors 目录统一命名：factor_{run_id:04d}.pkl ──────────────────

_FACTOR_COUNTER_FILE = Path(__file__).parent / 'data' / 'factors' / '.run_id_counter'


def _next_factor_run_id(factors_dir: Path) -> int:
    """
    返回下一个可用的因子文件编号（从 1 开始）。
    优先读取计数器文件（O(1)），避免随文件增多的 glob 线性扫描。
    如计数器文件不存在则从目录扫描初始化（向后兼容）。
    """
    factors_dir.mkdir(parents=True, exist_ok=True)
    counter_path = factors_dir / '.run_id_counter'
    try:
        if counter_path.exists():
            current = int(counter_path.read_text().strip())
            next_id = current + 1
        else:
            # 初次运行：从目录扫描确定起始值（向后兼容已有文件）
            existing = list(factors_dir.glob('factor_*.pkl'))
            if existing:
                ids = []
                for p in existing:
                    try:
                        ids.append(int(p.stem.split('_')[1]))
                    except (IndexError, ValueError):
                        pass
                next_id = (max(ids) + 1) if ids else 1
            else:
                next_id = 1
        counter_path.write_text(str(next_id))
        return next_id
    except Exception:
        # 降级：全量扫描
        existing = list(factors_dir.glob('factor_*.pkl'))
        if not existing:
            return 1
        ids = []
        for p in existing:
            try:
                ids.append(int(p.stem.split('_')[1]))
            except (IndexError, ValueError):
                pass
        return (max(ids) + 1) if ids else 1


def _save_factor(result: dict, factors_dir: Path) -> str:
    """
    将搜索结果统一保存为 factor_{run_id:04d}.pkl，返回保存路径字符串。
    run_id 自动递增，每次调用只写一个文件。
    """
    factors_dir.mkdir(parents=True, exist_ok=True)
    run_id    = _next_factor_run_id(factors_dir)
    save_path = factors_dir / f"factor_{run_id:04d}.pkl"

    # 处理 model 可能为 None 的情况
    model = result.get('model')
    if model is None:
        # 对于没有模型的策略，保存一个空字典
        model = {}

    joblib.dump(
        {
            'model':              model,
            'meta':               result.get('meta', {}),
            'config':             result.get('config', {}),
            'run_id':             run_id,
            'cum_return':         result.get('cum_return', 0),
            'annualized_return':  result.get('annualized_return', float('nan')),
            'sharpe_ratio':       result.get('sharpe_ratio', float('nan')),
            'max_drawdown':       result.get('max_drawdown', float('nan')),
            'volatility':         result.get('volatility', float('nan')),
            'win_rate':           result.get('win_rate', 0),
            'profit_loss_ratio':  result.get('profit_loss_ratio', 0),
            'calmar_ratio':       result.get('calmar_ratio', float('nan')),
            'sortino_ratio':      result.get('sortino_ratio', float('nan')),
            'total_trades':       result.get('total_trades', 0),
            # 验证标记（double / double_no_wf / val_only）
            'validated':          result.get('validated', 'unknown'),
            'holdout':            result.get('holdout', {}),
            'wf_summary':         result.get('wf_summary', {}),
            'val_period':         result.get('val_period', ''),
            'saved_at':           datetime.now().isoformat(),
        },
        save_path,
    )
    return str(save_path)


def _latest_factor_path(factors_dir: Path) -> str | None:
    """返回 factors_dir 下编号最大的 factor_XXXX.pkl 路径，不存在则 None。"""
    candidates = list(factors_dir.glob('factor_*.pkl'))
    if not candidates:
        return None
    return str(max(candidates, key=lambda p: int(p.stem.split('_')[1])))


# ══════════════════════════════════════════════════════════════════
#  步骤 1 : 数据就绪检查
# ══════════════════════════════════════════════════════════════════

def step1_ensure_data(sources_override=None, ticker: str = None, skip_download: bool = False):
    """
    确保历史日线数据已就绪。

    Returns
    -------
    hist_data : pd.DataFrame   历史日线（Close 等标准列）
    hist_path : str            历史文件路径
    """
    logger.info("步骤1/3: 数据就绪检查开始")

    cfg = load_config()
    effective_ticker = ticker or cfg.get('ticker', '0700.hk')

    # ── 1a. 历史日线 ────────────────────────────────────────────
    hist_dir = Path(__file__).parent / 'data' / 'historical'

    # 只匹配目标 ticker 的文件，避免误读 HSI 其他成分股数据
    # 文件名格式：{ticker}_{period}.csv 或 {safe_name}_{period}.csv
    # 例如 0700.hk_5y.csv 或 0700_HK_3y.csv
    _ticker_lower = effective_ticker.lower()
    _ticker_safe  = effective_ticker.replace('.', '_').lower()
    def _is_ticker_file(p: Path) -> bool:
        stem = p.stem.lower()
        return stem.startswith(_ticker_lower + '_') or stem.startswith(_ticker_safe + '_')

    hist_files = sorted(
        [f for f in hist_dir.glob('*.csv') if _is_ticker_file(f)],
        key=lambda p: p.stat().st_mtime, reverse=True,
    )

    mgr = DataManager()

    if hist_files and not _hist_data_is_stale(str(hist_files[0])):
        hist_path = str(hist_files[0])
        hist_data = pd.read_csv(hist_path, index_col=0, parse_dates=True)
        latest_date = hist_data.index.max()
        logger.info("历史日线数据已是最新", extra={
            "hist_path": hist_path,
            "records": len(hist_data),
            "latest_date": str(latest_date.date())
        })
    else:
        if skip_download:
            if hist_files:
                hist_path = str(hist_files[0])
                hist_data = pd.read_csv(hist_path, index_col=0, parse_dates=True)
                latest_date = hist_data.index.max()
                logger.warning("使用过期本地缓存（%s），--skip-data-download 已设置", latest_date.date())
            else:
                logger.error("本地无 %s 的历史数据缓存，且 --skip-data-download 已设置，流程终止", effective_ticker)
                sys.exit(1)
            return hist_data, hist_path
        if hist_files:
            # 显示数据内容的最新日期，而不是文件修改时间
            from datetime import date as _date, time as _time
            try:
                df_tmp = pd.read_csv(hist_files[0], index_col=0, parse_dates=True)
                latest_date = df_tmp.index.max()
            except Exception:
                latest_date = pd.Timestamp('1970-01-01')
            today = _date.today()
            now = datetime.now()
            if now.time() < _time(18, 0):
                target = _prev_hk_trading_day(today)
            else:
                target = today if _is_hk_trading_day(today) else _prev_hk_trading_day(today)
            hint = str(target)
            logger.warning("历史日线数据已过期，正在更新", extra={
                "latest_date": str(latest_date.date()),
                "target_date": hint
            })
        else:
            logger.warning("本地无历史日线数据，正在下载")
        hist_data, hist_path = mgr.download(
            effective_ticker,
            period=cfg.get('period', '3y'),
            sources_override=sources_override,
        )
        if hist_data is None or hist_data.empty:
            # 下载失败时降级使用旧文件（如果存在）
            if hist_files:
                hist_path = str(hist_files[0])
                hist_data = pd.read_csv(hist_path, index_col=0, parse_dates=True)
                logger.warning("数据更新失败，继续使用旧数据", extra={
                    "hist_path": hist_path,
                    "records": len(hist_data)
                })
            else:
                logger.critical("历史数据下载失败，流程终止")
                sys.exit(1)
        else:
            logger.info("历史数据已更新", extra={
                "hist_path": hist_path,
                "records": len(hist_data)
            })

    return hist_data, hist_path


def _ensure_hsi_data(cfg: dict = None) -> None:
    """HSI成分股增量更新 — 单独提取，供 main() 和 train_portfolio_tickers() 调用一次。"""
    if cfg is None:
        cfg = load_config()
    hsi_period = cfg.get('hsi_period', '3y')
    logger.info("开始增量更新HSI成分股数据", extra={"period": hsi_period})
    try:
        mgr = DataManager()
        hsi_result = mgr.download_hsi_incremental(period=hsi_period)
        logger.info("HSI成分股更新完成", extra={
            "total":          hsi_result['total'],
            "skipped":        hsi_result['skipped'],
            "updated":        hsi_result['updated'],
            "failed_count":   len(hsi_result['failed']),
            "failed_tickers": hsi_result['failed'],
        })
    except Exception as e:
        logger.warning("HSI成分股更新失败", extra={"error": str(e)})


# ══════════════════════════════════════════════════════════════════
#  步骤 2 : 多策略 × 100 次超参搜索
# ══════════════════════════════════════════════════════════════════

def step2_train(
    hist_data: pd.DataFrame,
    use_optuna: bool = False,
    optuna_trials: int = 50,
    strategy_type: str = None,
    factors_dir_override: Path = None,
):
    if use_optuna and OPTUNA_AVAILABLE:
        return step2_train_optuna(
            hist_data, n_trials=optuna_trials,
            strategy_type=strategy_type,
            factors_dir_override=factors_dir_override,
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
):
    """
    对每个策略执行随机超参搜索（原生方法）。
    """
    logger.info("步骤2/3: 多策略超参搜索开始", extra={"search_method": "random_search"})

    cfg = load_config()

    # on_result 仅在搜索中途满足阈值时触发（用于实时预览）
    best_result, sorted_results, test_df = run_search(hist_data, cfg, strategy_type=strategy_type)

    # ── Hold-Out + Walk-Forward 选优（统一入口） ──────────────────
    strategy_mods = _discover_strategies(strategy_type=strategy_type)
    if sorted_results:
        best_result = _select_best_with_holdout(
            sorted_results, test_df, cfg, strategy_mods, full_data=hist_data
        )


    # ── 确定保存目录 ──────────────────────────────────────────────
    factors_dir = (
        factors_dir_override
        if factors_dir_override is not None
        else Path(__file__).parent / 'data' / 'factors'
    )
    factor_path   = None

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
    n_trials: int = 50,
    strategy_type: str = None,
    factors_dir_override: Path = None,
):
    """
    使用 Optuna 贝叶斯优化进行超参搜索。
    """
    logger.info("步骤2/3: 多策略超参搜索开始", extra={
        "search_method": "optuna",
        "n_trials_per_strategy": n_trials,
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

    # 准备回测配置
    backtest_config = {
        'initial_capital': cfg.get('initial_capital', 100000),
        'fees_rate': cfg.get('fees_rate', 0.00088),
        'stamp_duty': cfg.get('stamp_duty', 0.001),
        'test_days': cfg.get('test_days', 5),
        'drawdown_pct': cfg.get('drawdown_pct', 0.02),
        # 阈值参数（必须传递，否则 optimize_with_optuna 使用默认值 0.10）
        'min_return': cfg.get('min_return', 0.10),
        'min_sharpe_ratio': cfg.get('min_sharpe_ratio', 1.0),
        'max_drawdown': cfg.get('max_drawdown', -0.15),
        'min_total_trades': cfg.get('min_total_trades', 5),
        # 策略训练配置（支持 multi/single/custom 分类）
        'strategy_training': cfg.get('strategy_training', {}),
        'lookback_months': cfg.get('lookback_months', 3),
        'train_years': cfg.get('train_years', 5),
        # ML 策略专用配置
        'ml_strategies': cfg.get('ml_strategies', {}),
        # 三段切割 / Hold-Out 相关
        'test_months': cfg.get('test_months', 6),
        'early_stop_threshold': cfg.get('early_stop_threshold', 0.03),
        'wf_min_window_win_rate': cfg.get('wf_min_window_win_rate', 0.5),
    }

    # 使用 Vectorbt
    use_vectorbt = cfg.get('backtest_engine', 'native') == 'vectorbt'

    # ── 三段切割：与 run_search 保持一致，提前确定 test_start ──
    import pandas as _pd
    _df_sorted = hist_data.copy().sort_index()
    if not _pd.api.types.is_datetime64_any_dtype(_df_sorted.index):
        _df_sorted.index = _pd.to_datetime(_df_sorted.index)
    _test_months_opt = int(backtest_config.get('test_months', 6))
    _data_end        = _df_sorted.index.max()
    _test_start_opt  = _data_end - _pd.DateOffset(months=_test_months_opt)
    _test_df_opt     = _df_sorted.loc[_df_sorted.index >= _test_start_opt]
    _search_data_opt = _df_sorted.loc[_df_sorted.index < _test_start_opt]
    # 数据量保护
    _min_months = int(backtest_config.get('lookback_months', 3)) + 6
    _actual_months = (_search_data_opt.index.max() - _search_data_opt.index.min()).days // 30 if not _search_data_opt.empty else 0
    if _search_data_opt.empty or _actual_months < _min_months:
        _search_data_opt = _df_sorted
        _test_df_opt = _pd.DataFrame()
    else:
        logger.info("三段切割完成", extra={
            "search_end": str(_test_start_opt.date()),
            "holdout_records": len(_test_df_opt)
        })
    optuna_data = _search_data_opt   # Optuna 只见这部分数据

    # 优化每个策略
    all_results = []
    best_of_all = None
    best_value = float('-inf')

    for mod in strategy_mods:
        logger.info("开始优化策略", extra={"strategy_module": mod.NAME})

        result = optimize_strategy(
            data=optuna_data,
            strategy_mod=mod,
            config=backtest_config,
            n_trials=n_trials,
            metric='sharpe_ratio',
            direction='maximize',
            use_vectorbt=use_vectorbt,
            verbose=True,
        )

        # 记录结果
        if result['best_value'] is not None and result['best_value'] > best_value:
            best_value = result['best_value']
            best_of_all = result

        # 收集所有试验结果
        for r in result.get('all_results', []):
            r['strategy_name'] = mod.NAME
            all_results.append(r)

    # 排序结果
    sorted_results = sorted(all_results, key=lambda x: x.get('value', float('-inf')), reverse=True)

    # ── 绘制最优解结果图 ─────────────────────────────────────────
    if best_of_all and best_of_all.get('best_params'):
        logger.info("绘制最优解结果图", extra={
            "strategy_name": best_of_all['strategy_name'],
            "sharpe": best_of_all['best_value']
        })
        try:
            trial_cfg = backtest_config.copy()
            trial_cfg.update(best_of_all['best_params'])

            # 找到对应策略模块（避免依赖循环变量 mod）
            best_mod = next((m for m in strategy_mods if m.NAME == best_of_all['strategy_name']), None)
            if best_mod is None:
                raise ValueError(f"找不到策略模块: {best_of_all['strategy_name']}")

            signal, _, meta = best_mod.run(hist_data.copy(), trial_cfg)

            if use_vectorbt:
                from backtest_vectorbt import backtest_vectorbt as bt_vbt
                detail = bt_vbt(hist_data, signal, trial_cfg)
            else:
                detail = backtest(hist_data, signal, trial_cfg)

            detail_df = detail.get('detail') if isinstance(detail, dict) else detail

        except Exception as e:
            logger.warning("回测失败", extra={"error": str(e)})

    # ── Hold-Out + Walk-Forward 选优（统一入口） ──────────────────
    factor_path = None
    factors_dir = (
        factors_dir_override
        if factors_dir_override is not None
        else Path(__file__).parent / 'data' / 'factors'
    )

    # 将 Optuna all_results 转换成与 run_search 兼容的格式
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
            'detail':         r.get('detail', _pd.DataFrame()),
        })

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

    # 打印排行榜
    if sorted_results:
        logger.info("优化结果排行榜Top10", extra={"total_results": len(sorted_results)})
        for i, r in enumerate(sorted_results[:10], 1):
            logger.info("排名", extra={
                "rank": i,
                "strategy_name": r.get('strategy_name', 'unknown'),
                "sharpe": r.get('value', 0),
                "params": r.get('params', {})
            })

    return factor_path, best_result, sorted_results


# ══════════════════════════════════════════════════════════════════
#  分层混合组合训练入口
# ══════════════════════════════════════════════════════════════════

def train_portfolio_tickers(
    tickers: list[str],
    use_optuna: bool = False,
    optuna_trials: int = 50,
    sources_override: list[str] = None,
    n_days: int = 3,
    skip_download: bool = False,
) -> list[dict]:
    """
    分层混合训练：
      1. ML 全局训练（strategy_type='multi'）运行一次 → data/factors/
      2. 对每只 ticker 跑规则策略训练（strategy_type='single'）→ data/factors/{TICKER_SAFE}/
      3. 生成每只 ticker 的信号报告
    返回每只 ticker 的结果列表。
    """
    config = load_config()
    if not skip_download:
        _ensure_hsi_data(config)

    base_factors_dir = Path(__file__).parent / 'data' / 'factors'
    results: list[dict] = []

    # ── 步骤 A：ML 全局训练（一次）────────────────────────────────
    # ML 策略内部加载 HSI 全量数据，只需任意一只股票的历史数据作为验证集
    ref_ticker = tickers[0] if tickers else config.get('ticker', '0700.HK')
    logger.info("ML全局训练开始（使用 %s 作为验证参考）", ref_ticker)
    ml_status = "ok"
    try:
        ref_hist, _ = step1_ensure_data(sources_override=sources_override, ticker=ref_ticker, skip_download=skip_download)
        step2_train(
            ref_hist,
            use_optuna=use_optuna,
            optuna_trials=optuna_trials,
            strategy_type='multi',
            factors_dir_override=None,   # 全局目录
        )
        logger.info("ML全局训练完成，因子已存入 %s", base_factors_dir)
    except Exception as e:
        ml_status = "failed"
        logger.warning("ML全局训练失败（将继续进行规则策略训练）: %s", e)

    # ── 步骤 B：每只 ticker 的规则策略训练 ────────────────────────
    for ticker in tickers:
        ticker_safe = ticker.replace('.', '_').upper()
        ticker_factors_dir = base_factors_dir / ticker_safe

        logger.info("规则策略训练: %s → %s", ticker, ticker_factors_dir,
                    extra={"ticker": ticker})

        # 下载数据
        try:
            hist_data, _ = step1_ensure_data(
                sources_override=sources_override, ticker=ticker, skip_download=skip_download
            )
        except Exception as e:
            logger.error("数据获取失败，跳过 %s: %s", ticker, e,
                         extra={"ticker": ticker})
            results.append({"ticker": ticker, "status": "data_failed",
                             "factor_path": None, "error": str(e)})
            continue

        # 规则策略训练
        try:
            factor_path, best_result, _ = step2_train(
                hist_data,
                use_optuna=use_optuna,
                optuna_trials=optuna_trials,
                strategy_type='single',
                factors_dir_override=ticker_factors_dir,
            )
        except Exception as e:
            logger.error("规则训练失败，跳过 %s: %s", ticker, e,
                         extra={"ticker": ticker})
            results.append({"ticker": ticker, "status": "train_failed",
                             "factor_path": None, "error": str(e)})
            continue

        if factor_path is None:
            factor_path = _latest_factor_path(ticker_factors_dir)

        # 信号报告
        report_md = ""
        if factor_path:
            try:
                report_md = generate_signal_report(hist_data, factor_path, n_days=n_days)
            except Exception as e:
                logger.warning("信号报告失败 %s: %s", ticker, e,
                               extra={"ticker": ticker})

        sharpe    = best_result.get('sharpe_ratio', float('nan')) if best_result else float('nan')
        validated = best_result.get('validated', 'unknown') if best_result else 'unknown'

        results.append({
            "ticker":     ticker,
            "status":     "ok",
            "factor_path": factor_path,
            "factors_dir": str(ticker_factors_dir),
            "sharpe_ratio": sharpe,
            "validated":   validated,
            "report_md":   report_md,
            "ml_status":   ml_status,
        })
        logger.info("规则训练完成: %s",
                    ticker,
                    extra={"ticker": ticker, "sharpe_ratio": sharpe})

    return results


# ══════════════════════════════════════════════════════════════════
#  辅助: 加载因子，附加策略模块引用（ML + 规则策略统一入口）
# ══════════════════════════════════════════════════════════════════

def _resolve_artifact(factor_path: str) -> dict:
    """
    加载 factor_path 对应的 .pkl，并附加对应的策略模块引用。
    - ML 策略（model != None）：直接用模型做数值预测
    - 规则策略（model = None）：通过 meta['name'] 找到策略模块，重新调用 run() 生成信号
    """
    try:
        art = joblib.load(factor_path)
    except Exception as e:
        logger.warning("加载因子失败", extra={"error": str(e), "factor_path": factor_path})
        return {}

    strategy_name = art.get('meta', {}).get('name', '')
    strategy_mod  = None
    for mod in _discover_strategies():
        if mod.NAME == strategy_name:
            strategy_mod = mod
            break
    art['strategy_mod'] = strategy_mod
    return art


def _signal_to_direction(signal_series: pd.Series) -> int:
    """取信号序列最后一个值作为方向（1=看涨，0=看跌）。"""
    if signal_series is None or signal_series.empty:
        return 0
    return int(signal_series.iloc[-1])


def _historical_price_range(
    returns: pd.Series,
    last_close: float,
    horizon_days: int,
    lower_pct: float = 0.10,
    upper_pct: float = 0.90,
) -> tuple[float, float]:
    """
    基于历史滚动 horizon_days 窗口累计收益率分布，
    返回 (price_lo, price_hi)。

    - 数据充足（≥ 20 个滚动窗口）：使用历史分位数（非对称，反映右偏分布）
    - 数据不足：退回正态近似，z 值查找表 + 线性插值，零新依赖
    - 返回值是"历史统计区间"，不是价格预测，调用方需在报告中注明
    """
    rolling = returns.rolling(horizon_days).sum().dropna()
    if len(rolling) >= 20:
        p_lo = float(rolling.quantile(lower_pct))
        p_hi = float(rolling.quantile(upper_pct))
    else:
        # 标准正态分位数查找表（覆盖常用百分位），无 scipy 依赖
        _Z = {0.01: -2.326, 0.05: -1.645, 0.10: -1.282, 0.20: -0.842,
              0.25: -0.674, 0.75:  0.674,  0.80:  0.842, 0.90:  1.282,
              0.95:  1.645, 0.99:  2.326}
        mu  = float(returns.mean() * horizon_days)
        sig = float(returns.std() * (horizon_days ** 0.5))

        def _z(p: float) -> float:
            if p in _Z:
                return _Z[p]
            keys = sorted(_Z.keys())
            lo_k = max((k for k in keys if k <= p), default=keys[0])
            hi_k = min((k for k in keys if k >= p), default=keys[-1])
            if lo_k == hi_k:
                return _Z[lo_k]
            t = (p - lo_k) / (hi_k - lo_k)
            return _Z[lo_k] + t * (_Z[hi_k] - _Z[lo_k])

        p_lo = mu + _z(lower_pct) * sig
        p_hi = mu + _z(upper_pct) * sig
    return last_close * (1 + p_lo), last_close * (1 + p_hi)


def _signal_confidence(artifact: dict, is_ml: bool) -> tuple[str, str]:
    """
    根据策略类型和验证等级，返回 (置信度标签, emoji)。

    ML 策略经过额外的模型泛化验证，同等验证等级下置信度高于规则策略：
      ML   + double       → 高   🟢
      ML   + double_no_wf → 中高  🟡
      ML   + val_only     → 低   🔴
      rule + double       → 中   🟡
      rule + double_no_wf → 中低  🟡
      rule + val_only     → 低   🔴
      其他                → 未知  ⚪
    """
    validated = artifact.get('validated', 'unknown')
    level_map = {
        ('ml',   'double'):       ('高',  '🟢'),
        ('ml',   'double_no_wf'): ('中高', '🟡'),
        ('ml',   'val_only'):     ('低',  '🔴'),
        ('rule', 'double'):       ('中',  '🟡'),
        ('rule', 'double_no_wf'): ('中低', '🟡'),
        ('rule', 'val_only'):     ('低',  '🔴'),
    }
    key = ('ml' if is_ml else 'rule', validated)
    return level_map.get(key, ('未知', '⚪'))


def _build_eval_table(
    train_period: str, val_period: str,
    val_cum_return: float, val_sharpe: float, val_max_dd: float,
    val_trades: int, val_win_rate: float,
    ann_return: float, calmar: float, sortino: float,
    volatility: float, daily_vol: float,
    artifact: dict,
) -> str:
    """生成训练/验证两阶段评估 Markdown 表格"""
    import math as _m

    def _f(v, fmt='.2%'):
        if v is None or (isinstance(v, float) and _m.isnan(v)):
            return '—'
        return f"{v:{fmt}}"

    def _badge(ok: bool) -> str:
        return '✅' if ok else '❌'

    # 判断补充指标是否有有效值（验证窗口交易次数过少时往往为 NaN）
    has_ann      = not (ann_return  is None or (isinstance(ann_return,  float) and _m.isnan(ann_return)))
    has_vol      = not (volatility  is None or (isinstance(volatility,  float) and _m.isnan(volatility)))
    has_calmar   = not (calmar      is None or (isinstance(calmar,      float) and _m.isnan(calmar)))
    has_sortino  = not (sortino     is None or (isinstance(sortino,     float) and _m.isnan(sortino)))

    ann_str      = _f(ann_return)     if has_ann     else "—（交易次数不足）"
    vol_str      = _f(volatility)     if has_vol     else "—（数据不足）"
    calmar_str   = _f(calmar, '.4f')  if has_calmar  else "—（需回撤>0）"
    sortino_str  = _f(sortino, '.4f') if has_sortino else "—（需负收益样本）"

    lines = [
        "| 阶段 | 时间范围 | 累计收益 | 夏普率¹ | 最大回撤² | 交易次数 | 胜率 | 达标 |",
        "|------|---------|---------|--------|---------|---------|------|------|",
        f"| 🔵 阶段1 训练 | {train_period} | — | — | — | — | — | —（仅拟合期，不评估指标） |",
        f"| 🟡 阶段2 验证 | {val_period} | {_f(val_cum_return)} | {_f(val_sharpe, '.4f')} "
        f"| {_f(val_max_dd)} | {val_trades} | {_f(val_win_rate)} "
        f"| {_badge(artifact.get('meets_threshold', False))} |",
        "",
        f"> ¹ **夏普率**：超额收益与波动率之比，衡量每单位风险所获得的收益，>1 为合格，>2 为优秀。  ",
        f"> ² **最大回撤**：持仓期间净值从最高点到最低点的最大跌幅，体现策略最坏情形下的亏损幅度。",
        "",
        f"**补充指标**  年化³ {ann_str} | 波动率⁴ {vol_str} | "
        f"近60日波动 {daily_vol:.2%} | 卡玛⁵ {calmar_str} | 索提诺⁶ {sortino_str}",
        "",
        f"> ³ **年化收益**：将验证期累计收益折算为1年的等效收益率（复利）。  ",
        f"> ⁴ **年化波动率**：日收益率标准差×√252，衡量收益的不稳定程度，越低越稳。  ",
        f"> ⁵ **卡玛比率（Calmar）**：年化收益率 ÷ 最大回撤绝对值，衡量承受回撤所换来的收益，>0.5 为合格。  ",
        f"> ⁶ **索提诺比率（Sortino）**：类似夏普率，但分母只计算下行（亏损）波动，对策略更友好，>1 为合格。",
        "",
    ]

    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════
#  步骤 3a : 信号报告（交易方向 + 统计参考区间）
# ══════════════════════════════════════════════════════════════════

def generate_signal_report(data: pd.DataFrame, factor_path: str, n_days: int = 3) -> str:
    """
    生成未来 n_days 个交易日的交易信号报告。

    输出内容：
    - 当前持仓信号（做多 / 空仓）及置信度等级
    - 未来 n_days 日的信号预期（基于当前信号保持的估算）
    - 价格参考区间（历史统计 P10~P90，明确标注"非预测"，区间随天数自然扩大）
    - ML 策略额外输出：模型原始预测收益率（单步，不累乘）

    不输出：单调外推的"预测价格"序列。
    返回 markdown 格式的报告内容。
    """
    logger.info("生成信号报告", extra={
        "n_days": n_days,
        "factor_path": factor_path
    })

    artifact     = _resolve_artifact(factor_path)
    if not artifact:
        logger.warning("无法加载因子文件，跳过信号报告")
        return ""

    model        = artifact.get('model')
    meta         = artifact.get('meta', {})
    config       = artifact.get('config', {})
    strategy_mod = artifact.get('strategy_mod')
    feat_cols    = meta.get('feat_cols', [])
    strategy     = meta.get('name', 'unknown')
    params       = meta.get('params', {})
    is_ml        = (model is not None and len(feat_cols) > 0)
    sharpe       = artifact.get('sharpe_ratio', float('nan'))
    sharpe_str   = f"{sharpe:.4f}" if not np.isnan(sharpe) else "N/A"

    # 置信度
    confidence_label, confidence_emoji = _signal_confidence(artifact, is_ml)

    logger.info("信号报告", extra={
        "strategy": strategy,
        "type": 'ML' if is_ml else '规则',
        "params": params,
        "sharpe": sharpe_str,
        "cum_return": f"{artifact.get('cum_return', float('nan')):.2%}",
        "confidence_label": confidence_label,
        "confidence_emoji": confidence_emoji,
        "validated": artifact.get('validated', 'unknown')
    })

    df = data.copy().sort_index()
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)

    last_date  = df.index.max()
    last_close = float(df['Close'].iloc[-1])
    returns    = df['Close'].pct_change().dropna()
    daily_vol  = float(returns.tail(60).std())

    logger.info("市场状态", extra={
        "last_date": str(last_date.date()),
        "last_close": last_close,
        "daily_volatility_60d": f"{daily_vol:.2%}"
    })

    # ── 情感分析 ──────────────────────────────────────────────────
    logger.info("开始情感分析")
    sentiment_result = analyze_stock_sentiment(config.get('ticker', '0700.HK'))
    sentiment_signal = get_sentiment_signal(sentiment_result)
    sentiment_emoji  = ("🟢" if sentiment_result['sentiment'] == "positive"
                        else "🔴" if sentiment_result['sentiment'] == "negative" else "⚪")
    logger.info("情感分析完成", extra={
        "sentiment": sentiment_result['sentiment'],
        "polarity": sentiment_result['polarity'],
        "positive_count": sentiment_result['positive_count'],
        "negative_count": sentiment_result['negative_count'],
        "neutral_count": sentiment_result['neutral_count'],
        "sentiment_emoji": sentiment_emoji
    })

    # ── 未来交易日（跳过周末）──────────────────────────────────────
    future_dates = []
    d = last_date
    while len(future_dates) < n_days:
        d += pd.Timedelta(days=1)
        if d.weekday() < 5:
            future_dates.append(d)

    # ── 循环外预计算各天统计区间（horizon=k，区间随天数自然扩大）──────
    price_ranges = []
    for k in range(1, n_days + 1):
        p_lo, p_hi = _historical_price_range(returns, last_close, horizon_days=k)
        price_ranges.append((p_lo, p_hi))

    # ── ML 策略：准备最新特征（单步推断，不滚动累乘）─────────────────
    latest_feats       = None
    pred_signal_series = None
    if is_ml:
        logger.info("ML模型推断", extra={
            "model_type": type(model).__name__,
            "feature_count": len(feat_cols)
        })
        if strategy_mod is not None and hasattr(strategy_mod, 'predict'):
            try:
                pred_signal_series = strategy_mod.predict(model, df, config, meta)
            except Exception as e:
                logger.warning("predict()接口失败", extra={"error": str(e)})

        if pred_signal_series is None:
            test_days = len(feat_cols)
            df['returns'] = df['Close'].pct_change()
            for i in range(1, test_days + 1):
                df[f'ret_{i}'] = df['returns'].shift(i)
            df_clean = df.dropna()
            if not df_clean.empty and all(c in df_clean.columns for c in feat_cols):
                latest_feats = list(df_clean[feat_cols].iloc[-1].values)
    else:
        logger.info("规则信号驱动")

    # ── 规则策略：取当前信号 ───────────────────────────────────────
    rule_direction = 1
    current_signal = "空仓"
    if not is_ml:
        if strategy_mod is not None:
            try:
                sig, _, _ = strategy_mod.run(data.copy(), config)
                rule_direction = _signal_to_direction(sig)
            except Exception as e:
                logger.warning("规则策略信号生成失败", extra={"error": str(e)})
        current_signal = "做多" if rule_direction == 1 else "空仓"
        logger.info("当前信号", extra={
            "signal": current_signal,
            "direction": rule_direction
        })

    # ML 当前方向
    ml_direction = 1
    if is_ml and pred_signal_series is not None and not pred_signal_series.empty:
        ml_direction = int(pred_signal_series.dropna().iloc[-1])

    # ── 单步 ML 预测值（所有天复用，不更新 latest_feats）─────────────
    ml_pred_ret_raw = None
    if is_ml:
        if latest_feats is not None:
            X_pred        = np.array(latest_feats).reshape(1, -1)
            ml_pred_ret_raw = float(model.predict(X_pred)[0])
        else:
            ml_pred_ret_raw = 0.0   # predict() 路径只有方向，幅度用 0

    # ── 预测循环 ──────────────────────────────────────────────────
    logger.info("未来交易日预测", extra={
        "is_ml": is_ml,
        "n_days": n_days
    })

    predictions = []
    for i, fd in enumerate(future_dates):
        price_lo, price_hi = price_ranges[i]

        if is_ml:
            pred_ret = ml_pred_ret_raw if ml_pred_ret_raw is not None else 0.0
            signal_val  = 1 if pred_ret > 0 else 0
            signal_str  = "做多" if pred_ret > 0 else "空仓"
            pred_display = f"{pred_ret:+.4f}"
        else:
            # 规则策略：pred_ret = 0.0，方向只体现在 signal_str，不偏移价格区间基准
            pred_ret     = 0.0
            signal_val   = rule_direction
            signal_str   = "做多" if rule_direction == 1 else "空仓"
            pred_display = "—"

        predictions.append({
            'date':         fd.date(),
            'signal':       signal_val,
            'signal_str':   signal_str,
            'pred_ret_raw': pred_display,
            'price_lo':     price_lo,
            'price_hi':     price_hi,
            'confidence':   f"{confidence_emoji} {confidence_label}",
        })

        logger.info("预测", extra={
            "date": str(fd.date()),
            "signal": signal_str,
            "pred_ret_raw": pred_display,
            "price_lo": price_lo,
            "price_hi": price_hi,
            "confidence_label": confidence_label
        })

    logger.warning("统计区间说明", extra={
        "note": "统计区间基于历史滚动收益率分布（P10/P90），非价格预测，不代表未来走势"
    })
    logger.warning("多日信号说明", extra={
        "note": "多日信号基于当前最新数据推算，未来市况变化可能导致信号翻转"
    })

    # ── 完整指标（来自因子文件 - 验证窗口） ──────────────────────
    ann_return        = artifact.get('annualized_return', float('nan'))
    max_drawdown      = artifact.get('max_drawdown', float('nan'))
    volatility        = artifact.get('volatility', float('nan'))
    win_rate          = artifact.get('win_rate', 0)
    profit_loss_ratio = artifact.get('profit_loss_ratio', 0)
    calmar_ratio      = artifact.get('calmar_ratio', float('nan'))
    sortino_ratio     = artifact.get('sortino_ratio', float('nan'))
    total_trades      = artifact.get('total_trades', 0)
    val_period        = artifact.get('val_period', '—')
    val_cum_return    = artifact.get('cum_return', float('nan'))
    val_sharpe        = artifact.get('sharpe_ratio', float('nan'))

    ann_return_str = f"{ann_return:.2%}" if not np.isnan(ann_return) else "N/A"
    max_dd_str     = f"{max_drawdown:.2%}" if not np.isnan(max_drawdown) else "N/A"
    vol_str        = f"{volatility:.2%}" if not np.isnan(volatility) else "N/A"
    calmar_str     = f"{calmar_ratio:.4f}" if not np.isnan(calmar_ratio) else "N/A"
    sortino_str    = f"{sortino_ratio:.4f}" if not np.isnan(sortino_ratio) else "N/A"
    win_rate_str   = f"{win_rate:.2%}" if win_rate > 0 else "N/A"
    pl_ratio_str   = f"{profit_loss_ratio:.2f}" if profit_loss_ratio > 0 else "N/A"

    # ── 推算训练窗口时间范围 ───────────────────────────────────────
    lookback_months = int(config.get('lookback_months', 3))
    train_years     = int(config.get('train_years', 2))
    try:
        _val_end   = pd.to_datetime(val_period.split('~')[-1].strip()) if '~' in val_period else df.index.max()
        _val_start = pd.to_datetime(val_period.split('~')[0].strip()) if '~' in val_period else (_val_end - pd.DateOffset(months=lookback_months))
        _train_end   = _val_start - pd.Timedelta(days=1)
        _train_start = _val_start - pd.DateOffset(years=train_years)
        train_period_str = f"{_train_start.date()} ~ {_train_end.date()}"
    except Exception:
        train_period_str = '—'

    # ── BS 点（买卖点）────────────────────────────────────────────
    bs_points = []
    backtest_engine = config.get('backtest_engine', 'native')
    if strategy_mod is not None and not is_ml:
        try:
            sig, _, _ = strategy_mod.run(data.copy(), config)
            if backtest_engine == 'vectorbt' and backtest_vbt is not None:
                bt_result = backtest_vbt(data, sig, config)
                if 'portfolio' in bt_result:
                    try:
                        trades = bt_result['portfolio'].get_trades()
                        if trades is not None:
                            for t in trades:
                                bs_points.append({
                                    'date':   str(t.entry_date.date()) if hasattr(t, 'entry_date') else 'N/A',
                                    'price':  t.entry_price,
                                    'action': '买入',
                                    'pv':     t.return_,
                                })
                    except Exception:
                        pass
            else:
                bt_result = backtest(data, sig, config)
                detail = bt_result.get('detail')
                if detail is not None and 'trade' in detail.columns:
                    for idx, row in detail.iterrows():
                        if row['trade'] != 0:
                            bs_points.append({
                                'date':   idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)[:10],
                                'price':  row['Close'],
                                'action': '买入' if row['trade'] == 1 else '卖出',
                                'pv':     row['pv'],
                            })
        except Exception:
            pass

    # ── Markdown 报告 ─────────────────────────────────────────────
    ticker_label = config.get('ticker', '0700.hk').upper()
    md_content = f"""# {ticker_label} 股票分析报告

## 基本信息

| 项目 | 值 |
|------|-----|
| 分析日期 | {datetime.now().strftime('%Y-%m-%d %H:%M')} |
| 最后交易日 | {last_date.date()} |
| 收盘价 | {last_close:.2f} HKD |

## 策略信息

| 项目 | 值 |
|------|-----|
| 策略名称 | {strategy} |
| 策略类型 | {'机器学习 (ML)' if is_ml else '规则策略'} |
| 策略参数 | {params} |
| 当前信号 | {current_signal} |
| 信号置信度 | {confidence_emoji} {confidence_label} ({artifact.get('validated','unknown')}) |

> **信号置信度** = 综合夏普率、验证达标状态和验证等级评出的信号可信程度：🟢 高置信 / 🟡 中置信 / 🔴 低置信。  
> **验证等级**：`double` = Hold-Out + Walk-Forward 双重验证通过（最可靠）；`double_no_wf` = Hold-Out 通过但 WF 窗口不足；`val_only` = 仅验证集达标（需谨慎参考）。

## 策略回测评估

> **训练** = 策略拟合期，**验证** = 超参筛选期（因子文件指标来源）

{_build_eval_table(
    train_period_str, val_period, val_cum_return, val_sharpe, max_drawdown, total_trades, win_rate,
    ann_return, calmar_ratio, sortino_ratio, volatility, daily_vol,
    artifact
)}

## 信号与参考区间（非价格预测）

| 日期 | 信号 | 模型输出 | 统计区间 (P10~P90) | 置信度 |
|------|------|---------|-----------------|------|
"""
    for p in predictions:
        md_content += (f"| {p['date']} | {p['signal_str']} | {p['pred_ret_raw']} "
                       f"| [{p['price_lo']:.2f}, {p['price_hi']:.2f}] "
                       f"| {p['confidence']} |\n")

    md_content += f"""
> ⚠️ **P10~P90 统计区间**：基于过去相同持仓时长的历史日收益率分布，取第10、90百分位价格区间，**非价格预测**，不代表未来走势。  
> ⚠️ **模型输出**：ML策略输出预测收益率（单步），规则策略显示 "—"。  
> ⚠️ 多日信号基于当前最新数据推算，未来市况变化可能导致信号翻转。
"""

    # ── 持仓管理与建议（含统一风控层） ──────────────────────────────
    current_config = load_config()
    risk_cfg       = current_config.get('risk_management', {})
    portfolio_val  = float(risk_cfg.get('portfolio_value', 100_000.0))

    # 默认值（无持仓时也保持定义）
    signal_for_rec = rule_direction if not is_ml else (1 if (ml_pred_ret_raw or 0) > 0 else 0)
    rec: dict = {}

    pm       = PositionManager(portfolio_value=portfolio_val, risk_config=risk_cfg)
    position = load_position_from_config(current_config)
    if position:
        position.current_price = last_close
        pm.position = position

        # 计算当前 ATR（复用内置 calc_atr，不依赖 ta 库）
        atr_period  = int(risk_cfg.get('atr_period', 14))
        current_atr = calc_atr(data, period=atr_period)

        # peak_price：从 config 读取，0 时退化为当前入场价（保守）
        peak_price  = float(current_config.get('position_peak_price', 0.0))
        entry_price = float(current_config.get('position_avg_cost', last_close))
        if peak_price <= 0:
            peak_price = max(entry_price, last_close)

        # 当日盈亏率（用今日收盘 vs 昨日收盘估算）
        if len(data) >= 2:
            prev_close      = float(data['Close'].iloc[-2])
            today_pnl_pct   = (last_close - prev_close) / prev_close if prev_close > 0 else 0.0
        else:
            today_pnl_pct   = 0.0

        trade_date_str   = str(last_date.date())

        rec = pm.apply_risk_controls(
            signal           = signal_for_rec,
            price            = last_close,
            atr              = current_atr,
            entry_price      = entry_price,
            peak_price       = peak_price,
            today_pnl_pct    = today_pnl_pct,
            capital          = portfolio_val,
            win_rate         = win_rate,
            profit_loss_ratio= profit_loss_ratio,
            trade_date       = trade_date_str,
        )
        # 补充持仓基本字段（供报告/飞书使用）
        rec.setdefault('shares',        position.shares)
        rec.setdefault('avg_cost',      position.avg_cost)
        rec.setdefault('current_price', last_close)
        rec.setdefault('profit',        position.profit)
        rec.setdefault('profit_pct',    position.profit_pct)

        logger.info("持仓状态与建议", extra={
            "shares": rec['shares'],
            "avg_cost": rec['avg_cost'],
            "current_price": rec['current_price'],
            "profit": rec['profit'],
            "profit_pct": f"{rec['profit_pct']:+.2f}%",
            "atr": current_atr,
            "stop_price": rec['stop_price'],
            "peak_price": peak_price,
            "kelly_shares": rec['kelly_shares'],
            "circuit_breaker": rec['circuit_breaker'],
            "action": rec['action'],
            "reason": rec['reason'],
            "signal": rec['signal'],
        })

        feishu_webhook = current_config.get('feishu_webhook')
        # 注意：飞书推送已移至下方统一发送（含 4 阶段验证数据），此处不重复发送

        stop_str   = f"{rec['stop_price']:.2f}" if rec.get('stop_price', 0) > 0 else "—"
        kelly_str  = f"{rec['kelly_shares']} 股（≈ {rec['kelly_amount']:.0f} 元）" if rec.get('kelly_shares', 0) > 0 else "—"
        cb_str     = f"⚠️ 已触发（连续亏损 {rec.get('consecutive_loss_days',0)} 天）" if rec.get('circuit_breaker') else f"✅ 正常（连续亏损 {rec.get('consecutive_loss_days',0)} 天）"

        md_content += f"""
## 持仓状态与建议

| 项目 | 值 |
|------|-----|
| 持股数量 | {rec['shares']} 股 |
| 平均成本 | {rec['avg_cost']:.2f} 元 |
| 当前价格 | {rec['current_price']:.2f} 元 |
| 市值 | {rec['current_price'] * rec['shares']:.2f} 元 |
| 盈亏金额 | {rec['profit']:+.2f} 元 |
| 盈亏比例 | {rec['profit_pct']:+.2f}% |

## 风控状态

| 项目 | 值 |
|------|-----|
| ATR({atr_period})⁷ | {current_atr:.4f} |
| 建议止损价⁸ | {stop_str} |
| Kelly 建议股数⁹ | {kelly_str} |
| 熔断状态¹⁰ | {cb_str} |

> ⁷ **ATR（Average True Range）**：平均真实波幅，衡量近{atr_period}日价格波动范围，用于动态设置止损距离。  
> ⁸ **建议止损价**：基于 ATR 动态计算的止损位（当前价 - N×ATR），触及时建议离场控制亏损。  
> ⁹ **Kelly 建议股数**：凯利公式根据策略胜率与盈亏比计算的最优仓位，可最大化长期几何增长率。  
> ¹⁰ **熔断**：连续亏损保护机制，连续亏损达到阈值后自动暂停交易信号，直至市况改善。

### 交易建议

- **操作**: {rec['action']}
- **原因**: {rec['reason']}
- **信号**: {"持仓 🟢" if rec['signal'] == 1 else "空仓 🔴"}

"""


    # ── 发送飞书 ──────────────────────────────────────────────────
    feishu_webhook = config.get('feishu_webhook')
    if feishu_webhook:
        signal_text = "做多" if (signal_for_rec == 1) else "空仓"
        _rec = rec
        feishu_report_data = {
            'ticker':           config.get('ticker', '0700.hk'),
            'current_price':    last_close,
            'last_date':        str(last_date.date()),
            'strategy':         strategy,
            'params':           params,
            'is_ml':            is_ml,
            'signal':           signal_text,
            'confidence_label': confidence_label,
            'confidence_emoji': confidence_emoji,
            'train_period':     train_period_str,
            'val_period':       val_period,
            'cum_return':       val_cum_return,
            'sharpe':           val_sharpe,
            'annualized_return': ann_return,
            'max_drawdown':     max_drawdown,
            'volatility':       volatility,
            'total_trades':     total_trades,
            'win_rate':         win_rate,
            'calmar_ratio':     calmar_ratio,
            'sortino_ratio':    sortino_ratio,
            'avg_volatility':   daily_vol,
            'predictions':      predictions,
            'sentiment':        sentiment_result,
            'position': {
                'shares':          _rec.get('shares', 0),
                'avg_cost':        _rec.get('avg_cost', 0),
                'current_price':   _rec.get('current_price', last_close),
                'profit':          _rec.get('profit', 0),
                'profit_pct':      _rec.get('profit_pct', 0),
                'stop_price':      _rec.get('stop_price', 0),
                'kelly_shares':    _rec.get('kelly_shares', 0),
                'kelly_amount':    _rec.get('kelly_amount', 0),
                'circuit_breaker': _rec.get('circuit_breaker', False),
            },
            'recommendation': _rec,
        }
        send_full_report_to_feishu(feishu_webhook, feishu_report_data)
        logger.info("飞书报告已发送")

    # ── 因子有效性分析（Issue 7）──────────────────────────────────
    try:
        # 用当前最新信号序列作为代理因子
        _fa_signal = pred_signal_series if (is_ml and pred_signal_series is not None) else None
        if _fa_signal is None and not is_ml and strategy_mod is not None:
            try:
                _fa_signal, _, _ = strategy_mod.run(data.copy(), config)
            except Exception:
                pass
        if _fa_signal is not None:
            fa_result = run_factor_analysis(df, _fa_signal, config)
            md_content += fa_result.get('summary_md', '')
    except Exception as _fa_e:
        logger.warning("因子分析失败", extra={"error": str(_fa_e)})

    md_content += """
## 风险提示

⚠️ 以上分析仅供参考，不构成投资建议。策略基于历史数据，实际市场走势可能存在较大差异。

---
*本报告由自动分析系统生成*
"""

    report_dir  = Path(__file__).parent / 'data' / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report_path.write_text(md_content, encoding='utf-8')
    logger.info("Markdown报告已保存", extra={"report_path": str(report_path)})

    return md_content


# ══════════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════════

def _print_portfolio_summary(results: list[dict], config: dict) -> None:
    """打印组合训练汇总表，可选飞书通知。"""
    import math as _m
    ok     = [r for r in results if r['status'] == 'ok']
    failed = [r for r in results if r['status'] != 'ok']
    badge_map = {
        'double':       '🏅 双验证',
        'double_no_wf': '🥈 WF不足',
        'val_only':     '⚠️  验证集',
        'unknown':      '❓ 未知',
    }
    lines = [
        "", "=" * 66, "  分层混合组合训练完成汇总", "=" * 66,
        f"  成功: {len(ok)}  失败: {len(failed)}  共: {len(results)}",
        "-" * 66,
        f"  {'股票代码':<12s}  {'状态':<8s}  {'Sharpe':>8s}  {'验证等级':<14s}",
        "-" * 66,
    ]
    for r in results:
        if r['status'] == 'ok':
            sh     = r.get('sharpe_ratio', float('nan'))
            sh_str = f"{sh:8.4f}" if not _m.isnan(sh) else '     N/A'
            badge  = badge_map.get(r.get('validated', 'unknown'), '❓')
            lines.append(f"  {r['ticker']:<12s}  {'✅ OK':<8s}  {sh_str}  {badge}")
        else:
            err = str(r.get('error', ''))[:30]
            lines.append(f"  {r['ticker']:<12s}  {'❌ FAIL':<8s}  {'':>8s}  {err}")
    lines += ["=" * 66, ""]
    logger.info("组合训练汇总\n%s", "\n".join(lines))
    logger.info("组合训练汇总", extra={"ok": len(ok), "failed": len(failed)})

    feishu_webhook = config.get('feishu_webhook')
    if feishu_webhook and ok:
        rows = []
        for r in results:
            sh = r.get('sharpe_ratio', float('nan'))
            if r['status'] == 'ok':
                rows.append(
                    f"- {r['ticker']}  Sharpe={'N/A' if _m.isnan(sh) else f'{sh:.4f}'}"
                    f"  {badge_map.get(r.get('validated','unknown'), '❓')}"
                )
            else:
                rows.append(f"- {r['ticker']}  ❌ {r['status']}")
        msg = (
            f"**分层混合组合训练完成**  {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
            f"成功 {len(ok)} / 共 {len(results)}\n\n"
            + "\n".join(rows)
        )
        try:
            from feishu_notify import send_feishu_message
            send_feishu_message(feishu_webhook, msg, msg_type="markdown")
        except Exception as e:
            logger.warning("飞书通知失败", extra={"error": str(e)})


def main():
    parser = argparse.ArgumentParser(description='腾讯股票分析一体化流程')
    parser.add_argument(
        '--sources', type=str, default=None,
        help='数据源优先级（逗号分隔），例如 "yahooquery,yfinance"'
    )
    parser.add_argument(
        '--skip-train', action='store_true',
        help='跳过超参搜索，直接使用 data/factors/ 中最新的因子做预测'
    )
    parser.add_argument(
        '--n-days', type=int, default=3,
        help='日线预测天数（默认 3）'
    )
    parser.add_argument(
        '--use-optuna', action='store_true',
        help='使用 Optuna 贝叶斯优化替代随机搜索'
    )
    parser.add_argument(
        '--optuna-trials', type=int, default=50,
        help='Optuna 搜索次数（默认 50）'
    )
    parser.add_argument(
        '--strategy-type', type=str, default=None,
        choices=['single', 'multi', 'custom'],
        help='只运行指定类型的策略 (single/multi/custom)'
    )
    parser.add_argument(
        '--portfolio', action='store_true',
        help='分层混合模式：ML全局训练一次 + 对 portfolio.yaml 每只股票训练规则策略',
    )
    parser.add_argument(
        '--skip-data-download', action='store_true',
        help='跳过数据下载，强制使用本地缓存文件（如无缓存则报错退出）',
    )
    args = parser.parse_args()

    # 加载配置
    config = load_config()

    sources_override = None
    if args.sources:
        sources_override = [s.strip() for s in args.sources.split(',') if s.strip()]

    if args.portfolio:
        from engine.portfolio_state import load_portfolio
        portfolio_state = load_portfolio()
        tickers = portfolio_state.all_tickers()
        if not tickers:
            default_ticker = config.get('ticker', '0700.HK').upper()
            tickers = [default_ticker]
            logger.warning("portfolio.yaml 无标的，降级使用 config.yaml ticker",
                           extra={"ticker": default_ticker})
        logger.info("分层混合组合训练启动",
                    extra={"ticker_count": len(tickers), "tickers": tickers})
        use_optuna = args.use_optuna or config.get('use_optuna', False)
        results = train_portfolio_tickers(
            tickers=tickers,
            use_optuna=use_optuna,
            optuna_trials=args.optuna_trials,
            sources_override=sources_override,
            n_days=args.n_days,
            skip_download=args.skip_data_download,
        )
        _print_portfolio_summary(results, config)
        return

    logger.info("腾讯股票分析流程启动")

    # ── 步骤 1 ────────────────────────────────────────────────────
    hist_data, hist_path = step1_ensure_data(sources_override, skip_download=args.skip_data_download)

    if not args.skip_data_download:
        _ensure_hsi_data(config)

    # ── 步骤 2 ────────────────────────────────────────────────────
    factors_dir = Path(__file__).parent / 'data' / 'factors'
    factor_path = None

    if args.skip_train:
        factor_path = _latest_factor_path(factors_dir)
        if factor_path:
            logger.info("跳过训练，使用现有因子", extra={"factor_file": Path(factor_path).name})
        else:
            logger.warning("--skip-train指定但无因子文件，执行正常训练")
            args.skip_train = False

    if not args.skip_train:
        # 默认使用配置中的 use_optuna，可以通过命令行参数覆盖
        use_optuna = args.use_optuna if args.use_optuna else config.get('use_optuna', False)
        factor_path, _, _ = step2_train(hist_data, use_optuna=use_optuna, optuna_trials=args.optuna_trials, strategy_type=args.strategy_type)
        if factor_path is None:
            # 保存失败时兜底取最新已有文件
            factor_path = _latest_factor_path(factors_dir)
            if factor_path:
                logger.info("使用已有最新因子", extra={"factor_file": Path(factor_path).name})

    # ── 步骤 3 ────────────────────────────────────────────────────
    logger.info("步骤3/3: 信号报告开始")

    if factor_path is None:
        logger.critical("没有可用的因子/模型，无法进行预测")
        return

    # 3a. 信号报告（日线）
    generate_signal_report(hist_data, factor_path, n_days=args.n_days)



    logger.info("分析流程完成")


if __name__ == "__main__":
    main()

