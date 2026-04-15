"""
main.py — 腾讯股票分析训练流程（入口）

步骤 1 : 数据就绪检查
步骤 2 : 多策略超参搜索 + 因子保存（因子库更新）

推断 / 操作建议请运行: python3 daily_run.py

训练逻辑已拆分到 pipeline/ 子包：
  pipeline/data_prep.py       — step1_ensure_data, _ensure_hk_data
  pipeline/select.py          — _save_factor, _next_factor_run_id
  pipeline/train.py           — step2_train, step2_train_native, step2_train_optuna
  pipeline/train_portfolio.py — train_portfolio_tickers, _print_portfolio_summary
"""

import argparse
import sys
import numpy as np
import pandas as pd
import joblib
import yaml
from pathlib import Path
from datetime import datetime, timedelta

from log_config import get_logger
from config_loader import load_config

# ── pipeline 子包（主逻辑） ────────────────────────────────────────
from pipeline.data_prep import step1_ensure_data, _ensure_hk_data, _hist_data_is_stale
from pipeline.select import _save_factor, _next_factor_run_id, _latest_factor_path
from pipeline.train import step2_train, step2_train_native, step2_train_optuna
from pipeline.train_portfolio import (
    train_portfolio_tickers,
    _train_meta_model,
    _print_portfolio_summary,
)

# ── 向后兼容：其余模块可能直接从 main 导入 ────────────────────────
from analyze_factor import (
    _discover_strategies,
    run_search,
    backtest,
    _select_best_with_holdout,
    run_factor_analysis,
)
from data.calendar import prev_trading_day as _prev_hk_trading_day, is_trading_day as _is_hk_trading_day
from data.factor_registry import FactorRegistry, _get_training_type
from position_manager import PositionManager, load_position_from_config, calc_atr

try:
    from backtest_vectorbt import backtest_vectorbt as backtest_vbt
except ImportError:
    backtest_vbt = None

try:
    from optimize_with_optuna import optimize_strategy, optimize_all_strategies
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    optimize_strategy = None
    optimize_all_strategies = None

logger = get_logger(__name__)

# backward-compat alias
_ensure_hsi_data = _ensure_hk_data


def _last_trading_day(ref: datetime | None = None) -> datetime:
    d = (ref or datetime.now()).date()
    d = _prev_hk_trading_day(d)
    return datetime(d.year, d.month, d.day, 16, 10)


# ══════════════════════════════════════════════════════════════════
#  Utility helpers (kept for backward compat / generate_signal_report)
# ══════════════════════════════════════════════════════════════════

def _resolve_artifact(factor_path: str) -> dict:
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
    rolling = returns.rolling(horizon_days).sum().dropna()
    if len(rolling) >= 20:
        p_lo = float(rolling.quantile(lower_pct))
        p_hi = float(rolling.quantile(upper_pct))
    else:
        from scipy.stats import norm as _norm
        mu  = float(returns.mean() * horizon_days)
        sig = float(returns.std() * (horizon_days ** 0.5))
        p_lo = mu + _norm.ppf(lower_pct) * sig
        p_hi = mu + _norm.ppf(upper_pct) * sig
    return last_close * (1 + p_lo), last_close * (1 + p_hi)


def _signal_confidence(artifact: dict, is_ml: bool) -> tuple[str, str]:
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
    import math as _m

    def _f(v, fmt='.2%'):
        if v is None or (isinstance(v, float) and _m.isnan(v)):
            return '—'
        return f"{v:{fmt}}"

    def _badge(ok: bool) -> str:
        return '✅' if ok else '❌'

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
#  步骤 3a : 信号报告（已移至 daily_run.py — 此处保留供调试）
# ══════════════════════════════════════════════════════════════════

def generate_signal_report(data: pd.DataFrame, factor_path: str, n_days: int = 3) -> str:
    # 暂停使用：预测职责已移至 daily_run.py。本函数保留供调试，不在主流程中调用。
    raise NotImplementedError(
        "generate_signal_report 已移至 daily_run.py。请运行 python3 daily_run.py。"
    )


# ══════════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='港股分析训练流程')
    parser.add_argument('--sources', type=str, default=None,
                        help='数据源优先级（逗号分隔），例如 "yahooquery,yfinance"')
    parser.add_argument('--use-optuna', action='store_true',
                        help='使用 Optuna 贝叶斯优化替代随机搜索')
    parser.add_argument('--strategy-type', type=str, default=None,
                        choices=['single', 'multi', 'custom'],
                        help='只运行指定类型的策略 (single/multi/custom)')
    parser.add_argument('--portfolio', action='store_true',
                        help='分层混合模式：ML全局训练一次 + 对 portfolio.yaml 每只股票训练规则策略')
    parser.add_argument('--skip-data-download', action='store_true',
                        help='跳过数据下载，强制使用本地缓存文件（如无缓存则报错退出）')
    args = parser.parse_args()

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
            sources_override=sources_override,
            skip_download=args.skip_data_download,
        )
        _print_portfolio_summary(results, config)
        return

    logger.info("训练流程启动")

    hist_data, hist_path = step1_ensure_data(sources_override, skip_download=args.skip_data_download)

    if not args.skip_data_download:
        _ensure_hk_data(config)

    use_optuna = args.use_optuna if args.use_optuna else config.get('use_optuna', False)
    factor_path, best_result, _ = step2_train(
        hist_data,
        use_optuna=use_optuna,
        strategy_type=args.strategy_type,
    )
    if factor_path is None:
        factor_path = _latest_factor_path(Path(__file__).parent / 'data' / 'factors')
        if factor_path:
            logger.info("使用已有最新因子", extra={"factor_file": Path(factor_path).name})

    ticker = config.get('ticker', '0700.HK').upper()

    _train_meta_model(
        ticker=ticker,
        data=hist_data,
        config=config,
        factors_dir=Path(__file__).parent / "data" / "factors",
        meta_dir=Path(__file__).parent / "data" / "meta",
    )

    result = {
        "ticker":      ticker,
        "status":      "ok" if best_result is not None else "no_factor",
        "sharpe_ratio": best_result.get('sharpe_ratio', float('nan')) if best_result else float('nan'),
        "validated":   best_result.get('validated', 'unknown') if best_result else 'unknown',
        "ml_status":   "n/a",
    }
    _print_portfolio_summary([result], config)
    logger.info("训练流程完成")


if __name__ == "__main__":
    main()
