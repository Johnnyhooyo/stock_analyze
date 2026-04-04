import importlib
from typing import Optional
import pandas as pd
import numpy as np
import yaml
import shutil
from scipy import stats as _scipy_stats
from sklearn.metrics import accuracy_score, roc_auc_score, log_loss
from pathlib import Path
from config_loader import load_config
from log_config import get_logger

logger = get_logger(__name__)

# Vectorbt 引擎（可选）
try:
    from backtest_vectorbt import backtest_vectorbt as backtest_vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False

# ── 多股票数据缓存（P3-A：改用 joblib.Memory 磁盘缓存） ──────────────
# 旧做法：进程级 dict，Optuna 多进程时每个 worker 各自重复加载磁盘数据。
# 新做法：joblib.Memory 写到 data/cache/，所有进程共享同一份缓存文件。
import joblib as _joblib
_CACHE_DIR = Path(__file__).parent / 'data' / 'cache'
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
_memory = _joblib.Memory(location=str(_CACHE_DIR), verbose=0)


@_memory.cache
def _load_multi_stock_data_cached(period: str = '5y', min_days: int = 300) -> pd.DataFrame:
    """实际加载逻辑，结果由 joblib.Memory 缓存到磁盘。"""
    try:
        from train_multi_stock import load_all_hk_data
        data = load_all_hk_data(period=period, min_days=min_days)
        if not data.empty and 'ticker' in data.columns:
            data = data.drop(columns=['ticker'])
        return data
    except ImportError:
        return pd.DataFrame()


def _load_multi_stock_data(period: str = '5y', min_days: int = 300) -> pd.DataFrame:
    """加载多股票数据（磁盘缓存，多进程安全）。"""
    return _load_multi_stock_data_cached(period, min_days)


def clear_multi_stock_cache():
    """清除多股票数据磁盘缓存。"""
    _memory.clear(warn=False)



# ══════════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════════

def _save_config(cfg: dict) -> None:
    config_path = Path(__file__).parent / 'config.yaml'
    tmp = config_path.with_suffix('.yaml.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, allow_unicode=True)
    shutil.move(str(tmp), str(config_path))


def _discover_strategies(strategy_type: str = None, cfg: dict = None) -> list:
    """自动发现 strategies/ 包下的所有策略模块，返回模块列表。

    Args:
        strategy_type: 可选，按类型过滤策略
            - 'single': 只返回单股票训练策略
            - 'multi': 只返回多股票训练策略
            - 'custom': 只返回自定义训练策略
            - None: 返回所有策略
        cfg: 可选，直接传入配置字典（避免重复加载）
    """
    config = cfg if cfg is not None else load_config()
    train_config = config.get('strategy_training', {})

    # 合并 single, multi, custom 中的所有策略
    strategy_list = set()
    if strategy_type is None:
        # 返回所有策略
        for key in ['single', 'multi', 'custom']:
            strategy_list.update(train_config.get(key, []))
    else:
        # 只返回指定类型的策略
        strategy_list.update(train_config.get(strategy_type, []))

    # 如果配置为空，使用默认策略
    if not strategy_list:
        strategy_list = {'bollinger_rsi_trend', 'macd_rsi_trend', 'rsi_divergence',
                        'xgboost_enhanced', 'lightgbm_enhanced'}

    modules = []
    for name in strategy_list:
        try:
            mod = importlib.import_module(f'strategies.{name}')
            if hasattr(mod, 'run') and hasattr(mod, 'NAME'):
                modules.append(mod)
        except ImportError as e:
            import logging
            logging.getLogger(__name__).warning(f"策略 '{name}' 导入失败: {e}")
    return modules


# ══════════════════════════════════════════════════════════════════
#  真正的因子有效性分析（IC / ICIR / 分层收益 / 因子衰减）
# ══════════════════════════════════════════════════════════════════

def compute_ic(factor: pd.Series, forward_returns: pd.Series) -> float:
    """
    计算 Rank IC（信息系数）。
    factor          : 因子截面值（任意数值序列）
    forward_returns : 因子对应的未来收益率（与 factor 对齐）
    返回 Spearman 相关系数；样本不足或无变化时返回 nan。
    """
    combined = pd.concat([factor, forward_returns], axis=1).dropna()
    if len(combined) < 5:
        return float('nan')
    r, _ = _scipy_stats.spearmanr(combined.iloc[:, 0], combined.iloc[:, 1])
    return float(r)


def compute_icir(ic_series: pd.Series) -> float:
    """
    计算 ICIR = IC均值 / IC标准差（年化信息比率的滚动估计）。
    ic_series : 一组时序 IC 值
    返回 ICIR；标准差为 0 时返回 nan。
    """
    ic_series = ic_series.dropna()
    if len(ic_series) < 2:
        return float('nan')
    std = float(ic_series.std(ddof=1))
    if std == 0:
        return float('nan')
    return float(ic_series.mean() / std)


def quintile_analysis(
    factor: pd.Series,
    forward_returns: pd.Series,
    n_quantiles: int = 5,
) -> pd.DataFrame:
    """
    因子分层（分位数）收益分析。
    factor          : 因子值序列
    forward_returns : 对应的未来收益率
    n_quantiles     : 分层数量（默认 5）

    返回 DataFrame，index 为分层编号（1=最小因子，n=最大因子），
    columns = ['mean_return', 'std_return', 'count', 'sharpe']。
    """
    combined = pd.concat([factor.rename('factor'), forward_returns.rename('ret')], axis=1).dropna()
    if len(combined) < n_quantiles * 3:
        return pd.DataFrame()

    try:
        combined['quantile'] = pd.qcut(combined['factor'], q=n_quantiles, labels=False, duplicates='drop') + 1
    except Exception:
        return pd.DataFrame()

    rows = []
    for q in sorted(combined['quantile'].unique()):
        subset = combined.loc[combined['quantile'] == q, 'ret']
        mu  = float(subset.mean())
        std = float(subset.std(ddof=1)) if len(subset) > 1 else float('nan')
        sharpe = mu / std * np.sqrt(252) if (not np.isnan(std) and std > 0) else float('nan')
        rows.append({'quantile': int(q), 'mean_return': mu, 'std_return': std,
                     'count': len(subset), 'sharpe': sharpe})

    return pd.DataFrame(rows).set_index('quantile')


def factor_decay(
    factor: pd.Series,
    close: pd.Series,
    max_lag: int = 20,
) -> pd.Series:
    """
    计算因子在不同预测期（1…max_lag）的 Rank IC，
    反映因子信号随时间的衰减速度。

    factor : 因子截面值（时间序列）
    close  : 收盘价序列（与 factor 同索引）
    max_lag: 最大预测期（天数）

    返回 pd.Series，index 为 lag（1..max_lag），values 为 Rank IC。
    """
    results = {}
    returns = close.pct_change()
    for lag in range(1, max_lag + 1):
        fwd_ret = returns.shift(-lag)
        ic = compute_ic(factor, fwd_ret)
        results[lag] = ic
    return pd.Series(results, name='IC')


def run_factor_analysis(
    data: pd.DataFrame,
    signal: pd.Series,
    config: dict,
    max_decay_lag: int = 20,
) -> dict:
    """
    综合因子有效性分析入口（供 main.py / generate_signal_report 调用）。

    Parameters
    ----------
    data    : 历史 DataFrame（必须含 Close 列）
    signal  : 策略信号序列（1/0），作为代理因子
    config  : 配置字典
    max_decay_lag : 衰减分析的最大预测期

    Returns
    -------
    dict，包含：
      ic_mean, ic_std, icir        — 时序 IC 统计
      quintile_df                  — 分层收益 DataFrame
      decay_series                 — 因子衰减 pd.Series
      ic_ttest_p                   — IC 是否显著异于 0 的 t 检验 p 值
      summary_md                   — 供报告嵌入的 Markdown 摘要
    """
    if 'Close' not in data.columns:
        return {'error': '缺少 Close 列'}

    close   = data['Close'].copy()
    returns = close.pct_change()

    # 1 天预测期前向收益
    fwd_ret_1d = returns.shift(-1)

    # ── 滚动时序 IC（按月）──────────────────────────────────────────
    # 将信号重采样为月度，计算每月的 IC
    combined = pd.concat([signal.rename('factor'), fwd_ret_1d.rename('ret')], axis=1).dropna()
    if len(combined) >= 5:
        # 整体 IC（单个截面值）
        ic_single = compute_ic(combined['factor'], combined['ret'])
        # 滚动月度 IC
        monthly_ic = []
        combined_m = combined.resample('ME').apply(lambda g: compute_ic(g['factor'], g['ret']))
        # resample 对每列独立处理，取 factor 列后重新计算
        combined2 = combined.copy()
        combined2.index = pd.to_datetime(combined2.index)
        for month, grp in combined2.groupby(combined2.index.to_period('M')):
            if len(grp) >= 3:
                ic_m = compute_ic(grp['factor'], grp['ret'])
                monthly_ic.append(ic_m)
        ic_series = pd.Series(monthly_ic).dropna()
    else:
        ic_single = float('nan')
        ic_series = pd.Series(dtype=float)

    ic_mean = float(ic_series.mean()) if not ic_series.empty else ic_single
    ic_std  = float(ic_series.std(ddof=1)) if len(ic_series) > 1 else float('nan')
    icir    = compute_icir(ic_series)

    # ── IC t 检验 ──────────────────────────────────────────────────
    ic_ttest_p = float('nan')
    if len(ic_series) >= 5:
        t_stat, ic_ttest_p = _scipy_stats.ttest_1samp(ic_series.dropna(), 0)
        ic_ttest_p = float(ic_ttest_p)

    # ── 分层收益 ───────────────────────────────────────────────────
    quintile_df = quintile_analysis(combined['factor'] if len(combined) >= 5 else signal, fwd_ret_1d)

    # ── 因子衰减 ───────────────────────────────────────────────────
    decay_series = factor_decay(signal, close, max_lag=max_decay_lag)

    # ── Markdown 摘要 ──────────────────────────────────────────────
    def _fmt(v, fmt='.4f'):
        return f"{v:{fmt}}" if isinstance(v, float) and not np.isnan(v) else '—'

    md_rows = ''
    if not quintile_df.empty:
        for q, row in quintile_df.iterrows():
            md_rows += (f"| Q{q} | {row['mean_return']:.4f} | {row['std_return']:.4f} "
                        f"| {int(row['count'])} | {_fmt(row['sharpe'])} |\n")

    decay_top5 = '  '.join(f"L{i}={_fmt(decay_series.get(i, float('nan')))}"
                            for i in range(1, min(6, max_decay_lag+1)))

    summary_md = f"""
## 因子有效性分析

| 指标 | 值 |
|------|-----|
| IC 均值（月度滚动）ᵃ | {_fmt(ic_mean)} |
| IC 标准差 | {_fmt(ic_std)} |
| ICIRᵇ | {_fmt(icir)} |
| IC t 检验 p 值 | {_fmt(ic_ttest_p)} {'✅ 显著(p<0.05)' if isinstance(ic_ttest_p, float) and ic_ttest_p < 0.05 else '❌ 不显著'} |

> ᵃ **IC（信息系数）**：策略信号与未来1日收益率的 Spearman 秩相关系数，反映信号对收益的预测能力；|IC| > 0.05 有参考价值，> 0.1 为较强信号。  
> ᵇ **ICIR**：IC均值 ÷ IC标准差，衡量信号预测能力的稳定性；> 0.5 为较优，> 1.0 为优秀。

### 分层收益ᶜ（Q1=信号最小，Q{len(quintile_df) if not quintile_df.empty else 'N'}=最大）

| 分层 | 均值收益 | 标准差 | 样本数 | 夏普 |
|------|---------|--------|--------|------|
{md_rows if md_rows else '| — | — | — | — | — |\n'}

> ᶜ **分层收益**：将因子值从小到大分5组，观察各组未来1日的平均收益，若呈单调递增说明因子有效。

### 因子衰减ᵈ（前5天 Rank IC）

{decay_top5}

> ᵈ **因子衰减**：信号在未来1~5日的 IC，反映信号的持续有效期；衰减越慢说明信号持效越长。  
> ⚠️ 以上因子分析基于策略信号作为代理因子，IC 为时序 Rank IC，非截面 IC。
"""

    return {
        'ic_mean':    ic_mean,
        'ic_std':     ic_std,
        'icir':       icir,
        'ic_ttest_p': ic_ttest_p,
        'quintile_df': quintile_df,
        'decay_series': decay_series,
        'summary_md': summary_md,
    }


def _check_meets_threshold(bt: dict, min_return: float, min_sharpe: float,
                           max_dd: float, min_trades: int,
                           train_acc: float = float('nan'),
                           val_acc: float = float('nan')) -> bool:
    """
    多维度验证策略是否满足阈值要求（防止过拟合）

    Args:
        bt: 回测结果字典
        min_return: 最低累计收益要求
        min_sharpe: 最低夏普比率要求
        max_dd: 最大回撤限制（负值，如 -0.15 表示最多回撤15%）
        min_trades: 最少交易次数要求
        train_acc: 训练集准确率（ML 策略专用，来自 meta['train_acc']）
        val_acc: 验证集准确率（ML 策略专用，由 analyze_factor 计算）

    Returns:
        bool: 是否满足所有阈值
    """
    cum_return = float(bt.get('cum_return', 0))
    sharpe = float(bt.get('sharpe_ratio', float('nan')))
    max_drawdown = float(bt.get('max_drawdown', 0))
    total_trades = int(bt.get('total_trades', 0))

    # 检查各项指标
    checks = {
        '收益': cum_return > min_return,
        '夏普率': not np.isnan(sharpe) and sharpe > min_sharpe,
        '回撤': max_drawdown >= max_dd,  # max_drawdown是负值
        '交易次数': total_trades >= min_trades,
    }

    # ── 修复项2：ML 质量门禁（仅对 ML 策略生效，val_acc 有值时才检查） ──
    if not np.isnan(val_acc):
        checks['ML验证准确率'] = val_acc > 0.52  # 高于随机基准 +2%
    if not np.isnan(train_acc) and not np.isnan(val_acc):
        checks['ML过拟合间隙'] = (train_acc - val_acc) < 0.10  # 训练/验证差 < 10%

    # 所有条件必须同时满足
    passed = all(checks.values())

    # 调试信息
    if not passed:
        failed = [k for k, v in checks.items() if not v]
        logger.debug(f"[验证失败] {' '.join(failed)}")

    return passed


# ══════════════════════════════════════════════════════════════════
#  回测引擎（纯函数，与策略无关）
# ══════════════════════════════════════════════════════════════════

def backtest(data: pd.DataFrame, signal: pd.Series, config: dict) -> dict:
    """
    在 data 上执行回测模拟。
    signal : int Series (1=做多, 0=观望), 索引与 data 对齐
    返回    : dict 包含 cum_return, annualized_return, buy_cnt, sell_cnt,
                       portfolio_value Series, strategy Series
    """
    initial_capital = float(config.get('initial_capital', 100_000.0))
    invest_fraction = float(config.get('invest_fraction', 0.95))  # 默认 0.95，预留现金缓冲
    lookback_months = int(config.get('lookback_months', 3))
    slippage        = float(config.get('slippage', 0.001))          # 滑点，与 vectorbt 引擎统一

    # 港股费率设置
    fees_rate = float(config.get('fees_rate', 0.00088))  # 买入费率 ~0.088%
    stamp_duty = float(config.get('stamp_duty', 0.001))   # 印花税 ~0.1%（仅卖出）

    # ── ATR 止损回测模拟（Issue #9 修复） ──────────────────────────
    risk_cfg = config.get('risk_management', {})
    if risk_cfg.get('simulate_in_backtest', True) and risk_cfg.get('use_atr_stop', True):
        try:
            from position_manager import simulate_atr_stoploss
            signal = simulate_atr_stoploss(
                data, signal,
                atr_period=int(risk_cfg.get('atr_period', 14)),
                atr_multiplier=float(risk_cfg.get('atr_multiplier', 2.0)),
                trailing=bool(risk_cfg.get('trailing_stop', True)),
                cooldown_bars=int(risk_cfg.get('cooldown_bars', 0)),
            )
        except Exception as _e:
            import logging as _logging
            _logging.getLogger(__name__).warning(f"[backtest] ATR 止损模拟失败（已跳过）: {_e}")
    # ────────────────────────────────────────────────────────────────

    bt = data.copy()
    # ⚠️ 前视偏差修复：信号在第 T 天收盘后生成，最早在第 T+1 天开盘执行。
    # 将信号整体后移 1 个交易日，确保回测中不会在生成信号的同一根 K 线上成交。
    bt['signal']    = signal.reindex(bt.index).fillna(0).shift(1).fillna(0).astype(int)
    bt['position']  = 0
    bt['trade']     = 0
    bt['shares']    = 0
    bt['cash']      = 0.0
    bt['pv']        = 0.0
    bt['strategy']  = 0.0

    cash, shares, position, prev_pv = initial_capital, 0, 0, None

    for idx, row in bt.iterrows():
        price   = float(row['Close'])
        desired = int(row['signal'])
        trade   = 0

        if desired == 1 and position == 0:
            # 买入：价格加滑点，扣除手续费
            exec_price = price * (1 + slippage)
            available_cash = cash / (1 + fees_rate)
            n = int(available_cash * invest_fraction // exec_price)
            if n > 0:
                cost = n * exec_price
                fees = cost * fees_rate
                cash = cash - cost - fees
                shares += n
                position = 1
                trade = 1
        elif desired == 0 and position == 1:
            # 卖出：价格减滑点，扣除手续费和印花税
            exec_price = price * (1 - slippage)
            proceeds = shares * exec_price
            fees = proceeds * (fees_rate + stamp_duty)
            cash = cash + proceeds - fees
            shares = 0
            position = 0
            trade = -1

        pv        = cash + shares * price
        daily_ret = 0.0 if prev_pv is None else (pv / prev_pv) - 1.0

        bt.at[idx, 'position'] = position
        bt.at[idx, 'trade']    = trade
        bt.at[idx, 'shares']   = shares
        bt.at[idx, 'cash']     = cash
        bt.at[idx, 'pv']       = pv
        bt.at[idx, 'strategy'] = daily_ret
        prev_pv = pv

    bt['strategy'] = bt['strategy'].astype(float)
    cum_return      = (1 + bt['strategy'].fillna(0)).prod() - 1
    try:
        # 使用实际交易天数计算年化收益，避免 lookback_months 近似误差。
        # 当 (1+cum_return) ≤ 0（本金归零）时，年化无意义，直接返回 -1。
        trading_days_total = max(1, len(bt))
        base = 1 + cum_return
        if base <= 0:
            ann = -1.0
        else:
            ann = base ** (252.0 / trading_days_total) - 1
    except Exception:
        ann = float('nan')

    # ── 夏普率（年化，无风险利率取 0） ──
    strat_rets = bt['strategy'].fillna(0)
    mean_ret   = float(strat_rets.mean()) if len(strat_rets) > 0 else 0.0  # 提前赋值，避免后续引用未定义
    try:
        trading_days = max(1, len(strat_rets))
        ann_factor   = 252 / trading_days          # 年化因子（以 252 交易日/年为准）
        mean_ret     = float(strat_rets.mean())
        std_ret      = float(strat_rets.std(ddof=1))
        if std_ret > 0:
            sharpe = mean_ret / std_ret * np.sqrt(252)
        else:
            sharpe = float('nan')
    except Exception:
        sharpe = float('nan')

    # ── 最大回撤 ──
    pv = bt['pv']
    running_max = pv.expanding().max()
    drawdown = (pv - running_max) / running_max
    max_drawdown = float(drawdown.min())

    # ── 波动率 ──
    volatility = float(strat_rets.std(ddof=1) * np.sqrt(252))

    # ── 胜率 & 盈亏比 ──
    trades = bt[bt['trade'] != 0].copy()
    if len(trades) > 1:
        trade_returns = []
        for i in range(len(trades) - 1):
            if trades.iloc[i]['trade'] == 1:  # 买入
                entry_price = trades.iloc[i]['Close']
                # 找下一个卖出
                for j in range(i + 1, len(trades)):
                    if trades.iloc[j]['trade'] == -1:  # 卖出
                        exit_price = trades.iloc[j]['Close']
                        ret = (exit_price - entry_price) / entry_price
                        trade_returns.append(ret)
                        break
        if trade_returns:
            wins = [r for r in trade_returns if r > 0]
            losses = [r for r in trade_returns if r <= 0]
            win_rate = len(wins) / len(trade_returns) if trade_returns else 0
            avg_win = np.mean(wins) if wins else 0
            avg_loss = np.mean(losses) if losses else 0
            profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        else:
            win_rate = 0
            profit_loss_ratio = 0
    else:
        win_rate = 0
        profit_loss_ratio = 0

    # ── 卡玛比率 (Calmar Ratio) ──
    if max_drawdown != 0:
        calmar_ratio = float(ann) / abs(max_drawdown)
    else:
        calmar_ratio = float('nan')

    # ── 索提诺比率 (Sortino Ratio) ──
    downside_returns = strat_rets[strat_rets < 0]
    if len(downside_returns) > 0 and downside_returns.std(ddof=1) > 0:
        sortino_ratio = mean_ret / (downside_returns.std(ddof=1) * np.sqrt(252))
    else:
        sortino_ratio = float('nan')

    return {
        'cum_return':         float(cum_return),
        'annualized_return':  float(ann),
        'sharpe_ratio':       sharpe,
        'max_drawdown':       max_drawdown,
        'volatility':         volatility,
        'win_rate':           win_rate,
        'profit_loss_ratio':  profit_loss_ratio,
        'calmar_ratio':       calmar_ratio,
        'sortino_ratio':      sortino_ratio,
        'buy_cnt':            int(bt['trade'].eq(1).sum()),
        'sell_cnt':           int(bt['trade'].eq(-1).sum()),
        'total_trades':       int(bt['trade'].abs().sum()),
        'portfolio_value':    bt['pv'],
        'strategy':           bt['strategy'],
        'detail':             bt,
    }


# ══════════════════════════════════════════════════════════════════
#  单次试验（训练 + 验证 + 回测）
# ══════════════════════════════════════════════════════════════════

def _get_strategy_training_type(strategy_name: str, config: dict) -> str:
    """获取策略的训练类型: single, multi, 或 custom"""
    train_config = config.get('strategy_training', {})

    # 优先使用配置中的分类
    if strategy_name in train_config.get('single', []):
        return 'single'
    if strategy_name in train_config.get('multi', []):
        return 'multi'
    if strategy_name in train_config.get('custom', []):
        return 'custom'

    # 默认：xgboost/lightgbm/rnn 相关的是 multi，其他是 single
    if 'xgboost' in strategy_name or 'lightgbm' in strategy_name or \
       'ridge' in strategy_name or 'linear' in strategy_name or \
       'forest' in strategy_name or 'rnn' in strategy_name:
        return 'multi'
    return 'single'


def run_trial(strategy_mod, data: pd.DataFrame, config: dict,
              trial_num: int = 0, test_start: pd.Timestamp = None) -> Optional[dict]:
    """
    对一个策略模块执行完整的 train / val / backtest 流程。

    训练/验证策略:
    - single: 使用当前股票数据训练和验证
    - multi: 使用多股票数据训练，但验证和回测使用目标股票数据
    - custom: 使用策略自定义的训练逻辑

    验证时间统一: 所有策略使用相同的验证时长 (lookback_months)
    时间对齐: multi策略的训练数据截止时间与目标股票验证开始时间对齐

    Args:
        strategy_mod: 策略模块
        data:         历史数据（不含 test 段，由调用方在传入前切除）
        config:       配置字典
        trial_num:    当前试验编号（供早停判断，0 表示不启用早停）
        test_start:   Hold-Out 测试段起始日期；非 None 时 data 应已去掉 test 段

    返回 result dict，包含所有指标；若失败返回 None。
    """
    lookback_months = int(config.get('lookback_months', 3))
    train_years     = int(config.get('train_years', 2))
    min_return      = float(config.get('min_return', 0.10))
    min_sharpe      = float(config.get('min_sharpe_ratio', 1.0))
    max_dd          = float(config.get('max_drawdown', -0.15))
    min_trades      = int(config.get('min_total_trades', 5))
    ticker          = config.get('ticker', 'UNKNOWN')

    # 获取策略训练类型
    strategy_name = getattr(strategy_mod, 'NAME', '')
    train_type = _get_strategy_training_type(strategy_name, config)

    # ── MIN_BARS 守卫：数据不足时跳过该策略 ──────────────────────
    min_bars = getattr(strategy_mod, 'MIN_BARS', 50)
    if len(data) < min_bars:
        logger.debug(f"[{strategy_name}] 数据不足 ({len(data)} < MIN_BARS={min_bars})，跳过")
        return None

    # ── 合并 ML 策略专用配置 ──
    # 例如: xgboost_enhanced 会读取 ml_strategies.xgboost_enhanced 中的参数
    # 策略专用配置优先级: 命令行 > 全局配置 > 策略专用配置
    ml_strategies_config = config.get('ml_strategies', {})
    if strategy_name in ml_strategies_config:
        # 先复制全局配置
        merged_cfg = config.copy()
        # 再合并策略专用配置 (策略配置作为基础)
        strategy_base = ml_strategies_config[strategy_name].copy()
        # 全局配置覆盖策略配置中的同名参数 (runtime overrides)
        for key in list(strategy_base.keys()):
            if key in config:
                strategy_base[key] = config[key]
        merged_cfg.update(strategy_base)
        config = merged_cfg

    # ── 先确定验证集时间范围 (统一使用 lookback_months) ──
    # 验证集始终使用目标股票数据
    df_target = data.copy().sort_index()
    if not pd.api.types.is_datetime64_any_dtype(df_target.index):
        try:
            df_target.index = pd.to_datetime(df_target.index)
        except Exception:
            return None

    df_target['returns'] = df_target['Close'].pct_change(fill_method=None)
    df_target = df_target.dropna(subset=['returns'])
    if df_target.empty:
        return None

    # ── 三段切割：若提供了 test_start，则 val/train 段均不得越过该边界 ──
    # test_start 由 run_search() 统一计算（全量数据末尾 test_months 之前），
    # 确保 test 段在整个超参搜索阶段完全封存，不参与任何评分。
    if test_start is not None:
        # data 应由调用方在传入前已切除 test 段，此处做二重保险
        df_target = df_target.loc[df_target.index < test_start]
        if df_target.empty:
            return None

    # 验证时间范围（val 结束在封存边界之前）
    target_end_date = df_target.index.max()
    val_start = target_end_date - pd.DateOffset(months=lookback_months)
    val_df = df_target.loc[df_target.index >= val_start]

    if val_df.empty:
        return None

    # ── 准备训练数据 ──
    train_df = None

    if train_type == 'multi':
        # 多股票训练 + 单股票验证
        # 关键: 训练数据截止时间与目标股票验证开始时间对齐
        multi_data = _load_multi_stock_data(period='5y', min_days=300)
        if not multi_data.empty and 'Close' in multi_data.columns:
            df_train = multi_data.copy()

            # 多股票数据的时间切分
            df_train = df_train.sort_index()
            df_train['returns'] = df_train['Close'].pct_change(fill_method=None)
            df_train = df_train.dropna(subset=['returns'])

            # 训练数据结束时间与目标股票验证开始时间对齐
            aligned_end_date = val_start  # 使用目标股票验证开始时间
            train_end = aligned_end_date
            train_start = train_end - pd.DateOffset(years=train_years)

            train_df = df_train.loc[(df_train.index >= train_start) & (df_train.index < train_end)]
            if train_df.empty:
                train_df = df_train.loc[df_train.index < train_end]

            # ── 修复项5：截断训练集末尾 label_period 行，消除边界数据泄露 ──
            # shift(-label_period) 标签在末尾 label_period 行会引用 val_df 的未来价格
            _label_period = int(config.get('label_period', 1))
            if len(train_df) > _label_period:
                train_df = train_df.iloc[:-_label_period]

            logger.info("使用多股票数据训练", extra={
                "records": len(train_df),
                "train_end": str(train_end.date()),
                "train_type": "multi"
            })
        else:
            logger.warning("多股票数据加载失败，回退到单股票训练")
            train_type = 'single'

    if train_type != 'multi':
        # 单股票训练 + 单股票验证
        end_date = df_target.index.max()
        val_start = end_date - pd.DateOffset(months=lookback_months)
        train_start = val_start - pd.DateOffset(years=train_years)

        train_df = df_target.loc[(df_target.index >= train_start) & (df_target.index < val_start)]

        if train_df.empty:
            train_df = df_target.loc[df_target.index < val_start]

    # 确保训练数据有效
    if train_df is None or train_df.empty:
        return None

    # ── 训练集：运行策略，拟合模型 ──
    try:
        train_signal, model, meta = strategy_mod.run(train_df, config)
    except Exception as e:
        logger.warning("策略训练异常", extra={"strategy": strategy_mod.NAME, "error": str(e)})
        return None

    # ── 验证集：用同一模型在 val_df 上生成信号 ──
    # ⚠️ 前视偏差修复：验证集必须使用训练集已拟合的模型进行推断，
    # 绝不能在 val_df 上重新执行 fit（无论 single 还是 multi 策略）。
    try:
        if model is not None and hasattr(strategy_mod, 'predict'):
            # ML 策略：直接用已训练模型独立推断，不重新 fit
            val_signal = strategy_mod.predict(model, val_df, config, meta)
            val_signal = val_signal.reindex(val_df.index).fillna(0)
            val_meta = meta.copy()
            val_meta['indicators'] = {}  # predict() 不返回 indicators，置空
        else:
            # 规则策略：设置 no_internal_split=True，确保用全量 val_df 生成信号（不内部重分割）
            val_config = config.copy()
            val_config['no_internal_split'] = True
            val_signal, _, val_meta = strategy_mod.run(val_df, val_config)
    except Exception as e:
        logger.warning("验证集推理异常", extra={"strategy": meta.get("name"), "error": str(e)})
        return None

    # val_meta 的 indicators 与 val_df 索引对齐，用于绘图；
    # 训练集 meta 的 params/name 信息合并进来保持完整
    val_meta['params']    = meta.get('params', val_meta.get('params', {}))
    val_meta['feat_cols'] = meta.get('feat_cols', [])

    # ── ML 分类验证指标（仅对有模型的策略计算） ──
    # ⚠️ 修复项1：使用分类指标（accuracy/AUC/log_loss），原回归指标(r2/MAE)对分类任务无意义
    val_acc = train_acc_from_meta = roc_auc = ml_log_loss = float('nan')
    # 从策略 meta 中提取训练集准确率（由 xgboost_enhanced / lightgbm_enhanced 的 run() 写入）
    if model is not None and meta.get('train_acc') is not None:
        train_acc_from_meta = float(meta['train_acc'])

    feat_cols = meta.get('feat_cols', [])
    if model is not None and feat_cols:
        try:
            vdf = val_df.copy()
            vdf['returns'] = vdf['Close'].pct_change(fill_method=None)
            label_period_val = int(config.get('label_period', 1))
            vdf['label'] = np.where(
                vdf['Close'].shift(-label_period_val) > vdf['Close'], 1, 0
            )
            for i in range(1, int(config.get('test_days', 5)) + 1):
                vdf[f'ret_{i}'] = vdf['returns'].shift(i)
            vdf = vdf.dropna()

            if not vdf.empty and all(c in vdf.columns for c in feat_cols):
                X_val   = vdf[feat_cols]
                y_val   = vdf['label']
                y_pred  = model.predict(X_val)
                val_acc = float(accuracy_score(y_val, y_pred))
                # AUC 和 log_loss 需要概率输出
                if hasattr(model, 'predict_proba'):
                    y_proba   = model.predict_proba(X_val)[:, 1]
                    roc_auc   = float(roc_auc_score(y_val, y_proba))
                    ml_log_loss = float(log_loss(y_val, y_proba))
        except Exception:
            pass

    # ── 回测（验证集） ──
    # ✅ 1-C 确认：两个回测引擎均在内部对信号做 shift(1)，即"T日收盘生成信号 → T+1日开盘执行"。
    #    - analyze_factor.backtest()：   bt['signal'] = signal.shift(1)  (约第163行)
    #    - backtest_vectorbt():          signal_shifted = signal.shift(1) (约第41行)
    #    此处传入原始策略信号，由各引擎独立处理，不存在双重偏移或当日执行偏差。
    backtest_engine = config.get('backtest_engine', 'vectorbt')
    if backtest_engine == 'vectorbt' and VECTORBT_AVAILABLE:
        bt = backtest_vbt(val_df, val_signal, config)
    else:
        bt = backtest(val_df, val_signal, config)

    # ── 早停机制：trial >= 10 后，val 软阈值未达标直接返回 None ──
    # 目的：快速淘汰明显无效的超参组合，减少无意义的 Hold-Out 评估资源消耗。
    # 软阈值比正式阈值宽松（early_stop_threshold 默认 0.03），不会误杀潜力策略。
    early_stop_thr = float(config.get('early_stop_threshold', 0.03))
    if trial_num >= 10 and bt.get('cum_return', 0) < early_stop_thr:
        return None

    # 因子保存由外层 (main.py / run_search) 统一处理，此处不再写磁盘
    return {
        'strategy_name':      val_meta['name'],
        'params':            val_meta.get('params', {}),
        'train_rows':        len(train_df),
        'val_rows':          len(val_df),
        # ── ML 分类指标（修复项1：替换原 r2/mae/direction_acc 回归指标） ──
        'train_acc':         train_acc_from_meta,
        'val_acc':           val_acc,
        'roc_auc':           roc_auc,
        'log_loss':          ml_log_loss,
        'cum_return':        bt['cum_return'],
        'annualized_return': bt['annualized_return'],
        'sharpe_ratio':      bt['sharpe_ratio'],
        'max_drawdown':      bt.get('max_drawdown', float('nan')),
        'volatility':        bt.get('volatility', float('nan')),
        'win_rate':          bt.get('win_rate', 0),
        'profit_loss_ratio': bt.get('profit_loss_ratio', 0),
        'calmar_ratio':      bt.get('calmar_ratio', float('nan')),
        'sortino_ratio':     bt.get('sortino_ratio', float('nan')),
        'total_trades':      bt.get('total_trades', 0),
        # 兼容两种回测引擎的结果格式
        'buy_cnt':           bt.get('buy_cnt', bt.get('total_trades', 0) // 2),
        'sell_cnt':          bt.get('sell_cnt', bt.get('total_trades', 0) // 2),
        # 多维度验证：回测阈值 + ML 质量门禁（修复项2）
        'meets_threshold': _check_meets_threshold(
            bt, min_return, min_sharpe, max_dd, min_trades,
            train_acc=train_acc_from_meta, val_acc=val_acc
        ),
        # 验证段时间范围（用于报告展示）
        'val_period': f"{val_df.index.min().date()} ~ {val_df.index.max().date()}",
        'factor_path':       None,
        'model':             model,
        'meta':              val_meta,   # ← 验证集 meta，indicators 与 detail 索引对齐
        'detail':            bt.get('detail', pd.DataFrame()),
        'config':            config,
    }


# ══════════════════════════════════════════════════════════════════
#  对外接口（供 main.py 调用）
# ══════════════════════════════════════════════════════════════════

def _holdout_test(result: dict, test_df: pd.DataFrame, cfg: dict) -> dict:
    """
    在封存的 Hold-Out 测试段上对候选策略做一次独立回测。

    - ML 策略：直接用 result['model'] 推断信号，不重新训练
    - 规则策略：以 result['detail'] 末尾行作为预热上下文（不 fit），
                在 test_df 上用 no_internal_split=True 生成信号

    返回回测指标字典；失败时返回 {'success': False, 'message': ...}
    """
    if test_df is None or test_df.empty:
        return {'success': False, 'message': 'test_df 为空'}

    strategy_mod = result.get('_strategy_mod')   # 由 run_search 注入，见下方
    model        = result.get('model')
    meta         = result.get('meta', {})
    config       = result.get('config', cfg)

    if strategy_mod is None:
        return {'success': False, 'message': '未找到策略模块引用'}

    try:
        if model is not None and hasattr(strategy_mod, 'predict'):
            # ML 策略：直接推断
            sig = strategy_mod.predict(model, test_df, config, meta)
            sig = sig.reindex(test_df.index).fillna(0)
        else:
            # 规则策略：取 detail（val 回测明细）末尾若干行做预热，不参与 fit
            warmup_rows = max(int(config.get('lookback', 60)), 60)
            detail_df   = result.get('detail', pd.DataFrame())
            if not detail_df.empty:
                warmup_part = detail_df.iloc[-warmup_rows:][['Open', 'High', 'Low', 'Close', 'Volume']]
                warmup_part = warmup_part.reindex(columns=test_df.columns, fill_value=np.nan)
                combined    = pd.concat([warmup_part, test_df]).sort_index()
            else:
                combined = test_df
            hld_cfg = config.copy()
            hld_cfg['no_internal_split'] = True
            sig, _, _ = strategy_mod.run(combined, hld_cfg)
            sig = sig.reindex(test_df.index).fillna(0)
    except Exception as e:
        return {'success': False, 'message': f'信号生成失败: {e}'}

    try:
        backtest_engine = config.get('backtest_engine', 'vectorbt')
        if backtest_engine == 'vectorbt' and VECTORBT_AVAILABLE:
            bt = backtest_vbt(test_df, sig, config)
        else:
            bt = backtest(test_df, sig, config)
    except Exception as e:
        return {'success': False, 'message': f'回测失败: {e}'}

    return {
        'success':      True,
        'period':       f"{test_df.index.min().date()} ~ {test_df.index.max().date()}",
        'cum_return':   bt.get('cum_return', float('nan')),
        'sharpe_ratio': bt.get('sharpe_ratio', float('nan')),
        'max_drawdown': bt.get('max_drawdown', float('nan')),
        'win_rate':     bt.get('win_rate', 0),
        'total_trades': bt.get('total_trades', 0),
    }


def _select_best_with_holdout(
    sorted_results: list,
    test_df: pd.DataFrame,
    cfg: dict,
    strategy_mods: list,
    full_data: pd.DataFrame = None,
) -> dict:
    """
    在 run_search() 结束后，对候选策略依次执行 Hold-Out 评估 + Walk-Forward 检验，
    选出最高置信度的策略并打上验证标记。

    Parameters
    ----------
    sorted_results : 按 val 收益降序排列的 trial 结果列表
    test_df        : 封存的 Hold-Out 测试段（三段切割模式），降级时为空 DataFrame
    cfg            : 配置字典
    strategy_mods  : 策略模块列表（用于注入 _strategy_mod 引用）
    full_data      : 全量历史数据（含 test 段），供 WF 滚动窗口使用；
                     为 None 时降级为仅用 test_df（若也为空则 WF 跳过）

    优先级（从高到低）：
      1. Val 达标 + Hold-Out 达标 + WF 窗口胜率 >= wf_min  → validated: 'double'
      2. Val 达标 + Hold-Out 达标（WF 不足）               → validated: 'double_no_wf'
      3. Val 达标 + Hold-Out 不达标 + WF 通过              → validated: 'val_only'
      4. 以上均无，取 sorted_results[0]                    → validated: 'val_only'（降级）

    返回选出的 result dict（含 'validated'、'holdout'、'wf_summary' 字段）。
    """
    from validate_strategy import walk_forward_analysis

    if not sorted_results:
        return None

    min_return  = float(cfg.get('min_return', 0.10))
    min_sharpe  = float(cfg.get('min_sharpe_ratio', 1.0))
    max_dd      = float(cfg.get('max_drawdown', -0.15))
    min_trades  = int(cfg.get('min_total_trades', 5))
    wf_min      = float(cfg.get('wf_min_window_win_rate', 0.5))
    # Hold-Out 阈值宽松 30%（对缩短的测试段给予容忍）
    hld_min_ret = min_return * 0.7

    # 先评估前 N 名候选，避免全量评估耗时
    MAX_CANDIDATES = min(10, len(sorted_results))
    candidates = [r for r in sorted_results if r.get('meets_threshold')][:MAX_CANDIDATES]
    if not candidates:
        candidates = sorted_results[:MAX_CANDIDATES]

    # 为每个候选注入策略模块引用
    name_to_mod = {m.NAME: m for m in strategy_mods}
    for r in candidates:
        r['_strategy_mod'] = name_to_mod.get(r['strategy_name'])

    logger.info("开始Hold-Out验证", extra={"candidate_count": len(candidates)})

    best_double   = None
    best_val_only = None

    for r in candidates:
        name = r['strategy_name']

        # ── Hold-Out 回测 ──
        if not test_df.empty:
            holdout = _holdout_test(r, test_df, cfg)
        else:
            holdout = {'success': False, 'message': '无封存数据（降级模式）'}

        holdout_ok = (
            holdout.get('success', False) and
            holdout.get('cum_return', float('-inf')) >= hld_min_ret
        )
        holdout_str = (f"收益={holdout['cum_return']:.2%}" if holdout.get('success')
                       else holdout.get('message', '失败'))
        logger.info("Hold-Out验证候选", extra={
            "strategy_name": name,
            "holdout_ok": holdout_ok,
            "holdout_str": holdout_str,
        })

        # ── Walk-Forward ──
        # WF 必须在全量历史数据上跑，才能覆盖足够多的滚动窗口。
        # full_data 优先；降级模式（test_df 为空）时 full_data 应由调用方传入；
        # 两者都为空才真正跳过。
        wf_result    = None
        wf_ok        = False
        strat_mod    = r.get('_strategy_mod')
        wf_data      = full_data if (full_data is not None and not full_data.empty) else test_df
        if strat_mod is not None:
            try:
                if wf_data is not None and not wf_data.empty:
                    wf_result = walk_forward_analysis(
                        wf_data, strat_mod, r.get('config', cfg),
                        train_months=12, test_months=3, step_months=3
                    )
                    if wf_result.get('success'):
                        wwin = wf_result['summary'].get('window_win_rate', 0)
                        wf_ok = wwin >= wf_min
                        logger.debug("WF窗口验证", extra={
                            "strategy_name": name,
                            "window_win_rate": f"{wwin:.0%}",
                            "wf_ok": wf_ok
                        })
                    else:
                        logger.debug("WF跳过", extra={
                            "strategy_name": name,
                            "reason": wf_result.get('message', '失败')
                        })
                else:
                    logger.debug("WF跳过", extra={"strategy_name": name, "reason": "全量数据未提供"})
            except Exception as e:
                logger.warning("WF异常", extra={"strategy_name": name, "error": str(e)})
        else:
            logger.debug("WF跳过", extra={"strategy_name": name, "reason": "未找到策略模块"})

        r['holdout']    = holdout
        r['wf_result']  = wf_result
        r['wf_summary'] = wf_result['summary'] if (wf_result and wf_result.get('success')) else {}

        if holdout_ok:
            r['validated'] = 'double' if wf_ok else 'double_no_wf'
        else:
            r['validated'] = 'val_only'

        # 选出最高优先级候选
        if r['validated'] == 'double' and best_double is None:
            best_double = r
        elif r['validated'] == 'double_no_wf' and best_double is None:
            best_double = r   # double_no_wf 也优于 val_only
        elif r['validated'] == 'val_only' and best_val_only is None:
            best_val_only = r

        # 找到最高优先级就不必继续
        if best_double is not None and best_double['validated'] == 'double':
            break

    chosen = best_double if best_double is not None else best_val_only
    if chosen is None:
        # 兜底：取排名第一的候选，打降级标记
        chosen = sorted_results[0]
        chosen['validated'] = 'val_only'
        chosen['holdout']   = {'success': False, 'message': '所有候选均未通过'}
        chosen['wf_summary'] = {}

    badge = {'double': '🏅 双验证通过', 'double_no_wf': '🥈 双验证（WF不足）',
             'val_only': '⚠️  仅验证集达标'}.get(chosen.get('validated', ''), '❓')
    logger.info("选定策略", extra={
        "strategy_name": chosen['strategy_name'],
        "badge": badge,
        "val_cum_return": f"{chosen['cum_return']:.2%}",
        "sharpe_ratio": chosen.get('sharpe_ratio', float('nan')),
        "holdout_success": chosen.get('holdout', {}).get('success', False),
        "holdout_cum_return": f"{chosen.get('holdout', {}).get('cum_return', 0):.2%}",
        "holdout_period": chosen.get('holdout', {}).get('period', '?'),
    })

    return chosen


def test_factor(data: pd.DataFrame):
    """
    兼容旧接口：使用默认配置跑所有策略一次，返回最佳结果。
    返回 (data, factor_path, total_return)
    """
    config = load_config()
    strategy_mods = _discover_strategies()
    best = None

    for mod in strategy_mods:
        result = run_trial(mod, data, config)
        if result is None:
            continue
        if best is None or result['cum_return'] > best['cum_return']:
            best = result

    if best is None:
        return data, None, 0.0

    _log_result(best)
    return data, best['factor_path'], best['cum_return']


def _log_result(r: dict) -> None:
    """记录策略试验结果到日志。"""
    extra = {
        "strategy_name": r['strategy_name'],
        "params": r['params'],
        "train_rows": r['train_rows'],
        "val_rows": r['val_rows'],
        "cum_return": f"{r['cum_return']:.2%}",
        "annualized_return": f"{r['annualized_return']:.2%}",
        "sharpe_ratio": r.get('sharpe_ratio', float('nan')),
        "buy_cnt": r['buy_cnt'],
        "sell_cnt": r['sell_cnt'],
        "factor_saved": bool(r['factor_path']),
    }
    _val_acc = r.get('val_acc', float('nan'))
    _train_acc = r.get('train_acc', float('nan'))
    _roc_auc = r.get('roc_auc', float('nan'))
    if not np.isnan(_val_acc):
        extra["val_acc"] = f"{_val_acc:.2%}"
        extra["train_acc"] = f"{_train_acc:.2%}"
        if not np.isnan(_train_acc):
            extra["overfit_gap"] = f"{_train_acc - _val_acc:.2%}"
        if not np.isnan(_roc_auc):
            extra["roc_auc"] = f"{_roc_auc:.4f}"
    logger.info("策略试验结果", extra=extra)


# ══════════════════════════════════════════════════════════════════
#  共享超参采样（analyze_factor.__main__ 和 main.step2_train 共用）
# ══════════════════════════════════════════════════════════════════

def _sample_hyperparams(rng: np.random.Generator, base_cfg: dict) -> dict:
    """
    基于 base_cfg 生成一组随机超参，返回新的 trial_cfg dict。
    所有策略的可调超参统一在此处维护，避免 main.py 与本文件重复。
    """
    max_days  = int(base_cfg.get('max_test_days', 20))
    base_days = int(base_cfg.get('test_days', 5))
    try_days  = int(rng.integers(base_days, max_days + 1))

    trial_cfg = base_cfg.copy()
    trial_cfg['test_days']       = try_days
    # 通用 ML 超参
    trial_cfg['ridge_alpha']     = float(rng.choice([0.01, 0.1, 1.0, 10.0, 100.0]))
    trial_cfg['rf_n_estimators'] = int(rng.choice([50, 100, 200]))
    trial_cfg['rf_max_depth']    = int(rng.choice([3, 5, 10, 20])) if rng.random() > 0.3 else None
    # MA 策略
    trial_cfg['ma_fast']         = int(rng.integers(3, 10))
    trial_cfg['ma_slow']         = int(rng.integers(15, 60))
    # RSI 策略
    trial_cfg['rsi_period']      = int(rng.integers(7, 21))
    trial_cfg['rsi_oversold']    = float(rng.integers(20, 35))
    trial_cfg['rsi_overbought']  = float(rng.integers(65, 80))
    # 回撤止损
    trial_cfg['drawdown_pct']    = round(float(rng.choice([0.01, 0.015, 0.02, 0.03, 0.05])), 3)
    # KDJ 超参
    trial_cfg['kdj_period']      = int(rng.integers(5, 14))
    trial_cfg['kdj_oversold']    = float(rng.integers(15, 30))
    trial_cfg['kdj_overbought']  = float(rng.integers(70, 90))
    # 量能指标移动平均超参
    trial_cfg['obv_ma_period']   = int(rng.integers(10, 40))
    trial_cfg['pvt_ma_period']   = int(rng.integers(10, 40))
    # 斐波那契回撤窗口超参（rsi_obv / rsi_pvt / kdj_obv / kdj_pvt）
    trial_cfg['fib_period']      = int(rng.integers(20, 120))
    return trial_cfg


# ══════════════════════════════════════════════════════════════════
#  核心搜索入口（供 __main__ 和 main.step2_train 共同调用）
# ══════════════════════════════════════════════════════════════════

def run_search(
    data: pd.DataFrame,
    cfg: Optional[dict] = None,
    on_result=None,
    strategy_type: str = None,
) -> tuple:
    """
    对所有已发现策略执行随机超参搜索。

    Parameters
    ----------
    data      : 历史日线 DataFrame（全量，含 test 段）
    cfg       : 配置字典；为 None 时从 config.yaml 自动加载
    on_result : 可选回调 fn(result) —— 每次找到满足阈值的结果时触发

    Returns
    -------
    best_result    : 所有 trial 中累计收益最高的结果 dict（可能为 None）
    sorted_results : 所有 trial 结果按累计收益降序排列的列表
    test_df        : 封存的 Hold-Out 测试段（由 _select_best_with_holdout 使用）
    """
    if cfg is None:
        cfg = load_config()

    base_cfg  = cfg.copy()
    min_ret   = float(cfg.get('min_return', 0.03))
    max_tries = int(cfg.get('max_tries', 300))

    # ── 三段切割：在搜索开始前确定 test_start，封存最近 test_months 个月 ──
    test_months = int(cfg.get('test_months', 6))
    df_sorted = data.copy().sort_index()
    if not pd.api.types.is_datetime64_any_dtype(df_sorted.index):
        df_sorted.index = pd.to_datetime(df_sorted.index)

    data_end    = df_sorted.index.max()
    test_start  = data_end - pd.DateOffset(months=test_months)
    test_df     = df_sorted.loc[df_sorted.index >= test_start]
    search_data = df_sorted.loc[df_sorted.index < test_start]   # 搜索只见这部分

    # 数据量保护：若剩余可搜索数据不足 lookback_months + 6 个月，降级为两段切割
    min_search_months = int(cfg.get('lookback_months', 3)) + 6
    actual_months = (search_data.index.max() - search_data.index.min()).days // 30 if not search_data.empty else 0
    if search_data.empty or actual_months < min_search_months:
        logger.warning("可搜索数据不足，降级为两段切割", extra={
            "actual_months": actual_months,
            "min_search_months": min_search_months
        })
        search_data = df_sorted
        test_start  = None
        test_df     = pd.DataFrame()
    else:
        logger.info("三段切割完成", extra={
            "search_end": str(test_start.date()),
            "holdout_start": str(test_start.date()),
            "holdout_end": str(data_end.date()),
            "holdout_records": len(test_df)
        })

    strategy_mods = _discover_strategies(strategy_type=strategy_type, cfg=base_cfg)
    if not strategy_mods:
        logger.critical("未发现任何策略模块，请检查 strategies/ 目录")
        _save_config(base_cfg)
        return None, [], test_df

    logger.info("策略搜索开始", extra={
        "strategy_count": len(strategy_mods),
        "strategy_names": [m.NAME for m in strategy_mods],
        "max_trials_per_strategy": max_tries,
        "min_return_threshold": f"{min_ret:.2%}"
    })

    all_results = []
    found_any   = False

    for mod in strategy_mods:
        logger.info("开始搜索策略模块", extra={"strategy_module": mod.NAME})

        best_of_strategy = None
        mod_seed_base    = abs(hash(mod.NAME)) % (2**31)

        for trial in range(1, max_tries + 1):
            rng       = np.random.default_rng(seed=mod_seed_base + trial)
            trial_cfg = _sample_hyperparams(rng, cfg)

            logger.debug("试验超参", extra={
                "trial": trial,
                "max_trials": max_tries,
                "params": {
                    "days": trial_cfg['test_days'],
                    "rsi": f"({trial_cfg['rsi_period']},{trial_cfg['rsi_oversold']:.0f}/{trial_cfg['rsi_overbought']:.0f})",
                    "kdj": f"({trial_cfg['kdj_period']},{trial_cfg['kdj_oversold']:.0f}/{trial_cfg['kdj_overbought']:.0f})",
                    "obv_ma": trial_cfg['obv_ma_period'],
                    "pvt_ma": trial_cfg['pvt_ma_period'],
                    "fib": trial_cfg['fib_period'],
                    "dd": f"{trial_cfg['drawdown_pct']:.1%}",
                    "ma": f"({trial_cfg['ma_fast']}/{trial_cfg['ma_slow']})",
                }
            })

            result = run_trial(mod, search_data.copy(), trial_cfg,
                               trial_num=trial, test_start=test_start)
            if result is None:
                logger.debug("试验跳过", extra={"trial": trial, "reason": "数据不足或早停"})
                continue

            logger.debug("试验结果", extra={
                "trial": trial,
                "strategy_name": result['strategy_name'],
                "cum_return": f"{result['cum_return']:.2%}"
            })

            if best_of_strategy is None or result['cum_return'] > best_of_strategy['cum_return']:
                best_of_strategy = result

            all_results.append(result)

            if result['meets_threshold']:
                logger.info("策略满足阈值", extra={
                    "strategy_name": result['strategy_name'],
                    "cum_return": f"{result['cum_return']:.2%}"
                })
                _log_result(result)
                if on_result is not None:
                    try:
                        on_result(result)
                    except Exception as e:
                        logger.warning("on_result回调异常", extra={"error": str(e)})
                found_any = True
                # 不退出，继续训练所有参数组合

        if best_of_strategy:
            logger.info("策略模块搜索完成", extra={
                "strategy_module": mod.NAME,
                "best_cum_return": f"{best_of_strategy['cum_return']:.2%}"
            })

    # ── 恢复原始配置 ──
    _save_config(base_cfg)

    # ── 排行榜 ──
    sorted_results = sorted(all_results, key=lambda r: r['cum_return'], reverse=True)
    logger.info("策略搜索完成，排行榜Top10", extra={
        "total_trials": len(all_results),
        "min_return_threshold": f"{min_ret:.2%}",
    })
    for rank, r in enumerate(sorted_results[:10], 1):
        logger.info("排行榜", extra={
            "rank": rank,
            "strategy_name": r['strategy_name'],
            "meets_threshold": r['meets_threshold'],
            "cum_return": f"{r['cum_return']:>7.2%}",
            "sharpe_ratio": r.get('sharpe_ratio', float('nan')),
            "params": r['params']
        })

    if not found_any:
        logger.warning("所有策略均未达到阈值", extra={
            "min_return_threshold": f"{min_ret:.2%}",
            "best_strategy": sorted_results[0]['strategy_name'] if sorted_results else None,
            "best_cum_return": f"{sorted_results[0]['cum_return']:.2%}" if sorted_results else None,
        })
    else:
        hit_cnt = sum(1 for r in sorted_results if r['meets_threshold'])
        logger.info("搜索完成", extra={
            "hit_count": hit_cnt,
            "total_trials": len(all_results)
        })

    # 优先返回满足阈值且收益最高的结果；无则取全局最佳
    threshold_results = [r for r in sorted_results if r['meets_threshold']]
    best_for_predict  = threshold_results[0] if threshold_results else (sorted_results[0] if sorted_results else None)
    return best_for_predict, sorted_results, test_df


# ══════════════════════════════════════════════════════════════════
#  主程序：直接调用 run_search
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    _cfg = load_config()

    # 自动发现数据文件
    _hist_dir  = Path(__file__).parent / 'data' / 'historical'
    _csv_files = list(_hist_dir.glob('*.csv'))
    if not _csv_files:
        logger.critical("data/historical/ 下没有 CSV 文件，请先下载数据")
        raise SystemExit(1)
    _data_file = max(_csv_files, key=lambda p: p.stat().st_mtime)
    logger.info("使用数据文件", extra={"data_file": str(_data_file)})
    _raw_data = pd.read_csv(_data_file, index_col=0, parse_dates=True)

    _best, _sorted, _test_df = run_search(_raw_data, _cfg)

    # ── Hold-Out + Walk-Forward 选优 ──
    _strategy_mods = _discover_strategies()
    if _sorted:
        _best = _select_best_with_holdout(
            _sorted, _test_df, _cfg, _strategy_mods, full_data=_raw_data
        )


    # ── 统一保存一个带编号的因子文件（复用 main._save_factor 避免重复逻辑）──
    if _best is not None:
        try:
            from main import _save_factor, _next_factor_run_id
            from data.factor_registry import FactorRegistry, _get_training_type
            _factors_dir = Path(__file__).parent / 'data' / 'factors'
            _factor_path = _save_factor(_best, _factors_dir)
            logger.info("因子已保存", extra={
                "factor_path": Path(_factor_path).name,
                "strategy_name": _best['strategy_name'],
                "cum_return": f"{_best['cum_return']:.2%}",
                "sharpe_ratio": _best.get('sharpe_ratio', float('nan')),
                "validated": _best.get('validated', '?')
            })
            try:
                registry = FactorRegistry()
                _run_id = int(Path(_factor_path).stem.split('_')[1])
                _strategy = _best.get('meta', {}).get('name', 'unknown')
                _ticker = _best.get('config', {}).get('ticker')
                _ttype = _get_training_type(_strategy)
                registry.register(
                    factor_id=_run_id,
                    filename=Path(_factor_path).name,
                    subdir=None,
                    strategy_name=_strategy,
                    ticker=_ticker,
                    training_type=_ttype,
                    sharpe_ratio=_best.get('sharpe_ratio', 0.0),
                    cum_return=_best.get('cum_return', 0.0),
                    max_drawdown=_best.get('max_drawdown', 0.0),
                    total_trades=_best.get('total_trades', 0),
                )
            except Exception as _re:
                logger.warning("因子注册失败（非阻塞）", extra={"error": str(_re)})
        except Exception as _e:
            logger.warning("因子保存失败", extra={"error": str(_e)})


