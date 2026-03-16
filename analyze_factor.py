import importlib
import pkgutil
from typing import Optional
import pandas as pd
import numpy as np
import yaml
import shutil
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
from pathlib import Path
from visualize import plot_strategy_result

# Vectorbt 引擎（可选）
try:
    from backtest_vectorbt import backtest_vectorbt as backtest_vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False



# ══════════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════════

def _load_config() -> dict:
    # 加载主配置
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}

    # 加载密钥配置（如果存在）
    keys_path = Path(__file__).parent / 'keys.yaml'
    if keys_path.exists():
        with open(keys_path, encoding='utf-8') as f:
            keys = yaml.safe_load(f) or {}
            config.update(keys)

    return config


def _save_config(cfg: dict) -> None:
    config_path = Path(__file__).parent / 'config.yaml'
    tmp = config_path.with_suffix('.yaml.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        yaml.dump(cfg, f, allow_unicode=True)
    shutil.move(str(tmp), str(config_path))


def _discover_strategies() -> list:
    """自动发现 strategies/ 包下的所有策略模块，返回模块列表。"""
    # 只使用新添加的两个策略
    new_strategies = ['bollinger_rsi_trend', 'macd_rsi_trend']
    pkg_path = Path(__file__).parent / 'strategies'
    modules = []
    for name in new_strategies:
        try:
            mod = importlib.import_module(f'strategies.{name}')
            if hasattr(mod, 'run') and hasattr(mod, 'NAME'):
                modules.append(mod)
        except ImportError:
            pass
    return modules


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
    invest_fraction = float(config.get('invest_fraction', 1.0))
    lookback_months = int(config.get('lookback_months', 3))

    bt = data.copy()
    bt['signal']    = signal.reindex(bt.index).fillna(0).astype(int)
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
            n = int(cash * invest_fraction // price)
            if n > 0:
                cash -= n * price; shares += n; position = 1; trade = 1
        elif desired == 0 and position == 1:
            cash += shares * price; shares = 0; position = 0; trade = -1

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
        ann = (1 + cum_return) ** (12.0 / max(1, lookback_months)) - 1
    except Exception:
        ann = float('nan')

    # ── 夏普率（年化，无风险利率取 0） ──
    strat_rets = bt['strategy'].fillna(0)
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

def run_trial(strategy_mod, data: pd.DataFrame, config: dict) -> Optional[dict]:
    """
    对一个策略模块执行完整的 train / val / backtest 流程。
    返回 result dict，包含所有指标；若失败返回 None。
    """
    lookback_months = int(config.get('lookback_months', 3))
    train_years     = int(config.get('train_years', 2))
    min_return      = float(config.get('min_return', 0.03))
    ticker          = config.get('ticker', 'UNKNOWN')

    # 基础校验
    if data is None or data.empty or 'Close' not in data.columns:
        return None

    df = data.copy().sort_index()
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            return None

    df['returns'] = df['Close'].pct_change()
    df = df.dropna(subset=['returns'])
    if df.empty:
        return None

    # ── 时间切分 ──
    end_date    = df.index.max()
    val_start   = end_date - pd.DateOffset(months=lookback_months)
    train_start = val_start - pd.DateOffset(years=train_years)

    train_df = df.loc[(df.index >= train_start) & (df.index < val_start)]
    val_df   = df.loc[df.index >= val_start]

    if train_df.empty:
        train_df = df.loc[df.index < val_start]
    if train_df.empty or val_df.empty:
        return None

    # ── 训练集：运行策略，拟合模型 ──
    try:
        train_signal, model, meta = strategy_mod.run(train_df, config)
    except Exception as e:
        print(f"    [{strategy_mod.NAME}] 训练异常: {e}")
        return None

    # ── 验证集：用同一模型在 val_df 上生成信号 ──
    try:
        val_signal, _, val_meta = strategy_mod.run(val_df, config)
    except Exception as e:
        print(f"    [{meta['name']}] 验证集推理异常: {e}")
        return None
    # val_meta 的 indicators 与 val_df 索引对齐，用于绘图；
    # 训练集 meta 的 params/name 信息合并进来保持完整
    val_meta['params']    = meta.get('params', val_meta.get('params', {}))
    val_meta['feat_cols'] = meta.get('feat_cols', [])

    # ── 回归验证指标（仅对有模型的策略计算） ──
    r2 = mae = direction_acc = float('nan')
    feat_cols = meta.get('feat_cols', [])
    if model is not None and feat_cols:
        try:
            vdf = val_df.copy()
            vdf['returns'] = vdf['Close'].pct_change()
            for i in range(1, int(config.get('test_days', 5)) + 1):
                vdf[f'ret_{i}'] = vdf['returns'].shift(i)
            vdf = vdf.dropna()

            if not vdf.empty and all(c in vdf.columns for c in feat_cols):
                X_val  = vdf[feat_cols]
                y_val  = vdf['returns']
                y_pred = model.predict(X_val)
                r2             = float(r2_score(y_val, y_pred))
                mae            = float(mean_absolute_error(y_val, y_pred))
                direction_acc  = float(np.mean(np.sign(y_pred) == np.sign(y_val.values)))
        except Exception:
            pass

    # ── 回测（验证集） ──
    # 根据配置选择回测引擎
    backtest_engine = config.get('backtest_engine', 'native')
    if backtest_engine == 'vectorbt' and VECTORBT_AVAILABLE:
        bt = backtest_vbt(val_df, val_signal, config)
    else:
        bt = backtest(val_df, val_signal, config)

    # 因子保存由外层 (main.py / run_search) 统一处理，此处不再写磁盘
    return {
        'strategy_name':      val_meta['name'],
        'params':            val_meta.get('params', {}),
        'train_rows':        len(train_df),
        'val_rows':          len(val_df),
        'r2':                r2,
        'mae':               mae,
        'direction_acc':     direction_acc,
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
        'meets_threshold':   bt['cum_return'] > min_return,
        'factor_path':       None,
        'model':             model,
        'meta':              val_meta,   # ← 验证集 meta，indicators 与 detail 索引对齐
        'detail':            bt.get('detail', pd.DataFrame()),
        'config':            config,
    }


# ══════════════════════════════════════════════════════════════════
#  对外接口（供 main.py 调用）
# ══════════════════════════════════════════════════════════════════

def test_factor(data: pd.DataFrame):
    """
    兼容旧接口：使用默认配置跑所有策略一次，返回最佳结果。
    返回 (data, factor_path, total_return)
    """
    config = _load_config()
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

    _print_result(best)
    return data, best['factor_path'], best['cum_return']


def _print_result(r: dict) -> None:
    print(f"\n  策略: {r['strategy_name']}  参数: {r['params']}")
    print(f"  训练集: {r['train_rows']} 条  验证集: {r['val_rows']} 条")
    if not np.isnan(r['r2']):
        print(f"  R²={r['r2']:.4f}  MAE={r['mae']:.6f}  方向准确率={r['direction_acc']:.2%}")
    sharpe_str = f"{r['sharpe_ratio']:.4f}" if not np.isnan(r.get('sharpe_ratio', float('nan'))) else "N/A"
    print(f"  累计收益: {r['cum_return']:.2%}  年化(估算): {r['annualized_return']:.2%}  夏普率: {sharpe_str}")
    print(f"  买入: {r['buy_cnt']} 次  卖出: {r['sell_cnt']} 次")
    if r['factor_path']:
        print(f"  ✅ 因子已保存: {r['factor_path']}")
    else:
        print(f"  ❌ 未达到保存阈值")


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


def _print_trial_header(trial: int, max_tries: int, cfg: dict) -> None:
    """打印单次 trial 的超参摘要行（不含换行，调用方补充结尾）。"""
    print(
        f"  [{trial:>3}/{max_tries}]"
        f"  days={cfg['test_days']}"
        f"  rsi=({cfg['rsi_period']},{cfg['rsi_oversold']:.0f}/{cfg['rsi_overbought']:.0f})"
        f"  kdj=({cfg['kdj_period']},{cfg['kdj_oversold']:.0f}/{cfg['kdj_overbought']:.0f})"
        f"  obv_ma={cfg['obv_ma_period']}  pvt_ma={cfg['pvt_ma_period']}"
        f"  fib={cfg['fib_period']}"
        f"  dd={cfg['drawdown_pct']:.1%}"
        f"  ma=({cfg['ma_fast']}/{cfg['ma_slow']})",
        end='  ',
    )


# ══════════════════════════════════════════════════════════════════
#  核心搜索入口（供 __main__ 和 main.step2_train 共同调用）
# ══════════════════════════════════════════════════════════════════

def run_search(
    data: pd.DataFrame,
    cfg: Optional[dict] = None,
    on_result=None,
) -> tuple:
    """
    对所有已发现策略执行随机超参搜索。

    Parameters
    ----------
    data      : 历史日线 DataFrame
    cfg       : 配置字典；为 None 时从 config.yaml 自动加载
    on_result : 可选回调 fn(result) —— 每次找到满足阈值的结果时触发
                （例如 main.py 中用于绘图）

    Returns
    -------
    best_result    : 所有 trial 中累计收益最高的结果 dict（可能为 None）
    sorted_results : 所有 trial 结果按累计收益降序排列的列表
    """
    if cfg is None:
        cfg = _load_config()

    base_cfg  = cfg.copy()
    min_ret   = float(cfg.get('min_return', 0.03))
    max_tries = int(cfg.get('max_tries', 300))

    strategy_mods = _discover_strategies()
    if not strategy_mods:
        print("  ❌ 未发现任何策略模块，请检查 strategies/ 目录")
        _save_config(base_cfg)
        return None, []

    print(f"  发现 {len(strategy_mods)} 个策略: {[m.NAME for m in strategy_mods]}")
    print(f"  搜索上限: 每策略最多 {max_tries} 次  目标收益 > {min_ret:.2%}")

    all_results = []
    found_any   = False

    for mod in strategy_mods:
        print(f"\n  {'━'*56}")
        print(f"  策略模块: {mod.NAME}")
        print(f"  {'━'*56}")

        best_of_strategy = None
        mod_seed_base    = abs(hash(mod.NAME)) % (2**31)

        for trial in range(1, max_tries + 1):
            rng       = np.random.default_rng(seed=mod_seed_base + trial)
            trial_cfg = _sample_hyperparams(rng, cfg)

            _print_trial_header(trial, max_tries, trial_cfg)

            result = run_trial(mod, data.copy(), trial_cfg)
            if result is None:
                print("跳过（数据不足）")
                continue

            print(f"收益={result['cum_return']:.2%}", end='')

            if best_of_strategy is None or result['cum_return'] > best_of_strategy['cum_return']:
                best_of_strategy = result

            all_results.append(result)

            if result['meets_threshold']:
                print("  ✅ 满足条件！")
                _print_result(result)
                if on_result is not None:
                    try:
                        on_result(result)
                    except Exception as e:
                        print(f"  ⚠️  on_result 回调异常: {e}")
                found_any = True
                # 不退出，继续训练所有参数组合
            else:
                print()

        if best_of_strategy:
            print(f"\n  [{mod.NAME}] 本策略最佳: 收益={best_of_strategy['cum_return']:.2%}")

    # ── 恢复原始配置 ──
    _save_config(base_cfg)

    # ── 排行榜 ──
    sorted_results = sorted(all_results, key=lambda r: r['cum_return'], reverse=True)
    print(f"\n  {'═'*56}")
    print("  所有试验排行榜 Top 10（按验证集累计收益）")
    print(f"  {'═'*56}")
    for rank, r in enumerate(sorted_results[:10], 1):
        flag = "✅" if r['meets_threshold'] else "  "
        sharpe_str = f"{r['sharpe_ratio']:.2f}" if not np.isnan(r.get('sharpe_ratio', float('nan'))) else " N/A"
        print(f"  {rank:>2}. {flag} {r['strategy_name']:<22}"
              f"  收益={r['cum_return']:>7.2%}  夏普={sharpe_str:>6}  参数={r['params']}")

    if not found_any:
        print(f"\n  ❌ 所有策略均未达到 {min_ret:.2%} 阈值")
        if sorted_results:
            b = sorted_results[0]
            print(f"     历史最佳: {b['strategy_name']} 收益={b['cum_return']:.2%}  参数={b['params']}")
    else:
        hit_cnt = sum(1 for r in sorted_results if r['meets_threshold'])
        print(f"\n  🎉 共有 {hit_cnt} 个策略满足阈值，最终因子由调用方统一保存")

    # 优先返回满足阈值且收益最高的结果；无则取全局最佳
    threshold_results = [r for r in sorted_results if r['meets_threshold']]
    best_for_predict  = threshold_results[0] if threshold_results else (sorted_results[0] if sorted_results else None)
    return best_for_predict, sorted_results


# ══════════════════════════════════════════════════════════════════
#  主程序：直接调用 run_search
# ══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import datetime as _dt

    _cfg = _load_config()

    # 自动发现数据文件
    _hist_dir  = Path(__file__).parent / 'data' / 'historical'
    _csv_files = list(_hist_dir.glob('*.csv'))
    if not _csv_files:
        print("❌ data/historical/ 下没有 CSV 文件，请先下载数据")
        raise SystemExit(1)
    _data_file = max(_csv_files, key=lambda p: p.stat().st_mtime)
    print(f"使用数据文件: {_data_file}")
    _raw_data = pd.read_csv(_data_file, index_col=0, parse_dates=True)

    def _on_result_plot(result):
        plot_strategy_result(result['detail'], result['meta'], result['config'])

    _best, _ = run_search(_raw_data, _cfg, on_result=_on_result_plot)

    # ── 绘制最优解结果图（确保最终选定的最优解一定有图） ──
    if _best is not None:
        print(f"\n  📊 绘制最优解结果图 ({_best['strategy_name']}  收益={_best['cum_return']:.2%})…")
        try:
            plot_strategy_result(_best['detail'], _best['meta'], _best['config'])
        except Exception as e:
            print(f"  ⚠️  绘图失败: {e}")

    # ── 统一保存一个带编号的因子文件 ──
    if _best is not None:
        _factors_dir = Path(__file__).parent / 'data' / 'factors'
        _factors_dir.mkdir(parents=True, exist_ok=True)
        _existing = list(_factors_dir.glob('factor_*.pkl'))
        _ids = []
        for _p in _existing:
            try:
                _ids.append(int(_p.stem.split('_')[1]))
            except (IndexError, ValueError):
                pass
        _run_id   = max(_ids) + 1 if _ids else 1
        _out_path = _factors_dir / f"factor_{_run_id:04d}.pkl"
        joblib.dump(
            {
                'model':              _best['model'],
                'meta':               _best['meta'],
                'config':             _best.get('config', {}),
                'run_id':             _run_id,
                'cum_return':         _best['cum_return'],
                'annualized_return':  _best.get('annualized_return', float('nan')),
                'sharpe_ratio':      _best.get('sharpe_ratio', float('nan')),
                'max_drawdown':       _best.get('max_drawdown', float('nan')),
                'volatility':         _best.get('volatility', float('nan')),
                'win_rate':           _best.get('win_rate', 0),
                'profit_loss_ratio':  _best.get('profit_loss_ratio', 0),
                'calmar_ratio':       _best.get('calmar_ratio', float('nan')),
                'sortino_ratio':      _best.get('sortino_ratio', float('nan')),
                'total_trades':       _best.get('total_trades', 0),
                'saved_at':           _dt.datetime.now().isoformat(),
            },
            _out_path,
        )
        print(f"\n  💾 因子已保存: {_out_path.name}"
              f"  (策略={_best['strategy_name']}  收益={_best['cum_return']:.2%}"
              f"  夏普={_best.get('sharpe_ratio', float('nan')):.4f})")


