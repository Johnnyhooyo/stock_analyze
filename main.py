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
from fetch_data import download_stock_data
from analyze_factor import (
    _load_config, _discover_strategies,
    run_search, backtest, VECTORBT_AVAILABLE,
)
try:
    from backtest_vectorbt import backtest_vectorbt as backtest_vbt
except ImportError:
    backtest_vbt = None
from position_manager import PositionManager, load_position_from_config
from feishu_notify import send_full_report_to_feishu
from sentiment_analysis import analyze_stock_sentiment, get_sentiment_signal
from validate_strategy import generate_test_report, out_of_sample_test, walk_forward_analysis
from visualize import plot_strategy_result, plot_yearly_trades


# ══════════════════════════════════════════════════════════════════
#  辅助函数
# ══════════════════════════════════════════════════════════════════

def _last_trading_day(ref: datetime | None = None) -> datetime:
    """
    返回 ref（默认今天）之前最近一个港股交易日（仅排除周末，不处理节假日）。
    结果为当天 16:10（收市时间），表示该交易日数据应已完整。
    """
    d = (ref or datetime.now()).date()
    # 往回找最近一个工作日（不含今天）
    d -= timedelta(days=1)
    while d.weekday() >= 5:   # 5=Saturday, 6=Sunday
        d -= timedelta(days=1)
    return datetime(d.year, d.month, d.day, 16, 10)


def _hist_data_is_stale(hist_file_path: str) -> bool:
    """
    判断历史数据文件是否过期：
    - 18点前：当天数据不可用，最新应该是昨天
    - 18点后：当天数据可能可用，最新应该是今天

    注意：不能用文件修改时间判断，因为周末运行程序会更新mtime但数据没变化。
    """
    import pandas as pd
    from datetime import date, datetime, time

    try:
        df = pd.read_csv(hist_file_path, parse_dates=['date'])
        latest_date = df['date'].max()

        now = datetime.now()
        # 港股收盘时间16:10，18点后当天数据应该可用
        cutoff_time = time(18, 0)

        if now.time() < cutoff_time:
            # 18点前：当天数据不可用，需要上一个交易日（昨天或上周五）
            target_date = date.today() - timedelta(days=1)
            while target_date.weekday() >= 5:  # 周末
                target_date -= timedelta(days=1)
        else:
            # 18点后：当天数据可用，需要今天的数据
            target_date = date.today()
            # 但如果是周末，需要找上一个交易日
            if target_date.weekday() >= 5:
                target_date -= timedelta(days=1)
                while target_date.weekday() >= 5:
                    target_date -= timedelta(days=1)

        return latest_date.date() < target_date
    except Exception:
        # 出错时默认过期，需要更新
        return True


# ── factors 目录统一命名：factor_{run_id:04d}.pkl ──────────────────

def _next_factor_run_id(factors_dir: Path) -> int:
    """
    扫描 factors_dir 下所有 factor_XXXX.pkl，返回下一个可用编号（从 1 开始）。
    """
    existing = list(factors_dir.glob('factor_*.pkl'))
    if not existing:
        return 1
    ids = []
    for p in existing:
        try:
            ids.append(int(p.stem.split('_')[1]))
        except (IndexError, ValueError):
            pass
    return max(ids) + 1 if ids else 1


def _save_factor(result: dict, factors_dir: Path) -> str:
    """
    将搜索结果统一保存为 factor_{run_id:04d}.pkl，返回保存路径字符串。
    run_id 自动递增，每次调用只写一个文件。
    """
    factors_dir.mkdir(parents=True, exist_ok=True)
    run_id    = _next_factor_run_id(factors_dir)
    save_path = factors_dir / f"factor_{run_id:04d}.pkl"
    joblib.dump(
        {
            'model':              result['model'],
            'meta':               result['meta'],
            'config':             result.get('config', {}),
            'run_id':             run_id,
            'cum_return':         result['cum_return'],
            'annualized_return':  result.get('annualized_return', float('nan')),
            'sharpe_ratio':       result.get('sharpe_ratio', float('nan')),
            'max_drawdown':       result.get('max_drawdown', float('nan')),
            'volatility':         result.get('volatility', float('nan')),
            'win_rate':           result.get('win_rate', 0),
            'profit_loss_ratio':  result.get('profit_loss_ratio', 0),
            'calmar_ratio':       result.get('calmar_ratio', float('nan')),
            'sortino_ratio':      result.get('sortino_ratio', float('nan')),
            'total_trades':       result.get('total_trades', 0),
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


def _load_config_full() -> dict:
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


# ══════════════════════════════════════════════════════════════════
#  步骤 1 : 数据就绪检查
# ══════════════════════════════════════════════════════════════════

def step1_ensure_data(sources_override=None):
    """
    确保历史日线数据已就绪。

    Returns
    -------
    hist_data : pd.DataFrame   历史日线（Close 等标准列）
    hist_path : str            历史文件路径
    """
    print("\n" + "="*60)
    print("  步骤 1 / 3 : 数据就绪检查")
    print("="*60)

    cfg = _load_config_full()
    ticker = cfg.get('ticker', '0700.hk')

    # ── 1a. 历史日线 ────────────────────────────────────────────
    hist_dir = Path(__file__).parent / 'data' / 'historical'
    hist_files = sorted(hist_dir.glob('*.csv'), key=lambda p: p.stat().st_mtime, reverse=True)

    if hist_files and not _hist_data_is_stale(str(hist_files[0])):
        hist_path = str(hist_files[0])
        hist_data = pd.read_csv(hist_path, index_col=0, parse_dates=True)
        latest_date = hist_data.index.max()
        print(f"  ✅ 历史日线数据已是最新: {hist_path}  ({len(hist_data)} 条, 最新 {latest_date:%Y-%m-%d})")
    else:
        if hist_files:
            # 显示数据内容的最新日期，而不是文件修改时间
            from datetime import date, datetime, time, timedelta
            df = pd.read_csv(hist_files[0], parse_dates=['date'])
            latest_date = df['date'].max()

            # 根据时间判断应该需要的数据日期
            now = datetime.now()
            if now.time() < time(18, 0):
                # 18点前：需要上一个交易日（昨天或上周五）
                target = date.today() - timedelta(days=1)
                while target.weekday() >= 5:
                    target -= timedelta(days=1)
                hint = f"昨天 {target}"
            else:
                # 18点后：需要今天的数据（如果今天不是周末）
                target = date.today()
                if target.weekday() >= 5:
                    target -= timedelta(days=1)
                    while target.weekday() >= 5:
                        target -= timedelta(days=1)
                hint = f"今天/昨天 {target}"

            print(f"  ⚠️  历史日线数据已过期（最新 {latest_date:%Y-%m-%d}  < 需要 {hint}），正在更新…")
        else:
            print("  ⚠️  本地无历史日线数据，正在下载…")
        hist_data, hist_path = download_stock_data(sources_override=sources_override)
        if hist_data is None or hist_data.empty:
            # 下载失败时降级使用旧文件（如果存在）
            if hist_files:
                hist_path = str(hist_files[0])
                hist_data = pd.read_csv(hist_path, index_col=0, parse_dates=True)
                print(f"  ⚠️  更新失败，继续使用旧数据: {hist_path}  ({len(hist_data)} 条)")
            else:
                print("  ❌ 历史数据下载失败，流程终止")
                sys.exit(1)
        else:
            print(f"  ✅ 历史数据已更新: {hist_path}  ({len(hist_data)} 条)")

    return hist_data, hist_path


# ══════════════════════════════════════════════════════════════════
#  步骤 2 : 多策略 × 100 次超参搜索
# ══════════════════════════════════════════════════════════════════

def step2_train(hist_data: pd.DataFrame):
    """
    对每个策略执行随机超参搜索。
    搜索结束后将最佳结果统一保存为一个 factor_{run_id:04d}.pkl。
    返回 (factor_path, best_result, sorted_results)。
    """
    print("\n" + "="*60)
    print("  步骤 2 / 3 : 多策略超参搜索（每策略最多 max_tries 次）")
    print("="*60)

    cfg = _load_config()

    def _on_result(result):
        try:
            plot_strategy_result(result['detail'], result['meta'], result['config'])
        except Exception as e:
            print(f"  ⚠️  绘图失败: {e}")

    # on_result 仅在搜索中途满足阈值时触发（用于实时预览）
    # 最终最优解的图统一在 run_search 返回后绘制，确保一定有图输出
    best_result, sorted_results = run_search(hist_data, cfg, on_result=_on_result)

    # ── 绘制最优解结果图 ─────────────────────────────────────────
    if best_result is not None:
        print(f"\n  📊 绘制最优解结果图 ({best_result['strategy_name']}  收益={best_result['cum_return']:.2%})…")
        try:
            plot_strategy_result(best_result['detail'], best_result['meta'], best_result['config'])
        except Exception as e:
            print(f"  ⚠️  绘图失败: {e}")

    # ── 统一保存一个因子文件 ────────────────────────────────────
    factor_path   = None
    factors_dir   = Path(__file__).parent / 'data' / 'factors'

    if best_result is not None:
        try:
            factor_path = _save_factor(best_result, factors_dir)
            best_result['factor_path'] = factor_path
            sharpe_str = f"{best_result['sharpe_ratio']:.4f}" if not np.isnan(best_result.get('sharpe_ratio', float('nan'))) else "N/A"
            print(f"\n  💾 因子已保存: {Path(factor_path).name}"
                  f"  (策略={best_result['strategy_name']}"
                  f"  收益={best_result['cum_return']:.2%}"
                  f"  夏普={sharpe_str})")
        except Exception as e:
            print(f"  ⚠️  因子保存失败: {e}")

    return factor_path, best_result, sorted_results


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
        print(f"  ⚠️  加载因子失败: {e}")
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


# ══════════════════════════════════════════════════════════════════
#  步骤 3a : 未来 n 个交易日预测（日线）
# ══════════════════════════════════════════════════════════════════

def predict_next_days(data: pd.DataFrame, factor_path: str, n_days: int = 3) -> str:
    """
    预测未来 n_days 个交易日收盘价。
    - ML 策略：用模型滚动预测每日收益率
    - 规则策略：重跑 run() 取最新信号方向，结合历史均值振幅估算涨跌幅
    返回 markdown 格式的报告内容
    """
    print(f"\n  {'─'*50}")
    print(f"  日线预测：未来 {n_days} 个交易日")
    print(f"  {'─'*50}")

    artifact     = _resolve_artifact(factor_path)
    if not artifact:
        print("  ❌ 无法加载因子文件，跳过日线预测")
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

    print(f"  策略: {strategy}  类型: {'ML' if is_ml else '规则'}  参数: {params}")
    print(f"  夏普率: {sharpe_str}  累计收益: {artifact.get('cum_return', float('nan')):.2%}")

    df = data.copy().sort_index()
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)

    last_date   = df.index.max()
    last_close  = float(df['Close'].iloc[-1])
    returns     = df['Close'].pct_change().dropna()
    daily_vol   = float(returns.tail(60).std())
    avg_abs_ret = float(returns.tail(20).abs().mean())

    print(f"  最后交易日: {last_date.date()}  收盘价: {last_close:.2f}")
    print(f"  近60日波动率: {daily_vol:.2%}  近20日平均振幅: {avg_abs_ret:.2%}")

    # ── 情感分析 ─────────────────────────────────────────────────────
    print(f"  正在分析市场情感...")
    sentiment_result = analyze_stock_sentiment(config.get('ticker', '0700.HK'))
    sentiment_signal = get_sentiment_signal(sentiment_result)
    sentiment_emoji = "🟢" if sentiment_result['sentiment'] == "positive" else "🔴" if sentiment_result['sentiment'] == "negative" else "⚪"
    print(f"  情感分析: {sentiment_emoji} {sentiment_result['sentiment']} (分数: {sentiment_result['polarity']:.3f})")
    print(f"  新闻统计: 正面 {sentiment_result['positive_count']} | 负面 {sentiment_result['negative_count']} | 中性 {sentiment_result['neutral_count']}")

    # 未来交易日（跳过周末）
    future_dates = []
    d = last_date
    while len(future_dates) < n_days:
        d += pd.Timedelta(days=1)
        if d.weekday() < 5:
            future_dates.append(d)

    # ── ML 策略：准备滚动特征 ──
    latest_feats = None
    if is_ml:
        test_days = len(feat_cols)
        df['returns'] = df['Close'].pct_change()
        for i in range(1, test_days + 1):
            df[f'ret_{i}'] = df['returns'].shift(i)
        df = df.dropna()
        latest_feats = list(df[feat_cols].iloc[-1].values)
        print(f"\n  模型: {type(model).__name__}  特征窗口: {test_days} 天")
    else:
        print(f"\n  规则信号驱动，方向=最新信号，振幅=历史均值")

    # ── 规则策略：用最新数据跑一次 run() 取当前信号 ──
    rule_direction = 1
    current_signal = "震荡"
    if not is_ml:
        if strategy_mod is not None:
            try:
                sig, _, _ = strategy_mod.run(data.copy(), config)
                rule_direction = _signal_to_direction(sig)
            except Exception as e:
                print(f"  ⚠️  规则策略信号生成失败: {e}")
        current_signal = "上涨" if rule_direction == 1 else "下跌"
        direction_str = "上涨 🟢" if rule_direction == 1 else "下跌 🔴"
        print(f"  当前信号: {direction_str}")

    print(f"\n  {'日期':<14} {'预测收盘价':>12} {'价格区间':>26} {'方向':>8} {'预测收益率':>12}")
    print(f"  {'-'*80}")

    # 存储预测结果用于生成 markdown
    predictions = []
    sim_close = last_close
    for fd in future_dates:
        if is_ml:
            test_days = len(feat_cols)
            X_pred    = np.array(latest_feats[-test_days:]).reshape(1, -1)
            pred_ret  = float(model.predict(X_pred)[0])
            direction = "上涨" if pred_ret > 0 else "下跌"
            latest_feats = [pred_ret] + latest_feats[:-1]
        else:
            pred_ret  = avg_abs_ret * (1 if rule_direction == 1 else -1)
            direction = "上涨" if rule_direction == 1 else "下跌"

        price_lo  = sim_close * (1 + pred_ret - daily_vol)
        price_hi  = sim_close * (1 + pred_ret + daily_vol)
        sim_close = sim_close * (1 + pred_ret)

        predictions.append({
            'date': fd.date(),
            'price': sim_close,
            'low': price_lo,
            'high': price_hi,
            'direction': direction,
            'return': pred_ret
        })

        direction_emoji = "🟢" if direction == "上涨" else "🔴"
        print(f"  {str(fd.date()):<14} {sim_close:>12.2f}"
              f"  [{price_lo:>8.2f}, {price_hi:>8.2f}]"
              f"  {direction_emoji:>8}  {pred_ret:>+11.2%}")

    print(f"\n  ⚠️  以上仅供参考，不构成投资建议。")

    # ── 获取完整指标 ─────────────────────────────────────────────
    ann_return = artifact.get('annualized_return', float('nan'))
    max_drawdown = artifact.get('max_drawdown', float('nan'))
    volatility = artifact.get('volatility', float('nan'))
    win_rate = artifact.get('win_rate', 0)
    profit_loss_ratio = artifact.get('profit_loss_ratio', 0)
    calmar_ratio = artifact.get('calmar_ratio', float('nan'))
    sortino_ratio = artifact.get('sortino_ratio', float('nan'))
    total_trades = artifact.get('total_trades', 0)

    # 格式化指标
    ann_return_str = f"{ann_return:.2%}" if not np.isnan(ann_return) else "N/A"
    max_dd_str = f"{max_drawdown:.2%}" if not np.isnan(max_drawdown) else "N/A"
    vol_str = f"{volatility:.2%}" if not np.isnan(volatility) else "N/A"
    calmar_str = f"{calmar_ratio:.4f}" if not np.isnan(calmar_ratio) else "N/A"
    sortino_str = f"{sortino_ratio:.4f}" if not np.isnan(sortino_ratio) else "N/A"
    win_rate_str = f"{win_rate:.2%}" if win_rate > 0 else "N/A"
    pl_ratio_str = f"{profit_loss_ratio:.2f}" if profit_loss_ratio > 0 else "N/A"

    # ── 提取 BS 点（买卖点）──────────────────────────────────────────
    bs_points = []
    backtest_engine = config.get('backtest_engine', 'native')
    if strategy_mod is not None and not is_ml:
        # 重新运行回测获取交易记录
        try:
            sig, _, _ = strategy_mod.run(data.copy(), config)
            # 根据配置选择回测引擎
            if backtest_engine == 'vectorbt' and backtest_vbt is not None:
                bt_result = backtest_vbt(data, sig, config)
                # Vectorbt 的交易记录需要从 portfolio 获取
                if 'portfolio' in bt_result:
                    portfolio = bt_result['portfolio']
                    # 尝试从交易记录中提取
                    try:
                        trades = portfolio.get_trades()
                        if trades is not None:
                            for t in trades:
                                bs_points.append({
                                    'date': str(t.entry_date.date()) if hasattr(t, 'entry_date') else 'N/A',
                                    'price': t.entry_price,
                                    'action': '买入',
                                    'pv': t.return_
                                })
                    except:
                        pass
            else:
                bt_result = backtest(data, sig, config)
                detail = bt_result.get('detail')
                if detail is not None and 'trade' in detail.columns:
                    for idx, row in detail.iterrows():
                        trade = row['trade']
                        if trade != 0:
                            bs_points.append({
                                'date': idx.strftime('%Y-%m-%d') if hasattr(idx, 'strftime') else str(idx)[:10],
                                'price': row['Close'],
                                'action': '买入' if trade == 1 else '卖出',
                                'pv': row['pv']
                            })
        except Exception as e:
            pass

    # ── 生成 Markdown 报告 ─────────────────────────────────────────
    md_content = f"""# 腾讯控股 (0700.hk) 股票分析报告

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

## 收益指标

| 指标 | 值 |
|------|-----|
| 累计收益率 | {artifact.get('cum_return', float('nan')):.2%} |
| 年化收益率 | {ann_return_str} |
| 夏普比率 | {sharpe_str} |
| 索提诺比率 | {sortino_str} |
| 卡玛比率 | {calmar_str} |

## 风险指标

| 指标 | 值 |
|------|-----|
| 最大回撤 | {max_dd_str} |
| 年化波动率 | {vol_str} |
| 近60日波动率 | {daily_vol:.2%} |

## 交易统计

| 指标 | 值 |
|------|-----|
| 总交易次数 | {total_trades} |
| 胜率 | {win_rate_str} |
| 盈亏比 | {pl_ratio_str} |
| 近20日平均振幅 | {avg_abs_ret:.2%} |

## 交易信号 (BS点)

| 日期 | 价格 | 操作 | 账户净值 |
|------|------|------|----------|
"""

    # 添加 BS 点
    if bs_points:
        for bs in bs_points:
            emoji = "🟢" if bs['action'] == "买入" else "🔴"
            md_content += f"| {bs['date']} | {bs['price']:.2f} | {emoji} {bs['action']} | {bs['pv']:.2f} |\n"
    else:
        md_content += f"| - | - | 无交易信号 | - |\n"

    md_content += f"""
## 预测结果

| 日期 | 预测收盘价 | 价格区间 | 方向 | 预测收益率 |
|------|-----------|----------|------|------------|
"""

    for p in predictions:
        direction_emoji = "🟢" if p['direction'] == "上涨" else "🔴"
        md_content += f"| {p['date']} | {p['price']:.2f} | [{p['low']:.2f}, {p['high']:.2f}] | {direction_emoji} {p['direction']} | {p['return']:+.2%} |\n"

    # ── 持仓管理与建议 ────────────────────────────────────────────────
    # 加载最新配置（确保获取最新的持仓设置）
    current_config = _load_config_full()
    pm = PositionManager()
    position = load_position_from_config(current_config)
    if position:
        # 更新当前价格
        position.current_price = last_close
        pm.position = position

        # 获取信号和预测收益率
        signal = rule_direction if not is_ml else (1 if predictions[0]['return'] > 0 else 0)
        predicted_return = predictions[0]['return'] if predictions else 0

        # 生成建议
        rec = pm.get_recommendation(signal, predicted_return)

        # 控制台输出
        print(f"\n  {'─'*50}")
        print(f"  持仓状态与建议")
        print(f"  {'─'*50}")
        print(f"  持股数量: {rec['shares']} 股")
        print(f"  平均成本: {rec['avg_cost']:.2f} 元")
        print(f"  当前价格: {rec['current_price']:.2f} 元")
        print(f"  盈亏金额: {rec['profit']:+.2f} 元 ({rec['profit_pct']:+.2f}%)")
        print(f"\n  交易建议: {rec['action']}")
        print(f"  原因: {rec['reason']}")
        print(f"  信号: {'持仓 🟢' if rec['signal'] == 1 else '空仓 🔴'}")
        print(f"  预测收益率: {rec['predicted_return']:+.2%}")

        # 发送到飞书（完整报告）
        feishu_webhook = current_config.get('feishu_webhook')
        if feishu_webhook:
            ticker = current_config.get('ticker', '0700.hk')
            signal_text = "上涨" if rule_direction == 1 else "下跌"

            # 构建完整报告数据
            report_data = {
                'ticker': ticker,
                'current_price': last_close,
                'last_date': str(last_date.date()) if hasattr(last_date, 'date') else str(last_date)[:10],
                'strategy': strategy,
                'params': params,
                'is_ml': is_ml,
                'signal': signal_text,
                'cum_return': artifact.get('cum_return', 0),
                'sharpe': sharpe,
                'annualized_return': artifact.get('annualized_return', 0),
                'max_drawdown': artifact.get('max_drawdown', 0),
                'volatility': artifact.get('volatility', 0),
                'total_trades': artifact.get('total_trades', 0),
                'win_rate': artifact.get('win_rate', 0),
                'calmar_ratio': artifact.get('calmar_ratio', 0),
                'avg_volatility': daily_vol,
                'predictions': predictions,
                'sentiment': sentiment_result,  # 情感分析结果
                'sentiment_signal': sentiment_signal,  # 情感信号：1=看涨, 0=看跌, -1=无信号
                'position': {
                    'shares': rec.get('shares', 0),
                    'avg_cost': rec.get('avg_cost', 0),
                    'current_price': rec.get('current_price', 0),
                    'profit': rec.get('profit', 0),
                    'profit_pct': rec.get('profit_pct', 0),
                },
                'recommendation': rec,
                'validation': {},  # 验证数据稍后填充
            }

            send_full_report_to_feishu(feishu_webhook, report_data)
            print(f"  📱 已发送到飞书群聊")

        md_content += f"""
## 持仓状态与建议

| 项目 | 值 |
|------|-----|
| 持股数量 | {rec['shares']} 股 |
| 平均成本 | {rec['avg_cost']:.2f} 元 |
| 当前价格 | {rec['current_price']:.2f} 元 |
| 持仓成本 | {rec['current_price'] * rec['shares']:.2f} 元 |
| 市值 | {rec['current_price'] * rec['shares']:.2f} 元 |
| 盈亏金额 | {rec['profit']:+.2f} 元 |
| 盈亏比例 | {rec['profit_pct']:+.2f}% |

### 交易建议

- **操作**: {rec['action']}
- **原因**: {rec['reason']}
- **信号**: {"持仓 🟢" if rec['signal'] == 1 else "空仓 🔴"}
- **预测收益率**: {rec['predicted_return']:+.2%}

"""

    # ── 添加验证报告内容 ────────────────────────────────────────────────
    # 运行样本外测试和Walk-Forward分析
    if artifact and artifact.get('strategy_mod'):
        strategy_mod = artifact['strategy_mod']
        params = artifact.get('meta', {}).get('params', {})
        config = artifact.get('config', {})

        # 样本外测试
        oos_result = out_of_sample_test(
            data, strategy_mod, params, config,
            train_months=12, test_months=3
        )

        # Walk-Forward分析
        wf_result = walk_forward_analysis(
            data, strategy_mod, config,
            train_months=12, test_months=3, step_months=3
        )

        # 添加验证内容到报告
        if oos_result.get('success'):
            oos = oos_result
            md_content += f"""
## 策略验证（样本外测试）

| 指标 | 值 |
|------|-----|
| 训练期 | {oos['train_period']} |
| 测试期 | {oos['test_period']} |
| 策略收益 | {oos['cum_return']:.2%} |
| 买入持有收益 | {oos['buy_hold_return']:.2%} |
| 超额收益 | {oos['excess_return']:+.2%} |
| 夏普比率 | {oos['sharpe_ratio']:.4f} |
| 最大回撤 | {oos['max_drawdown']:.2%} |
| 交易次数 | {oos['total_trades']} |
"""

        # 添加Walk-Forward分析
        if wf_result.get('success'):
            wf = wf_result
            md_content += f"""
## Walk-Forward 分析

| 指标 | 值 |
|------|-----|
| 总窗口数 | {wf.get('total_windows', 0)} |
| 盈利窗口数 | {wf.get('profitable_windows', 0)} |
| 窗口胜率 | {wf.get('win_rate', 0):.2%} |
| 平均收益 | {wf.get('avg_return', 0):.2%} |
| 平均夏普率 | {wf.get('avg_sharpe', 0):.4f} |
"""

        # 生成年度交易图
        yearly_plot_path = plot_yearly_trades(data, strategy_mod, config)
        if yearly_plot_path:
            # 添加图片到报告
            md_content += f"""
## 过去一年交易记录

![年度交易图]({yearly_plot_path})

"""

    md_content += f"""
## 风险提示

⚠️ 以上预测仅供参考，不构成投资建议。预测基于历史数据和算法模型，实际市场走势可能存在较大差异。

---
*本报告由自动分析系统生成*
"""

    # 保存 markdown 文件
    report_dir = Path(__file__).parent / 'data' / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    report_path.write_text(md_content, encoding='utf-8')
    print(f"\n  📄 Markdown 报告已保存: {report_path}")

    return md_content


# ══════════════════════════════════════════════════════════════════
#  步骤 3b : 明天按小时预测（分时外推）
# ══════════════════════════════════════════════════════════════════

def predict_next_day_hourly(
    hist_data: pd.DataFrame,
    kline_df: pd.DataFrame,
    factor_path: str,
) -> None:
    """
    对明天港股交易时段按小时预测价格走势。
    港股交易时段：09:30-12:00, 13:00-16:00（每小时一个点）
    - ML 策略：模型预测小时收益率（日线预测值 / √7）
    - 规则策略：最新信号方向 × 历史小时均值振幅
    """
    print(f"\n  {'─'*50}")
    print(f"  小时预测：明天各交易时段价格")
    print(f"  {'─'*50}")

    artifact     = _resolve_artifact(factor_path)
    if not artifact:
        print("  ❌ 无法加载因子文件，跳过小时预测")
        return

    model        = artifact.get('model')
    meta         = artifact.get('meta', {})
    config       = artifact.get('config', {})
    strategy_mod = artifact.get('strategy_mod')
    feat_cols    = meta.get('feat_cols', [])
    is_ml        = (model is not None and len(feat_cols) > 0)

    # ── 基准价：优先分时末价，其次日线收盘 ──
    base_price = None
    if kline_df is not None and not kline_df.empty and 'price' in kline_df.columns:
        valid = kline_df['price'].dropna()
        if not valid.empty:
            base_price = float(valid.iloc[-1])
            print(f"  基准价（今日分时末价）: {base_price:.2f}")
    if base_price is None:
        df_h = hist_data.copy().sort_index()
        if not pd.api.types.is_datetime64_any_dtype(df_h.index):
            df_h.index = pd.to_datetime(df_h.index)
        base_price = float(df_h['Close'].iloc[-1])
        print(f"  基准价（历史日线末收盘）: {base_price:.2f}")

    # ── 波动率 ──
    df_hist = hist_data.copy().sort_index()
    if not pd.api.types.is_datetime64_any_dtype(df_hist.index):
        df_hist.index = pd.to_datetime(df_hist.index)
    returns        = df_hist['Close'].pct_change().dropna()
    daily_vol      = float(returns.tail(60).std())
    hourly_vol     = daily_vol / np.sqrt(7)
    avg_abs_ret    = float(returns.tail(20).abs().mean())
    hourly_avg_abs = avg_abs_ret / np.sqrt(7)

    # ── ML 策略：准备特征 ──
    latest_feats = None
    if is_ml:
        test_days = len(feat_cols)
        df_hist['returns'] = df_hist['Close'].pct_change()
        for i in range(1, test_days + 1):
            df_hist[f'ret_{i}'] = df_hist['returns'].shift(i)
        df_hist = df_hist.dropna()
        latest_feats = list(df_hist[feat_cols].iloc[-1].values)
        print(f"\n  模型: {type(model).__name__}  日线特征窗口: {test_days} 天")
    else:
        print(f"\n  规则信号驱动，方向=最新信号，振幅=历史小时均值")

    # ── 规则策略：取最新信号 ──
    rule_direction = 1
    if not is_ml:
        if strategy_mod is not None:
            try:
                sig, _, _ = strategy_mod.run(hist_data.copy(), config)
                rule_direction = _signal_to_direction(sig)
            except Exception as e:
                print(f"  ⚠️  规则策略信号生成失败: {e}")
        direction_str = "上涨 🟢" if rule_direction == 1 else "下跌 🔴"
        print(f"  当前信号: {direction_str}")

    print(f"  每小时波动率估算: {hourly_vol:.2%}")

    # ── 港股明天交易时间点 ──
    tomorrow = (pd.Timestamp.today() + pd.Timedelta(days=1)).normalize()
    while tomorrow.weekday() >= 5:
        tomorrow += pd.Timedelta(days=1)

    hour_slots = [
        tomorrow.replace(hour=9,  minute=30),
        tomorrow.replace(hour=10, minute=30),
        tomorrow.replace(hour=11, minute=30),
        tomorrow.replace(hour=13, minute=0),
        tomorrow.replace(hour=14, minute=0),
        tomorrow.replace(hour=15, minute=0),
        tomorrow.replace(hour=16, minute=0),
    ]

    print(f"\n  {'时间':<20} {'预测价格':>12} {'价格区间':>26} {'方向':>8} {'预测变动率':>12}")
    print(f"  {'-'*82}")

    sim_price = base_price
    for slot in hour_slots:
        if is_ml:
            test_days = len(feat_cols)
            X_pred    = np.array(latest_feats[-test_days:]).reshape(1, -1)
            pred_ret  = float(model.predict(X_pred)[0]) / np.sqrt(7)
            direction = "上涨 🟢" if pred_ret > 0 else "下跌 🔴"
            latest_feats = [pred_ret * np.sqrt(7)] + latest_feats[:-1]
        else:
            pred_ret  = hourly_avg_abs * (1 if rule_direction == 1 else -1)
            direction = "上涨 🟢" if rule_direction == 1 else "下跌 🔴"

        price_lo  = sim_price * (1 + pred_ret - hourly_vol)
        price_hi  = sim_price * (1 + pred_ret + hourly_vol)
        sim_price = sim_price * (1 + pred_ret)

        print(f"  {str(slot):<20} {sim_price:>12.2f}"
              f"  [{price_lo:>8.2f}, {price_hi:>8.2f}]"
              f"  {direction:>8}  {pred_ret:>+11.2%}")

    print(f"\n  ⚠️  小时预测基于日线信号/模型外推，误差较大，仅供参考。")


# ══════════════════════════════════════════════════════════════════
#  主流程
# ══════════════════════════════════════════════════════════════════

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
    args = parser.parse_args()

    sources_override = None
    if args.sources:
        sources_override = [s.strip() for s in args.sources.split(',') if s.strip()]

    print("=" * 60)
    print("  腾讯股票分析流程  （自动化三步版）")
    print("=" * 60)

    # ── 步骤 1 ────────────────────────────────────────────────────
    hist_data, hist_path = step1_ensure_data(sources_override)

    # ── 步骤 2 ────────────────────────────────────────────────────
    factors_dir = Path(__file__).parent / 'data' / 'factors'
    factor_path = None

    if args.skip_train:
        factor_path = _latest_factor_path(factors_dir)
        if factor_path:
            print(f"\n  [跳过训练] 使用现有因子: {Path(factor_path).name}")
        else:
            print("\n  ⚠️  --skip-train 指定但 data/factors/ 中无因子文件，执行正常训练")
            args.skip_train = False

    if not args.skip_train:
        factor_path, _, _ = step2_train(hist_data)
        if factor_path is None:
            # 保存失败时兜底取最新已有文件
            factor_path = _latest_factor_path(factors_dir)
            if factor_path:
                print(f"  ℹ️  使用已有最新因子: {Path(factor_path).name}")

    # ── 步骤 3 ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  步骤 3 / 3 : 预测")
    print("="*60)

    if factor_path is None:
        print("  ❌ 没有可用的因子/模型，无法进行预测")
        return

    # 3a. 未来 n 天日线预测
    predict_next_days(hist_data, factor_path, n_days=args.n_days)

    # 3b. 生成策略验证报告（样本外测试 + Walk-Forward 分析）
    print(f"\n  {'─'*50}")
    print(f"  策略验证报告")
    print(f"  {'─'*50}")

    artifact = _resolve_artifact(factor_path)
    validation_data = {}
    if artifact and artifact.get('strategy_mod'):
        strategy_mod = artifact['strategy_mod']
        params = artifact.get('meta', {}).get('params', {})
        config = artifact.get('config', {})

        validation_md, report_path, validation_data = generate_test_report(
            hist_data, strategy_mod, params, config
        )
        print(f"  📊 验证报告已保存: {report_path.name}")
        print(f"\n{validation_md}")

        # 发送到飞书（带验证数据）
        feishu_webhook = config.get('feishu_webhook')
        if feishu_webhook:
            # 构建报告数据
            report_data = {
                'ticker': config.get('ticker', '0700.hk'),
                'current_price': float(hist_data['Close'].iloc[-1]),
                'last_date': str(hist_data.index.max().date()),
                'strategy': artifact.get('meta', {}).get('name', ''),
                'params': params,
                'is_ml': False,
                'signal': "震荡",
                'cum_return': artifact.get('cum_return', 0),
                'sharpe': artifact.get('sharpe_ratio', 0),
                'annualized_return': artifact.get('annualized_return', 0),
                'max_drawdown': artifact.get('max_drawdown', 0),
                'volatility': artifact.get('volatility', 0),
                'total_trades': artifact.get('total_trades', 0),
                'win_rate': artifact.get('win_rate', 0),
                'calmar_ratio': artifact.get('calmar_ratio', 0),
                'avg_volatility': 0,
                'predictions': [],
                'position': {},
                'recommendation': {},
                'validation': validation_data,
            }
            send_full_report_to_feishu(feishu_webhook, report_data)
            print(f"  📱 验证报告已发送到飞书群聊")
    else:
        print("  ⚠️ 无法加载策略模块，跳过验证报告")

    # 3c. 明天按小时预测（已禁用，专注日线预测）
    # predict_next_day_hourly(hist_data, kline_df, factor_path)

    print("\n" + "="*60)
    print("  🎉 分析流程完成！")
    print("="*60)


if __name__ == "__main__":
    main()

