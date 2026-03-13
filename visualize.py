import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yaml
from pathlib import Path


# Try a sensible set of sans-serif fonts that include CJK glyphs on macOS/linux
plt.rcParams['font.sans-serif'] = [
    'PingFang HK', 'PingFang SC', 'AppleGothic', 'Arial Unicode MS', 'SimHei', 'DejaVu Sans'
]
plt.rcParams['axes.unicode_minus'] = False


# Default config used when reading config.yaml fails or isn't present
DEFAULT_CONFIG = {}


def plot_trades(data):
    if data is None or data.empty:
        print("没有数据可视化")
        return None

    # 读取配置（相对于脚本位置）
    config_path = Path(__file__).parent / 'config.yaml'
    config = DEFAULT_CONFIG.copy()
    try:
        with open(config_path) as f:
            loaded = yaml.safe_load(f) or {}
            if isinstance(loaded, dict):
                config.update(loaded)
    except Exception:
        # 使用默认配置
        pass

    title = config.get('title', '股票交易信号')
    ticker = config.get('ticker', 'UNKNOWN')
    lookback_months = int(config.get('lookback_months', 3))

    # 验证必要列
    if 'Close' not in data.columns:
        raise ValueError("输入数据缺少 'Close' 列，无法绘图")

    if 'signal' not in data.columns:
        print("警告: 输入数据缺少 'signal' 列，绘图仍会显示价格曲线")

    # 确保索引为日期时间
    if not pd.api.types.is_datetime64_any_dtype(data.index):
        try:
            data.index = pd.to_datetime(data.index)
        except Exception:
            raise ValueError("数据索引无法解析为日期时间，无法绘图")

    # 计算策略净值曲线（从 1.0 起始）
    if 'strategy' in data.columns:
        equity = (1 + data['strategy'].fillna(0)).cumprod()
    else:
        equity = pd.Series(1.0, index=data.index)

    plt.figure(figsize=(14, 9))

    # 上图: 价格和买卖点
    ax1 = plt.subplot(2, 1, 1)
    ax1.plot(data.index, data['Close'], label='价格', color='black', alpha=0.7)

    # 优先使用显式的 trade 列（1=buy, -1=sell），如果不存在再回退到 signal
    if 'trade' in data.columns:
        buy_signals = data[data['trade'] == 1]
        sell_signals = data[data['trade'] == -1]
    elif 'signal' in data.columns:
        buy_signals = data[data['signal'] == 1]
        # 不再把所有 signal==0 都当作卖出，仅用于回退兼容
        sell_signals = pd.DataFrame(columns=data.columns)
    else:
        buy_signals = pd.DataFrame(columns=data.columns)
        sell_signals = pd.DataFrame(columns=data.columns)

    if not buy_signals.empty:
        ax1.scatter(buy_signals.index, buy_signals['Close'],
                    marker='^', color='g', s=80, label='买入')
    if not sell_signals.empty:
        ax1.scatter(sell_signals.index, sell_signals['Close'],
                    marker='v', color='r', s=80, label='卖出')

    ax1.set_title(title)
    ax1.set_ylabel('价格')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    # 下图: 策略净值
    ax2 = plt.subplot(2, 1, 2, sharex=ax1)
    ax2.plot(equity.index, equity.values, label='策略净值', color='blue')
    # 如果存在 portfolio_value，也绘制为参考（标准化显示）
    if 'portfolio_value' in data.columns:
        # 归一化 portfolio_value 到起始 1.0 的净值曲线用于对比
        pv = data['portfolio_value'].dropna()
        if not pv.empty:
            pv_norm = pv / pv.iloc[0]
            ax2.plot(pv_norm.index, pv_norm.values, label='组合净值(实际)', color='orange', linestyle='--')
    ax2.set_ylabel('净值')
    ax2.set_xlabel('日期')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    # 保存图片到 data/plots
    # 合并两个子图的图例到图像顶部，便于阅读
    fig = plt.gcf()
    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2

    # 去重（保持出现顺序），例如避免重复的 '价格' 或 '策略净值' 标签
    seen = set()
    unique_handles = []
    unique_labels = []
    for h, l in zip(all_handles, all_labels):
        if l not in seen:
            unique_handles.append(h)
            unique_labels.append(l)
            seen.add(l)

    if unique_handles:
        # 在图像顶部居中显示合并后的图例
        fig.legend(unique_handles, unique_labels,
                   loc='upper center', ncol=max(1, len(unique_labels)),
                   bbox_to_anchor=(0.5, 1.02), frameon=False, fontsize='medium')

    out_dir = Path(__file__).parent / 'data' / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"{ticker}_trade_signals_{lookback_months}m.png"
    # 给顶部图例留出空间
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(out_path, bbox_inches='tight')
    print(f"交易信号图已保存至: {out_path}")
    plt.close()
    return str(out_path)


def plot_strategy_result(detail: pd.DataFrame, meta: dict, config: dict) -> str:
    """
    为单次满足条件的策略结果绘制多联图并保存。
    子图布局自动根据 meta['indicators'] 中的指标动态生成：
      - 子图1 (必有): 收盘价 + 买卖点 + Fib 支撑/阻力线（若有）+ MA 均线（若有）
      - 子图2 (RSI 系):  RSI + 超买/超卖参考线
      - 子图2 (KDJ 系):  K/D/J + 超买/超卖参考线
      - 子图3 (OBV 系):  OBV + OBV_MA
      - 子图3 (PVT 系):  PVT + PVT_MA
      - 子图4 (持仓):   position 0/1 填充
      - 子图5 (净值):   策略累计净值曲线
    """
    strategy_name   = meta.get('name', 'unknown')
    params          = meta.get('params', {})
    indicators      = meta.get('indicators', {})
    ticker          = config.get('ticker', 'UNKNOWN')
    lookback_months = int(config.get('lookback_months', 3))

    df = detail.copy()
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        df.index = pd.to_datetime(df.index)

    equity   = (1 + df['strategy'].fillna(0)).cumprod()
    buy_pts  = df[df['trade'] ==  1] if 'trade' in df.columns else pd.DataFrame()
    sell_pts = df[df['trade'] == -1] if 'trade' in df.columns else pd.DataFrame()
    param_str = '  '.join(f"{k}={v}" for k, v in params.items())

    # ── 决定有哪些指标子图 ──────────────────────────────────────
    has_rsi       = 'rsi'         in indicators
    has_kdj       = 'K'           in indicators
    has_obv       = 'obv'         in indicators
    has_pvt       = 'pvt'         in indicators
    has_ma        = 'ma_fast'     in indicators
    has_fib       = 'fib_0618'    in indicators
    has_pred_ret  = 'pred_return' in indicators   # Ridge / LinearReg
    has_pred_prob = 'pred_proba'  in indicators   # RandomForest
    has_pos       = 'position'    in df.columns

    # 子图列表：[(label, height_ratio), ...]
    panel_defs = [('price', 3)]
    if has_rsi:
        panel_defs.append(('rsi', 1.5))
    if has_kdj:
        panel_defs.append(('kdj', 1.5))
    if has_obv:
        panel_defs.append(('obv', 1.2))
    if has_pvt:
        panel_defs.append(('pvt', 1.2))
    if has_pred_ret:
        panel_defs.append(('pred_ret', 1.2))
    if has_pred_prob:
        panel_defs.append(('pred_prob', 1.2))
    if has_pos:
        panel_defs.append(('pos', 0.6))
    panel_defs.append(('equity', 2))

    n_panels     = len(panel_defs)
    height_total = sum(r for _, r in panel_defs)
    fig_height   = max(10, 2.5 * n_panels)

    fig, axes = plt.subplots(
        n_panels, 1,
        figsize=(15, fig_height),
        sharex=True,
        gridspec_kw={'height_ratios': [r for _, r in panel_defs]},
    )
    if n_panels == 1:
        axes = [axes]
    fig.suptitle(f"{ticker}  {strategy_name}\n{param_str}", fontsize=11, y=1.005)

    panel_axes = {label: axes[i] for i, (label, _) in enumerate(panel_defs)}

    # ── 辅助：取指标 Series，reindex 后前向/后向填充，消除 NaN 空洞 ──
    def _ind(key: str) -> pd.Series:
        s = indicators[key]
        if not isinstance(s, pd.Series):
            return pd.Series(dtype=float)
        return s.reindex(df.index).ffill().bfill()

    def _vlines(ax):
        for idx in buy_pts.index:
            ax.axvline(idx, color='#00aa00', alpha=0.25, linewidth=0.7)
        for idx in sell_pts.index:
            ax.axvline(idx, color='#dd0000', alpha=0.25, linewidth=0.7)

    # ── 子图1: 价格 + 买卖点 + Fib + MA ──────────────────────────
    ax_p = panel_axes['price']
    ax_p.plot(df.index, df['Close'], color='black', linewidth=1, label='收盘价')

    # MA 均线叠加在价格图
    if has_ma:
        ma_f = _ind('ma_fast')
        ma_s = _ind('ma_slow')
        ax_p.plot(df.index, ma_f, color='#e67e22', linewidth=1, label=f"MA快({params.get('ma_fast','')})")
        ax_p.plot(df.index, ma_s, color='#8e44ad', linewidth=1, label=f"MA慢({params.get('ma_slow','')})")

    # Fib 支撑/阻力线叠加在价格图
    if has_fib:
        f618 = _ind('fib_0618')
        f382 = _ind('fib_0382')
        ax_p.plot(df.index, f618, color='#27ae60', linewidth=0.9,
                  linestyle='--', alpha=0.75, label='Fib 0.618 支撑')
        ax_p.plot(df.index, f382, color='#e74c3c', linewidth=0.9,
                  linestyle='--', alpha=0.75, label='Fib 0.382 阻力')

    # 买卖点标记
    if not buy_pts.empty:
        ax_p.scatter(buy_pts.index, buy_pts['Close'],
                     marker='^', color='#00aa00', s=90, zorder=5, label='买入')
    if not sell_pts.empty:
        ax_p.scatter(sell_pts.index, sell_pts['Close'],
                     marker='v', color='#dd0000', s=90, zorder=5, label='卖出')
    ax_p.set_ylabel('价格')
    ax_p.legend(loc='upper left', fontsize=8, ncol=2)
    ax_p.grid(True, linestyle='--', alpha=0.4)

    # ── RSI 子图 ──────────────────────────────────────────────────
    if has_rsi:
        ax_r  = panel_axes['rsi']
        rsi_s = _ind('rsi')
        ob    = float(indicators.get('rsi_overbought', 70))
        os_   = float(indicators.get('rsi_oversold',   30))
        ax_r.plot(df.index, rsi_s, color='#2980b9', linewidth=1, label='RSI')
        ax_r.axhline(ob,  color='#e74c3c', linestyle='--', linewidth=0.8, label=f'超买 {ob:.0f}')
        ax_r.axhline(os_, color='#27ae60', linestyle='--', linewidth=0.8, label=f'超卖 {os_:.0f}')
        ax_r.axhline(50,  color='gray',    linestyle=':',  linewidth=0.6)
        mask = rsi_s.notna()
        ax_r.fill_between(df.index, os_, rsi_s, where=mask & (rsi_s <= os_), alpha=0.20, color='#27ae60')
        ax_r.fill_between(df.index, ob,  rsi_s, where=mask & (rsi_s >= ob),  alpha=0.20, color='#e74c3c')
        _vlines(ax_r)
        ax_r.set_ylabel('RSI')
        ax_r.set_ylim(0, 100)
        ax_r.legend(loc='upper left', fontsize=8, ncol=3)
        ax_r.grid(True, linestyle='--', alpha=0.4)

    # ── KDJ 子图 ──────────────────────────────────────────────────
    if has_kdj:
        ax_k = panel_axes['kdj']
        K_s  = _ind('K'); D_s = _ind('D'); J_s = _ind('J')
        ob   = float(indicators.get('kdj_overbought', 80))
        os_  = float(indicators.get('kdj_oversold',   20))
        ax_k.plot(df.index, K_s, color='#2980b9', linewidth=1,   label='K')
        ax_k.plot(df.index, D_s, color='#e67e22', linewidth=1,   label='D')
        ax_k.plot(df.index, J_s, color='#8e44ad', linewidth=0.8, label='J', alpha=0.8)
        ax_k.axhline(ob,  color='#e74c3c', linestyle='--', linewidth=0.8, label=f'超买 {ob:.0f}')
        ax_k.axhline(os_, color='#27ae60', linestyle='--', linewidth=0.8, label=f'超卖 {os_:.0f}')
        ax_k.axhline(50,  color='gray',    linestyle=':',  linewidth=0.6)
        _vlines(ax_k)
        ax_k.set_ylabel('KDJ')
        ax_k.legend(loc='upper left', fontsize=8, ncol=4)
        ax_k.grid(True, linestyle='--', alpha=0.4)

    # ── OBV 子图 ──────────────────────────────────────────────────
    if has_obv:
        ax_o   = panel_axes['obv']
        obv_s  = _ind('obv'); obvm_s = _ind('obv_ma')
        mask   = obv_s.notna() & obvm_s.notna()
        ax_o.plot(df.index, obv_s,  color='#16a085', linewidth=1,   label='OBV')
        ax_o.plot(df.index, obvm_s, color='#e67e22', linewidth=1,   label='OBV MA', linestyle='--')
        ax_o.fill_between(df.index, obv_s, obvm_s, where=mask & (obv_s >= obvm_s), alpha=0.15, color='#27ae60')
        ax_o.fill_between(df.index, obv_s, obvm_s, where=mask & (obv_s <  obvm_s), alpha=0.15, color='#e74c3c')
        _vlines(ax_o)
        ax_o.set_ylabel('OBV')
        ax_o.legend(loc='upper left', fontsize=8, ncol=2)
        ax_o.grid(True, linestyle='--', alpha=0.4)

    # ── PVT 子图 ──────────────────────────────────────────────────
    if has_pvt:
        ax_v   = panel_axes['pvt']
        pvt_s  = _ind('pvt'); pvtm_s = _ind('pvt_ma')
        mask   = pvt_s.notna() & pvtm_s.notna()
        ax_v.plot(df.index, pvt_s,  color='#8e44ad', linewidth=1,   label='PVT')
        ax_v.plot(df.index, pvtm_s, color='#e67e22', linewidth=1,   label='PVT MA', linestyle='--')
        ax_v.fill_between(df.index, pvt_s, pvtm_s, where=mask & (pvt_s >= pvtm_s), alpha=0.15, color='#27ae60')
        ax_v.fill_between(df.index, pvt_s, pvtm_s, where=mask & (pvt_s <  pvtm_s), alpha=0.15, color='#e74c3c')
        _vlines(ax_v)
        ax_v.set_ylabel('PVT')
        ax_v.legend(loc='upper left', fontsize=8, ncol=2)
        ax_v.grid(True, linestyle='--', alpha=0.4)

    # ── 预测收益率子图（Ridge / LinearReg） ──────────────────────
    if has_pred_ret:
        ax_pr = panel_axes['pred_ret']
        pr_s  = _ind('pred_return')
        mask  = pr_s.notna()
        ax_pr.plot(df.index, pr_s, color='#2980b9', linewidth=1, label='预测收益率')
        ax_pr.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax_pr.fill_between(df.index, 0, pr_s, where=mask & (pr_s >= 0), alpha=0.2, color='#27ae60')
        ax_pr.fill_between(df.index, 0, pr_s, where=mask & (pr_s <  0), alpha=0.2, color='#e74c3c')
        _vlines(ax_pr)
        ax_pr.set_ylabel('预测收益')
        ax_pr.legend(loc='upper left', fontsize=8)
        ax_pr.grid(True, linestyle='--', alpha=0.4)

    # ── 上涨概率子图（RandomForest） ────────────────────────────
    if has_pred_prob:
        ax_pp = panel_axes['pred_prob']
        pp_s  = _ind('pred_proba')
        mask  = pp_s.notna()
        ax_pp.plot(df.index, pp_s, color='#8e44ad', linewidth=1, label='上涨概率')
        ax_pp.axhline(0.5, color='gray', linestyle='--', linewidth=0.8, label='50%')
        ax_pp.fill_between(df.index, 0.5, pp_s, where=mask & (pp_s >= 0.5), alpha=0.2, color='#27ae60')
        ax_pp.fill_between(df.index, 0.5, pp_s, where=mask & (pp_s <  0.5), alpha=0.2, color='#e74c3c')
        _vlines(ax_pp)
        ax_pp.set_ylim(0, 1)
        ax_pp.set_ylabel('上涨概率')
        ax_pp.legend(loc='upper left', fontsize=8, ncol=2)
        ax_pp.grid(True, linestyle='--', alpha=0.4)

    # ── 持仓子图 ──────────────────────────────────────────────────
    if has_pos:
        ax_pos = panel_axes['pos']
        ax_pos.fill_between(df.index, df['position'].astype(float),
                            step='post', alpha=0.55, color='steelblue', label='持仓')
        ax_pos.set_ylim(-0.1, 1.5)
        ax_pos.set_yticks([0, 1])
        ax_pos.set_yticklabels(['空', '持'])
        ax_pos.set_ylabel('持仓')
        ax_pos.legend(loc='upper left', fontsize=8)
        ax_pos.grid(True, linestyle='--', alpha=0.4)

    # ── 净值子图 ──────────────────────────────────────────────────
    ax_eq = panel_axes['equity']
    ax_eq.plot(equity.index, equity.values, color='royalblue', linewidth=1.5, label='策略净值')
    ax_eq.axhline(1.0, color='gray', linestyle='--', linewidth=0.8)
    ax_eq.fill_between(equity.index, 1.0, equity.values,
                       where=equity.values >= 1.0, alpha=0.22, color='green', label='盈利')
    ax_eq.fill_between(equity.index, 1.0, equity.values,
                       where=equity.values <  1.0, alpha=0.22, color='red',   label='亏损')
    cum_ret = equity.iloc[-1] - 1
    ax_eq.set_title(f"累计收益: {cum_ret:.2%}", fontsize=10, pad=3)
    ax_eq.set_ylabel('净值')
    ax_eq.set_xlabel('日期')
    ax_eq.legend(loc='upper left', fontsize=8, ncol=3)
    ax_eq.grid(True, linestyle='--', alpha=0.4)

    plt.tight_layout()

    out_dir  = Path(__file__).parent / 'data' / 'plots'
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_name = strategy_name.replace('/', '_')
    out_path  = out_dir / f"{ticker}_{safe_name}_{lookback_months}m.png"
    fig.savefig(out_path, bbox_inches='tight', dpi=130)
    plt.close(fig)
    print(f"  📊 策略图已保存: {out_path}")
    return str(out_path)


if __name__ == "__main__":
    # 示例使用
    data_file = Path(__file__).parent / "data" / "historical" / "0700.HK_1mo.csv"
    if data_file.exists():
        data = pd.read_csv(data_file, index_col=0, parse_dates=True)
        data['signal'] = np.where(data['Close'].pct_change() > 0, 1, 0)
        data['strategy'] = data['signal'].shift(1) * data['Close'].pct_change()
        plot_trades(data)
    else:
        print("示例数据文件不存在，跳过示例运行")
