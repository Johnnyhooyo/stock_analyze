"""
策略: RNN 深度学习趋势预测 (GRU/LSTM)
------------------------------------------------------
基于 OHLCV 序列数据的深度学习策略，使用 GRU/LSTM 捕捉时序模式。

特征工程（严格无前视）:
  - 价格衍生特征 (ret_1d, ret_5d, log_hl, log_co, volume_z)
  - 技术指标 (RSI, MACD, Bollinger Bands, EMA, ATR)

模型: GRU (默认) 或 LSTM
  - 2层网络: hidden=64 → hidden=32
  - Dropout=0.2
  - 分类阈值输出 0/1 信号

配置选项 (config.yaml → ml_strategies.rnn_trend):
  rnn_hidden_size: 64
  rnn_num_layers: 2
  rnn_dropout: 0.2
  rnn_window: 30
  rnn_label_period: 5
  rnn_label_threshold: 0.02
  rnn_upper_threshold: 0.60
  rnn_lower_threshold: 0.40
  rnn_epochs: 50
  rnn_batch_size: 128
  rnn_lr: 0.001
  rnn_cell_type: "gru"  # "gru" | "lstm"
"""

import base64
import io
import copy
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from log_config import get_logger

logger = get_logger(__name__)

NAME = "rnn_trend"

MIN_BARS = 100

FEATURE_COLS = [
    "ret_1d", "ret_5d", "log_hl", "log_co", "volume_z",
    "rsi_14", "bb_upper_dist", "bb_lower_dist",
    "macd_line", "macd_signal",
    "ema_20_dist", "ema_60_dist", "atr_14_pct",
]

_PARAM_KEYS = [
    "rnn_hidden_size", "rnn_num_layers", "rnn_dropout",
    "rnn_window", "rnn_label_period", "rnn_label_threshold",
    "rnn_upper_threshold", "rnn_lower_threshold",
    "rnn_epochs", "rnn_batch_size", "rnn_lr", "rnn_cell_type",
]

TORCH_AVAILABLE = False
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    nn = None


def _serialize_state_dict(model: "torch.nn.Module") -> str:
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return base64.b64encode(buf.getvalue()).decode()


def _deserialize_state_dict(state_dict_b64: str) -> dict:
    buf = io.BytesIO(base64.b64decode(state_dict_b64))
    return torch.load(buf, map_location="cpu", weights_only=True)


class _GRUModel(nn.Module if TORCH_AVAILABLE else object):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        cell_type: str = "gru",
    ):
        if not TORCH_AVAILABLE:
            raise ImportError("torch is not installed. Run: pip install torch")

        super().__init__()
        self.cell_type = cell_type
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if cell_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
            )
        else:
            self.rnn = nn.GRU(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout if num_layers > 1 else 0,
                batch_first=True,
            )

        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        if self.cell_type == "lstm":
            out, _ = self.rnn(x)
        else:
            out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)


class _SequenceDataset(Dataset if TORCH_AVAILABLE else object):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        if not TORCH_AVAILABLE:
            raise ImportError("torch is not installed")
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple["torch.Tensor", "torch.Tensor"]:
        return self.X[idx], self.y[idx]


def _compute_features_for_timestep(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    open_price: np.ndarray,
    volume: np.ndarray,
    t: int,
) -> np.ndarray:
    """为时间步 t 计算 13 维特征向量（仅使用 t 及之前的数据）。"""
    c = close[t]
    h = high[t]
    l = low[t]
    o = open_price[t]
    v = volume[t]

    ret_1d = (c / close[t - 1] - 1) if t >= 1 and close[t - 1] != 0 else 0.0
    ret_5d = (c / close[t - 5] - 1) if t >= 5 and close[t - 5] != 0 else 0.0
    log_hl = np.log(h / l) if l != 0 else 0.0
    log_co = np.log(c / o) if o != 0 else 0.0

    vol_window = min(20, t + 1)
    vol_start = max(0, t - vol_window + 1)
    vol_mean = np.mean(volume[vol_start:t + 1])
    vol_std = np.std(volume[vol_start:t + 1]) + 1e-9
    volume_z = (v - vol_mean) / vol_std

    rsi_period = min(14, t + 1)
    rsi_start = max(0, t - rsi_period + 1)
    delta_arr = np.diff(close[rsi_start:t + 1], prepend=close[rsi_start])
    gain_arr = np.maximum(delta_arr, 0)
    loss_arr = np.maximum(-delta_arr, 0)
    avg_gain = np.mean(gain_arr)
    avg_loss = np.mean(loss_arr)
    rsi_14 = 50.0 if avg_loss == 0 else 100.0 - 100.0 / (1.0 + avg_gain / (avg_loss + 1e-9))

    bb_period = min(20, t + 1)
    bb_start = max(0, t - bb_period + 1)
    bb_close = close[bb_start:t + 1]
    ma20_arr = np.mean(bb_close)
    std20_arr = np.std(bb_close) + 1e-9
    bb_upper_val = ma20_arr + 2 * std20_arr
    bb_lower_val = ma20_arr - 2 * std20_arr
    bb_upper_dist = (c - bb_upper_val) / c if c != 0 else 0.0
    bb_lower_dist = (c - bb_lower_val) / c if c != 0 else 0.0

    ema_data = close[:t + 1]
    span12 = min(12, t + 1)
    span26 = min(26, t + 1)
    ema12_val = float(pd.Series(ema_data).ewm(span=span12, adjust=False).mean().iloc[-1])
    ema26_val = float(pd.Series(ema_data).ewm(span=span26, adjust=False).mean().iloc[-1])
    macd_line = ema12_val - ema26_val

    macd_hist_len = min(9, t + 1)
    macd_series_vals = np.full(macd_hist_len, macd_line)
    macd_signal_val = float(pd.Series(macd_series_vals).ewm(span=macd_hist_len, adjust=False).mean().iloc[-1]) if macd_hist_len > 0 else 0.0

    span20 = min(20, t + 1)
    ema20_val = float(pd.Series(ema_data).ewm(span=span20, adjust=False).mean().iloc[-1])
    span60 = min(60, t + 1)
    ema60_val = float(pd.Series(ema_data).ewm(span=span60, adjust=False).mean().iloc[-1])
    ema_20_dist = (c - ema20_val) / c if c != 0 else 0.0
    ema_60_dist = (c - ema60_val) / c if c != 0 else 0.0

    atr_val = 0.0
    if t >= 1:
        tr1 = h - l
        prev_close = close[t - 1]
        tr2 = abs(h - prev_close)
        tr3 = abs(l - prev_close)
        atr_val = max(tr1, tr2, tr3)
        if t >= 14:
            tr_arr = np.array([
                max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
                for i in range(t - 13, t + 1)
            ])
            atr_val = float(np.mean(tr_arr))
    atr_14_pct = atr_val / c if c != 0 else 0.0

    return np.array([
        ret_1d, ret_5d, log_hl, log_co, volume_z,
        rsi_14 / 100.0,
        bb_upper_dist,
        bb_lower_dist,
        macd_line / c if c != 0 else 0.0,
        macd_signal_val / c if c != 0 else 0.0,
        ema_20_dist,
        ema_60_dist,
        atr_14_pct,
    ], dtype=np.float32)


# ── 特征矩阵缓存 ─────────────────────────────────────────────────
# _build_feature_matrix 是 O(N × window) 的 Python 双循环，每个时间步还要
# 调用 pd.Series.ewm，多股票训练下单次耗时很高。Optuna 每个 trial 都会
# 重建相同的数据 × window 组合的矩阵，在此加一层按 (数据指纹, window)
# 键控的 LRU 缓存。cache 是进程级，外部若切换数据源请调用
# clear_rnn_feature_cache()。
_RNN_FEATURE_CACHE: dict = {}
_RNN_FEATURE_CACHE_MAX = 2048


def _rnn_cache_fingerprint(df: pd.DataFrame, window: int) -> tuple | None:
    """轻量指纹，失败返回 None 表示不缓存。"""
    try:
        if df is None or df.empty or "Close" not in df.columns:
            return None
        close = df["Close"]
        return (
            len(df),
            str(df.index.min()),
            str(df.index.max()),
            float(close.iloc[0]),
            float(close.iloc[-1]),
            int(window),
        )
    except Exception:
        return None


def clear_rnn_feature_cache() -> None:
    _RNN_FEATURE_CACHE.clear()


def _compute_flat_features(df: pd.DataFrame) -> np.ndarray:
    """
    向量化计算 (N, feat_dim) 的扁平特征矩阵。

    与逐步调用 ``_compute_features_for_timestep(t)`` 在语义上等价
    （rolling 使用 min_periods=1 对应原函数的 ``min(period, t+1)``
    自适应窗口），但把复杂度从 O(N²) 降到 O(N)。

    注意：保留了原实现中的 MACD signal 退化行为——原代码对一段
    常数数组做 EMA，结果恒等于 macd_line。这里显式令 macd_signal
    = macd_line 以保持训练出来的模型特征分布不变；若将来要修复
    这个 bug，换成 ``pd.Series(macd_line).ewm(span=9).mean()``。
    """
    close = df["Close"].values.astype(np.float64)
    high = df["High"].values.astype(np.float64) if "High" in df.columns else close
    low = df["Low"].values.astype(np.float64) if "Low" in df.columns else close
    open_p = df["Open"].values.astype(np.float64) if "Open" in df.columns else close
    volume = (
        df["Volume"].values.astype(np.float64)
        if "Volume" in df.columns
        else np.ones_like(close)
    )

    n = len(close)
    if n == 0:
        return np.zeros((0, len(FEATURE_COLS)), dtype=np.float32)

    close_s = pd.Series(close)

    # ret_1d / ret_5d
    ret_1d = close_s.pct_change(1).fillna(0.0).to_numpy()
    ret_5d = close_s.pct_change(5).fillna(0.0).to_numpy()

    # log_hl / log_co（避免除零）
    with np.errstate(divide="ignore", invalid="ignore"):
        log_hl = np.where(low > 0, np.log(np.where(low > 0, high / np.where(low == 0, 1, low), 1)), 0.0)
        log_co = np.where(open_p > 0, np.log(np.where(open_p == 0, 1, close / np.where(open_p == 0, 1, open_p))), 0.0)
    log_hl = np.nan_to_num(log_hl, nan=0.0, posinf=0.0, neginf=0.0)
    log_co = np.nan_to_num(log_co, nan=0.0, posinf=0.0, neginf=0.0)

    # volume_z: (v - rolling_mean_20) / rolling_std_20
    vol_s = pd.Series(volume)
    vol_mean = vol_s.rolling(20, min_periods=1).mean().to_numpy()
    vol_std = vol_s.rolling(20, min_periods=1).std(ddof=0).fillna(0.0).to_numpy() + 1e-9
    volume_z = (volume - vol_mean) / vol_std

    # RSI 14（与原实现一致：delta 前补 0，rolling mean 用 min_periods=1）
    delta = np.diff(close, prepend=close[0])
    gain = np.maximum(delta, 0.0)
    loss = np.maximum(-delta, 0.0)
    avg_gain = pd.Series(gain).rolling(14, min_periods=1).mean().to_numpy()
    avg_loss = pd.Series(loss).rolling(14, min_periods=1).mean().to_numpy()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi_14 = np.where(avg_loss == 0.0, 50.0, 100.0 - 100.0 / (1.0 + rs))

    # Bollinger 20
    ma20 = close_s.rolling(20, min_periods=1).mean().to_numpy()
    std20 = close_s.rolling(20, min_periods=1).std(ddof=0).fillna(0.0).to_numpy() + 1e-9
    bb_upper = ma20 + 2.0 * std20
    bb_lower = ma20 - 2.0 * std20
    safe_close = np.where(close != 0, close, 1.0)
    bb_upper_dist = np.where(close != 0, (close - bb_upper) / safe_close, 0.0)
    bb_lower_dist = np.where(close != 0, (close - bb_lower) / safe_close, 0.0)

    # EMA 12/26/20/60（对整段序列一次性计算，每个 t 的值等价于原实现的
    # pd.Series(close[:t+1]).ewm(...).mean().iloc[-1]，因为 adjust=False
    # 的 EWM 是递推，不依赖未来值）
    ema12 = close_s.ewm(span=12, adjust=False).mean().to_numpy()
    ema26 = close_s.ewm(span=26, adjust=False).mean().to_numpy()
    ema20 = close_s.ewm(span=20, adjust=False).mean().to_numpy()
    ema60 = close_s.ewm(span=60, adjust=False).mean().to_numpy()
    macd_line = ema12 - ema26
    # 原实现 bug：MACD signal 恒等于 macd_line（对常数序列做 EMA）。保留
    macd_signal = macd_line

    macd_line_norm = np.where(close != 0, macd_line / safe_close, 0.0)
    macd_signal_norm = np.where(close != 0, macd_signal / safe_close, 0.0)
    ema_20_dist = np.where(close != 0, (close - ema20) / safe_close, 0.0)
    ema_60_dist = np.where(close != 0, (close - ema60) / safe_close, 0.0)

    # ATR 14：TR = max(H-L, |H-prev_close|, |L-prev_close|)，rolling 14 均值
    prev_close = np.empty_like(close)
    prev_close[0] = close[0]
    prev_close[1:] = close[:-1]
    tr = np.maximum.reduce(
        [high - low, np.abs(high - prev_close), np.abs(low - prev_close)]
    )
    # 原实现：t<1 时 ATR=0；1<=t<14 时 ATR=当日 TR；t>=14 时 14 日 TR 均值
    atr = pd.Series(tr).rolling(14, min_periods=1).mean().to_numpy()
    atr[0] = 0.0  # 原实现 t=0 时 ATR 强制为 0
    atr_14_pct = np.where(close != 0, atr / safe_close, 0.0)

    F = np.stack(
        [
            ret_1d, ret_5d, log_hl, log_co, volume_z,
            rsi_14 / 100.0,
            bb_upper_dist, bb_lower_dist,
            macd_line_norm, macd_signal_norm,
            ema_20_dist, ema_60_dist,
            atr_14_pct,
        ],
        axis=1,
    ).astype(np.float32)
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    return F


def _build_feature_matrix(df: pd.DataFrame, window: int = 30) -> np.ndarray:
    """
    构建 3D 特征矩阵 (N_samples, window, n_features)。
    每个样本对应时间 t 时刻的历史窗口 [t-window+1, t]。
    每个时间步的特征仅使用该时刻及之前的数据，严格无前视。

    向量化实现：先算出扁平 (N, feat_dim) 特征矩阵，再用滑窗索引
    填充 3D 张量，把复杂度从 O(N² × window) 降到 O(N × window)。
    """
    cache_key = _rnn_cache_fingerprint(df, window)
    if cache_key is not None:
        cached = _RNN_FEATURE_CACHE.get(cache_key)
        if cached is not None:
            return cached.copy()

    n = len(df)
    feat_dim = len(FEATURE_COLS)
    X = np.zeros((n, window, feat_dim), dtype=np.float32)

    if n == 0 or window <= 0:
        return X

    F = _compute_flat_features(df)  # (n, feat_dim)

    # X[t, w_idx, :] = F[t - (window - 1 - w_idx)] = F[t - window + 1 + w_idx]
    # 只在 t >= window-1 时写入，前 window-1 行保留 0（与原实现一致）
    for t in range(window - 1, n):
        X[t] = F[t - window + 1 : t + 1]

    if cache_key is not None:
        if len(_RNN_FEATURE_CACHE) >= _RNN_FEATURE_CACHE_MAX:
            _RNN_FEATURE_CACHE.pop(next(iter(_RNN_FEATURE_CACHE)))
        _RNN_FEATURE_CACHE[cache_key] = X.copy()

    return X


def _build_labels(
    close: np.ndarray,
    label_period: int,
    label_threshold: float,
) -> np.ndarray:
    """构建标签：未来 label_period 天收益 > label_threshold → 1，否则 0。"""
    n = len(close)
    y = np.zeros(n, dtype=np.float32)
    for t in range(n - label_period):
        ret = close[t + label_period] / close[t] - 1
        y[t] = 1.0 if ret > label_threshold else 0.0
    return y


def _train(
    model: "torch.nn.Module",
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    config: dict,
) -> "torch.nn.Module":
    """训练 GRU/LSTM 模型，带 Early Stopping。"""
    if not TORCH_AVAILABLE:
        raise ImportError("torch is not installed")

    epochs = int(config.get("rnn_epochs", 50))
    batch_size = int(config.get("rnn_batch_size", 128))
    lr = float(config.get("rnn_lr", 0.001))

    train_dataset = _SequenceDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = _SequenceDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    best_state = None
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze(-1)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                outputs = model(X_batch).squeeze(-1)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def run(data: pd.DataFrame, config: dict):
    """
    运行 RNN 策略。

    训练类型由 data 中的 ticker 列决定:
    - 多股票输入（含 ticker 列，多只股票）: multi-stock 路径
      run_trial() 调用此函数进行多股票训练，predict() 在目标股 OOS 上推理
    - 单股票输入（无 ticker 列）: 单股票后备路径（单元测试/smoke_test）

    在 multi-stock 路径下，run() 返回的 signal 全为 0，
    真正的 OOS 信号由 run_trial() 调用 predict() 生成。
    """
    if not TORCH_AVAILABLE:
        raise ImportError(
            "torch is not installed. Run: pip install torch>=2.0 "
            "(CPU-only recommended: pip install torch --index-url https://download.pytorch.org/whl/cpu)"
        )

    window = int(config.get("rnn_window", 30))
    label_period = int(config.get("rnn_label_period", 5))
    label_threshold = float(config.get("rnn_label_threshold", 0.02))
    hidden_size = int(config.get("rnn_hidden_size", 64))
    num_layers = int(config.get("rnn_num_layers", 2))
    dropout = float(config.get("rnn_dropout", 0.2))
    cell_type = config.get("rnn_cell_type", "gru")

    df = data.copy()
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

    is_multi = "ticker" in df.columns and df["ticker"].nunique() > 1

    if is_multi:
        X_list, y_list = [], []
        for ticker_val, stock_df in df.groupby("ticker"):
            stock_df = stock_df.sort_index()
            if "ticker" in stock_df.columns:
                stock_df = stock_df.drop(columns=["ticker"])
            if len(stock_df) < window + label_period + 10:
                continue
            X_stock = _build_feature_matrix(stock_df, window=window)
            y_stock = _build_labels(stock_df["Close"].values, label_period, label_threshold)
            valid = (np.arange(len(stock_df)) >= window - 1) & \
                    (np.arange(len(stock_df)) < len(stock_df) - label_period)
            X_list.append(X_stock[valid])
            y_list.append(y_stock[valid])

        if not X_list:
            logger.warning("多股票数据不足，返回零信号")
            signal = pd.Series(0, index=df.index)
            meta = {
                "name": NAME,
                "params": {k: config.get(k) for k in _PARAM_KEYS if config.get(k) is not None},
                "feat_cols": FEATURE_COLS,
                "indicators": {},
                "model_type": cell_type,
            }
            return signal, None, meta

        X_all = np.concatenate(X_list, axis=0)
        y_all = np.concatenate(y_list, axis=0)

        split = int(len(X_all) * 0.8)
        X_train, y_train = X_all[:split], y_all[:split]
        X_es, y_es = X_all[split:], y_all[split:]

        signal = pd.Series(0, index=df.index)
    else:
        no_split = config.get("no_internal_split", False)

        close = df["Close"].values
        y_all = _build_labels(close, label_period, label_threshold)
        X_all = _build_feature_matrix(df, window=window)

        valid_start = window
        valid_mask = np.zeros(len(df), dtype=bool)
        valid_mask[valid_start:] = True

        split_idx = int(len(df) * 0.8) if not no_split else len(df)
        train_mask = valid_mask & (np.arange(len(df)) < split_idx)
        val_mask = valid_mask & (np.arange(len(df)) >= split_idx)

        X_train = X_all[train_mask]
        y_train = y_all[train_mask]
        X_es = X_all[val_mask]
        y_es = y_all[val_mask]

        if len(X_train) < 10 or len(X_es) < 5:
            logger.warning("训练数据不足，返回零信号")
            signal = pd.Series(0, index=df.index)
            meta = {
                "name": NAME,
                "params": {k: config.get(k) for k in _PARAM_KEYS if config.get(k) is not None},
                "feat_cols": FEATURE_COLS,
                "indicators": {},
                "model_type": cell_type,
            }
            return signal, None, meta

    model = _GRUModel(
        input_size=len(FEATURE_COLS),
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
        cell_type=cell_type,
    )

    model = _train(model, X_train, y_train, X_es, y_es, config)

    if not is_multi:
        all_signals = np.zeros(len(df), dtype=np.int32)
        if len(X_es) > 0:
            model.eval()
            with torch.no_grad():
                X_val_tensor = torch.FloatTensor(X_es)
                probs = np.atleast_1d(model(X_val_tensor).squeeze(-1).numpy())
                val_indices = np.where(val_mask)[0]
                upper = float(config.get("rnn_upper_threshold", 0.60))
                lower = float(config.get("rnn_lower_threshold", 0.40))
                all_signals[val_indices] = 0
                all_signals[val_indices[probs > upper]] = 1
                all_signals[val_indices[probs < lower]] = 0
        signal = pd.Series(all_signals, index=df.index)

    meta = {
        "name": NAME,
        "params": {k: config.get(k) for k in _PARAM_KEYS if config.get(k) is not None},
        "feat_cols": FEATURE_COLS,
        "indicators": {},
        "state_dict_b64": _serialize_state_dict(model),
        "model_type": cell_type,
        "input_size": len(FEATURE_COLS),
        "hidden_size": hidden_size,
        "num_layers": num_layers,
        "dropout": dropout,
    }

    return signal, model, meta


def predict(model, data: pd.DataFrame, config: dict, meta: dict) -> pd.Series:
    """
    独立推断函数：只做特征工程 + model.forward()，不做训练。
    """
    if not TORCH_AVAILABLE:
        raise ImportError("torch is not installed")

    window = int(config.get("rnn_window", 30))

    if meta.get("state_dict_b64") is None and model is None:
        raise ValueError("model is None and state_dict_b64 not in meta")

    input_size = meta.get("input_size", len(FEATURE_COLS))
    hidden_size = meta.get("hidden_size", 64)
    num_layers = meta.get("num_layers", 2)
    dropout = meta.get("dropout", 0.2)
    cell_type = meta.get("model_type", "gru")

    if model is None:
        model = _GRUModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            cell_type=cell_type,
        )
        state_dict = _deserialize_state_dict(meta["state_dict_b64"])
        model.load_state_dict(state_dict)

    model.eval()

    df = data.copy()
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated(keep="first")]

    X = _build_feature_matrix(df, window=window)

    valid_start = window
    valid_mask = np.zeros(len(df), dtype=bool)
    valid_mask[valid_start:] = True

    signals = np.zeros(len(df), dtype=np.int32)
    signals[:valid_start] = 0

    if X.shape[0] > 0:
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            probs = np.atleast_1d(model(X_tensor).squeeze(-1).numpy())

            upper = float(config.get("rnn_upper_threshold", 0.60))
            lower = float(config.get("rnn_lower_threshold", 0.40))
            valid_indices = np.where(valid_mask)[0]
            probs_valid = probs[valid_indices]
            signals[valid_indices[probs_valid > upper]] = 1
            signals[valid_indices[probs_valid < lower]] = 0

    return pd.Series(signals, index=df.index)
