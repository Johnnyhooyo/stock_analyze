"""
技术指标计算模块 - 使用 ta-lib

ta-lib (https://github.com/bukosabino/ta) 封装了200+技术指标，
提供快速、标准化的计算，比纯pandas实现更高效。

用法:
    from strategies.indicators import add_ta_features
    df_with_features = add_ta_features(df)
"""

import numpy as np
import pandas as pd
from typing import Optional

# ta-lib 库
try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False


def add_ta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    使用 ta-lib 计算所有技术指标特征

    计算的特征:
    - RSI (6, 14, 21)
    - MACD (line, signal, histogram)
    - Bollinger Bands (10, 20, 30 周期)
    - KDJ (K, D, J)
    - ATR (7, 14, 21)
    - OBV, PVT
    - 波动率
    - 成交量特征
    - 移动平均
    - 动量
    """
    if not TA_AVAILABLE:
        raise ImportError("ta library not installed. Run: pip install ta")

    df = df.copy()

    # ========== 收益特征 ==========
    df['returns'] = df['Close'].pct_change()
    for i in range(1, 21):
        df[f'ret_{i}'] = df['Close'].pct_change(i)

    # ========== RSI ==========
    for period in [6, 14, 21]:
        df[f'rsi_{period}'] = ta.momentum.RSIIndicator(df['Close'], period).rsi()

    # ========== MACD ==========
    macd = ta.trend.MACD(df['Close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()

    # ========== Bollinger Bands ==========
    for period, std in [(10, 1.5), (20, 2.0), (30, 2.5)]:
        bb = ta.volatility.BollingerBands(df['Close'], period, std)
        df[f'bb_upper_{period}'] = bb.bollinger_hband()
        df[f'bb_mid_{period}'] = bb.bollinger_mavg()
        df[f'bb_lower_{period}'] = bb.bollinger_lband()
        df[f'bb_position_{period}'] = bb.bollinger_pband()
        df[f'bb_width_{period}'] = bb.bollinger_wband()

    # ========== KDJ (Stochastic) ==========
    # ta 的 StochasticOscillator 对应 KDJ 指标
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'])
    df['kdj_k'] = stoch.stoch()
    df['kdj_d'] = stoch.stoch_signal()
    # J = 3K - 2D
    df['kdj_j'] = 3 * df['kdj_k'] - 2 * df['kdj_d']
    df['kdj_overbought'] = (df['kdj_j'] > 80).astype(int)
    df['kdj_oversold'] = (df['kdj_j'] < 20).astype(int)

    # ========== ATR ==========
    atr = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'])
    for period in [7, 14, 21]:
        # ta 的 ATR 周期是固定的 14，我们需要手动计算不同周期
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - df['Close'].shift(1)).abs()
        tr3 = (df['Low'] - df['Close'].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df[f'atr_{period}'] = tr.rolling(period).mean()
        df[f'atr_pct_{period}'] = df[f'atr_{period}'] / df['Close']

    # ========== 波动率 ==========
    for period in [5, 10, 20]:
        df[f'volatility_{period}'] = df['returns'].rolling(period).std()
        df[f'volatility_change_{period}'] = df[f'volatility_{period}'].pct_change()

    # ========== 成交量特征 ==========
    df['volume'] = df['Volume']
    for period in [5, 10, 20]:
        df[f'volume_ma_{period}'] = df['Volume'].rolling(period).mean()
        df[f'volume_ratio_{period}'] = df['Volume'] / df[f'volume_ma_{period}']

    # ========== OBV & PVT ==========
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()

    # PVT (Price Volume Trend)
    pvt = ((df['Close'].diff() / df['Close'].shift(1)) * df['Volume']).fillna(0).cumsum()
    df['pvt'] = pvt

    for period in [5, 10]:
        df[f'obv_ma_{period}'] = df['obv'].rolling(period).mean()
        df[f'pvt_ma_{period}'] = df['pvt'].rolling(period).mean()

    # ========== 移动平均 ==========
    for period in [5, 10, 20, 50]:
        df[f'ma_{period}'] = df['Close'].rolling(period).mean()
        df[f'price_vs_ma_{period}'] = df['Close'] / df[f'ma_{period}']

    # ========== 趋势特征 ==========
    for period in [5, 10, 20]:
        df[f'high_{period}'] = df['High'].rolling(period).max()
        df[f'low_{period}'] = df['Low'].rolling(period).min()
        df[f'high_ratio_{period}'] = df['Close'] / df[f'high_{period}']
        df[f'low_ratio_{period}'] = df['Close'] / df[f'low_{period}']

    # ========== 动量 ==========
    for period in [5, 10, 20]:
        df[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1

    return df


def add_ta_features_fallback(df: pd.DataFrame) -> pd.DataFrame:
    """
    纯 pandas 实现的技术指标（当 ta 不可用时的 fallback）
    """
    df = df.copy()

    # RSI
    def _rsi(series, period=14):
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - 100 / (1 + rs)

    for period in [6, 14, 21]:
        df[f'rsi_{period}'] = _rsi(df['Close'], period)

    # MACD
    ema_fast = df['Close'].ewm(span=12, adjust=False).mean()
    ema_slow = df['Close'].ewm(span=26, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df['macd'] = macd_line
    df['macd_signal'] = signal_line
    df['macd_hist'] = macd_line - signal_line

    # Bollinger Bands
    for period, std in [(10, 1.5), (20, 2.0), (30, 2.5)]:
        ma = df['Close'].rolling(period).mean()
        std_val = df['Close'].rolling(period).std()
        upper = ma + std * std_val
        lower = ma - std * std_val
        df[f'bb_upper_{period}'] = upper
        df[f'bb_mid_{period}'] = ma
        df[f'bb_lower_{period}'] = lower
        df[f'bb_position_{period}'] = (df['Close'] - lower) / (upper - lower).replace(0, np.nan)
        df[f'bb_width_{period}'] = (upper - lower) / ma

    # KDJ
    lowest_low = df['Low'].rolling(9).min()
    highest_high = df['High'].rolling(9).max()
    rsv = (df['Close'] - lowest_low) / (highest_high - lowest_low).replace(0, np.nan) * 100
    k = rsv.ewm(com=2, adjust=False).mean()
    d = k.ewm(com=2, adjust=False).mean()
    df['kdj_k'] = k
    df['kdj_d'] = d
    df['kdj_j'] = 3 * k - 2 * d
    df['kdj_overbought'] = (df['kdj_j'] > 80).astype(int)
    df['kdj_oversold'] = (df['kdj_j'] < 20).astype(int)

    # ATR
    for period in [7, 14, 21]:
        tr1 = df['High'] - df['Low']
        tr2 = (df['High'] - df['Close'].shift(1)).abs()
        tr3 = (df['Low'] - df['Close'].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df[f'atr_{period}'] = tr.rolling(period).mean()
        df[f'atr_pct_{period}'] = df[f'atr_{period}'] / df['Close']

    # Volatility
    for period in [5, 10, 20]:
        df[f'volatility_{period}'] = df['Close'].pct_change().rolling(period).std()
        df[f'volatility_change_{period}'] = df[f'volatility_{period}'].pct_change()

    # Volume
    df['volume'] = df['Volume']
    for period in [5, 10, 20]:
        df[f'volume_ma_{period}'] = df['Volume'].rolling(period).mean()
        df[f'volume_ratio_{period}'] = df['Volume'] / df[f'volume_ma_{period}']

    # OBV & PVT
    df['obv'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    df['pvt'] = ((df['Close'].diff() / df['Close'].shift(1)) * df['Volume']).fillna(0).cumsum()
    for period in [5, 10]:
        df[f'obv_ma_{period}'] = df['obv'].rolling(period).mean()
        df[f'pvt_ma_{period}'] = df['pvt'].rolling(period).mean()

    # MA
    for period in [5, 10, 20, 50]:
        df[f'ma_{period}'] = df['Close'].rolling(period).mean()
        df[f'price_vs_ma_{period}'] = df['Close'] / df[f'ma_{period}']

    # Trend
    for period in [5, 10, 20]:
        df[f'high_{period}'] = df['High'].rolling(period).max()
        df[f'low_{period}'] = df['Low'].rolling(period).min()
        df[f'high_ratio_{period}'] = df['Close'] / df[f'high_{period}']
        df[f'low_ratio_{period}'] = df['Close'] / df[f'low_{period}']

    # Momentum
    for period in [5, 10, 20]:
        df[f'momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1

    return df


# ═══════════════════════════════════════════════════════════════════════════
#  公共指标函数（纯 pandas，不依赖 ta 库）
#  供各规则策略统一使用，消除重复实现
# ═══════════════════════════════════════════════════════════════════════════

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """RSI（相对强弱指标）。"""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0):
    """布林带：返回 (upper, middle, lower)。"""
    middle = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def ema(series: pd.Series, period: int) -> pd.Series:
    """指数移动平均线。"""
    return series.ewm(span=period, adjust=False).mean()


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """MACD：返回 (macd_line, signal_line, histogram)。"""
    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def kdj(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 9):
    """KDJ：返回 (K, D, J)。"""
    lowest_low = low.rolling(period).min()
    highest_high = high.rolling(period).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    rsv = (close - lowest_low) / denom * 100
    K = rsv.ewm(alpha=1/3, adjust=False).mean()
    D = K.ewm(alpha=1/3, adjust=False).mean()
    J = 3 * K - 2 * D
    return K, D, J


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """OBV（能量潮）。"""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """ATR（平均真实波幅）。"""
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def pvt(close: pd.Series, volume: pd.Series) -> pd.Series:
    """PVT（价量趋势）。"""
    return ((close.diff() / close.shift(1)) * volume).fillna(0).cumsum()


def fibonacci(high: pd.Series, low: pd.Series, period: int = 60):
    """斐波那契回撤：返回 (fib_0618_support, fib_0382_resistance)。"""
    swing_high = high.rolling(period).max()
    swing_low = low.rolling(period).min()
    diff = swing_high - swing_low
    fib_0618 = swing_high - 0.618 * diff  # 支撑位
    fib_0382 = swing_high - 0.382 * diff  # 阻力位
    return fib_0618, fib_0382
