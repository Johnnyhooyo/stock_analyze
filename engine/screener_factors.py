"""
选股因子模块 — 轻量级多因子评分

与交易策略使用的深度技术指标不同，选股因子：
- 纯 pandas 实现（无 ta-lib 依赖，启动快）
- 计算密集度低（全市场扫描用）
- 输出归一化分数 [0, 100]，便于线性加权

因子分类：
  Tier 1 - 动量与趋势（速度快）
  Tier 2 - 量价配合（中等计算量）

Usage:
    from engine.screener_factors import ScreenerFactors
    factors = ScreenerFactors()
    scores = factors.calc_all(df)
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import TypedDict

from strategies.indicators import rsi, macd, obv


class FactorScores(TypedDict):
    momentum: float
    trend: float
    volume: float


SIGNALS_LIST = list[str]


@dataclass
class FactorResult:
    momentum_score: float
    trend_score: float
    volume_score: float
    signals: SIGNALS_LIST
    change_pct_5d: float
    change_pct_20d: float
    avg_volume_ratio: float
    rsi_14: float
    macd_hist: float
    obv_slope: float
    valuation_score: float = 0.0
    sentiment_score: float = 50.0


class ScreenerFactors:

    def calc_all(
        self,
        df: pd.DataFrame,
        enable_valuation: bool = False,
        enable_sentiment: bool = False,
        ticker: str = None,
    ) -> FactorResult:
        last = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else last

        mom = self._calc_momentum(df)
        trend = self._calc_trend(df)
        vol = self._calc_volume(df)
        sigs = self._detect_signals(df, mom, trend, vol)

        val_score = 0.0
        sent_score = 50.0

        if enable_valuation and ticker:
            val_score = self._score_valuation(ticker)

        if enable_sentiment and ticker:
            sent_score = self._score_sentiment(ticker)

        return FactorResult(
            momentum_score=mom,
            trend_score=trend,
            volume_score=vol,
            signals=sigs,
            change_pct_5d=self._pct_change(df, 5),
            change_pct_20d=self._pct_change(df, 20),
            avg_volume_ratio=self._avg_volume_ratio(df, 5, 20),
            rsi_14=float(last.get("rsi_14", 50.0)),
            macd_hist=float(last.get("macd_hist", 0.0)),
            obv_slope=float(self._obv_slope(df, 20)),
            valuation_score=val_score,
            sentiment_score=sent_score,
        )

    def _score_valuation(self, ticker: str) -> float:
        """估值得分：PE分位 + 股息率（网络请求，较慢）"""
        try:
            import yfinance
            info = yfinance.Ticker(ticker).info
            pe = info.get("trailingPE")
            div_yield = info.get("dividendYield")
            score = 50.0
            if pe and pe > 0:
                score = min(pe / 30 * 50, 100.0)
            if div_yield and div_yield > 0:
                score = min(score + div_yield * 1000, 100.0)
            return max(0.0, min(score, 100.0))
        except Exception:
            return 50.0

    def _score_sentiment(self, ticker: str) -> float:
        """情感得分：复用 sentiment_analysis 模块"""
        try:
            from sentiment_analysis import get_sentiment
            result = get_sentiment(ticker)
            if result and isinstance(result, dict):
                polarity = result.get("polarity", 0.0)
                return max(0.0, min((polarity + 1) / 2 * 100, 100.0))
            return 50.0
        except Exception:
            return 50.0

    def _pct_change(self, df: pd.DataFrame, period: int) -> float:
        if len(df) <= period:
            return 0.0
        return (df["Close"].iloc[-1] / df["Close"].iloc[-period - 1] - 1) * 100

    def _avg_volume_ratio(self, df: pd.DataFrame, short: int, long: int) -> float:
        if len(df) < long:
            return 1.0
        vol = df["Volume"]
        short_ma = vol.rolling(short).mean()
        long_ma = vol.rolling(long).mean()
        recent_avg = vol.iloc[-short:].mean()
        base = long_ma.iloc[-1] if long_ma.iloc[-1] > 0 else 1.0
        return recent_avg / base

    def _obv_slope(self, df: pd.DataFrame, period: int) -> float:
        if len(df) < period + 1:
            return 0.0
        o = obv(df["Close"], df["Volume"])
        recent = o.iloc[-period:]
        if len(recent) < 2:
            return 0.0
        x = np.arange(len(recent))
        slope = np.polyfit(x, recent.values, 1)[0]
        return slope

    def _calc_momentum(self, df: pd.DataFrame) -> float:
        close = df["Close"]
        scores = []

        for period, weight in [(5, 0.25), (10, 0.25), (20, 0.30), (60, 0.20)]:
            if len(df) <= period:
                continue
            ret = close.iloc[-1] / close.iloc[-period - 1] - 1
            score = (ret + 0.10) / 0.20 * 100
            score = max(0.0, min(100.0, score))
            scores.append((score, weight))

        if not scores:
            return 50.0
        total_weight = sum(w for _, w in scores)
        return sum(s * w for s, w in scores) / total_weight

    def _calc_trend(self, df: pd.DataFrame) -> float:
        close = df["Close"]
        scores = []

        ma5 = close.rolling(5).mean()
        ma10 = close.rolling(10).mean()
        ma20 = close.rolling(20).mean()
        ma60 = close.rolling(60).mean() if len(df) >= 60 else ma20

        if len(df) >= 20:
            ma_bull = (ma5.iloc[-1] > ma10.iloc[-1] > ma20.iloc[-1])
            long_bull = ma5.iloc[-1] > ma60.iloc[-1] if len(df) >= 60 else True
            scores.append((100.0 if ma_bull and long_bull else 30.0, 0.35))

        rsi_val = rsi(close, 14)
        rsi_last = rsi_val.iloc[-1] if not rsi_val.isna().iloc[-1] else 50.0
        if rsi_last < 30:
            rsi_score = 100.0 - rsi_last
        elif rsi_last > 70:
            rsi_score = 100.0 - (rsi_last - 70) * 2
        else:
            rsi_score = rsi_last
        scores.append((max(0.0, min(100.0, rsi_score)), 0.30))

        macd_line, signal_line, hist = macd(close)
        macd_hist = hist.iloc[-1]
        macd_score = (macd_hist + 0.02) / 0.04 * 100 if macd_hist > 0 else (macd_hist + 0.02) / 0.04 * 100
        scores.append((max(0.0, min(100.0, macd_score)), 0.35))

        if not scores:
            return 50.0
        total_weight = sum(w for _, w in scores)
        return sum(s * w for s, w in scores) / total_weight

    def _calc_volume(self, df: pd.DataFrame) -> float:
        close = df["Close"]
        vol = df["Volume"]
        scores = []

        if len(df) >= 21:
            vol_ma20 = vol.rolling(20).mean()
            vol_ratio = vol.iloc[-1] / vol_ma20.iloc[-1] if vol_ma20.iloc[-1] > 0 else 1.0
            vol_score = min(vol_ratio / 3.0 * 100, 100.0) if vol_ratio >= 1.0 else max(0.0, (1.0 - vol_ratio) * 50)
            scores.append((max(0.0, min(100.0, vol_score)), 0.40))

        close_20_high = close.rolling(20).max().iloc[-1]
        is_breakout = close.iloc[-1] >= close_20_high * 0.98
        scores.append((100.0 if is_breakout else 30.0, 0.25))

        if len(df) >= 21:
            vol_ma5 = vol.rolling(5).mean()
            vol_ma20 = vol.rolling(20).mean()
            is_contraction = vol_ma5.iloc[-1] < vol_ma20.iloc[-1] * 0.7
            in_uptrend = close.iloc[-1] > close.rolling(20).mean().iloc[-1]
            scores.append((100.0 if is_contraction and in_uptrend else 40.0, 0.20))

        o = obv(close, vol)
        if len(df) >= 21:
            obv_20_high = o.rolling(20).max().iloc[-1]
            is_obv_high = o.iloc[-1] >= obv_20_high * 0.95
            scores.append((100.0 if is_obv_high else 40.0, 0.15))

        if not scores:
            return 50.0
        total_weight = sum(w for _, w in scores)
        return sum(s * w for s, w in scores) / total_weight

    def _detect_signals(
        self,
        df: pd.DataFrame,
        mom_score: float,
        trend_score: float,
        vol_score: float,
    ) -> SIGNALS_LIST:
        close = df["Close"]
        vol = df["Volume"]
        signals: SIGNALS_LIST = []

        if len(df) >= 21:
            high_20 = close.rolling(20).max().iloc[-1]
            if close.iloc[-1] >= high_20 * 0.98:
                signals.append("突破20日新高")

        if len(df) >= 10:
            ma5 = close.rolling(5).mean()
            ma10 = close.rolling(10).mean()
            ma20 = close.rolling(20).mean()
            ma60 = close.rolling(60).mean() if len(df) >= 60 else ma20
            if ma5.iloc[-1] > ma10.iloc[-1] > ma20.iloc[-1] > ma60.iloc[-1]:
                signals.append("均线多头排列")

        rsi_val = rsi(close, 14)
        rsi_last = rsi_val.iloc[-1]
        rsi_prev = rsi_val.iloc[-2] if len(rsi_val) >= 2 else rsi_last
        if rsi_last < 30 and rsi_last > rsi_prev:
            signals.append("RSI超卖反弹")

        macd_line, signal_line, hist = macd(close)
        if len(df) >= 2:
            if hist.iloc[-1] > 0 and hist.iloc[-2] <= 0:
                signals.append("MACD金叉")
            elif hist.iloc[-1] < 0 and hist.iloc[-2] >= 0:
                signals.append("MACD死叉")

        if len(df) >= 21:
            vol_ma20 = vol.rolling(20).mean()
            vol_ratio = vol.iloc[-1] / vol_ma20.iloc[-1] if vol_ma20.iloc[-1] > 0 else 1.0
            if vol_ratio >= 2.0 and close.iloc[-1] > close.iloc[-2]:
                signals.append("放量上涨")

        if len(df) >= 21:
            vol_ma5 = vol.rolling(5).mean()
            vol_ma20 = vol.rolling(20).mean()
            if vol_ma5.iloc[-1] < vol_ma20.iloc[-1] * 0.5:
                uptrend = close.iloc[-1] > close.rolling(20).mean().iloc[-1]
                if uptrend:
                    signals.append("缩量回调")

        o = obv(close, vol)
        if len(df) >= 21:
            obv_20_high = o.rolling(20).max().iloc[-1]
            if o.iloc[-1] >= obv_20_high * 0.98:
                signals.append("OBV趋势新高")

        return signals
