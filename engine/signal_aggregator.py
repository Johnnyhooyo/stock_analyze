"""
engine/signal_aggregator.py — 多策略共识信号聚合

对每只股票加载 data/factors/ 下所有有效 factor_*.pkl，
分别运行信号生成，再按各因子的 sharpe_ratio 加权投票，
输出共识信号（BUY / SELL / HOLD）及置信度。

设计目标：
  - 委员会投票（多策略共识）比单策略更鲁棒
  - 只做轻量推断（不重新训练）
  - 因子文件不存在时优雅降级
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from log_config import get_logger
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from engine.meta_aggregator import MetaAggregator

logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────
#  结果数据类
# ──────────────────────────────────────────────────────────────────

@dataclass
class AggregatedSignal:
    """单只股票的多策略共识信号"""
    ticker: str
    consensus_signal: int          # 1 = 看涨, 0 = 看跌
    bullish_count: int = 0         # 投票看涨的策略数
    bearish_count: int = 0         # 投票看跌的策略数
    total_strategies: int = 0      # 参与投票的策略总数
    confidence_pct: float = 0.0    # 加权置信度 [0, 1]
    top_strategy: str = ""         # 权重最高的策略名称
    top_strategy_sharpe: float = float("nan")
    ml_signal: Optional[int] = None    # ML 策略信号（如有）
    rule_signal: Optional[int] = None  # 规则策略共识信号（如有）
    raw_votes: list = field(default_factory=list)  # 原始投票记录

    @property
    def signal_label(self) -> str:
        if self.consensus_signal == 1:
            return "📈 看涨"
        return "📉 看跌"

    @property
    def confidence_label(self) -> str:
        pct = self.confidence_pct
        if pct >= 0.75:
            return "高"
        if pct >= 0.55:
            return "中"
        return "低"

    @property
    def action_emoji(self) -> str:
        """配合持仓状态使用，纯信号层面的 emoji。"""
        return "🟢" if self.consensus_signal == 1 else "🔴"


# ──────────────────────────────────────────────────────────────────
#  SignalAggregator
# ──────────────────────────────────────────────────────────────────

class SignalAggregator:
    """
    多策略信号聚合器。

    用法：
        agg = SignalAggregator(factors_dir="data/factors")
        signal = agg.aggregate("0700.HK", hist_df, config)
    """

    def __init__(
        self,
        factors_dir: Optional[Path] = None,
        min_sharpe_weight: float = 0.0,
        max_factors: int = 20,
        use_registry: bool = True,
        aggregation_method: str = "vote",
        meta_dir: Optional[Path] = None,
    ):
        """
        Args:
            factors_dir:         factor_*.pkl 所在目录（默认 data/factors/）
            min_sharpe_weight:   参与投票的最低 sharpe 阈值（默认 0，全部参与）
            max_factors:         最多加载的因子数量（取编号最大的 N 个）
            use_registry:        是否使用因子注册表过滤（默认 True，只加载 active 因子）
            aggregation_method:  聚合方式，"vote"（默认）或 "stacking"
            meta_dir:            meta-model 存储目录（默认 data/meta/）
        """
        self.factors_dir = factors_dir or (
            Path(__file__).parent.parent / "data" / "factors"
        )
        self.min_sharpe_weight = min_sharpe_weight
        self.max_factors = max_factors
        self._use_registry = use_registry
        self._registry = None
        # 预加载策略模块 dict（NAME → module），避免每个因子推断时重复 import
        self._strategy_modules: dict = {}
        self._aggregation_method = aggregation_method
        self._meta_dir = meta_dir or (Path(__file__).parent.parent / "data" / "meta")
        self._meta_cache: dict[str, Optional["MetaAggregator"]] = {}

    # ── 内部：加载因子列表 ────────────────────────────────────────

    def _load_factors(self, factors_dir: Optional[Path] = None) -> list[dict]:
        """
        加载指定目录中所有 factor_*.pkl。
        factors_dir 为 None 时使用 self.factors_dir（向后兼容）。
        """
        target = factors_dir or self.factors_dir
        candidates = sorted(
            target.glob("factor_*.pkl"),
            key=lambda p: int(p.stem.split("_")[1]),
            reverse=True,
        )
        artifacts = []
        for path in candidates[: self.max_factors]:
            try:
                art = joblib.load(path)
                art["_path"] = str(path)
                artifacts.append(art)
            except Exception as e:
                logger.warning("加载因子失败 %s: %s", path.name, e, extra={"factor": path.name})
        return artifacts

    # ── 内部：从单个因子生成信号 ──────────────────────────────────

    def _get_signal_from_artifact(
        self, art: dict, data: pd.DataFrame, config: dict
    ) -> Optional[int]:
        """
        从单个因子文件推断当前信号（1 或 0）。
        ML 策略优先用 predict()，规则策略用 strategy_mod.run()。
        失败时返回 None（投弃权票）。
        """
        model = art.get("model")
        meta = art.get("meta", {})
        strategy_name = meta.get("name", "")
        feat_cols = meta.get("feat_cols", [])
        is_ml = model is not None and len(feat_cols) > 0

        # 找到对应策略模块（懒加载并缓存，只在第一次调用时 import）
        if not self._strategy_modules:
            from analyze_factor import _discover_strategies
            self._strategy_modules = {mod.NAME: mod for mod in _discover_strategies()}
        strategy_mod = self._strategy_modules.get(strategy_name)

        art_config = {**config, **(art.get("config", {}))}
        # 保留 ticker
        art_config["ticker"] = config.get("ticker", art_config.get("ticker", "0700.HK"))

        df = data.copy().sort_index()

        try:
            if is_ml and strategy_mod is not None and hasattr(strategy_mod, "predict"):
                sig_series = strategy_mod.predict(model, df, art_config, meta)
                if sig_series is not None and not sig_series.empty:
                    return int(sig_series.dropna().iloc[-1])

            elif not is_ml and strategy_mod is not None:
                sig_series, _, _ = strategy_mod.run(df, art_config)
                if sig_series is not None and not sig_series.empty:
                    return int(sig_series.iloc[-1])

        except Exception as e:
            logger.warning("策略 %s 信号生成失败: %s", strategy_name, e,
                           extra={"strategy": strategy_name})

        return None

    # ── 内部：加载 meta-model ──────────────────────────────────────

    def _load_meta(self, ticker: str) -> Optional["MetaAggregator"]:
        """Lazily load meta-model for ticker. Returns None if unavailable."""
        if ticker not in self._meta_cache:
            try:
                from engine.meta_aggregator import MetaAggregator
                self._meta_cache[ticker] = MetaAggregator.load(ticker, self._meta_dir)
            except Exception as e:
                logger.debug("meta-model 加载异常: %s", e)
                self._meta_cache[ticker] = None
        return self._meta_cache[ticker]

    # ── 主接口 ────────────────────────────────────────────────────

    def _get_registry(self):
        """懒加载注册表"""
        if self._registry is None and self._use_registry:
            try:
                from data.factor_registry import FactorRegistry
                self._registry = FactorRegistry()
            except Exception as e:
                logger.debug("因子注册表加载失败，降级为全量磁盘扫描", extra={"error": str(e)})
                self._registry = None
        return self._registry

    def _filter_by_registry(self, artifacts: list[dict], subdir: str | None) -> list[dict]:
        """根据注册表过滤因子，只保留 active 的记录。

        Fallback 规则：
        - 注册表不可用 → 返回全部（原有行为）
        - 注册表为空（无任何 active 记录）→ 返回全部（原有行为）
        - 注册表非空但过滤后为空 → 记录 warning，返回全部（防止注册表与磁盘不同步时误过滤）
        """
        registry = self._get_registry()
        if registry is None:
            return artifacts
        try:
            active = registry.active_records()
            if not active:
                return artifacts
            # subdir 大小写不敏感匹配，兼容历史注册记录
            active_keys = {
                (r.filename, (r.subdir or "").upper() if r.subdir else None)
                for r in active
            }
            normalized_subdir = (subdir or "").upper() if subdir else None
            filtered = [
                a for a in artifacts
                if (Path(a["_path"]).name, normalized_subdir) in active_keys
            ]
            if not filtered and artifacts:
                logger.warning(
                    "注册表过滤后因子为空（注册表 %d 条 active，磁盘 %d 个文件，subdir=%s），"
                    "回退使用全部磁盘因子",
                    len(active), len(artifacts), subdir,
                )
                return artifacts
            if len(filtered) < len(artifacts):
                logger.debug("注册表过滤: %d -> %d 个因子", len(artifacts), len(filtered))
            return filtered
        except Exception as e:
            logger.warning("注册表过滤异常，回退使用全部因子: %s", e)
            return artifacts

    def aggregate(
        self, ticker: str, data: pd.DataFrame, config: dict
    ) -> AggregatedSignal:
        """
        对指定股票运行所有可用因子，返回共识信号。

        Args:
            ticker: 股票代码（用于日志，config["ticker"] 才是真正的行情代码）
            data:   该股票的历史 OHLCV DataFrame
            config: 包含 ticker、risk_management 等的配置字典

        Returns:
            AggregatedSignal
        """
        ticker_safe = ticker.replace(".", "_").upper()
        per_ticker_dir = self.factors_dir / ticker_safe

        if per_ticker_dir.is_dir() and any(per_ticker_dir.glob("factor_*.pkl")):
            per_ticker_arts = self._load_factors(per_ticker_dir)
            per_ticker_arts = self._filter_by_registry(per_ticker_arts, subdir=ticker_safe)
            global_arts_all = self._load_factors(self.factors_dir)
            global_ml_arts = [
                a for a in global_arts_all
                if len(a.get("meta", {}).get("feat_cols", [])) > 0
            ]
            global_ml_arts = self._filter_by_registry(global_ml_arts, subdir=None)
            artifacts = per_ticker_arts + global_ml_arts
            artifacts = artifacts[:self.max_factors]
            logger.debug(
                "混合加载: per-ticker %d 个规则因子 + 全局 %d 个ML因子",
                len(per_ticker_arts), len(global_ml_arts),
                extra={"ticker": ticker},
            )
        else:
            artifacts = self._load_factors(self.factors_dir)
            artifacts = self._filter_by_registry(artifacts, subdir=None)
            logger.debug("全局因子加载: %d 个",
                         len(artifacts), extra={"ticker": ticker})

        if not artifacts:
            logger.warning("%s: 无可用因子，返回默认空仓信号", ticker, extra={"ticker": ticker})
            return AggregatedSignal(
                ticker=ticker,
                consensus_signal=0,
                confidence_pct=0.0,
            )

        votes: list[dict] = []   # {signal, weight, strategy_name, is_ml}
        top_strategy = ""
        top_weight = float("-inf")

        for art in artifacts:
            sharpe = art.get("sharpe_ratio", float("nan"))
            if math.isnan(sharpe):
                sharpe = 0.0

            # Rank IC 优先作为权重；IC 缺失或 NaN 时退回 Sharpe
            rank_ic = art.get("rank_ic", float("nan"))
            if not math.isnan(rank_ic):
                weight = max(abs(rank_ic), 0.01)   # IC 越高权重越大
            else:
                weight = max(sharpe, 0.01)          # 旧因子兜底

            sig = self._get_signal_from_artifact(art, data, config)
            if sig is None:
                continue

            strategy_name = art.get("meta", {}).get("name", "unknown")
            model = art.get("model")
            feat_cols = art.get("meta", {}).get("feat_cols", [])
            is_ml = model is not None and len(feat_cols) > 0

            votes.append({
                "signal": sig,
                "weight": weight,
                "strategy_name": strategy_name,
                "sharpe": sharpe,
                "rank_ic": rank_ic,
                "is_ml": is_ml,
            })

            if weight > top_weight:
                top_weight = weight
                top_strategy = strategy_name

        if not votes:
            return AggregatedSignal(
                ticker=ticker,
                consensus_signal=0,
                confidence_pct=0.0,
            )

        # 加权投票
        bull_w = sum(v["weight"] for v in votes if v["signal"] == 1)
        bear_w = sum(v["weight"] for v in votes if v["signal"] == 0)
        total_w = bull_w + bear_w

        consensus = 1 if bull_w >= bear_w else 0
        confidence = (max(bull_w, bear_w) / total_w) if total_w > 0 else 0.5

        # Stacking override (only when votes are available)
        if self._aggregation_method == "stacking" and votes:
            meta = self._load_meta(ticker)
            if meta is not None:
                try:
                    base_signals = {v["strategy_name"]: v["signal"] for v in votes}
                    feat = meta.build_feature_vector(base_signals, data)
                    consensus, confidence = meta.predict(feat)
                except Exception as e:
                    logger.warning(
                        "%s: stacking predict 失败，fallback 到 vote: %s", ticker, e,
                        extra={"ticker": ticker}
                    )
            else:
                logger.debug("%s: meta-model 不存在，fallback 到 vote", ticker,
                             extra={"ticker": ticker})

        # 分别统计 ML 和规则策略
        ml_votes = [v for v in votes if v["is_ml"]]
        rule_votes = [v for v in votes if not v["is_ml"]]

        ml_signal = None
        if ml_votes:
            ml_bull = sum(v["weight"] for v in ml_votes if v["signal"] == 1)
            ml_bear = sum(v["weight"] for v in ml_votes if v["signal"] == 0)
            ml_signal = 1 if ml_bull >= ml_bear else 0

        rule_signal = None
        if rule_votes:
            rule_bull = sum(1 for v in rule_votes if v["signal"] == 1)
            rule_bear = sum(1 for v in rule_votes if v["signal"] == 0)
            rule_signal = 1 if rule_bull >= rule_bear else 0

        return AggregatedSignal(
            ticker=ticker,
            consensus_signal=consensus,
            bullish_count=sum(1 for v in votes if v["signal"] == 1),
            bearish_count=sum(1 for v in votes if v["signal"] == 0),
            total_strategies=len(votes),
            confidence_pct=confidence,
            top_strategy=top_strategy,
            top_strategy_sharpe=top_weight,
            ml_signal=ml_signal,
            rule_signal=rule_signal,
            raw_votes=votes,
        )

