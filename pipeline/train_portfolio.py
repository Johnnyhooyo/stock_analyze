"""
pipeline/train_portfolio.py — 分层混合组合训练

包含:
  train_portfolio_tickers() — ML全局训练一次 + 每只股票规则训练
  _train_meta_model()       — Stacking 第二层 meta-model 训练
  _print_portfolio_summary() — 汇总打印 + 可选飞书通知
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

from log_config import get_logger
from config_loader import load_config, ticker_to_safe
from pipeline.data_prep import step1_ensure_data, _ensure_hk_data
from pipeline.train import step2_train
from pipeline.select import _latest_factor_path

logger = get_logger(__name__)


def _train_meta_model(
    ticker: str,
    data,
    config: dict,
    factors_dir: Path,
    meta_dir: Path,
) -> None:
    """
    训练并保存 meta-model（Stacking 第二层 Logistic Regression）。
    在基础策略训练完成、factor_*.pkl 已保存后调用。
    失败时记录 warning，不中断主流程。
    """
    try:
        from engine.meta_aggregator import MetaAggregator
        from engine.signal_aggregator import SignalAggregator

        agg = SignalAggregator(factors_dir=factors_dir, use_registry=False)
        artifacts = agg._load_factors()

        if len(artifacts) < 2:
            logger.warning(
                "meta-model 训练跳过：active 因子数量不足（需要 >= 2，当前 %d）",
                len(artifacts),
                extra={"ticker": ticker},
            )
            return

        stacking_cfg = config.get("stacking", {})
        n_splits = int(stacking_cfg.get("n_splits", 5))
        label_days = int(stacking_cfg.get("label_days", 5))

        ma = MetaAggregator(meta_dir=meta_dir)
        metrics = ma.train(
            ticker=ticker,
            data=data,
            artifacts=artifacts,
            config=config,
            n_splits=n_splits,
            label_days=label_days,
        )
        ma.save(ticker)
        logger.info(
            "meta-model 训练完成",
            extra={"ticker": ticker, **metrics},
        )
    except Exception as e:
        logger.warning(
            "meta-model 训练失败，不影响基础策略: %s", e,
            extra={"ticker": ticker},
        )


def train_portfolio_tickers(
    tickers: list[str],
    use_optuna: bool = False,
    optuna_trials: int = 50,
    sources_override: list[str] = None,
    skip_download: bool = False,
) -> list[dict]:
    """
    分层混合训练：
      1. ML 全局训练（strategy_type='multi'）运行一次 → data/factors/
      2. 对每只 ticker 跑规则策略训练（strategy_type='single'）→ data/factors/{TICKER_SAFE}/
    返回每只 ticker 的结果列表。
    """
    config = load_config()
    if not skip_download:
        _ensure_hk_data(config)

    base_factors_dir = Path(__file__).parent.parent / 'data' / 'factors'
    results: list[dict] = []

    # ── 步骤 A：ML 全局训练（一次）────────────────────────────────
    ref_ticker = tickers[0] if tickers else config.get('ticker', '0700.HK')
    logger.info("ML全局训练开始（使用 %s 作为验证参考）", ref_ticker)
    ml_status = "ok"
    try:
        ref_hist, _ = step1_ensure_data(
            sources_override=sources_override, ticker=ref_ticker, skip_download=skip_download
        )
        step2_train(
            ref_hist,
            use_optuna=use_optuna,
            optuna_trials=optuna_trials,
            strategy_type='multi',
            factors_dir_override=None,
            ticker=ref_ticker,
        )
        logger.info("ML全局训练完成，因子已存入 %s", base_factors_dir)
    except Exception as e:
        ml_status = "failed"
        logger.warning("ML全局训练失败（将继续进行规则策略训练）: %s", e)

    # ── 步骤 B：每只 ticker 的规则策略训练 ────────────────────────
    for ticker in tickers:
        ticker_safe = ticker_to_safe(ticker)
        ticker_factors_dir = base_factors_dir / ticker_safe

        logger.info("规则策略训练: %s → %s", ticker, ticker_factors_dir,
                    extra={"ticker": ticker})

        try:
            hist_data, _ = step1_ensure_data(
                sources_override=sources_override, ticker=ticker, skip_download=skip_download
            )
        except Exception as e:
            logger.error("数据获取失败，跳过 %s: %s", ticker, e, extra={"ticker": ticker})
            results.append({"ticker": ticker, "status": "data_failed",
                             "factor_path": None, "error": str(e)})
            continue

        try:
            factor_path, best_result, _ = step2_train(
                hist_data,
                use_optuna=use_optuna,
                optuna_trials=optuna_trials,
                strategy_type='single',
                factors_dir_override=ticker_factors_dir,
                ticker=ticker,
            )
        except Exception as e:
            logger.error("规则训练失败，跳过 %s: %s", ticker, e, extra={"ticker": ticker})
            results.append({"ticker": ticker, "status": "train_failed",
                             "factor_path": None, "error": str(e)})
            continue

        if factor_path is None:
            factor_path = _latest_factor_path(ticker_factors_dir)

        sharpe    = best_result.get('sharpe_ratio', float('nan')) if best_result else float('nan')
        validated = best_result.get('validated', 'unknown') if best_result else 'unknown'

        results.append({
            "ticker":      ticker,
            "status":      "ok",
            "factor_path": factor_path,
            "factors_dir": str(ticker_factors_dir),
            "sharpe_ratio": sharpe,
            "validated":   validated,
            "ml_status":   ml_status,
        })

        _train_meta_model(
            ticker=ticker,
            data=hist_data,
            config=config,
            factors_dir=ticker_factors_dir,
            meta_dir=Path(__file__).parent.parent / "data" / "meta",
        )
        logger.info("规则训练完成: %s", ticker,
                    extra={"ticker": ticker, "sharpe_ratio": sharpe})

    return results


def _print_portfolio_summary(results: list[dict], config: dict) -> None:
    """打印组合训练汇总表，可选飞书通知。"""
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
            sh_str = f"{sh:8.4f}" if not math.isnan(sh) else '     N/A'
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
                    f"- {r['ticker']}  Sharpe={'N/A' if math.isnan(sh) else f'{sh:.4f}'}"
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
