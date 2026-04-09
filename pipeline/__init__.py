"""
pipeline/ — 训练管道子包

公开 API:
  step1_ensure_data       — 数据就绪检查
  step2_train             — 超参搜索（native / optuna）
  train_portfolio_tickers — 分层混合组合训练
  _save_factor            — 因子保存 + 注册
  _latest_factor_path     — 最新因子路径
  _print_portfolio_summary — 汇总打印
"""

from pipeline.data_prep import step1_ensure_data, _ensure_hk_data, _hist_data_is_stale
from pipeline.select import _save_factor, _next_factor_run_id, _latest_factor_path
from pipeline.train import step2_train, step2_train_native, step2_train_optuna
from pipeline.train_portfolio import (
    train_portfolio_tickers,
    _train_meta_model,
    _print_portfolio_summary,
)

__all__ = [
    "step1_ensure_data",
    "_ensure_hk_data",
    "_hist_data_is_stale",
    "_save_factor",
    "_next_factor_run_id",
    "_latest_factor_path",
    "step2_train",
    "step2_train_native",
    "step2_train_optuna",
    "train_portfolio_tickers",
    "_train_meta_model",
    "_print_portfolio_summary",
]
