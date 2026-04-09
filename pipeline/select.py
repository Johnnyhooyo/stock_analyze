"""
pipeline/select.py — 因子保存与选优

包含:
  _next_factor_run_id()  — 自动递增因子编号
  _save_factor()         — 保存因子 .pkl 并注册到 FactorRegistry
  _latest_factor_path()  — 返回最新因子路径
"""

from __future__ import annotations

import joblib
from datetime import datetime
from pathlib import Path

from log_config import get_logger
from data.factor_registry import FactorRegistry, _get_training_type

logger = get_logger(__name__)


def _next_factor_run_id(factors_dir: Path) -> int:
    """
    返回下一个可用的因子文件编号（从 1 开始）。
    优先读取计数器文件（O(1)），避免随文件增多的 glob 线性扫描。
    如计数器文件不存在则从目录扫描初始化（向后兼容）。
    """
    factors_dir.mkdir(parents=True, exist_ok=True)
    counter_path = factors_dir / '.run_id_counter'
    try:
        if counter_path.exists():
            current = int(counter_path.read_text().strip())
            next_id = current + 1
        else:
            existing = list(factors_dir.glob('factor_*.pkl'))
            if existing:
                ids = []
                for p in existing:
                    try:
                        ids.append(int(p.stem.split('_')[1]))
                    except (IndexError, ValueError):
                        pass
                next_id = (max(ids) + 1) if ids else 1
            else:
                next_id = 1
        counter_path.write_text(str(next_id))
        return next_id
    except Exception as e:
        logger.debug("run_id 计数器读取失败，降级为全量扫描", extra={"error": str(e)})
        existing = list(factors_dir.glob('factor_*.pkl'))
        if not existing:
            return 1
        ids = []
        for p in existing:
            try:
                ids.append(int(p.stem.split('_')[1]))
            except (IndexError, ValueError):
                pass
        return (max(ids) + 1) if ids else 1


def _save_factor(result: dict, factors_dir: Path) -> str:
    """
    将搜索结果统一保存为 factor_{run_id:04d}.pkl，返回保存路径字符串。
    run_id 自动递增，每次调用只写一个文件。
    保存后自动将因子注册到 factor_registry.json。
    """
    factors_dir.mkdir(parents=True, exist_ok=True)
    run_id    = _next_factor_run_id(factors_dir)
    save_path = factors_dir / f"factor_{run_id:04d}.pkl"

    model = result.get('model')
    if model is None:
        model = {}

    joblib.dump(
        {
            'model':              model,
            'meta':               result.get('meta', {}),
            'config':             result.get('config', {}),
            'run_id':             run_id,
            'cum_return':         result.get('cum_return', 0),
            'annualized_return':  result.get('annualized_return', float('nan')),
            'sharpe_ratio':       result.get('sharpe_ratio', float('nan')),
            'max_drawdown':       result.get('max_drawdown', float('nan')),
            'volatility':         result.get('volatility', float('nan')),
            'win_rate':           result.get('win_rate', 0),
            'profit_loss_ratio':  result.get('profit_loss_ratio', 0),
            'calmar_ratio':       result.get('calmar_ratio', float('nan')),
            'sortino_ratio':      result.get('sortino_ratio', float('nan')),
            'total_trades':       result.get('total_trades', 0),
            'rank_ic':            result.get('rank_ic', float('nan')),
            'validated':          result.get('validated', 'unknown'),
            'holdout':            result.get('holdout', {}),
            'wf_summary':         result.get('wf_summary', {}),
            'val_period':         result.get('val_period', ''),
            'saved_at':           datetime.now().isoformat(),
        },
        save_path,
    )

    try:
        meta = result.get('meta', {})
        config = result.get('config', {})
        strategy_name = meta.get('name', 'unknown')
        ticker = config.get('ticker')
        training_type = _get_training_type(strategy_name)
        subdir = None
        if factors_dir.name.endswith('_HK') or factors_dir.name.endswith('_HK_s'):
            subdir = factors_dir.name
        elif factors_dir != (Path(__file__).parent.parent / 'data' / 'factors'):
            subdir = factors_dir.name

        registry = FactorRegistry()
        registry.register(
            factor_id=run_id,
            filename=f"factor_{run_id:04d}.pkl",
            subdir=subdir,
            strategy_name=strategy_name,
            ticker=ticker,
            training_type=training_type,
            sharpe_ratio=result.get('sharpe_ratio', 0.0),
            cum_return=result.get('cum_return', 0.0),
            max_drawdown=result.get('max_drawdown', 0.0),
            total_trades=result.get('total_trades', 0),
        )
    except Exception as e:
        logger.warning("因子注册失败（非阻塞）", extra={"factor": save_path.name, "error": str(e)})

    return str(save_path)


def _latest_factor_path(factors_dir: Path) -> str | None:
    """返回 factors_dir 下编号最大的 factor_XXXX.pkl 路径，不存在则 None。"""
    candidates = list(factors_dir.glob('factor_*.pkl'))
    if not candidates:
        return None
    return str(max(candidates, key=lambda p: int(p.stem.split('_')[1])))
