"""
速率限制器
==========
per-vendor 的令牌桶限速器，防止批量下载触发数据源限流。

对标 VN.py RestClient 的速率限制设计。

Usage::

    limiter = RateLimiter(max_calls=5, period=60)  # 5 次/分钟
    limiter.acquire()  # 阻塞直到有令牌可用
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    令牌桶速率限制器。

    Args:
        max_calls: 周期内最大调用次数
        period: 周期长度（秒）
        name: 限制器名称（用于日志）
    """

    def __init__(self, max_calls: int = 10, period: float = 60.0, name: str = "default") -> None:
        self.max_calls = max_calls
        self.period = period
        self.name = name
        self._lock = threading.Lock()
        self._calls: list[float] = []

    def acquire(self, timeout: Optional[float] = None) -> bool:
        """
        获取一个令牌。如果超过速率限制则阻塞等待。

        Args:
            timeout: 最大等待时间（秒），None 表示无限等待

        Returns:
            True 如果成功获取令牌，False 如果超时
        """
        start = time.monotonic()
        while True:
            with self._lock:
                now = time.monotonic()
                # 清理过期的调用记录
                cutoff = now - self.period
                self._calls = [t for t in self._calls if t > cutoff]

                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return True

                # 计算需要等待的时间
                wait_until = self._calls[0] + self.period
                wait_time = wait_until - now

            # 检查超时
            if timeout is not None:
                elapsed = time.monotonic() - start
                if elapsed + wait_time > timeout:
                    logger.warning(f"[RateLimiter:{self.name}] 等待超时 ({timeout}s)")
                    return False

            logger.debug(f"[RateLimiter:{self.name}] 等待 {wait_time:.1f}s…")
            time.sleep(min(wait_time, 1.0))  # 每次最多睡 1s，然后重新检查

    @property
    def available(self) -> int:
        """当前可用令牌数。"""
        with self._lock:
            now = time.monotonic()
            cutoff = now - self.period
            active = sum(1 for t in self._calls if t > cutoff)
            return max(0, self.max_calls - active)

    def __repr__(self) -> str:
        return f"RateLimiter(name={self.name}, max={self.max_calls}/{self.period}s, avail={self.available})"


# ── 默认 vendor 限流配置 ────────────────────────────────────────

_DEFAULT_LIMITS: Dict[str, Dict] = {
    "yfinance":         {"max_calls": 30, "period": 60},    # 30 req/min
    "yahooquery":       {"max_calls": 30, "period": 60},    # 30 req/min
    "pandas_datareader": {"max_calls": 20, "period": 60},   # 20 req/min
    "akshare":          {"max_calls": 20, "period": 60},    # 20 req/min
    "alpha_vantage":    {"max_calls": 5,  "period": 60},    # 5 req/min (free tier)
}

# ── 全局 limiter 注册表 ─────────────────────────────────────────
_limiters: Dict[str, RateLimiter] = {}
_global_lock = threading.Lock()


def get_limiter(vendor_name: str, custom_limits: Optional[Dict] = None) -> RateLimiter:
    """
    获取指定 vendor 的速率限制器（单例，线程安全）。

    Args:
        vendor_name: 数据源名称
        custom_limits: 自定义限流配置 {"max_calls": int, "period": float}

    Returns:
        RateLimiter 实例
    """
    with _global_lock:
        if vendor_name not in _limiters:
            limits = custom_limits or _DEFAULT_LIMITS.get(vendor_name, {"max_calls": 30, "period": 60})
            _limiters[vendor_name] = RateLimiter(
                max_calls=limits["max_calls"],
                period=limits["period"],
                name=vendor_name,
            )
        return _limiters[vendor_name]


def reset_all_limiters() -> None:
    """重置所有限制器（主要用于测试）。"""
    with _global_lock:
        _limiters.clear()

