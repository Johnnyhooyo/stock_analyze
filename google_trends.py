"""
Google Trends 数据获取模块
用于获取股票的搜索热度数据

数据持久化：每天只获取一次，保存到本地文件
"""

import pandas as pd
from datetime import datetime
from pathlib import Path
import time
import random

# Google Trends 数据目录
DATA_DIR = Path(__file__).parent / 'data' / 'trends'
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 数据文件路径
TRENDS_FILE = DATA_DIR / "tencent_trends.csv"


def get_google_trends(
    keyword: str = "腾讯",
    timeframe: str = 'today 3-m',
    geo: str = 'HK',
    refresh: bool = False
) -> pd.DataFrame:
    """
    获取 Google Trends 数据（优先读取本地缓存）

    Args:
        keyword: 搜索关键词
        timeframe: 时间范围
        geo: 地区代码
        refresh: 是否强制刷新

    Returns:
        DataFrame with date and trend index
    """
    # 检查本地缓存
    if not refresh and TRENDS_FILE.exists():
        try:
            df = pd.read_csv(TRENDS_FILE, index_col=0, parse_dates=True)
            # 检查是否今天的数据
            if not df.empty:
                last_date = df.index[-1]
                today = datetime.now().date()
                if last_date.date() >= today:
                    return df
        except Exception:
            pass

    # 尝试获取新数据
    try:
        from pytrends.request import TrendReq
        pytrends = TrendReq(hl='zh-CN', tz=360)
        time.sleep(random.uniform(1, 2))

        pytrends.build_payload(
            kw_list=[keyword],
            cat=0,
            timeframe=timeframe,
            geo=geo,
            gprop=''
        )

        interest = pytrends.interest_over_time()

        if interest is not None and not interest.empty:
            df = interest[[keyword]].copy()
            df.columns = ['trend']
            df.index.name = 'date'

            # 保存到本地文件
            df.to_csv(TRENDS_FILE)
            return df

    except Exception as e:
        print(f"  ⚠️ Google Trends 获取失败: {e}")

    # 返回空DataFrame
    return pd.DataFrame()


def get_tencent_trends(refresh: bool = False) -> pd.DataFrame:
    """获取腾讯控股的 Google Trends 数据"""
    return get_google_trends(
        keyword="腾讯",
        timeframe='today 3-m',
        geo='HK',
        refresh=refresh
    )


def get_trends_with_price(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    将 Google Trends 数据与价格数据合并

    Args:
        price_df: 价格数据（需要 Close 列）

    Returns:
        添加了 trend 列的 DataFrame
    """
    # 获取热度数据
    trends = get_tencent_trends()

    if trends.empty:
        # 使用备选方案：基于价格波动
        return _add_fallback_trend(price_df)

    # 合并数据
    df = price_df.copy()
    df['trend'] = trends['trend']

    # 前向填充
    df['trend'] = df['trend'].ffill()

    return df


def _add_fallback_trend(price_df: pd.DataFrame) -> pd.DataFrame:
    """备选方案：基于价格波动和成交量计算热度"""
    df = price_df.copy()

    # 价格波动
    df['returns'] = df['Close'].pct_change().abs()

    # 成交量（如果有）
    if 'Volume' in df.columns:
        df['volume_norm'] = df['Volume'] / df['Volume'].rolling(20).mean()
    else:
        df['volume_norm'] = 1

    # 计算热度（归一化到 0-100）
    df['trend'] = (df['returns'] * df['volume_norm'] * 100).clip(0, 100)
    df['trend'] = df['trend'].fillna(0)

    return df


if __name__ == "__main__":
    # 测试
    print("获取腾讯 Google Trends 数据...")
    trends = get_tencent_trends()
    print(f"数据条数: {len(trends)}")
    if not trends.empty:
        print(trends.tail(10))
