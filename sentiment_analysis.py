"""
情感分析模块
用于分析新闻、研报等文本数据的情感倾向

数据持久化：每天只获取一次，保存到本地文件
"""

import pandas as pd
from typing import Optional
from datetime import datetime
from pathlib import Path

# 情感数据目录
DATA_DIR = Path(__file__).parent / 'data' / 'sentiment'
DATA_DIR.mkdir(parents=True, exist_ok=True)

# 数据文件路径
SENTIMENT_FILE = DATA_DIR / "sentiment_cache.csv"

# 内存缓存
SENTIMENT_CACHE = {}


def _save_sentiment_cache(ticker: str, data: dict):
    """保存情感数据到本地文件"""
    df = pd.DataFrame([{
        'ticker': ticker,
        'date': datetime.now().date(),
        'sentiment': data.get('sentiment', 'neutral'),
        'polarity': data.get('polarity', 0),
        'news_count': data.get('news_count', 0),
        'positive_count': data.get('positive_count', 0),
        'negative_count': data.get('negative_count', 0),
        'neutral_count': data.get('neutral_count', 0),
    }])

    if SENTIMENT_FILE.exists():
        # 读取已有数据，去除当天的旧数据
        try:
            existing = pd.read_csv(SENTIMENT_FILE, parse_dates=['date'])
            existing = existing[existing['date'] != str(datetime.now().date())]
            df = pd.concat([existing, df], ignore_index=True)
        except Exception:
            pass

    df.to_csv(SENTIMENT_FILE, index=False)


def _load_sentiment_cache(ticker: str) -> Optional[dict]:
    """从本地文件加载情感数据"""
    if not SENTIMENT_FILE.exists():
        return None

    try:
        df = pd.read_csv(SENTIMENT_FILE, parse_dates=['date'])
        today = str(datetime.now().date())
        row = df[(df['ticker'] == ticker) & (df['date'] == today)]

        if not row.empty:
            return {
                'sentiment': row.iloc[0]['sentiment'],
                'polarity': float(row.iloc[0]['polarity']),
                'news_count': int(row.iloc[0]['news_count']),
                'positive_count': int(row.iloc[0]['positive_count']),
                'negative_count': int(row.iloc[0]['negative_count']),
                'neutral_count': int(row.iloc[0]['neutral_count']),
                'latest_news': []  # 新闻详情不缓存
            }
    except Exception:
        pass

    return None


def _is_mostly_chinese(text: str) -> bool:
    """判断文本是否以中文为主（CJK 字符占比 > 30%）"""
    if not text:
        return False
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    return cjk / max(len(text), 1) > 0.3


def analyze_sentiment(text: str) -> dict:
    """
    分析文本情感。
    - 中文文本：优先使用 SnowNLP，回退到关键词匹配
    - 英文文本：使用 TextBlob，回退到关键词匹配
    注：情感分析结果为参考信息，不直接参与交易决策（见 config.yaml sentiment_weight）
    """
    if _is_mostly_chinese(text):
        # ── 中文路径：SnowNLP ─────────────────────────────────────
        try:
            from snownlp import SnowNLP
            s = SnowNLP(text)
            polarity = float(s.sentiments) * 2 - 1  # [0,1] → [-1,1]
            if polarity > 0.1:
                label = "positive"
            elif polarity < -0.1:
                label = "negative"
            else:
                label = "neutral"
            return {'polarity': polarity, 'label': label, 'subjectivity': 0.5}
        except ImportError:
            pass  # SnowNLP 未安装，回退关键词匹配
        return _simple_sentiment(text)
    else:
        # ── 英文路径：TextBlob ────────────────────────────────────
        try:
            from textblob import TextBlob
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity  # -1 ~ 1
            if polarity > 0.1:
                label = "positive"
            elif polarity < -0.1:
                label = "negative"
            else:
                label = "neutral"
            return {
                'polarity': polarity,
                'label': label,
                'subjectivity': blob.sentiment.subjectivity
            }
        except ImportError:
            return _simple_sentiment(text)


def _simple_sentiment(text: str) -> dict:
    """
    简单的情感分析（基于关键词）
    """
    positive_words = ['涨', '上涨', '看好', '增持', '买入', '利多', '利好', '突破', '增长', '上升', '强势']
    negative_words = ['跌', '下跌', '看空', '减持', '卖出', '利空', '风险', '下跌', '下降', '弱势', '亏损']

    text_lower = text.lower()
    pos_count = sum(1 for w in positive_words if w in text)
    neg_count = sum(1 for w in negative_words if w in text)

    if pos_count > neg_count:
        polarity = min(0.5, (pos_count - neg_count) * 0.1)
        label = "positive"
    elif neg_count > pos_count:
        polarity = max(-0.5, (neg_count - pos_count) * -0.1)
        label = "negative"
    else:
        polarity = 0
        label = "neutral"

    return {
        'polarity': polarity,
        'label': label,
        'subjectivity': 0.5
    }


def fetch_stock_news(ticker: str = "0700.HK", max_results: int = 10) -> list:
    """
    获取股票新闻（使用 Yahoo Finance RSS）

    Args:
        ticker: 股票代码
        max_results: 最大结果数

    Returns:
        list: 新闻列表
    """
    news = []

    try:
        import yfinance as yf

        # 获取股票信息
        stock = yf.Ticker(ticker)
        news_data = stock.news

        if news_data:
            for item in news_data[:max_results]:
                try:
                    # 处理新版 yfinance 格式
                    content = item.get('content') or {}
                    title = content.get('title', '') or item.get('title', '')
                    summary = content.get('summary', '') or content.get('description', '') or item.get('summary', '')
                    provider = content.get('provider', {}).get('displayName', '') if isinstance(content.get('provider'), dict) else ''

                    if title:  # 只添加有标题的新闻
                        news.append({
                            'title': title,
                            'publisher': provider,
                            'link': content.get('clickThroughUrl', {}).get('url', '') if isinstance(content.get('clickThroughUrl'), dict) else '',
                            'pubDate': content.get('pubDate', ''),
                            'summary': summary
                        })
                except Exception:
                    continue
    except Exception as e:
        print(f"  ⚠️ 获取新闻失败: {e}")

    return news


def analyze_stock_sentiment(ticker: str = "0700.HK", force_refresh: bool = False) -> dict:
    """
    分析股票情感倾向

    Args:
        ticker: 股票代码
        force_refresh: 是否强制刷新

    Returns:
        dict: 情感分析结果
    """
    global SENTIMENT_CACHE

    cache_key = f"{ticker}_{datetime.now().date()}"

    # 1. 先检查内存缓存
    if not force_refresh and cache_key in SENTIMENT_CACHE:
        return SENTIMENT_CACHE[cache_key]

    # 2. 检查本地文件缓存
    if not force_refresh:
        cached = _load_sentiment_cache(ticker)
        if cached:
            # 补充新闻数据（需要实时获取）
            news = fetch_stock_news(ticker)
            if news:
                # 分析新闻
                analyzed_news = []
                for item in news[:5]:
                    text = item.get('title', '') + ' ' + item.get('summary', '')
                    result = analyze_sentiment(text)
                    analyzed_news.append({
                        'title': item.get('title', '')[:50] + '...' if len(item.get('title', '')) > 50 else item.get('title', ''),
                        'sentiment': result['label'],
                        'polarity': result['polarity']
                    })
                cached['latest_news'] = analyzed_news
            SENTIMENT_CACHE[cache_key] = cached
            return cached

    # 获取新闻
    news = fetch_stock_news(ticker)

    if not news:
        return {
            'sentiment': 'neutral',
            'polarity': 0,
            'news_count': 0,
            'positive_count': 0,
            'negative_count': 0,
            'neutral_count': 0,
            'latest_news': []
        }

    # 分析每条新闻
    polarities = []
    labels = []
    analyzed_news = []

    for item in news:
        text = item.get('title', '') + ' ' + item.get('summary', '')
        result = analyze_sentiment(text)
        polarities.append(result['polarity'])
        labels.append(result['label'])

        analyzed_news.append({
            'title': item.get('title', '')[:50] + '...' if len(item.get('title', '')) > 50 else item.get('title', ''),
            'sentiment': result['label'],
            'polarity': result['polarity']
        })

    # 统计
    positive_count = labels.count('positive')
    negative_count = labels.count('negative')
    neutral_count = labels.count('neutral')
    avg_polarity = sum(polarities) / len(polarities) if polarities else 0

    # 综合判断
    if avg_polarity > 0.1:
        sentiment = "positive"
    elif avg_polarity < -0.1:
        sentiment = "negative"
    else:
        sentiment = "neutral"

    result = {
        'sentiment': sentiment,
        'polarity': avg_polarity,
        'news_count': len(news),
        'positive_count': positive_count,
        'negative_count': negative_count,
        'neutral_count': neutral_count,
        'latest_news': analyzed_news[:5]
    }

    # 缓存结果到内存和文件
    SENTIMENT_CACHE[cache_key] = result
    _save_sentiment_cache(ticker, result)

    return result


def get_sentiment_signal(sentiment_result: dict) -> int:
    """
    将情感分析结果转换为交易信号

    Args:
        sentiment_result: 情感分析结果

    Returns:
        int: 1=看涨, 0=看跌/观望
    """
    polarity = sentiment_result.get('polarity', 0)
    sentiment = sentiment_result.get('sentiment', 'neutral')

    # 情感极强时给出明确信号
    if polarity > 0.3:
        return 1  # 强烈看涨
    elif polarity < -0.3:
        return 0  # 强烈看跌
    else:
        # 中性或轻微情感时不改变现有立场
        return -1  # 无信号


if __name__ == "__main__":
    print("分析腾讯控股情感倾向...")

    result = analyze_stock_sentiment("0700.HK")

    print(f"\n情感倾向: {result['sentiment']}")
    print(f"情感分数: {result['polarity']:.3f}")
    print(f"新闻数量: {result['news_count']}")
    print(f"正面: {result['positive_count']} | 负面: {result['negative_count']} | 中性: {result['neutral_count']}")

    print("\n最新新闻:")
    for news in result['latest_news'][:3]:
        emoji = "🟢" if news['sentiment'] == "positive" else "🔴" if news['sentiment'] == "negative" else "⚪"
        print(f"  {emoji} {news['title']}")
