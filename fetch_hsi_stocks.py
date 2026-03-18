"""
恒生指数成分股数据下载模块
下载所有恒生指数成分股的历史数据
"""

import yfinance as yf
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime, timedelta
import time
import logging

# Setup logging
LOG_DIR = Path(__file__).parent / 'data' / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / 'fetch_hsi.log'
logger = logging.getLogger('fetch_hsi')
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)


# 恒生指数成分股列表 (2024年)
# 包含主要蓝筹股
HSI_STOCKS = [
    # 金融/银行
    '0005.HK',  # 汇丰控股
    '0941.HK',  # 中国移动
    '0939.HK',  # 建设银行
    '0945.HK',  # 中国平安
    '2628.HK',  # 中国人寿
    '0388.HK',  # 香港交易所
    '3988.HK',  # 中国银行
    '3968.HK',  # 招商银行
    '6820.HK',  # 友邦保险
    '0003.HK',  # 香港中华煤气
    '0002.HK',  # 中电控股

    # 科技/互联网
    '0700.HK',  # 腾讯控股
    '9988.HK',  # 阿里巴巴-SW
    '3690.HK',  # 美团-W
    '9618.HK',  # 京东集团-SW
    '9888.HK',  # 百度集团-SW
    '9961.HK',  # 携程集团-SW
    '1810.HK',  # 小米集团-W

    # 地产
    '0016.HK',  # 新鸿基地产
    '0017.HK',  # 新世界发展
    '1109.HK',  # 华润置地
    '0688.HK',  # 中国海外发展
    '0823.HK',  # 领展房产基金

    # 消费
    '0027.HK',  # 银河娱乐
    '2319.HK',  # 蒙牛乳业
    '2269.HK',  # 农夫山泉
    '2020.HK',  # 安踏体育

    # 医药
    '1177.HK',  # 中国生物制药
    '2266.HK',  # 威高股份

    # 能源
    '0883.HK',  # 中国海洋石油
    '0857.HK',  # 中国石油

    # 工业/制造
    '1211.HK',  # 比亚迪股份
    '0175.HK',  # 吉利汽车
    '2382.HK',  # 舜宇光学科技

    # 公用事业
    '0066.HK',  # 港铁公司
    '0686.HK',  # 中国电力

    # 基建
    '1038.HK',  # 长江基建

    # 综合
    '0001.HK',  # 长和
    '0019.HK',  # 太古股份

    # 新股/次新股 (近年上市)
    '6611.HK',  # 泡泡玛特
    '6618.HK',  # 京东健康
    '6837.HK',  # 海底捞

    # 补充更多成分股
    '0285.HK',  # 比亚迪电子
    '0188.HK',  # 中国燃气
    '0233.HK',  # 洛阳钼业
    '0014.HK',  # 希慎兴业
    '0177.HK',  # 恒安国际
    '0013.HK',  # 九龙仓集团
    '0116.HK',  # 维达国际
    '0728.HK',  # 中国电信
    '3808.HK',  # 中国重汽
    '3908.HK',  # 华润水泥
    '3320.HK',  # 华润医药
    '0175.HK',  # 吉利汽车
    '1052.HK',  # 越秀交通基建
    '0489.HK',  # 东风集团
    '0186.HK',  # 宝龙地产
    '0012.HK',  # 恒生银行
    '0004.HK',  # 九龙仓置业
    '0813.HK',  # 嘉里建设
    '0011.HK',  # 恒生银行
    '0038.HK',  # 恒生银行
    '0006.HK',  # 港交所
]

# 去重
HSI_STOCKS = list(set(HSI_STOCKS))
logger.info(f"恒生指数成分股数量: {len(HSI_STOCKS)}")


def get_hsi_stocks():
    """获取恒生指数成分股列表"""
    return HSI_STOCKS


def download_hsi_stock(
    ticker: str,
    period: str = '5y',
    retry: int = 3
) -> pd.DataFrame:
    """
    下载单只股票数据

    Args:
        ticker: 股票代码
        period: 数据周期
        retry: 重试次数

    Returns:
        DataFrame 或 None
    """
    # 生成 ticker 变体
    variants = [ticker]
    if '.' not in ticker:
        variants.append(f"{ticker}.HK")

    # 方法1: 尝试 yfinance
    for variant in variants:
        for attempt in range(retry):
            try:
                data = yf.download(
                    variant,
                    period=period,
                    progress=False,
                    threads=False
                )
                if data is not None and not data.empty:
                    return data
            except Exception as e:
                if attempt < retry - 1:
                    time.sleep(1)
                    continue

    # 方法2: 尝试 akshare
    try:
        import akshare as ak

        # 提取数字代码
        symbol = ticker.split('.')[0] if '.' in ticker else ticker
        # 去掉前导0
        if len(symbol) > 4:
            symbol = symbol.lstrip('0')

        df = ak.stock_hk_daily(symbol=symbol)
        if df is not None and not df.empty:
            # 转换列名
            col_map = {}
            for c in df.columns:
                lc = str(c).lower()
                if '开' in lc:
                    col_map[c] = 'Open'
                elif '高' in lc:
                    col_map[c] = 'High'
                elif '低' in lc:
                    col_map[c] = 'Low'
                elif '收' in lc:
                    col_map[c] = 'Close'
                elif '量' in lc or '成交' in lc:
                    col_map[c] = 'Volume'
            df = df.rename(columns=col_map)

            # 设置日期索引
            if '日期' in df.columns:
                df.index = pd.to_datetime(df['日期'])
                df = df.drop(columns=['日期'])
            elif 'date' in df.columns:
                df.index = pd.to_datetime(df['date'])
                df = df.drop(columns=['date'])

            # 只保留需要的列
            cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            available_cols = [c for c in cols if c in df.columns]
            df = df[available_cols]

            return df
    except Exception as e:
        pass

    # 方法3: 尝试 yahooquery
    try:
        from yahooquery import Ticker as YQ
        yq = YQ(ticker)
        hist = yq.history(period=period)
        if hist is not None and not hist.empty:
            if isinstance(hist.index, pd.MultiIndex):
                hist = hist.xs(ticker, level=0) if ticker in hist.index.get_level_values(0) else hist.iloc[0]
            return hist
    except Exception as e:
        pass

    return None


def download_all_hsi(
    period: str = '5y',
    out_dir: Path = None,
    delay: float = 0.5,
    skip_existing: bool = True
) -> dict:
    """
    下载所有恒生指数成分股数据

    Args:
        period: 数据周期
        out_dir: 输出目录
        delay: 请求间隔(秒)
        skip_existing: 是否跳过已存在的文件

    Returns:
        下载结果统计
    """
    if out_dir is None:
        out_dir = Path(__file__).parent / 'data' / 'historical' / 'hsi'
    out_dir.mkdir(parents=True, exist_ok=True)

    stocks = get_hsi_stocks()
    total = len(stocks)
    success = 0
    failed = []

    logger.info(f"开始下载 {total} 只恒生指数成分股数据...")

    for i, ticker in enumerate(stocks, 1):
        # 检查是否已存在
        if skip_existing:
            file_path = out_dir / f"{ticker.replace('.', '_')}_{period}.csv"
            if file_path.exists():
                logger.info(f"[{i}/{total}] 跳过 {ticker} (已存在)")
                success += 1
                continue

        logger.info(f"[{i}/{total}] 下载 {ticker}...")
        data = download_hsi_stock(ticker, period)

        if data is not None and not data.empty:
            # 标准化列名
            try:
                data.columns = [c[0] if isinstance(c, tuple) else c for c in data.columns]
                if 'Close' not in data.columns and 'close' in data.columns:
                    data = data.rename(columns={'close': 'Close'})
                if 'Volume' not in data.columns and 'volume' in data.columns:
                    data = data.rename(columns={'volume': 'Volume'})
            except:
                pass

            # 保存
            file_path = out_dir / f"{ticker.replace('.', '_')}_{period}.csv"
            try:
                data.to_csv(file_path)
                success += 1
                logger.info(f"  -> 成功: {len(data)} 条记录")
            except Exception as e:
                failed.append(ticker)
                logger.warning(f"  -> 保存失败: {e}")
        else:
            failed.append(ticker)
            logger.warning(f"  -> 无数据")

        # 延迟，避免请求过快
        time.sleep(delay)

    result = {
        'total': total,
        'success': success,
        'failed': failed,
        'out_dir': str(out_dir)
    }

    logger.info(f"\n下载完成!")
    logger.info(f"  总数: {total}")
    logger.info(f"  成功: {success}")
    logger.info(f"  失败: {len(failed)}")
    if failed:
        logger.info(f"  失败列表: {failed}")

    return result


def get_hsi_data(ticker: str, period: str = '5y') -> pd.DataFrame:
    """
    读取本地恒生指数成分股数据

    Args:
        ticker: 股票代码
        period: 数据周期

    Returns:
        DataFrame
    """
    out_dir = Path(__file__).parent / 'data' / 'historical' / 'hsi'
    file_path = out_dir / f"{ticker.replace('.', '_')}_{period}.csv"

    if file_path.exists():
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        return data
    else:
        # 尝试下载
        data = download_hsi_stock(ticker, period)
        if data is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            data.to_csv(file_path)
        return data if data is not None else pd.DataFrame()


def list_hsi_files(period: str = '5y') -> list:
    """列出所有已下载的恒生指数成分股文件"""
    out_dir = Path(__file__).parent / 'data' / 'historical' / 'hsi'
    if not out_dir.exists():
        return []
    return list(out_dir.glob(f"*_{period}.csv"))


if __name__ == "__main__":
    # 下载所有数据
    result = download_all_hsi(period='5y', delay=0.3)
    print(f"\n结果: 成功 {result['success']}/{result['total']}")
