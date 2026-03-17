import yfinance as yf
import pandas as pd
import yaml
from pathlib import Path
from datetime import datetime, timedelta, date, time
import re
import logging
import sys

# Setup logging to file and stdout
LOG_DIR = Path(__file__).parent / 'data' / 'logs'
LOG_DIR.mkdir(parents=True, exist_ok=True)
log_file = LOG_DIR / 'fetch.log'
logger = logging.getLogger('fetch_data')
if not logger.handlers:
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s %(levelname)s: %(message)s')
    fh.setFormatter(fmt)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)


def _parse_period_to_days(period):
    # 支持简单 period 解析: '1y', '6mo', '3mo', '30d'
    days = 90
    if isinstance(period, str):
        m = re.match(r"(\d+)y", period)
        if m:
            days = int(m.group(1)) * 252
        else:
            m = re.match(r"(\d+)mo", period)
            if m:
                days = int(m.group(1)) * 21
            else:
                m = re.match(r"(\d+)d", period)
                if m:
                    days = int(m.group(1))
    return days


def _try_pandas_datareader(ticker, period):
    try:
        from pandas_datareader import data as pdr
    except Exception as e:
        logger.info(f"pandas_datareader 未安装或导入失败: {e}")
        return None

    days = _parse_period_to_days(period)
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)
    # 尝试 yahoo，然后 stooq
    try:
        logger.info(f"尝试通过 pandas_datareader/yahoo 下载 {ticker} from {start_date.date()} to {end_date.date()}")
        df = pdr.DataReader(ticker, 'yahoo', start=start_date, end=end_date)
        if df is not None and not df.empty:
            df = df.sort_index()
            return df
    except Exception as e:
        logger.info(f"pandas_datareader yahoo 下载失败: {e}")
    try:
        logger.info(f"尝试通过 pandas_datareader/stooq 下载 {ticker} from {start_date.date()} to {end_date.date()}")
        df = pdr.DataReader(ticker, 'stooq', start=start_date, end=end_date)
        if df is not None and not df.empty:
            df = df.sort_index()
            return df
    except Exception as e:
        logger.info(f"pandas_datareader stooq 下载失败: {e}")
    return None


def _try_akshare(ticker, period):
    try:
        import akshare as ak
    except Exception as e:
        logger.info(f"akshare 未安装或导入失败: {e}")
        return None

    # Prepare symbol for akshare: strip known suffixes like .HK
    symbol = ticker
    if isinstance(ticker, str):
        if ticker.endswith('.HK') or ticker.endswith('.hk'):
            symbol = ticker.split('.')[0]

    days = _parse_period_to_days(period)
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)
    try:
        logger.info(f"尝试通过 akshare 下载 {ticker} (symbol={symbol})（回溯 {days} 天）")
        # Try several akshare functions that might return HK data
        df = None
        # Try common HK functions; wrap each call
        if hasattr(ak, 'stock_hk_daily'):
            try:
                df = ak.stock_hk_daily(symbol=symbol)
            except Exception as e:
                logger.info(f"ak.stock_hk_daily 调用失败: {e}")
        if (df is None or df.empty) and hasattr(ak, 'stock_zh_a_hist'):
            try:
                df = ak.stock_zh_a_hist(symbol=symbol)
            except Exception as e:
                logger.info(f"ak.stock_zh_a_hist 调用失败: {e}")
        if (df is None or df.empty) and hasattr(ak, 'stock_zh_a_daily'):
            try:
                df = ak.stock_zh_a_daily(symbol=symbol)
            except Exception as e:
                logger.info(f"ak.stock_zh_a_daily 调用失败: {e}")

        if df is None or df.empty:
            logger.info("akshare 未获取到数据")
            return None

        # If index already datetime-like, ensure sorted and return
        if pd.api.types.is_datetime64_any_dtype(df.index):
            df = df.sort_index()
            return df

        # Normalize columns with heuristics including Chinese names
        col_map = {}
        # Some DataFrame columns might be non-string (e.g., ints); convert to str first
        lower_cols = {str(c).lower(): c for c in df.columns}
        mapping_candidates = {
            'open': ['open', '开盘', '开盘价'],
            'high': ['high', '最高', '最高价'],
            'low': ['low', '最低', '最低价'],
            'close': ['close', '收盘', '收盘价', 'close*'],
            'volume': ['volume', '成交量', 'volume_traded'],
            'date': ['date', 'time', '交易日期', 'date_time', '日期']
        }
        for std, variants in mapping_candidates.items():
            for v in variants:
                if v in lower_cols:
                    col_map[lower_cols[v]] = std.capitalize()
                    break

        df = df.rename(columns=col_map)

        # If there is a date-like column, set it as index
        date_col = None
        for c in ['Date', 'date', 'Time', 'time', '交易日期', '日期']:
            if c in df.columns:
                date_col = c
                break
        if date_col is None:
            # try to find any column with date-like strings
            for c in df.columns:
                sample = df[c].dropna().astype(str)
                if not sample.empty:
                    try:
                        pd.to_datetime(sample.iloc[0])
                        date_col = c
                        break
                    except Exception:
                        continue
        if date_col is not None:
            try:
                df.index = pd.to_datetime(df[date_col])
                df = df.drop(columns=[date_col], errors='ignore')
            except Exception as e:
                logger.info(f"将列 {date_col} 转为日期索引失败: {e}")

        # As a last resort, if first column looks like date strings, try parsing it
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            first_col = df.columns[0] if len(df.columns) > 0 else None
            if first_col is not None:
                try:
                    df.index = pd.to_datetime(df[first_col])
                    df = df.drop(columns=[first_col], errors='ignore')
                except Exception:
                    pass

        df = df.sort_index()
        return df
    except Exception as e:
        logger.info(f"akshare 下载失败: {e}")
    return None


def _try_alpha_vantage(ticker, period, api_key):
    if not api_key:
        logger.info("Alpha Vantage 未提供 API key，跳过")
        return None
    try:
        import importlib
        module = importlib.import_module('alpha_vantage.timeseries')
        TimeSeries = getattr(module, 'TimeSeries', None)
        if TimeSeries is None:
            logger.info("alpha_vantage.timeseries.TimeSeries 未找到，跳过")
            return None
    except Exception as e:
        logger.info(f"alpha_vantage 库未安装或导入失败: {e}")
        return None

    try:
        logger.info(f"尝试通过 Alpha Vantage 下载 {ticker}")
        ts = TimeSeries(key=api_key, output_format='pandas')
        data, meta = ts.get_daily_adjusted(symbol=ticker, outputsize='compact')
        # AlphaVantage 返回列如 '1. open', '4. close' 等
        data = data.rename(columns=lambda s: s.split('. ')[1] if '. ' in s else s)
        data.index = pd.to_datetime(data.index)
        data = data.sort_index()
        return data
    except Exception as e:
        logger.info(f"Alpha Vantage 下载失败: {e}")
    return None


def _try_yahooquery(ticker, period):
    try:
        from yahooquery import Ticker as YQ
    except Exception as e:
        logger.info(f"yahooquery 未安装或导入失败: {e}")
        return None

    days = _parse_period_to_days(period)
    end_date = datetime.today()
    start_date = end_date - timedelta(days=days)
    try:
        logger.info(f"尝试通过 yahooquery 下载 {ticker} from {start_date.date()} to {end_date.date()}")
        yq = YQ(ticker)
        # yahooquery.history returns DataFrame (if multi-ticker may be multiindex)
        hist = yq.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
        if hist is None or (hasattr(hist, 'empty') and hist.empty):
            logger.info("yahooquery 未返回历史数据")
            return None
        # If returned is MultiIndex (ticker, date), take the ticker slice
        if isinstance(hist.index, pd.MultiIndex):
            # select last ticker level or given ticker
            try:
                df = hist.xs(ticker, level=0)
            except Exception:
                # try base ticker
                base = ticker.split('.')[0]
                if base in hist.index.get_level_values(0):
                    df = hist.xs(base, level=0)
                else:
                    # try first ticker
                    df = hist.groupby(level=0).first().iloc[0]
        else:
            df = hist
        # Ensure datetime index
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            try:
                df.index = pd.to_datetime(df.index)
            except Exception:
                pass
        df = df.sort_index()
        return df
    except Exception as e:
        logger.info(f"yahooquery 下载失败: {e}")
    return None


def _generate_ticker_variants(ticker):
    variants = [ticker]
    if isinstance(ticker, str):
        if '.' in ticker:
            base = ticker.split('.')[0]
            variants.append(base)
        else:
            # try with .HK suffix
            variants.append(f"{ticker}.HK")
            # zero-pad to 4 digits
            if ticker.isdigit():
                if len(ticker) < 4:
                    variants.append(ticker.zfill(4))
                    variants.append(ticker.zfill(4) + '.HK')
    # ensure uniqueness and keep order
    seen = set()
    out = []
    for v in variants:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _normalize_df(df):
    """Normalize DataFrame columns to standard OHLCV names and ensure datetime index."""
    if df is None:
        return df
    # If yfinance group_by returns MultiIndex columns, try to extract the first ticker
    try:
        if isinstance(df.columns, pd.MultiIndex):
            # pick first top-level key
            try:
                top = df.columns.levels[0][0]
                df = df[top].copy()
            except Exception:
                # fallback: take first second-level columns
                df = df.iloc[:, :]
    except Exception:
        pass

    # If dataframe has a 'date' column, set it as index
    if 'date' in df.columns:
        try:
            df.index = pd.to_datetime(df['date'])
            df = df.drop(columns=['date'], errors='ignore')
        except Exception:
            pass
    if 'Date' in df.columns:
        try:
            df.index = pd.to_datetime(df['Date'])
            df = df.drop(columns=['Date'], errors='ignore')
        except Exception:
            pass

    # Normalize column names using lowercase keys
    col_map = {}
    for c in list(df.columns):
        # guard against non-string column names
        lc = str(c).lower().strip().replace(' ', '').replace('_', '')
        if lc in ('open', 'openprice'):
            col_map[c] = 'Open'
        elif lc in ('high', 'highprice'):
            col_map[c] = 'High'
        elif lc in ('low', 'lowprice'):
            col_map[c] = 'Low'
        elif lc in ('close', 'closeprice') or lc.startswith('close'):
            col_map[c] = 'Close'
        elif lc in ('adjclose', 'adj_close', 'adjcloseprice'):
            col_map[c] = 'Adj Close'
        elif lc in ('volume', 'vol', 'volumetraded'):
            col_map[c] = 'Volume'
    if col_map:
        df = df.rename(columns=col_map)

    # As a last attempt, if 'close' exists in any case variant, rename it
    if 'Close' not in df.columns:
        for c in df.columns:
            if c.lower() == 'close':
                df = df.rename(columns={c: 'Close'})
                break

    # Ensure datetime index
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass

    return df


def download_stock_data(retries=3, backoff=1.0, sources_override=None):
    # 读取配置（相对于脚本位置，避免工作目录问题）
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    ticker = config.get('ticker')
    period = config.get('period', '1y')
    # allow runtime override of data source priority
    data_sources = sources_override if sources_override is not None else config.get('data_sources', ['yfinance', 'pandas_datareader', 'akshare', 'alpha_vantage'])
    alpha_key = config.get('alpha_vantage_key')

    if not ticker:
        raise ValueError('config.yaml 中缺少 ticker 配置')

    # --- 新逻辑: 优先使用本地历史数据（如果存在且不过期则直接返回），否则再去网络下载 ---
    out_dir = Path(__file__).parent / 'data' / 'historical'
    out_dir.mkdir(parents=True, exist_ok=True)
    # 默认保存路径（如果从网络获取成功，会保存到这里）
    file_path = out_dir / f"{ticker}_{period}.csv"

    # 先查找是否已有本地历史文件
    candidates = list(out_dir.glob(f"{ticker}_*.csv"))
    if candidates:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        file_path = candidates[0]
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)

        if not data.empty:
            last_date = data.index.max().date()
            now = datetime.now()

            # 判断逻辑：
            # - 18点前：当天数据不可用，需要上一个交易日（昨天或上周五）
            # - 18点后：当天数据可能可用，需要今天的数据
            if now.time() < time(18, 0):
                # 18点前：需要上一个交易日
                target = date.today() - timedelta(days=1)
                while target.weekday() >= 5:
                    target -= timedelta(days=1)
            else:
                # 18点后：需要今天的数据（如果是周末则用上一个交易日）
                target = date.today()
                if target.weekday() >= 5:
                    target -= timedelta(days=1)
                    while target.weekday() >= 5:
                        target -= timedelta(days=1)

            if last_date >= target:
                # 有最新数据
                logger.info(f"本地历史数据有效（最后交易日: {last_date}），使用本地文件: {file_path}")
                return data, str(file_path)
            else:
                logger.info(f"本地历史数据过期（最后交易日: {last_date} < 需要: {target}），需要更新")

    # 若本地没有或已过期，则继续走网络下载逻辑
    data = pd.DataFrame()
    last_exc = None
    for source in data_sources:
        if source == 'yfinance':
            try:
                logger.info(f"尝试数据源: yfinance for {ticker} period={period}")
                # 尝试使用 start/end 代替 period，禁用 threads/progress 提高稳定性
                days = _parse_period_to_days(period)
                end_date = datetime.today()
                start_date = end_date - timedelta(days=days)

                ticker_variants = _generate_ticker_variants(ticker)
                data = None
                for variant in ticker_variants:
                    try:
                        logger.info(f"尝试下载 {variant}...")
                        # yf.download 的两个常用签名，尝试第一个，如果失败再尝试第二个
                        try:
                            data = yf.download(variant, start=start_date.date(), end=end_date.date(), progress=False, threads=False)
                        except Exception:
                            data = yf.download([variant], start=start_date.date(), end=end_date.date(), group_by='ticker', progress=False)
                        if data is not None and not data.empty:
                            logger.info("yfinance 获取成功")
                            break
                    except Exception as e:
                        last_exc = e
                        logger.info(f"yfinance 下载失败: {e}")
                if data is not None and not data.empty:
                    break
            except Exception as e:
                last_exc = e
                logger.info(f"yfinance 下载失败: {e}")
        elif source == 'pandas_datareader':
            df = _try_pandas_datareader(ticker, period)
            if df is not None and not df.empty:
                data = df
                break
        elif source == 'akshare':
            df = _try_akshare(ticker, period)
            if df is not None and not df.empty:
                data = df
                break
        elif source == 'alpha_vantage':
            df = _try_alpha_vantage(ticker, period, alpha_key)
            if df is not None and not df.empty:
                data = df
                break
        elif source == 'yahooquery':
            df = _try_yahooquery(ticker, period)
            if df is not None and not df.empty:
                data = df
                break
        else:
            logger.info(f"未知数据源配置: {source}，跳过")

    # 2) 若都失败，尝试使用 yf.Ticker.history 回退（保持原有行为）
    if data is None or data.empty:
        try:
            logger.info(f"尝试使用 yf.Ticker.history 回退 {ticker}...")
            t = yf.Ticker(ticker)
            data = t.history(period=period)
        except Exception as e:
            last_exc = e
            logger.info(f"使用 Ticker.history 回退也失败: {e}")

    # 3) 如果仍然没有数据：不要生成合成数据（移除合成回退），直接记录并返回空 DataFrame
    if data is None or data.empty:
        logger.info(f"未能从任何数据源获取到 {ticker} 的数据（最后错误: {last_exc}），且本地没有历史数据。")
        return pd.DataFrame(), None

    # 4) 保存成功获取的数据
    try:
        data = _normalize_df(data)
        # Ensure 'Close' exists (try lowercase fallback)
        if 'Close' not in data.columns:
            for c in data.columns:
                if c.lower() == 'close':
                    data = data.rename(columns={c: 'Close'})
                    break
        # Ensure datetime index and set its name
        if pd.api.types.is_datetime64_any_dtype(data.index):
            data.index.name = 'date'
        else:
            try:
                data.index = pd.to_datetime(data.index)
                data.index.name = 'date'
            except Exception:
                pass

        # 如果本地已有历史数据，合并新旧数据，保留最新值
        if file_path.exists():
            old_data = pd.read_csv(file_path, index_col=0, parse_dates=True)
            # 统一时区处理：移除时区信息进行比较
            if old_data.index.tz is not None:
                old_data.index = old_data.index.tz_localize(None)
            if data.index.tz is not None:
                data.index = data.index.tz_localize(None)
            # 合并：使用新数据更新旧数据，保留最新值
            combined = pd.concat([old_data, data])
            combined = combined[~combined.index.duplicated(keep='last')]
            combined = combined.sort_index()
            data = combined
            logger.info(f"合并历史数据: 旧 {len(old_data)} 条 + 新 {len(data)} 条 = 合并后 {len(data)} 条")

        data.to_csv(file_path, index_label='date')
        logger.info(f"数据已保存至: {file_path}")
        return data, str(file_path)
    except Exception as e:
        logger.info(f"保存数据时出错: {e}")
        return data, None


if __name__ == "__main__":
    download_stock_data()