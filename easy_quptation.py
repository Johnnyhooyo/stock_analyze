"""
easy_quptation.py — 港股实时行情与分时K线工具

独立工具模块，不属于核心回测 pipeline。用于获取：
- 实时行情（价格、涨跌、成交量等）
- 日K 线数据（保存至 data/historical/）
- 分时 K 线（1 分钟频率，用于盘中交易决策）

数据来源：
- 港股：easyquotation hkquote/daykline，或本地 time_kline 模块
- 其他标的：easyquotation timekline

使用方式（standalone）：
    from easy_quptation import get_realtime, fetch_and_save_daykline, get_timekline

    # 实时行情
    df = get_realtime(['00700', '09988'])

    # 日K 保存
    fetch_and_save_daykline('00700')

    # 分时K
    tk = get_timekline('00700')
"""

import easyquotation
import pandas as pd
from pathlib import Path
from typing import List, Optional, Union
import datetime as dt

# Prefer local HK timekline implementation when asked for HK tickers
try:
    from time_kline import get_hk_timekline
except Exception:
    get_hk_timekline = None

# Module provides utility wrappers around easyquotation for realtime and day k-line data.
# Do not perform network calls at import time; call the functions below explicitly.

DATA_DIR = Path(__file__).parent / 'data' / 'historical'
DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_realtime(tickers: Union[List[str], str]):
    """Fetch realtime quotes for given tickers using easyquotation hkquote.

    Args:
        tickers: a single ticker string (e.g. '00700' or '0700.HK') or a list of such strings.

    Returns:
        dict or pandas.DataFrame containing realtime data as returned by easyquotation.
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    q = easyquotation.use("hkquote")
    try:
        data = q.real(tickers)
    except Exception as e:
        # bubble up with context
        raise RuntimeError(f"获取实时行情失败: {e}")

    # Try to present as DataFrame when possible for convenience
    try:
        if isinstance(data, dict):
            # convert dict-of-dicts to DataFrame
            df = pd.DataFrame.from_dict(data, orient='index')
            return df
    except Exception:
        pass
    return data


def fetch_and_save_daykline(tickers: Union[List[str], str], file_prefix: Optional[str] = None):
    """Fetch day k-line data and save to CSV files under data/historical.

    Args:
        tickers: ticker or list of tickers to fetch (e.g. '00700').
        file_prefix: optional prefix for saved filenames; if not provided, ticker is used.

    Returns:
        dict mapping ticker -> saved file path (str)
    """
    if isinstance(tickers, str):
        tickers = [tickers]
    q = easyquotation.use("daykline")

    result_files = {}
    try:
        raw = q.real(tickers)
    except Exception as e:
        raise RuntimeError(f"获取日K失败: {e}")

    # raw may be a dict keyed by ticker or already a DataFrame for single ticker
    for ticker in tickers:
        # Determine data for this ticker
        ticker_key = ticker
        data_obj = None
        if isinstance(raw, dict):
            # many easyquotation backends return a dict keyed by ticker
            data_obj = raw.get(ticker) or raw.get(ticker_key)
        else:
            # fallback: assume raw itself is the table for single ticker
            data_obj = raw

        # Try to normalize to DataFrame
        df = None
        if isinstance(data_obj, pd.DataFrame):
            df = data_obj.copy()
        elif isinstance(data_obj, dict):
            # sometimes returned dict contains 'data' list or k/v pairs
            if 'data' in data_obj and isinstance(data_obj['data'], (list, tuple)):
                df = pd.DataFrame(data_obj['data'])
            else:
                # try to frame it directly
                try:
                    df = pd.DataFrame.from_dict(data_obj)
                except Exception:
                    df = pd.DataFrame([data_obj])
        elif isinstance(data_obj, (list, tuple)):
            df = pd.DataFrame(data_obj)
        else:
            # last resort: try to build DataFrame from raw if single ticker
            try:
                df = pd.DataFrame(raw)
            except Exception:
                df = pd.DataFrame()

        if df is None or df.empty:
            # skip saving empty results but include None to indicate failure
            result_files[ticker] = None
            continue

        # normalize column names (common names like date/Open/Close)
        # If there's a 'date' or 'time' column, set it as index
        # Some data sources use numeric column names; convert to str before checking
        # detect date column robustly (name match or sample parsing)
        date_col = None
        for c in df.columns:
            if str(c).lower() in ('date', 'time', 'datetime', '交易日期', '日期'):
                date_col = c
                break
        if date_col is None:
            # try first few columns for parseable dates
            for c in list(df.columns)[:3]:
                try:
                    sample = df[c].dropna().astype(str).iloc[0]
                except Exception:
                    continue
                try:
                    _ = pd.to_datetime(sample)
                    date_col = c
                    break
                except Exception:
                    continue
        if date_col is not None:
            try:
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df = df.dropna(subset=[date_col])
                df = df.set_index(date_col).sort_index()
            except Exception:
                pass
        # If no date column could be found, leave as-is (we'll still attempt to standardize numeric cols)

        # If the source uses positional numeric column names (common for easyquotation/daykline), map them.
        # Based on easyquotation.daykline format and observed CSV samples, the columns correspond to:
        # 1: Close, 2: Open, 3: High, 4: Low, 5: Volume, 6: Extra(metadata), 7: Change(pct), 8: Amount(turnover)
        col_strs = [str(c) for c in df.columns]
        numeric_name_set = set(col_strs)
        expected_pos = {str(i) for i in range(1, 9)}
        if expected_pos.issubset(numeric_name_set):
            rename_pos = {
                '1': 'Close',
                '2': 'Open',
                '3': 'High',
                '4': 'Low',
                '5': 'Volume',
                '6': 'Meta',
                '7': 'Change',
                '8': 'Amount',
            }
            # apply mapping for existing keys
            actual_rename = {c: rename_pos[str(c)] for c in df.columns if str(c) in rename_pos}
            if actual_rename:
                df = df.rename(columns=actual_rename)

        # Standardize OHLCV column names when possible
        # If Close is missing, try to infer from numeric columns
        if 'Close' not in df.columns:
            # map obvious name variants first
            mapped = False
            for c in df.columns:
                lc = str(c).lower()
                if lc in ('close', 'closeprice', '收盘', '收盘价'):
                    df = df.rename(columns={c: 'Close'})
                    mapped = True
                    break
            if not mapped:
                # infer from numeric columns: find volume (largest mean) and price columns
                # create a numeric view: coerce columns to numeric where possible
                numeric_df = df.apply(lambda s: pd.to_numeric(s, errors='coerce'))
                num_cols = numeric_df.select_dtypes(include=['number']).columns.tolist()
                if len(num_cols) >= 4:
                    means = {c: numeric_df[c].abs().mean() for c in num_cols}
                    # candidate volume is column with largest mean
                    vol_candidate = max(means, key=means.get)
                    # Decide if vol_candidate is really volume by comparing to others
                    sorted_means = sorted(means.values(), reverse=True)
                    vol_is_obvious = False
                    if len(sorted_means) > 1 and sorted_means[0] >= 10 * sorted_means[1]:
                        vol_is_obvious = True
                    if sorted_means and sorted_means[0] > 1e5:
                        vol_is_obvious = True

                    if vol_is_obvious:
                        volume_col = vol_candidate
                        price_cols = [c for c in num_cols if c != volume_col]
                    else:
                        # no obvious volume: assume last numeric col is volume
                        volume_col = num_cols[-1]
                        price_cols = num_cols[:-1]

                    # now identify High and Low by counting rowwise max/min occurrences
                    if len(price_cols) >= 2:
                        # use numeric_df (coerced numeric) for max/min operations
                        price_df = numeric_df[price_cols]
                        try:
                            # per-row max/min column names (on numeric data)
                            row_max = price_df.idxmax(axis=1)
                            row_min = price_df.idxmin(axis=1)
                            max_counts = row_max.value_counts()
                            min_counts = row_min.value_counts()
                            high_col = max_counts.idxmax()
                            low_col = min_counts.idxmax()
                        except Exception:
                            # fallback: pick highest-mean as High and lowest-mean as Low
                            pm = {c: numeric_df[c].mean() for c in price_cols}
                            high_col = max(pm, key=pm.get)
                            low_col = min(pm, key=pm.get)

                        remaining = [c for c in price_cols if c not in (high_col, low_col)]
                        # pick Close as remaining column with larger mean
                        close_col = None
                        open_col = None
                        if len(remaining) == 1:
                            close_col = remaining[0]
                        elif len(remaining) >= 2:
                            rem_means = {c: numeric_df[c].mean() for c in remaining}
                            close_col = max(rem_means, key=rem_means.get)
                            open_col = [c for c in remaining if c != close_col][0]
                        # final rename map
                        rename_map = {}
                        if open_col is not None:
                            rename_map[open_col] = 'Open'
                        if close_col is not None:
                            rename_map[close_col] = 'Close'
                        rename_map[high_col] = 'High'
                        rename_map[low_col] = 'Low'
                        if volume_col is not None:
                            rename_map[volume_col] = 'Volume'
                        if rename_map:
                            # rename columns on original df (not numeric_df)
                            df = df.rename(columns=rename_map)
        # build filename
        prefix = file_prefix or ticker
        fn = f"{prefix}.csv"
        out_path = DATA_DIR / fn
        try:
            # Post-process: coerce columns to numeric where possible and drop mostly-non-numeric cols
            coerced = df.apply(lambda s: pd.to_numeric(s, errors='coerce'))
            # keep columns that have at least 40% numeric values
            keep = [c for c in coerced.columns if coerced[c].count() >= 0.4 * len(coerced)]
            df_clean = df[keep].copy()

            # If standard OHLCV columns exist among kept columns, reorder them
            # Order columns explicitly for clarity.
            # Common fields (based on easyquotation/daykline):
            # Open, High, Low, Close, Volume, Amount (turnover), Change (pct), Meta (extra)
            desired_order = ['Open', 'High', 'Low', 'Close', 'Volume', 'Amount', 'Change', 'Meta']
            present_desired = [c for c in desired_order if c in df_clean.columns]
            other_cols = [c for c in df_clean.columns if c not in present_desired]
            final_cols = present_desired + other_cols

            df_clean = df_clean[final_cols]

            # ensure index label 'date' when saving
            if pd.api.types.is_datetime64_any_dtype(df_clean.index):
                df_clean.to_csv(out_path, index_label='date')
            else:
                df_clean.to_csv(out_path, index=True)
            result_files[ticker] = str(out_path)
        except Exception as e:
            result_files[ticker] = None
    return result_files


def plot_range(ticker: str, start, end):
    """已废弃：绘图功能已移除，此函数保留签名以兼容旧调用，直接返回 None。"""
    return None


def get_timekline(ticker: str, source: Optional[str] = None):
    """Fetch minute/time k-line for a given ticker.

    For HK tickers this uses the local time_kline implementation which fetches and caches
    today's minute payload from GTIMG/ifzq. For other tickers it falls back to easyquotation.

    Returns a pandas.DataFrame indexed by Timestamp with columns ['price','volume'], or
    an empty DataFrame if no data is available.
    """
    # If the ticker looks like a HK ticker, prefer the custom fetcher
    t_upper = str(ticker).upper()
    is_hk = False
    if t_upper.endswith('.HK'):
        is_hk = True
    else:
        s = str(ticker).split('.')[0]
        if s.isdigit() and len(s) <= 5:
            is_hk = True

    if is_hk and get_hk_timekline is not None:
        try:
            # forward source param to allow choosing the upstream URL
            df = get_hk_timekline(ticker, source=source)
            return df
        except Exception:
            # fall back to easyquotation if custom fetcher fails
            pass

    q = easyquotation.use("timekline")
    raw = None
    try:
        raw = q.real([ticker])
    except Exception:
        try:
            raw = q.real(ticker)
        except Exception:
            try:
                base = ticker.split('.')[0]
                raw = q.real([base])
            except Exception:
                raw = None
    if raw is None:
        print(f"警告: 获取分时数据失败或无数据: {ticker}")
        return pd.DataFrame()

    data_obj = None
    if isinstance(raw, dict):
        data_obj = raw.get(ticker) or raw.get(ticker.replace('.HK', ''))
        if data_obj is None and len(raw) == 1:
            data_obj = list(raw.values())[0]
    else:
        data_obj = raw

    if data_obj is None:
        return pd.DataFrame()

    # try to extract time rows
    time_rows = None
    data_date = None
    if isinstance(data_obj, dict) and 'time_data' in data_obj:
        time_rows = data_obj.get('time_data')
        data_date = data_obj.get('date')
    elif isinstance(data_obj, dict):
        for v in data_obj.values():
            if isinstance(v, list):
                time_rows = v
                break
    elif isinstance(data_obj, (list, tuple)):
        time_rows = list(data_obj)

    if not time_rows:
        return pd.DataFrame()

    parsed = []
    for row in time_rows:
        if row is None:
            continue
        if isinstance(row, (list, tuple)):
            parts = [str(x).strip() for x in row]
        else:
            parts = str(row).strip().split()
        if len(parts) >= 3:
            time_str, price_s, vol_s = parts[0], parts[1], parts[2]
        elif len(parts) == 2:
            time_str, price_s = parts[0], parts[1]
            vol_s = None
        else:
            continue
        parsed.append((time_str, price_s, vol_s))

    if not parsed:
        return pd.DataFrame()

    parsed_date = None
    if data_date is not None:
        try:
            parsed_date = pd.to_datetime(str(data_date)).normalize().date()
        except Exception:
            try:
                parsed_date = pd.to_datetime(str(data_date), format='%Y%m%d').normalize().date()
            except Exception:
                parsed_date = None

    use_date = parsed_date or pd.Timestamp.today().normalize().date()

    times = []
    prices = []
    vols = []
    for time_str, price_s, vol_s in parsed:
        ts_token = str(time_str)
        if len(ts_token) == 4 and ts_token.isdigit():
            t = ts_token[:2] + ':' + ts_token[2:]
        else:
            t = ts_token
        dt_str = pd.Timestamp(use_date).strftime('%Y-%m-%d') + ' ' + t
        try:
            ts_dt = pd.to_datetime(dt_str)
        except Exception:
            continue
        try:
            price = float(price_s)
        except Exception:
            price = pd.NA
        try:
            vol = int(float(vol_s)) if vol_s is not None else pd.NA
        except Exception:
            vol = pd.NA
        times.append(ts_dt)
        prices.append(price)
        vols.append(vol)

    if not times:
        return pd.DataFrame()

    df = pd.DataFrame({'price': prices, 'volume': vols}, index=pd.DatetimeIndex(times))
    df = df.sort_index()
    return df


if __name__ == "__main__":
    # Demo: fetch today's intraday (timekline) via get_timekline
    ticker = '00700'
    print(f"Fetching timekline for {ticker}...")
    try:
        tk = get_timekline(ticker)
    except Exception as e:
        print("获取分时数据失败:", e)
        tk = pd.DataFrame()

    if tk is None or tk.empty:
        print(f"未获取到 {ticker} 的分时数据")
    else:
        print(tk.tail(10))
