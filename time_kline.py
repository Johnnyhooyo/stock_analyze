"""
time_kline.py — 港股分时K线获取工具

独立工具模块，从 GTIMG/ifzq 等数据源获取港股分时K线数据并缓存到本地。

与 easy_quptation.py 的区别：
- easy_quptation.get_timekline() 优先调用本模块（港股专用）
- time_kline 更底层，专精港股分时数据，支持多数据源回退

使用方式（standalone）：
    from time_kline import get_hk_timekline

    # 获取今日分时（自动缓存）
    tk = get_hk_timekline('00700')

数据缓存：data/timekline/{ticker}_{date}.csv
"""

from pathlib import Path
from typing import Optional, Union
import datetime as dt
import re
import csv
import sys
import json

import pandas as pd

# Local cache directory
CACHE_DIR = Path(__file__).parent / 'data' / 'timekline'
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _fetch_url(url: str, timeout: float = 10.0) -> Optional[str]:
    """Fetch URL content; prefer requests when available, else use urllib."""
    try:
        import requests
        r = requests.get(url, timeout=timeout)
        if r.status_code == 200:
            return r.text
        return None
    except Exception:
        # fallback to urllib
        try:
            from urllib.request import urlopen
            with urlopen(url, timeout=timeout) as resp:
                raw = resp.read()
                try:
                    return raw.decode('utf-8')
                except Exception:
                    return raw.decode('latin1')
        except Exception:
            return None


def _normalize_ticker(ticker: str) -> str:
    # accept '00700', '700', '0700.HK', '0700' etc -> return zero-padded 5-digit string
    t = str(ticker).split('.')[0]
    t = t.strip()
    if t.isdigit():
        return t.zfill(5)
    # if contains letters, just return as-is
    return t


def _parse_minute_lines(text: str) -> tuple[Optional[str], list]:
    """Extract date string (YYYY-MM-DD) if present and list of (time, price, volume) tuples.

    Returns (date_str_or_None, rows)
    """
    # try to find explicit date patterns like 'date:YYYYMMDD' or 'date:YYMMDD' or 'date:YYYY-MM-DD'
    date_str = None
    m = re.search(r"date\s*[:=]\s*([0-9]{6,8})", text, flags=re.IGNORECASE)
    if m:
        s = m.group(1)
        if len(s) == 6:
            date_str = '20' + s[-6:]
            # format as YYYYMMDD -> YYYY-MM-DD
            date_str = pd.to_datetime(date_str, format='%Y%m%d').strftime('%Y-%m-%d')
        elif len(s) == 8:
            date_str = pd.to_datetime(s, format='%Y%m%d').strftime('%Y-%m-%d')
    # try also look for ISO date like 2020-01-23
    if date_str is None:
        m2 = re.search(r"(20[0-9]{2}-[01][0-9]-[0-3][0-9])", text)
        if m2:
            date_str = m2.group(1)

    # find minute lines like '0930 11.64 29727' or '09:30 11.64 29727'
    rows = []
    for line in re.findall(r"(^|\\n)\s*([0-2]?\d[:]?\d{2})\s+([0-9]+(?:\.[0-9]+)?)\s+([0-9]+)\\b", text):
        # regex returns groups with preceding newline; pick appropriate ones
        time_raw = line[1]
        price = line[2]
        vol = line[3]
        # normalize time
        t = time_raw.replace(':', '')
        if len(t) == 3:  # e.g. 930 -> 0930
            t = '0' + t
        if len(t) == 4 and t.isdigit():
            timestr = t[:2] + ':' + t[2:]
        else:
            timestr = time_raw
        rows.append((timestr, price, vol))

    # fallback: broader pattern matching (some files have different separators)
    if not rows:
        for line in text.splitlines():
            line = line.strip()
            if re.match(r"^\d{3,4}\s+", line):
                parts = line.split()
                if len(parts) >= 3 and re.match(r"^\d{3,4}$", parts[0]):
                    t = parts[0].zfill(4)
                    timestr = t[:2] + ':' + t[2:]
                    price = parts[1]
                    vol = parts[2]
                    rows.append((timestr, price, vol))

    return date_str, rows


def _save_to_csv(ticker_code: str, date_str: str, df: pd.DataFrame) -> Path:
    fn = CACHE_DIR / f"hk{ticker_code}_{date_str}.csv"
    df.to_csv(fn, index_label='datetime')
    return fn


def get_hk_timekline(ticker: str, source: Optional[str] = None) -> pd.DataFrame:
    """Get Hong Kong minute (timekline) data for ticker (today/latest).

    Behavior:
    - Check local cache for today's file hk{code}_{YYYY-MM-DD}.csv and return if present.
    - Otherwise try the GTIMG/ifzq endpoints to fetch the latest available minute payload.
    - Parse payload; if it contains a date use that when saving, otherwise use today's date.
    - Save parsed minute data to cache and return DataFrame with columns ['price','volume'].
    """
    code = _normalize_ticker(ticker)
    today = pd.Timestamp.today().normalize().date()
    today_str = today.strftime('%Y-%m-%d')

    # check today's cache first
    local_today = CACHE_DIR / f"hk{code}_{today_str}.csv"
    if local_today.exists():
        try:
            df = pd.read_csv(local_today, index_col=0, parse_dates=True)
            return df
        except Exception:
            try:
                local_today.unlink()
            except Exception:
                pass

    # candidate URLs mapped by key
    url_map = {
        'gtimg_minute': f"http://data.gtimg.cn/flashdata/hk/minute/hk{code}.js",
        'gtimg_4day': f"http://data.gtimg.cn/flashdata/hk/4day/hk{code}.js",
        'ifzq': f"https://web.ifzq.gtimg.cn/appstock/app/minute/query?code=hk{code}",
    }

    # small parsers for each source
    def parse_gtimg_minute(text: str):
        # GTIMG minute payload often is like: min_data="\n\
        # date:260225\n\
        # 0930 522.50 1020357\n\
        # ..."
        # Extract the quoted min_data content and unescape common sequences before parsing.
        m = re.search(r'min_data\s*=\s*"(.*?)"', text, flags=re.S)
        payload = None
        if m:
            payload = m.group(1)
            # payload may include literal backslashes followed by newlines ("\\\n\").
            # Normalize common escape patterns: replace '\\n' with '\n' and remove trailing '\\' at line ends.
            try:
                # First replace literal backslash+newline patterns ("\\\n") -> "\n"
                payload = payload.replace('\\\n', '\n')
                # Replace literal backslash-n sequences
                payload = payload.replace('\\n', '\n')
                # Remove stray backslashes
                payload = payload.replace('\\', '')
            except Exception:
                pass
        else:
            # fallback: if no min_data variable, try to parse full text
            payload = text
        return _parse_minute_lines(payload)

    def parse_gtimg_4day(text: str):
        rows = []
        date_found = None
        for m in re.finditer(r"\{\s*\"date\"\s*:\s*\"(\d{6,8})\"\s*,\s*\"data\"\s*:\s*\"([^\"]+)\"", text):
            d = m.group(1)
            try:
                if len(d) == 6:
                    date_s = '20' + d[-6:]
                else:
                    date_s = pd.to_datetime(d, format='%Y%m%d').strftime('%Y-%m-%d')
            except Exception:
                date_s = None
            data_blob = m.group(2)
            for entry in data_blob.strip('^').split('^'):
                parts = entry.split('~')
                if len(parts) >= 3:
                    timestr = parts[0]
                    if len(timestr) == 4 and timestr.isdigit():
                        timestr = timestr[:2] + ':' + timestr[2:]
                    price = parts[1]
                    vol = parts[2]
                    rows.append((timestr, price, vol))
            if rows and date_s:
                date_found = date_s
                break
        return date_found, rows

    def parse_ifzq_json(text: str):
        try:
            obj = json.loads(text)
        except Exception:
            j = re.search(r"(\{\s*\"code\".*\})", text, flags=re.S)
            if j:
                try:
                    obj = json.loads(j.group(1))
                except Exception:
                    return None, []
            else:
                return None, []
        try:
            data_root = obj.get('data', {})
            key = None
            for k in data_root.keys():
                if k.lower().startswith('hk') and code in k:
                    key = k
                    break
            if not key:
                key = f'hk{code}'
            entry = data_root.get(key, {})
            inner = entry.get('data') if isinstance(entry, dict) else None
            if inner and 'data' in inner:
                rows = []
                date_s = inner.get('date')
                for line in inner.get('data', []):
                    parts = str(line).split()
                    if len(parts) >= 3:
                        timestr = parts[0]
                        if len(timestr) == 4 and timestr.isdigit():
                            timestr = timestr[:2] + ':' + timestr[2:]
                        price = parts[1]
                        vol = parts[2]
                        rows.append((timestr, price, vol))
                return date_s, rows
        except Exception:
            return None, []
        return None, []

    # determine which sources to try
    if source is None:
        keys_to_try = ['gtimg_minute', 'gtimg_4day', 'ifzq']
    else:
        if source not in ('gtimg_minute', 'gtimg_4day', 'ifzq'):
            raise ValueError(f"unknown source key: {source}")
        keys_to_try = [source]

    for key in keys_to_try:
        url = url_map[key]
        text = _fetch_url(url)
        if not text:
            continue
        parsed_date = None
        rows = []
        if key == 'ifzq':
            parsed_date, rows = parse_ifzq_json(text)
        elif key == 'gtimg_4day':
            parsed_date, rows = parse_gtimg_4day(text)
        else:
            parsed_date, rows = parse_gtimg_minute(text)

        if not rows:
            parsed_date, rows = _parse_minute_lines(text)
        if not rows:
            quotes = re.findall(r'"([^"]{50,})"', text, flags=re.S)
            for q in quotes:
                pd_date, rows = _parse_minute_lines(q)
                if rows:
                    parsed_date = parsed_date or pd_date
                    break
        if not rows:
            continue

        use_date = parsed_date or today_str
        dts, prices, vols = [], [], []
        for time_s, price_s, vol_s in rows:
            try:
                dt_str = f"{use_date} {time_s}"
                ts = pd.to_datetime(dt_str)
                price = float(price_s)
                vol = int(float(vol_s)) if vol_s is not None and vol_s != '' else pd.NA
                dts.append(ts)
                prices.append(price)
                vols.append(vol)
            except Exception:
                continue
        if not dts:
            continue

        df = pd.DataFrame({'price': prices, 'volume': vols}, index=pd.DatetimeIndex(dts))
        df = df.sort_index()
        try:
            _save_to_csv(code, use_date, df)
        except Exception:
            pass
        return df

    # nothing fetched
    return pd.DataFrame()


def _find_minute_array_in_json(obj) -> tuple[Optional[str], list]:
    """Recursively search JSON-like object for a minute array: return (date, rows) if found.

    Rows are tuples (time, price, volume) where time like '09:30' or '0930'.
    """
    rows = []
    date_str = None

    def is_minute_row(item):
        if not isinstance(item, (list, tuple)):
            return False
        if len(item) < 2:
            return False
        t = str(item[0])
        if re.match(r"^\d{1,2}:?\d{2}$", t):
            return True
        return False

    def walk(o):
        nonlocal rows, date_str
        if isinstance(o, dict):
            # common fields might be 'date', 'time_data', 'data', 'minute'
            if 'date' in o and date_str is None:
                try:
                    date_str = pd.to_datetime(str(o['date'])).strftime('%Y-%m-%d')
                except Exception:
                    pass
            for k, v in o.items():
                if isinstance(v, (list, tuple)):
                    # check if this list appears to be minute rows
                    if v and is_minute_row(v[0]):
                        for item in v:
                            try:
                                t = str(item[0]).replace(':', '')
                                if len(t) == 3:
                                    t = '0' + t
                                if len(t) == 4:
                                    timestr = t[:2] + ':' + t[2:]
                                else:
                                    timestr = str(item[0])
                                price = str(item[1])
                                vol = str(item[2]) if len(item) > 2 else ''
                                rows.append((timestr, price, vol))
                            except Exception:
                                continue
        elif isinstance(o, (list, tuple)):
            for item in o:
                walk(item)

    walk(obj)
    return date_str, rows
