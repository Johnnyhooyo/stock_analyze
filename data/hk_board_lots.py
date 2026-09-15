"""HKEX board-lot master data with a daily local cache.

The official HKEX ``List of Securities`` workbook is parsed with the Python
standard library so daily execution does not require an Excel dependency.
Only rows whose Category is ``Equity`` are exposed to the stock workflow.
"""

from __future__ import annotations

import io
import json
import re
import urllib.request
import zipfile
from datetime import date, datetime
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

from log_config import get_logger


logger = get_logger(__name__)

HKEX_SECURITIES_URL = (
    "https://www.hkex.com.hk/eng/services/trading/securities/"
    "securitieslists/ListOfSecurities.xlsx"
)
_ROOT = Path(__file__).parent.parent
_CACHE_DIR = _ROOT / "cache"
_NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_MEMORY_CACHE: Optional[dict[str, int]] = None


def normalize_hk_ticker(stock_code: str) -> Optional[str]:
    """Convert HKEX's five-digit stock code to the project's Yahoo ticker."""
    digits = re.sub(r"\D", "", str(stock_code))
    if not digits:
        return None
    code = int(digits)
    if code <= 0:
        return None
    width = 4 if code <= 9999 else 5
    return f"{code:0{width}d}.HK"


def _cell_value(cell: ET.Element, shared_strings: list[str]) -> str:
    value = cell.find("x:v", _NS)
    if value is None or value.text is None:
        return ""
    if cell.attrib.get("t") == "s":
        try:
            return shared_strings[int(value.text)]
        except (ValueError, IndexError):
            return ""
    return value.text.strip()


def parse_hkex_board_lots(content: bytes) -> tuple[dict[str, int], str]:
    """Parse ticker -> board lot and the HKEX 'updated as at' label."""
    with zipfile.ZipFile(io.BytesIO(content)) as workbook:
        shared_root = ET.fromstring(workbook.read("xl/sharedStrings.xml"))
        shared_strings = [
            "".join(node.text or "" for node in item.findall(".//x:t", _NS))
            for item in shared_root.findall("x:si", _NS)
        ]
        sheet = ET.fromstring(workbook.read("xl/worksheets/sheet1.xml"))

    lots: dict[str, int] = {}
    updated_as_at = ""
    for row in sheet.findall(".//x:sheetData/x:row", _NS):
        values: dict[str, str] = {}
        for cell in row.findall("x:c", _NS):
            match = re.match(r"[A-Z]+", cell.attrib.get("r", ""))
            if match:
                values[match.group()] = _cell_value(cell, shared_strings)

        row_number = int(row.attrib.get("r", "0"))
        if row_number == 2:
            updated_as_at = values.get("A", "")
            continue
        if values.get("C") != "Equity":
            continue

        ticker = normalize_hk_ticker(values.get("A", ""))
        try:
            lot_size = int(float(values.get("E", "").replace(",", "")))
        except (TypeError, ValueError):
            continue
        if ticker and lot_size > 0:
            lots[ticker] = lot_size

    if not lots:
        raise ValueError("HKEX securities workbook contained no Equity board lots")
    return lots, updated_as_at


def _cache_path(fetch_date: date) -> Path:
    return _CACHE_DIR / f"hk_board_lots_{fetch_date.isoformat()}.json"


def _load_cache(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_lots = payload.get("lots", {})
    lots = {
        str(ticker).upper(): int(lot)
        for ticker, lot in raw_lots.items()
        if int(lot) > 0
    }
    if not lots:
        raise ValueError(f"empty board-lot cache: {path}")
    return lots


def refresh_board_lots(fetch_date: Optional[date] = None) -> dict[str, int]:
    """Download the official workbook, validate it, and atomically cache JSON."""
    target_date = fetch_date or date.today()
    request = urllib.request.Request(
        HKEX_SECURITIES_URL,
        headers={"User-Agent": "Mozilla/5.0 stock-analyze/1.0"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        content = response.read(10_000_001)
    if len(content) > 10_000_000:
        raise ValueError("HKEX securities workbook exceeded 10 MB")

    lots, updated_as_at = parse_hkex_board_lots(content)
    payload = {
        "source": HKEX_SECURITIES_URL,
        "updated_as_at": updated_as_at,
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "count": len(lots),
        "lots": lots,
    }
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(target_date)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)
    logger.info("HKEX 每手股数已更新: %s 只 (%s)", len(lots), updated_as_at)
    return lots


def get_board_lots(force_refresh: bool = False) -> dict[str, int]:
    """Return today's master data, falling back to the newest valid cache."""
    global _MEMORY_CACHE
    if _MEMORY_CACHE is not None and not force_refresh:
        return _MEMORY_CACHE.copy()

    today_path = _cache_path(date.today())
    if not force_refresh and today_path.exists():
        try:
            _MEMORY_CACHE = _load_cache(today_path)
            return _MEMORY_CACHE.copy()
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            logger.warning("每手股数当日缓存无效: %s", exc)

    try:
        _MEMORY_CACHE = refresh_board_lots()
        return _MEMORY_CACHE.copy()
    except Exception as exc:
        logger.warning("港交所每手股数下载失败，将尝试历史缓存: %s", exc)

    for path in sorted(_CACHE_DIR.glob("hk_board_lots_*.json"), reverse=True):
        try:
            _MEMORY_CACHE = _load_cache(path)
            logger.warning("使用每手股数历史缓存: %s", path.name)
            return _MEMORY_CACHE.copy()
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
    return {}


def get_board_lot(ticker: str, force_refresh: bool = False) -> Optional[int]:
    """Return one ticker's real board lot, or ``None`` when unavailable."""
    return get_board_lots(force_refresh=force_refresh).get(ticker.upper())
