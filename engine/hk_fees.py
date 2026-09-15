"""Hong Kong cash-equity transaction fee calculator.

Statutory rates are kept separate from broker-specific commission/platform fees.
All monetary components are rounded to cents; stock stamp duty is rounded up to
the next whole Hong Kong dollar.  Stamp-duty-exempt products can be configured
explicitly by ticker because a numeric HKEX code alone does not identify the
security type reliably.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from decimal import Decimal, ROUND_CEILING, ROUND_HALF_UP


CENT = Decimal("0.01")
ONE_DOLLAR = Decimal("1")


def _decimal(value) -> Decimal:
    return Decimal(str(value))


def _cent(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


@dataclass(frozen=True)
class HKFeeBreakdown:
    gross_amount: float
    stamp_duty: float
    sfc_levy: float
    afrc_levy: float
    hkex_trading_fee: float
    settlement_fee: float
    commission: float
    platform_fee: float
    total: float

    def to_dict(self) -> dict:
        return asdict(self)


def calculate_hk_stock_fees(
    gross_amount: float,
    ticker: str,
    config: dict | None = None,
) -> HKFeeBreakdown:
    """Calculate one side of an HK stock trade using 2026 fee rules."""
    root = config or {}
    cfg = root.get("hk_trading_fees", root)
    gross = max(_decimal(gross_amount), Decimal("0"))

    exempt = {str(t).upper() for t in cfg.get("stamp_duty_exempt_tickers", [])}
    stamp_rate = _decimal(cfg.get("stamp_duty_rate", "0.001"))
    stamp = Decimal("0") if ticker.upper() in exempt or gross == 0 else (
        gross * stamp_rate
    ).quantize(ONE_DOLLAR, rounding=ROUND_CEILING)

    sfc = _cent(gross * _decimal(cfg.get("sfc_levy_rate", "0.000027")))
    afrc = _cent(gross * _decimal(cfg.get("afrc_levy_rate", "0.0000015")))
    trading = _cent(gross * _decimal(cfg.get("hkex_trading_fee_rate", "0.0000565")))
    settlement = _cent(gross * _decimal(cfg.get("settlement_fee_rate", "0.000042")))

    commission_rate = _decimal(cfg.get("broker_commission_rate", "0"))
    commission_min = _decimal(cfg.get("broker_commission_min", "0"))
    commission = _cent(max(gross * commission_rate, commission_min if gross else Decimal("0")))
    platform = _cent(_decimal(cfg.get("platform_fee", "0"))) if gross else Decimal("0")
    total = stamp + sfc + afrc + trading + settlement + commission + platform

    return HKFeeBreakdown(
        gross_amount=float(gross), stamp_duty=float(stamp), sfc_levy=float(sfc),
        afrc_levy=float(afrc), hkex_trading_fee=float(trading),
        settlement_fee=float(settlement), commission=float(commission),
        platform_fee=float(platform), total=float(_cent(total)),
    )


def affordable_hk_shares(
    cash_limit: float,
    price: float,
    ticker: str,
    config: dict | None = None,
    lot_size: int | None = None,
) -> int:
    """Return the largest board-lot quantity whose gross plus fees fits cash."""
    if cash_limit <= 0 or price <= 0:
        return 0
    if lot_size is None:
        root = config or {}
        board_cfg = root.get("board_lots", {})
        fallback = max(1, int(board_cfg.get("fallback_lot_size", 1)))
        if board_cfg.get("enabled", False):
            overrides = {
                str(key).upper(): int(value)
                for key, value in board_cfg.get("overrides", {}).items()
            }
            lot_size = overrides.get(ticker.upper())
            if lot_size is None:
                from data.hk_board_lots import get_board_lot

                lot_size = get_board_lot(ticker)
            if lot_size is None and board_cfg.get("required", False):
                return 0
        lot_size = lot_size or fallback
    lot_size = max(1, int(lot_size))
    shares = int(cash_limit // price) // lot_size * lot_size
    while shares > 0:
        gross = shares * price
        if gross + calculate_hk_stock_fees(gross, ticker, config).total <= cash_limit:
            return shares
        shares -= lot_size
    return 0
