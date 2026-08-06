"""Historical price ingestion: yfinance with disk cache + resilient error handling.

Output contract: a DataFrame indexed by a tz-naive DatetimeIndex named 'Date',
ascending, with columns ['Open', 'High', 'Low', 'Close', 'Volume'] and no NaNs.

Audit note: Stooq (pandas-datareader and direct CSV) is non-functional in this
environment, so yfinance is the single source, made robust with caching, bounded
retries on transient/network errors (but NOT on invalid tickers), and a stale-cache
fallback when the network is down.
"""
from __future__ import annotations

from typing import Optional

import pandas as pd
from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from config.logging_config import get_logger
from config.settings import settings
from database import cache

logger = get_logger("data.prices")

_COLS = ["Open", "High", "Low", "Close", "Volume"]
_MIN_RAW_ROWS = 30  # anything less is unusable; likely a bad symbol


class PriceDataError(RuntimeError):
    """Raised when price history cannot be obtained."""


class InvalidTickerError(PriceDataError):
    """Empty/insufficient data — a bad symbol. NOT retried."""


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce yfinance's frame into the output contract."""
    if df is None or df.empty:
        raise InvalidTickerError("empty price frame")

    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    df = df.rename(columns={c: str(c).title() for c in df.columns})
    missing = [c for c in _COLS if c not in df.columns]
    if missing:
        raise InvalidTickerError(f"missing columns {missing}")

    out = df[_COLS].copy()
    out.index = pd.to_datetime(out.index)
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    out.index.name = "Date"
    out = out[~out.index.duplicated(keep="last")].sort_index()
    out = out.apply(pd.to_numeric, errors="coerce").dropna(how="any")
    if len(out) < _MIN_RAW_ROWS:
        raise InvalidTickerError(f"only {len(out)} rows returned")
    return out


@retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.5, min=0.5, max=8),
    retry=retry_if_not_exception_type(InvalidTickerError),  # retry network, not bad symbols
)
def _from_yfinance(ticker: str, period: str, interval: str) -> pd.DataFrame:
    import yfinance as yf

    hist = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=True)
    return _normalize(hist)


def fetch_prices(
    ticker: str,
    period: Optional[str] = None,
    interval: Optional[str] = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Return cleaned OHLCV history for `ticker`.

    Raises InvalidTickerError for bad symbols; PriceDataError if the network fails and
    no cache is available.
    """
    ticker = ticker.strip().upper()
    period = period or settings.price_period
    interval = interval or settings.price_interval
    key = f"prices_{ticker}_{period}_{interval}"

    if use_cache:
        cached = cache.read_df(key)
        if cached is not None and not cached.empty:
            logger.info("prices for %s from cache (%d rows)", ticker, len(cached))
            return cached

    try:
        df = _from_yfinance(ticker, period, interval)
    except InvalidTickerError as exc:
        raise InvalidTickerError(
            f"No price data for '{ticker}'. Check the symbol "
            f"(US like AAPL; NSE like TCS.NS)."
        ) from exc
    except Exception as exc:  # noqa: BLE001 - transient/network after retries
        stale = cache.read_df(key, ttl_minutes=10**9)  # any age
        if stale is not None and not stale.empty:
            logger.warning("network failed for %s; serving stale cache: %s", ticker, exc)
            return stale
        raise PriceDataError(f"Could not fetch prices for {ticker}: {exc}") from exc

    logger.info("prices for %s from yfinance (%d rows)", ticker, len(df))
    cache.write_df(key, df)
    return df


def resolve_company_name(ticker: str) -> str:
    """Best-effort human-readable company name (falls back to the ticker)."""
    ticker = ticker.strip().upper()
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info or {}
        return info.get("longName") or info.get("shortName") or ticker
    except Exception as exc:  # noqa: BLE001
        logger.warning("company name lookup failed for %s: %s", ticker, exc)
        return ticker
