"""Market context: company fundamentals (yfinance) + macro proxies (no-key).

All best-effort — any piece that fails is simply omitted with a note.
"""
from __future__ import annotations

from config.logging_config import get_logger
from orchestration.schemas import MarketContext

logger = get_logger("data.context")

# Macro proxies available via yfinance without an API key.
_MACRO_TICKERS = {"VIX": "^VIX", "USD_Index": "DX-Y.NYB", "US10Y": "^TNX"}

# Fundamental fields worth surfacing (yfinance `.info` keys → friendly label).
_FUND_FIELDS = {
    "trailingPE": "trailing_pe",
    "forwardPE": "forward_pe",
    "profitMargins": "profit_margin",
    "revenueGrowth": "revenue_growth",
    "earningsGrowth": "earnings_growth",
    "marketCap": "market_cap",
    "beta": "beta",
    "dividendYield": "dividend_yield",
    "targetMeanPrice": "analyst_target",
    "recommendationKey": "analyst_reco",
    "sector": "sector",
    "industry": "industry",
    "fiftyTwoWeekHigh": "week52_high",
    "fiftyTwoWeekLow": "week52_low",
}


def fetch_context(ticker: str) -> MarketContext:
    ticker = ticker.strip().upper()
    macro: dict[str, float] = {}
    fundamentals: dict[str, object] = {}
    notes: list[str] = []

    try:
        import yfinance as yf

        info = yf.Ticker(ticker).info or {}
        for src, label in _FUND_FIELDS.items():
            val = info.get(src)
            if val is not None and val != "":
                fundamentals[label] = val
    except Exception as exc:  # noqa: BLE001
        logger.warning("fundamentals failed for %s: %s", ticker, exc)
        notes.append("Company fundamentals were unavailable.")

    try:
        import yfinance as yf

        for label, t in _MACRO_TICKERS.items():
            try:
                hist = yf.Ticker(t).history(period="5d", interval="1d")
                if not hist.empty:
                    macro[label] = round(float(hist["Close"].iloc[-1]), 2)
            except Exception:  # noqa: BLE001
                continue
    except Exception as exc:  # noqa: BLE001
        logger.warning("macro proxies failed: %s", exc)

    if not macro:
        notes.append("Macro indicators were unavailable.")

    return MarketContext(macro=macro, fundamentals=fundamentals, notes=notes)
