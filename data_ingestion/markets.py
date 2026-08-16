"""Market/region helpers: ticker normalization, region inference, benchmarks, search.

Standalone (no dependency on the universe module) to avoid import cycles.
Regions are the strings "US" and "INDIA".
"""
from __future__ import annotations

from config.logging_config import get_logger

logger = get_logger("data.markets")

INDIA_SUFFIXES = (".NS", ".BO")   # NSE, BSE
BENCHMARKS = {"US": "^GSPC", "INDIA": "^NSEI"}
REGIONS = ("US", "INDIA")

_SEARCH_TTL_MINUTES = 24 * 60     # symbol↔name mappings barely change
_SEARCH_FETCH = 20                # ask for more than we show, so region ranking has material


def normalize_ticker(symbol: str, region: str) -> str:
    """Uppercase/strip; append `.NS` for Indian symbols that lack an exchange suffix."""
    s = (symbol or "").strip().upper()
    if not s:
        return s
    if region.upper() == "INDIA" and not s.endswith(INDIA_SUFFIXES):
        s = f"{s}.NS"
    return s


def infer_region(ticker: str) -> str:
    """Best-effort region from a ticker's suffix (defaults to US)."""
    t = (ticker or "").strip().upper()
    return "INDIA" if t.endswith(INDIA_SUFFIXES) else "US"


def benchmark_for(region: str) -> str:
    """Index ticker used as the beta benchmark for a region."""
    return BENCHMARKS.get(region.upper(), "^GSPC")


def _in_region(symbol: str, region: str) -> bool:
    """True when a symbol belongs to the given market."""
    s = symbol.upper()
    if region.upper() == "INDIA":
        return s.endswith(INDIA_SUFFIXES)
    return "." not in s          # US listings carry no exchange suffix on Yahoo


def search_symbols(query: str, region: str = "US", limit: int = 8) -> list:
    """Find stocks by **ticker or company name** (e.g. "apple" → AAPL).

    Wraps Yahoo's symbol search via yfinance, so it covers every listed company rather
    than only the bundled index lists. Results are filtered to ordinary equities and
    ranked so the user's currently selected market comes first — without hiding the
    others, since plenty of companies are cross-listed.

    Returns `list[SymbolHit]`; an empty list on any failure, so callers can fall back.
    """
    from database import cache
    from orchestration.schemas import SymbolHit

    q = (query or "").strip()
    if len(q) < 2:
        return []

    key = f"symsearch_{region.upper()}_{q.lower()}"
    cached = cache.read_json(key, ttl_minutes=_SEARCH_TTL_MINUTES)
    if cached is not None:
        return [SymbolHit(**h) for h in cached][:limit]

    try:
        import yfinance as yf

        quotes = yf.Search(q, max_results=_SEARCH_FETCH).quotes or []
    except Exception as exc:  # noqa: BLE001 - search is a convenience, never essential
        logger.warning("symbol search failed for %r: %s", q, exc)
        return []

    hits: list[SymbolHit] = []
    seen: set[str] = set()
    for item in quotes:
        symbol = (item.get("symbol") or "").strip().upper()
        if not symbol or symbol in seen:
            continue
        if (item.get("quoteType") or "").upper() != "EQUITY":
            continue
        seen.add(symbol)
        hits.append(SymbolHit(
            symbol=symbol,
            name=(item.get("longname") or item.get("shortname") or "").strip(),
            exchange=(item.get("exchDisp") or "").strip(),
            in_region=_in_region(symbol, region),
        ))

    # Stable ranking: current market first, original relevance order preserved within each group.
    hits.sort(key=lambda h: not h.in_region)
    cache.write_json(key, [h.model_dump(mode="json") for h in hits])
    return hits[:limit]
