"""Market/region helpers: ticker normalization, region inference, benchmarks.

Standalone (no dependency on the universe module) to avoid import cycles.
Regions are the strings "US" and "INDIA".
"""
from __future__ import annotations

INDIA_SUFFIXES = (".NS", ".BO")   # NSE, BSE
BENCHMARKS = {"US": "^GSPC", "INDIA": "^NSEI"}
REGIONS = ("US", "INDIA")


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
