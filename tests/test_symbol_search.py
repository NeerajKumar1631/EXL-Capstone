"""Symbol/company-name search — offline, with Yahoo's response mocked."""
from unittest.mock import MagicMock, patch

from data_ingestion.markets import search_symbols

_RELIANCE_QUOTES = [
    {"symbol": "RS", "longname": "Reliance, Inc.", "exchDisp": "NYSE", "quoteType": "EQUITY"},
    {"symbol": "RELIANCE.NS", "longname": "Reliance Industries Limited",
     "exchDisp": "NSE", "quoteType": "EQUITY"},
    {"symbol": "RELINFRA.BO", "longname": "Reliance Infrastructure Limited",
     "exchDisp": "Bombay", "quoteType": "EQUITY"},
    {"symbol": "REL-ETF", "longname": "Some Fund", "exchDisp": "NYSE", "quoteType": "ETF"},
    {"symbol": "RELIANCE.NS", "longname": "duplicate", "exchDisp": "NSE", "quoteType": "EQUITY"},
]


def _search(query: str, region: str, quotes=None, **kw):
    """Call search_symbols with Yahoo mocked and the disk cache bypassed."""
    fake = MagicMock()
    fake.quotes = _RELIANCE_QUOTES if quotes is None else quotes
    with patch("yfinance.Search", return_value=fake), \
         patch("database.cache.read_json", return_value=None), \
         patch("database.cache.write_json"):
        return search_symbols(query, region, **kw)


def test_finds_indian_stock_by_company_name_and_ranks_region_first():
    hits = _search("reliance", "INDIA")
    assert hits, "expected matches"
    # NSE/BSE listings must come before the NYSE one when India is selected
    assert hits[0].symbol == "RELIANCE.NS"
    assert all(h.in_region for h in hits[:2])
    assert hits[-1].symbol == "RS" and not hits[-1].in_region


def test_same_query_ranks_us_listing_first_for_us_region():
    hits = _search("reliance", "US")
    assert hits[0].symbol == "RS"
    assert hits[0].in_region


def test_non_equities_and_duplicates_are_dropped():
    symbols = [h.symbol for h in _search("reliance", "INDIA")]
    assert "REL-ETF" not in symbols          # ETF filtered out
    assert symbols.count("RELIANCE.NS") == 1  # de-duplicated


def test_label_is_human_readable():
    hit = next(h for h in _search("reliance", "INDIA") if h.symbol == "RELIANCE.NS")
    assert hit.label == "Reliance Industries Limited — RELIANCE.NS (NSE)"


def test_short_queries_and_failures_return_empty_without_raising():
    assert search_symbols("a", "US") == []      # too short to be useful
    assert search_symbols("", "US") == []
    with patch("yfinance.Search", side_effect=RuntimeError("network down")), \
         patch("database.cache.read_json", return_value=None):
        assert search_symbols("apple", "US") == []   # degrades, never raises


def test_limit_is_respected():
    assert len(_search("reliance", "INDIA", limit=2)) == 2
