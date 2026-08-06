"""News sources: RSS feeds and per-source credibility weights.

Credibility weights (0..1) reflect a rough editorial-reliability prior and are used
to weight the aggregate sentiment score. Tune as needed.
"""
from __future__ import annotations

from urllib.parse import quote

# RSS feeds that don't require an API key. `{ticker}` / `{query}` are substituted.
RSS_FEEDS: dict[str, str] = {
    "Yahoo Finance": "https://feeds.finance.yahoo.com/rss/2.0/headline?s={ticker}&region=US&lang=en-US",
    "CNBC": "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664",
    "MarketWatch": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "NASDAQ": "https://www.nasdaq.com/feed/rssoutbound?symbol={ticker}",
    "Seeking Alpha": "https://seekingalpha.com/api/sa/combined/{ticker}.xml",
}

# Credibility priors keyed by a lowercase substring of the source name/domain.
SOURCE_CREDIBILITY: dict[str, float] = {
    "reuters": 1.0,
    "bloomberg": 1.0,
    "wall street journal": 0.98,
    "wsj": 0.98,
    "financial times": 0.97,
    "cnbc": 0.9,
    "marketwatch": 0.88,
    "the economist": 0.95,
    "barron": 0.9,
    "forbes": 0.82,
    "business insider": 0.75,
    "yahoo": 0.75,
    "nasdaq": 0.78,
    "seeking alpha": 0.72,
    "motley fool": 0.65,
    "benzinga": 0.6,
}

DEFAULT_CREDIBILITY = 0.55


def credibility_for(source: str) -> float:
    """Return a credibility weight for a source name/domain (case-insensitive substring match)."""
    if not source:
        return DEFAULT_CREDIBILITY
    s = source.lower()
    for key, weight in SOURCE_CREDIBILITY.items():
        if key in s:
            return weight
    return DEFAULT_CREDIBILITY


def rss_urls(ticker: str, query: str) -> dict[str, str]:
    """Resolve RSS feed templates for a given ticker/query."""
    t, q = quote(ticker), quote(query)
    return {name: tpl.format(ticker=t, query=q) for name, tpl in RSS_FEEDS.items()}
