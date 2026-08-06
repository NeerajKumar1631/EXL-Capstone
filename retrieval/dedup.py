"""Deduplicate articles: exact-URL + near-duplicate titles (RapidFuzz)."""
from __future__ import annotations

from rapidfuzz import fuzz

from orchestration.schemas import Article

_TITLE_THRESHOLD = 88  # token-set ratio above which two titles are "the same story"


def deduplicate(articles: list[Article], threshold: int = _TITLE_THRESHOLD) -> list[Article]:
    """Keep the first occurrence of each unique story (prefers earlier/higher-ranked input)."""
    kept: list[Article] = []
    seen_urls: set[str] = set()
    for art in articles:
        url = (art.url or "").split("?")[0].rstrip("/").lower()
        if url and url in seen_urls:
            continue
        if any(fuzz.token_set_ratio(art.title, k.title) >= threshold for k in kept):
            continue
        kept.append(art)
        if url:
            seen_urls.add(url)
    return kept
