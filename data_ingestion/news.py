"""News ingestion.

Primary: Event Registry (newsapi.ai) — rich, full-text, on-topic with relevance sort.
Fallback: yfinance `.news` — no key, used when Event Registry is empty/quota-limited/down.
Output: a list of normalized `Article` objects (deduping happens downstream).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

import requests

from config.logging_config import get_logger
from config.settings import settings
from config.sources import credibility_for
from database import cache
from orchestration.schemas import Article

logger = get_logger("data.news")

_TIMEOUT = 25


def _parse_dt(value) -> Optional[datetime]:
    if not value:
        return None
    try:
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(value, tz=timezone.utc)
        s = str(value).replace("Z", "+00:00")
        return datetime.fromisoformat(s)
    except Exception:
        return None


def _from_event_registry(company: str, days: int, max_articles: int) -> list[Article]:
    date_start = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    payload = {
        "apiKey": settings.news_api_key,
        "keyword": company,
        "keywordLoc": "title",
        "articlesSortBy": "rel",
        "articlesCount": min(max_articles, 100),
        "resultType": "articles",
        "isDuplicateFilter": "skipDuplicates",
        "lang": "eng",
        "dataType": ["news"],
        "dateStart": date_start,
    }
    resp = requests.post(settings.event_registry_url, json=payload, timeout=_TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(f"Event Registry error: {data['error']}")

    results = (data.get("articles") or {}).get("results", []) or []
    articles: list[Article] = []
    for a in results:
        title = (a.get("title") or "").strip()
        url = a.get("url") or ""
        if not title or not url:
            continue
        source = ((a.get("source") or {}).get("title")) or "unknown"
        body = (a.get("body") or "").strip()
        articles.append(Article(
            title=title,
            url=url,
            source=source,
            published_at=_parse_dt(a.get("dateTime")),
            snippet=body[:500],
            content=body[:4000],
            credibility=credibility_for(source),
        ))
    return articles


def _from_yfinance_news(ticker: str, max_articles: int) -> list[Article]:
    import yfinance as yf

    raw = yf.Ticker(ticker).news or []
    articles: list[Article] = []
    for item in raw[:max_articles]:
        content = item.get("content", item) if isinstance(item, dict) else {}
        title = (content.get("title") or "").strip()
        if not title:
            continue
        url = ""
        for key in ("canonicalUrl", "clickThroughUrl"):
            node = content.get(key)
            if isinstance(node, dict) and node.get("url"):
                url = node["url"]
                break
        url = url or content.get("link") or content.get("url") or ""
        source = ((content.get("provider") or {}).get("displayName")) or "Yahoo Finance"
        body = (content.get("summary") or content.get("description") or "").strip()
        published = content.get("pubDate") or content.get("displayTime") or item.get("providerPublishTime")
        articles.append(Article(
            title=title,
            url=url or f"https://finance.yahoo.com/quote/{ticker}",
            source=source,
            published_at=_parse_dt(published),
            snippet=body[:500],
            content=body[:4000],
            credibility=credibility_for(source),
        ))
    return articles


def fetch_news(
    ticker: str,
    company: str,
    days: Optional[int] = None,
    max_articles: Optional[int] = None,
    use_cache: bool = True,
) -> list[Article]:
    """Return recent news articles for a company (Event Registry → yfinance fallback)."""
    days = days or settings.news_lookback_days
    max_articles = max_articles or settings.news_max_articles
    key = f"news_{ticker}_{days}_{max_articles}"

    if use_cache:
        cached = cache.read_json(key)
        if cached:
            logger.info("news for %s from cache (%d)", ticker, len(cached))
            return [Article(**a) for a in cached]

    articles: list[Article] = []
    if settings.has_news_api:
        try:
            articles = _from_event_registry(company, days, max_articles)
            logger.info("Event Registry returned %d articles for %s", len(articles), company)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Event Registry failed for %s: %s", company, exc)

    if not articles:
        try:
            articles = _from_yfinance_news(ticker, max_articles)
            logger.info("yfinance news returned %d articles for %s", len(articles), ticker)
        except Exception as exc:  # noqa: BLE001
            logger.warning("yfinance news failed for %s: %s", ticker, exc)

    if articles:
        cache.write_json(key, [a.model_dump(mode="json") for a in articles])
    return articles
