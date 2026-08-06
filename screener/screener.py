"""Index screener: batch-score constituents and rank them into a Leaderboard.

Concurrency-capped, per-ticker cached, and partial-failure tolerant — huge indices are
time-boxed via `settings.screener_max_constituents` and coverage is reported (never a
silent truncation).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

from config import universe
from config.logging_config import get_logger
from config.settings import settings
from orchestration.schemas import Leaderboard, ScoreCard
from screener.score import quick_score

logger = get_logger("screener")


def _safe_score(ticker: str, region: str) -> Optional[ScoreCard]:
    try:
        return quick_score(ticker, region)
    except Exception as exc:  # noqa: BLE001
        logger.warning("score failed for %s: %s", ticker, exc)
        return None


def screen(
    region: str,
    index_key: str,
    limit: Optional[int] = None,
    progress: Optional[callable] = None,
) -> Leaderboard:
    """Score and rank an index's constituents (highest composite first)."""
    info = universe.get_index(region, index_key)
    cap = limit or settings.screener_max_constituents
    tickers = list(info.tickers[:cap])
    capped = len(info.tickers) > len(tickers)

    cards: list[ScoreCard] = []
    failed = 0
    done = 0
    with ThreadPoolExecutor(max_workers=settings.screener_concurrency) as ex:
        futures = {ex.submit(_safe_score, t, region): t for t in tickers}
        for fut in as_completed(futures):
            done += 1
            card = fut.result()
            if card is None:
                failed += 1
            else:
                cards.append(card)
            if progress:
                progress(f"Scored {done}/{len(tickers)}…")

    cards.sort(key=lambda c: c.composite, reverse=True)
    logger.info("screened %s/%s: %d scored, %d failed%s",
                index_key, region, len(cards), failed, " (capped)" if capped else "")
    return Leaderboard(
        region=region.upper(), index_key=index_key, index_name=info.name, cards=cards,
        total_constituents=info.count, requested=len(tickers), scored=len(cards),
        failed=failed, capped=capped,
    )
