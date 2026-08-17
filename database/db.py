"""SQLite persistence for analysis runs (history view)."""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from datetime import datetime

from config.logging_config import get_logger
from config.settings import settings
from database.models import Base, DailySentiment, Run, Watch
from orchestration.schemas import AnalysisResult

logger = get_logger("database")


@lru_cache(maxsize=1)
def _session_factory():
    # Restore the database before the engine opens it — on an ephemeral host this is the
    # only chance to recover history from a previous run. No-op unless HF sync is configured.
    from database import sync

    sync.pull()
    engine = create_engine(f"sqlite:///{settings.db_path}", future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, class_=Session, future=True)


def _sync_push() -> None:
    """Queue a database upload after a write (no-op unless HF sync is configured)."""
    from database import sync

    sync.push()


def save_run(result: AnalysisResult) -> Optional[int]:
    """Persist a completed analysis. Returns the row id, or None if nothing to save."""
    if result.recommendation is None or result.forecast is None:
        return None
    try:
        nd = result.forecast.ensemble.next_day
        run = Run(
            created_at=result.as_of,
            ticker=result.ticker,
            company=result.company_name,
            action=result.recommendation.action,
            confidence=result.recommendation.confidence,
            last_close=result.forecast.last_close,
            next_day_price=nd.predicted_price if nd else result.forecast.last_close,
            beats_baseline=result.forecast.beats_baseline,
            best_model=result.forecast.best_model,
            sentiment_label=result.news.sentiment.label if result.news else "neutral",
            sentiment_score=result.news.sentiment.weighted_score if result.news else 0.0,
            thesis=result.recommendation.thesis,
        )
        with _session_factory()() as s:
            s.add(run)
            s.commit()
            run_id = run.id
        _sync_push()
        return run_id
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to persist run: %s", exc)
        return None


def recent_runs(limit: int = 25) -> list[dict]:
    """Most recent persisted runs as plain dicts (for a history table)."""
    try:
        with _session_factory()() as s:
            rows = s.query(Run).order_by(Run.created_at.desc()).limit(limit).all()
            return [
                {
                    "when": r.created_at.strftime("%Y-%m-%d %H:%M"),
                    "ticker": r.ticker,
                    "company": r.company,
                    "action": r.action,
                    "confidence": round(r.confidence, 2),
                    "last_close": round(r.last_close, 2),
                    "next_day": round(r.next_day_price, 2),
                    "sentiment": r.sentiment_label,
                    "beats_baseline": r.beats_baseline,
                }
                for r in rows
            ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to read runs: %s", exc)
        return []


def all_runs(limit: int = 500) -> list[dict]:
    """Every persisted run with the fields needed to grade it against what happened.

    Distinct from `recent_runs`, which formats for display; this keeps raw types so the
    track-record scorer can do arithmetic and date lookups on them.
    """
    try:
        with _session_factory()() as s:
            rows = s.query(Run).order_by(Run.created_at.desc()).limit(limit).all()
            return [
                {
                    "id": r.id,
                    "created_at": r.created_at,
                    "ticker": r.ticker,
                    "company": r.company,
                    "action": r.action,
                    "confidence": r.confidence,
                    "last_close": r.last_close,
                    "next_day_price": r.next_day_price,
                    "beats_baseline": r.beats_baseline,
                }
                for r in rows
            ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to read runs: %s", exc)
        return []


def latest_run_by_ticker() -> dict[str, dict]:
    """The most recent run for each ticker, keyed by ticker (for the watchlist cards)."""
    latest: dict[str, dict] = {}
    for row in all_runs():
        latest.setdefault(row["ticker"], row)   # all_runs() is newest-first
    return latest


# ── Daily sentiment history ───────────────────────────────
def record_sentiment(ticker: str, day: str, score: float, label: str, n_articles: int) -> None:
    """Store one day's sentiment reading (last write wins for a given ticker+day).

    The news API serves only ~4 weeks of history, so a sentiment feature can't be trained on
    the past — this accumulates it going forward instead.
    """
    try:
        with _session_factory()() as s:
            row = (s.query(DailySentiment)
                   .filter(DailySentiment.ticker == ticker, DailySentiment.day == day)
                   .first())
            if row:
                row.score, row.label = score, label
                row.n_articles, row.recorded_at = n_articles, datetime.now()
            else:
                s.add(DailySentiment(ticker=ticker, day=day, score=score, label=label,
                                     n_articles=n_articles, recorded_at=datetime.now()))
            s.commit()
        _sync_push()
    except Exception as exc:  # noqa: BLE001 - never break an analysis over bookkeeping
        logger.warning("record_sentiment failed for %s %s: %s", ticker, day, exc)


def sentiment_history(ticker: str) -> dict[str, float]:
    """Every stored sentiment reading for a ticker, as {'YYYY-MM-DD': score}."""
    try:
        with _session_factory()() as s:
            rows = s.query(DailySentiment).filter(DailySentiment.ticker == ticker).all()
            return {r.day: r.score for r in rows}
    except Exception as exc:  # noqa: BLE001
        logger.warning("sentiment_history failed for %s: %s", ticker, exc)
        return {}


# ── Watchlist ─────────────────────────────────────────────
def add_watch(ticker: str, region: str = "US") -> bool:
    """Add a ticker to the watchlist (idempotent). Returns True if added."""
    ticker = ticker.strip().upper()
    if not ticker:
        return False
    try:
        with _session_factory()() as s:
            if s.query(Watch).filter(Watch.ticker == ticker).first():
                return False
            s.add(Watch(ticker=ticker, region=region.upper(), added_at=datetime.now()))
            s.commit()
        _sync_push()
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("add_watch failed: %s", exc)
        return False


def remove_watch(ticker: str) -> None:
    ticker = ticker.strip().upper()
    try:
        with _session_factory()() as s:
            row = s.query(Watch).filter(Watch.ticker == ticker).first()
            if row:
                s.delete(row)
                s.commit()
        _sync_push()
    except Exception as exc:  # noqa: BLE001
        logger.warning("remove_watch failed: %s", exc)


def list_watch() -> list[dict]:
    try:
        with _session_factory()() as s:
            rows = s.query(Watch).order_by(Watch.added_at.desc()).all()
            return [{"ticker": r.ticker, "region": r.region} for r in rows]
    except Exception as exc:  # noqa: BLE001
        logger.warning("list_watch failed: %s", exc)
        return []


def is_watched(ticker: str) -> bool:
    ticker = ticker.strip().upper()
    try:
        with _session_factory()() as s:
            return s.query(Watch).filter(Watch.ticker == ticker).first() is not None
    except Exception:  # noqa: BLE001
        return False
