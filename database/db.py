"""SQLite persistence for analysis runs (history view)."""
from __future__ import annotations

from functools import lru_cache
from typing import Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from datetime import datetime

from config.logging_config import get_logger
from config.settings import settings
from database.models import Base, Run, Watch
from orchestration.schemas import AnalysisResult

logger = get_logger("database")


@lru_cache(maxsize=1)
def _session_factory():
    engine = create_engine(f"sqlite:///{settings.db_path}", future=True)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, class_=Session, future=True)


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
            return run.id
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
