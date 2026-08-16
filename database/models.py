"""SQLAlchemy ORM models for persisted analysis runs."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Run(Base):
    __tablename__ = "runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    created_at: Mapped[datetime] = mapped_column(DateTime)
    ticker: Mapped[str] = mapped_column(String(20), index=True)
    company: Mapped[str] = mapped_column(String(120), default="")
    action: Mapped[str] = mapped_column(String(8), default="Hold")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    last_close: Mapped[float] = mapped_column(Float, default=0.0)
    next_day_price: Mapped[float] = mapped_column(Float, default=0.0)
    beats_baseline: Mapped[bool] = mapped_column(Boolean, default=False)
    best_model: Mapped[str] = mapped_column(String(40), default="")
    sentiment_label: Mapped[str] = mapped_column(String(12), default="neutral")
    sentiment_score: Mapped[float] = mapped_column(Float, default=0.0)
    thesis: Mapped[str] = mapped_column(Text, default="")


class Watch(Base):
    __tablename__ = "watchlist"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(20), unique=True, index=True)
    region: Mapped[str] = mapped_column(String(10), default="US")
    added_at: Mapped[datetime] = mapped_column(DateTime)


class DailySentiment(Base):
    """One credibility-weighted sentiment reading per ticker per day.

    Exists because the news API only serves ~4 weeks of history, so a sentiment feature
    cannot be trained on the past — it has to be accumulated going forward. Every analysis
    writes one row here; once coverage is deep enough the forecaster can use it as a real
    model input (see `technical_analysis/features.py`).
    """

    __tablename__ = "daily_sentiment"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    ticker: Mapped[str] = mapped_column(String(20), index=True)
    day: Mapped[str] = mapped_column(String(10), index=True)     # YYYY-MM-DD
    score: Mapped[float] = mapped_column(Float, default=0.0)     # weighted, -1..1
    label: Mapped[str] = mapped_column(String(12), default="neutral")
    n_articles: Mapped[int] = mapped_column(Integer, default=0)
    recorded_at: Mapped[datetime] = mapped_column(DateTime)

    __table_args__ = (UniqueConstraint("ticker", "day", name="uq_daily_sentiment_ticker_day"),)
