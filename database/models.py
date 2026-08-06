"""SQLAlchemy ORM models for persisted analysis runs."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text
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
