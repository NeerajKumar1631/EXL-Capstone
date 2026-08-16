"""Typed data contracts shared across all modules/agents.

Every cross-module boundary passes one of these pydantic models (or a pandas
DataFrame for raw price/feature tables). Keeping them here makes the pipeline
easy to reason about and test in isolation.
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

Action = Literal["Buy", "Hold", "Sell"]
SentimentLabel = Literal["positive", "negative", "neutral"]

DISCLAIMER = (
    "This is an AI-generated analysis for educational purposes only and is NOT "
    "financial advice. Markets are risky; do your own research and consult a "
    "licensed advisor before investing."
)


# ── News ──────────────────────────────────────────────────────────────
class Article(BaseModel):
    title: str
    url: str
    source: str = "unknown"
    published_at: Optional[datetime] = None
    snippet: str = ""
    content: str = ""

    # enrichment (filled by later pipeline stages)
    relevance_score: float = 0.0
    sentiment_label: Optional[SentimentLabel] = None
    sentiment_score: float = 0.0          # signed compound in [-1, 1]
    sentiment_confidence: float = 0.0     # model probability of the chosen label
    event_type: Optional[str] = None
    credibility: float = 0.5              # source credibility weight in [0, 1]

    @property
    def text(self) -> str:
        body = self.snippet or self.content
        return (f"{self.title}. {body}").strip()


class SentimentSummary(BaseModel):
    weighted_score: float                 # credibility-weighted compound in [-1, 1]
    label: SentimentLabel
    n_articles: int
    n_positive: int = 0
    n_negative: int = 0
    n_neutral: int = 0


class NewsResult(BaseModel):
    summary: str = ""
    sentiment: SentimentSummary
    top_articles: list[Article] = Field(default_factory=list)
    n_collected: int = 0
    n_after_dedup: int = 0


# ── Forecasting ───────────────────────────────────────────────────────
class ModelMetrics(BaseModel):
    rmse: float
    mae: float
    mape: float
    r2: float
    directional_accuracy: float           # fraction of correct sign predictions, 0..1
    skill_vs_baseline: float              # (baseline_rmse - model_rmse) / baseline_rmse


class HorizonForecast(BaseModel):
    horizon: str                          # "1d", "1w", "1m"
    horizon_days: int
    predicted_return: float               # cumulative log-return over the horizon
    predicted_price: float
    lower: Optional[float] = None
    upper: Optional[float] = None


class ModelForecast(BaseModel):
    name: str
    weight: float = 1.0
    metrics: Optional[ModelMetrics] = None
    horizons: list[HorizonForecast] = Field(default_factory=list)

    @property
    def next_day(self) -> Optional[HorizonForecast]:
        for h in self.horizons:
            if h.horizon == "1d":
                return h
        return self.horizons[0] if self.horizons else None


class StrategyBacktest(BaseModel):
    """'Follow the forecast' vs buy-and-hold over the held-out window."""

    n_days: int
    strategy_return: float                # total, over the window
    buy_hold_return: float
    excess_return: float                  # strategy - buy_and_hold
    strategy_annualised: Optional[float] = None
    buy_hold_annualised: Optional[float] = None
    days_in_market: int = 0
    trades: int = 0
    cost_bps: float = 0.0
    win_rate: Optional[float] = None      # of the days it was long, how many rose
    beat_buy_and_hold: bool = False
    notes: list[str] = Field(default_factory=list)


class ForecastResult(BaseModel):
    ticker: str
    last_close: float
    as_of: datetime
    models: list[ModelForecast]           # individual models
    ensemble: ModelForecast               # weighted ensemble
    baseline_metrics: ModelMetrics        # naive persistence baseline
    beats_baseline: bool
    best_model: str
    # Prediction intervals (split conformal). `interval_coverage` is measured on data not
    # used to set the width; None means the holdout was too small to check honestly.
    interval_level: float = 0.0           # e.g. 0.80 for an 80% interval; 0 = unavailable
    interval_coverage: Optional[float] = None
    interval_n_calibration: int = 0
    strategy: Optional[StrategyBacktest] = None
    # backtest arrays for the "actual vs predicted" chart
    backtest_dates: list[str] = Field(default_factory=list)
    backtest_actual: list[float] = Field(default_factory=list)
    backtest_pred: list[float] = Field(default_factory=list)
    feature_importance: dict[str, float] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


# ── Context ───────────────────────────────────────────────────────────
class MarketContext(BaseModel):
    macro: dict[str, float] = Field(default_factory=dict)
    fundamentals: dict[str, object] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


# ── Risk & History ────────────────────────────────────────────────────
class BigMove(BaseModel):
    date: str
    pct: float                            # single-day % move


class RiskProfile(BaseModel):
    annual_volatility: float              # annualized std of daily returns
    max_drawdown: float                   # most negative peak-to-trough fraction (e.g. -0.35)
    drawdown_peak: Optional[str] = None
    drawdown_trough: Optional[str] = None
    beta: Optional[float] = None          # vs region benchmark
    benchmark: Optional[str] = None
    var_95: float = 0.0                   # historical 1-day VaR (negative daily return)
    var_99: float = 0.0
    week52_high: float = 0.0
    week52_low: float = 0.0
    price_position_52w: float = 0.0       # 0=at 52w low, 1=at 52w high
    sharpe_like: Optional[float] = None   # annualized mean/vol (rf=0)
    biggest_up: list[BigMove] = Field(default_factory=list)
    biggest_down: list[BigMove] = Field(default_factory=list)
    rolling_vol_dates: list[str] = Field(default_factory=list)
    rolling_vol: list[float] = Field(default_factory=list)
    drawdown_dates: list[str] = Field(default_factory=list)
    drawdown_series: list[float] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# ── Track record ──────────────────────────────────────────────────────
class GradedRun(BaseModel):
    """One past prediction, checked against what the price actually did."""

    ticker: str
    company: str = ""
    when: str
    action: str
    confidence: float = 0.0
    last_close: float
    predicted_price: float
    actual_price: float
    predicted_return: float               # fraction, e.g. +0.004
    actual_return: float
    direction_correct: bool
    abs_pct_error: float                  # |predicted - actual| / actual


class TrackRecord(BaseModel):
    """How the saved predictions have actually performed.

    `pending` and `unverifiable` are reported rather than hidden: a prediction whose next
    trading day has not happened yet is not a miss, and one we cannot line up against a
    price bar must not be silently counted either way.
    """

    graded: list[GradedRun] = Field(default_factory=list)
    total_runs: int = 0
    pending: int = 0                      # next trading day has not happened yet
    unverifiable: int = 0                 # could not match the run to a price bar
    n_graded: int = 0
    n_correct: int = 0
    hit_rate: Optional[float] = None      # None until at least one run is graded
    mean_abs_pct_error: Optional[float] = None
    notes: list[str] = Field(default_factory=list)


# ── Symbol search ─────────────────────────────────────────────────────
class SymbolHit(BaseModel):
    """One match from a symbol-or-company-name search."""

    symbol: str
    name: str = ""
    exchange: str = ""
    in_region: bool = False               # matches the user's currently selected market

    @property
    def label(self) -> str:
        """Human-readable option text, e.g. 'Apple Inc. — AAPL (NASDAQ)'."""
        bits = self.name or self.symbol
        suffix = f" ({self.exchange})" if self.exchange else ""
        return f"{bits} — {self.symbol}{suffix}"


# ── Screener ──────────────────────────────────────────────────────────
class ScoreCard(BaseModel):
    ticker: str
    company: str = ""
    composite: float = 0.0                # 0-100 overall
    momentum: float = 0.0                 # 0-100 sub-score
    trend: float = 0.0                    # 0-100 sub-score
    low_vol: float = 0.0                  # 0-100 sub-score (higher = calmer)
    last_close: float = 0.0
    ret_1m: float = 0.0
    ret_3m: float = 0.0
    rsi: float = 0.0
    annual_volatility: float = 0.0


class Leaderboard(BaseModel):
    region: str
    index_key: str
    index_name: str
    cards: list[ScoreCard] = Field(default_factory=list)
    total_constituents: int = 0
    requested: int = 0
    scored: int = 0
    failed: int = 0
    capped: bool = False


# ── Compare ───────────────────────────────────────────────────────────
class CompareItem(BaseModel):
    ticker: str
    company: str = ""
    ok: bool = True
    error: Optional[str] = None
    action: Optional[str] = None
    confidence: float = 0.0
    last_close: float = 0.0
    next_day_return: float = 0.0
    directional_accuracy: float = 0.0
    beats_baseline: bool = False
    sentiment_label: str = "neutral"
    sentiment_score: float = 0.0
    annual_volatility: float = 0.0
    max_drawdown: float = 0.0
    beta: Optional[float] = None
    dates: list[str] = Field(default_factory=list)      # rebased-to-100 overlay
    rebased: list[float] = Field(default_factory=list)


class ComparisonResult(BaseModel):
    items: list[CompareItem] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# ── Recommendation ────────────────────────────────────────────────────
class Recommendation(BaseModel):
    action: Action
    confidence: float                     # 0..1
    thesis: str                           # the user-requested "why buy / why not" summary
    positive_factors: list[str] = Field(default_factory=list)
    negative_factors: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)
    opportunities: list[str] = Field(default_factory=list)
    disclaimer: str = DISCLAIMER


# ── Top-level result ──────────────────────────────────────────────────
class AnalysisResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    ticker: str
    company_name: str = ""
    as_of: datetime
    forecast: Optional[ForecastResult] = None
    news: Optional[NewsResult] = None
    context: Optional[MarketContext] = None
    recommendation: Optional[Recommendation] = None
    risk: Optional[RiskProfile] = None         # v2: risk & history metrics
    prices: Optional[pd.DataFrame] = None      # OHLCV history for charts
    # `errors`/`warnings` are plain-English and shown to the user; `details` carries the
    # matching technical text (exception type + message) for a collapsed debug view.
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    details: list[str] = Field(default_factory=list)
