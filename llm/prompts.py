"""Prompt builders. All facts (numbers, article titles/sources) are supplied explicitly
so the model summarizes/reasons over given evidence and does not invent it.
"""
from __future__ import annotations

from orchestration.schemas import ForecastResult, MarketContext, Article, SentimentSummary


def _fmt_articles(articles: list[Article]) -> str:
    lines = []
    for i, a in enumerate(articles, 1):
        when = a.published_at.strftime("%Y-%m-%d") if a.published_at else "n/a"
        lines.append(
            f"[{i}] ({a.sentiment_label or 'n/a'}, {when}, {a.source}) {a.title}\n"
            f"    {a.snippet[:280]}"
        )
    return "\n".join(lines) if lines else "(no articles)"


def summary_prompt(company: str, ticker: str, articles: list[Article]) -> str:
    return (
        f"You are a financial news analyst. Summarize the recent news about {company} ({ticker}) "
        f"in 4-6 sentences. Focus on facts that could move the stock (earnings, guidance, products, "
        f"legal, macro). Only use the articles below; do not invent facts. End with the overall tone.\n\n"
        f"ARTICLES:\n{_fmt_articles(articles)}"
    )


def recommendation_prompt(
    company: str,
    ticker: str,
    forecast: ForecastResult,
    sentiment: SentimentSummary,
    context: MarketContext,
    articles: list[Article],
) -> str:
    e = forecast.ensemble
    nd = e.next_day
    horizon_txt = ", ".join(
        f"{h.horizon}: {h.predicted_return*100:+.2f}% (${h.predicted_price:.2f})" for h in e.horizons
    )
    fund = ", ".join(f"{k}={v}" for k, v in list(context.fundamentals.items())[:10]) or "n/a"
    macro = ", ".join(f"{k}={v}" for k, v in context.macro.items()) or "n/a"

    return (
        f"You are a prudent equity research assistant. Decide Buy, Hold, or Sell for {company} ({ticker}) "
        f"for a retail investor, using ONLY the evidence below. Never invent numbers or events. "
        f"If the model shows no skill over the naive baseline, weight the news/fundamentals more and lower your confidence.\n\n"
        f"=== QUANTITATIVE FORECAST (next-day log returns; honestly evaluated) ===\n"
        f"Last close: ${forecast.last_close:.2f}\n"
        f"Ensemble next-day: {nd.predicted_return*100:+.2f}% -> ${nd.predicted_price:.2f}\n"
        f"Multi-horizon: {horizon_txt}\n"
        f"Directional accuracy (holdout): {e.metrics.directional_accuracy*100:.0f}%  |  "
        f"Skill vs naive baseline: {e.metrics.skill_vs_baseline:+.3f}  |  "
        f"Beats baseline: {forecast.beats_baseline}\n"
        f"(If beats_baseline is False, the price model is essentially a coin-flip — say so.)\n\n"
        f"=== NEWS SENTIMENT (FinBERT, credibility-weighted) ===\n"
        f"Overall: {sentiment.label} (score {sentiment.weighted_score:+.3f}) across {sentiment.n_articles} articles "
        f"(+{sentiment.n_positive}/-{sentiment.n_negative}/~{sentiment.n_neutral}).\n\n"
        f"=== FUNDAMENTALS ===\n{fund}\n\n"
        f"=== MACRO ===\n{macro}\n\n"
        f"=== TOP ARTICLES ===\n{_fmt_articles(articles)}\n\n"
        f"Return JSON with: action (Buy/Hold/Sell); confidence (0-1, calibrated — low if the model lacks skill "
        f"and news is mixed); thesis (3-5 sentence plain-English 'why buy or why not' for a retail investor); "
        f"positive_factors, negative_factors, risks, opportunities (each a short list, each item grounded in a "
        f"number or an article above). Be balanced and never guarantee returns."
    )
