"""Tools the conversational analyst can call. Each returns a compact, grounded string
(real numbers + source URLs) and never raises — errors come back as readable text so the
agent can respond gracefully. Plain functions hold the logic (easy to unit-test); the
`@tool` wrappers expose them to LangChain.
"""
from __future__ import annotations

from langchain_core.tools import tool

from config.logging_config import get_logger

logger = get_logger("chat.tools")


def _analyze_text(ticker: str) -> str:
    try:
        from orchestration.pipeline import analyze

        r = analyze(ticker, use_llm=False)
        if r.forecast is None:
            return f"No usable data for {ticker}: {' '.join(r.errors) or 'unknown error'}."
        fc = r.forecast
        nd = fc.ensemble.next_day
        rec = r.recommendation
        parts = [
            f"{r.company_name} ({r.ticker}):",
            f"verdict={rec.action if rec else 'n/a'} (confidence {rec.confidence*100:.0f}%)" if rec else "",
            f"last_close=${fc.last_close:.2f}",
            f"next_day={nd.predicted_return*100:+.2f}% -> ${nd.predicted_price:.2f}",
            f"directional_accuracy={fc.ensemble.metrics.directional_accuracy*100:.0f}%",
            f"beats_naive_baseline={fc.beats_baseline}",
        ]
        if r.news:
            parts.append(f"news_sentiment={r.news.sentiment.label} ({r.news.sentiment.weighted_score:+.2f})")
        if r.risk:
            parts.append(f"annual_vol={r.risk.annual_volatility*100:.0f}% max_drawdown={r.risk.max_drawdown*100:.0f}% beta={r.risk.beta}")
        if r.news and r.news.top_articles:
            srcs = "; ".join(f"{a.title} ({a.url})" for a in r.news.top_articles[:3])
            parts.append(f"sources: {srcs}")
        return " | ".join(p for p in parts if p)
    except Exception as exc:  # noqa: BLE001
        return f"analyze failed for {ticker}: {exc}"


def _risk_text(ticker: str) -> str:
    try:
        from analytics.risk import compute_risk
        from data_ingestion.markets import benchmark_for, infer_region
        from data_ingestion.prices import fetch_prices

        prices = fetch_prices(ticker)
        region = infer_region(ticker)
        try:
            bench = fetch_prices(benchmark_for(region))
        except Exception:
            bench = None
        rk = compute_risk(prices, bench, benchmark_for(region))
        worst = rk.biggest_down[0] if rk.biggest_down else None
        return (f"{ticker} risk: annual_vol={rk.annual_volatility*100:.1f}%, "
                f"max_drawdown={rk.max_drawdown*100:.1f}% ({rk.drawdown_peak}->{rk.drawdown_trough}), "
                f"beta={rk.beta}, 1d_VaR95={rk.var_95*100:.2f}%, "
                f"52w_position={rk.price_position_52w*100:.0f}%"
                + (f", worst_day={worst.pct:.1f}% on {worst.date}" if worst else ""))
    except Exception as exc:  # noqa: BLE001
        return f"risk lookup failed for {ticker}: {exc}"


def _screen_text(region: str, index_key: str) -> str:
    try:
        from screener.screener import screen

        lb = screen(region, index_key, limit=15)
        top = lb.cards[:5]
        rows = "; ".join(f"{c.ticker} (score {c.composite:.0f}, 3m {c.ret_3m*100:+.0f}%)" for c in top)
        return (f"Top of {lb.index_name} [{region}] (scored {lb.scored}/{lb.requested}): {rows}"
                if top else f"No results for {index_key} in {region}.")
    except Exception as exc:  # noqa: BLE001
        return f"screen failed for {index_key}/{region}: {exc}"


def _compare_text(tickers_csv: str) -> str:
    try:
        from compare.compare import compare

        tickers = [t.strip() for t in tickers_csv.replace(" ", ",").split(",") if t.strip()]
        cmp = compare(tickers)
        out = []
        for i in cmp.items:
            if not i.ok:
                out.append(f"{i.ticker}: unavailable")
            else:
                out.append(f"{i.ticker}: verdict={i.action}, next_day={i.next_day_return*100:+.2f}%, "
                           f"sentiment={i.sentiment_label}, vol={i.annual_volatility*100:.0f}%, beta={i.beta}")
        return " || ".join(out) + ("  (" + " ".join(cmp.notes) + ")" if cmp.notes else "")
    except Exception as exc:  # noqa: BLE001
        return f"compare failed: {exc}"


def _resolve_text(query: str, region: str = "US") -> str:
    try:
        from data_ingestion.markets import normalize_ticker
        from data_ingestion.prices import PriceDataError, fetch_prices, resolve_company_name

        norm = normalize_ticker(query, region)
        try:
            fetch_prices(norm)
        except PriceDataError:
            return f"'{query}' -> '{norm}' looks invalid for {region}. Try a valid symbol."
        return f"'{query}' resolves to {norm} ({resolve_company_name(norm)})."
    except Exception as exc:  # noqa: BLE001
        return f"resolve failed for {query}: {exc}"


# ── LangChain tool wrappers ────────────────────────────────
@tool
def analyze_stock(ticker: str) -> str:
    """Full analysis of ONE stock ticker (US like AAPL, or NSE like RELIANCE.NS): verdict,
    forecast, sentiment, risk, and news sources."""
    return _analyze_text(ticker)


@tool
def risk_history(ticker: str) -> str:
    """Risk profile and history for a ticker: volatility, max drawdown, beta, VaR, worst day."""
    return _risk_text(ticker)


@tool
def screen_index(region: str, index_key: str) -> str:
    """Top-ranked stocks in an index. region is 'US' or 'INDIA'; index_key like 'nifty50',
    'sp500', 'dow30', 'sensex', 'niftybank', 'nasdaq100'."""
    return _screen_text(region, index_key)


@tool
def compare_stocks(tickers_csv: str) -> str:
    """Compare 2-3 tickers side by side. Pass a comma-separated list, e.g. 'AAPL,MSFT'."""
    return _compare_text(tickers_csv)


@tool
def resolve_ticker(query: str, region: str = "US") -> str:
    """Validate/normalize a symbol for a region ('US' or 'INDIA') and return its company name."""
    return _resolve_text(query, region)


ALL_TOOLS = [analyze_stock, risk_history, screen_index, compare_stocks, resolve_ticker]
