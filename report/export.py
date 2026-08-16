"""Export an AnalysisResult to a shareable Markdown / HTML report.

Every section is guarded so a partial result (e.g. no news) still renders. PDF export is
optional (fpdf2) and degrades to None if the library isn't installed.
"""
from __future__ import annotations

import html
from typing import Optional

from config.logging_config import get_logger
from orchestration.schemas import AnalysisResult

logger = get_logger("report.export")

# fpdf2's built-in fonts are latin-1 only, and these reports are full of typographic
# characters (em dashes, curly quotes, the disclaimer's warning sign). Rather than shipping
# a Unicode TTF just for this, fold them to ASCII equivalents before writing the PDF.
_PDF_TRANSLITERATIONS = {
    "—": "-", "–": "-", "−": "-",          # em/en dash, minus
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "…": "...", "·": "-", "•": "-",         # ellipsis, middot, bullet
    "→": "->", "≈": "~", "×": "x",
    "⚠": "!", "️": "", "₹": "INR ", "✅": "", "❌": "",
}


def _pdf_safe(text: str) -> str:
    """Make a line printable with fpdf2's built-in latin-1 fonts."""
    for src, dst in _PDF_TRANSLITERATIONS.items():
        text = text.replace(src, dst)
    # Anything still outside latin-1 is dropped rather than crashing the export.
    return text.encode("latin-1", errors="ignore").decode("latin-1")


def to_markdown(result: AnalysisResult) -> str:
    r = result
    lines: list[str] = []
    lines.append(f"# StockSense AI — {r.company_name or r.ticker} ({r.ticker})")
    lines.append(f"_Generated {r.as_of.strftime('%Y-%m-%d %H:%M')}_")
    lines.append("")
    lines.append("> ⚠️ Educational analysis — **not financial advice**.")
    lines.append("")

    if r.recommendation:
        rec = r.recommendation
        lines.append(f"## Recommendation: **{rec.action}** ({rec.confidence*100:.0f}% confidence)")
        lines.append("")
        lines.append(rec.thesis)
        lines.append("")
        for title, items in [("Positive factors", rec.positive_factors),
                             ("Negative factors", rec.negative_factors),
                             ("Risks", rec.risks), ("Opportunities", rec.opportunities)]:
            if items:
                lines.append(f"**{title}:**")
                lines += [f"- {x}" for x in items]
                lines.append("")

    if r.forecast:
        fc = r.forecast
        lines.append("## Forecast")
        lines.append(f"- Last close: ${fc.last_close:,.2f}")
        for h in fc.ensemble.horizons:
            lines.append(f"- {h.horizon}: {h.predicted_return*100:+.2f}% → ${h.predicted_price:,.2f}")
        lines.append(f"- Directional accuracy: {fc.ensemble.metrics.directional_accuracy*100:.0f}% · "
                     f"Beats naive baseline: {fc.beats_baseline}")
        if not fc.beats_baseline:
            lines.append("- ⚠️ The price model does not beat a naive baseline — treat the point forecast as low-confidence.")
        lines.append("")

    if r.risk:
        rk = r.risk
        lines.append("## Risk & History")
        lines.append(f"- Annualized volatility: {rk.annual_volatility*100:.1f}%")
        lines.append(f"- Max drawdown: {rk.max_drawdown*100:.1f}% ({rk.drawdown_peak} → {rk.drawdown_trough})")
        lines.append(f"- Beta: {rk.beta if rk.beta is not None else 'n/a'} · "
                     f"1-day VaR(95%): {rk.var_95*100:.2f}%")
        if rk.biggest_down:
            worst = rk.biggest_down[0]
            lines.append(f"- Worst single day: {worst.pct:.1f}% on {worst.date}")
        lines.append("")

    if r.news:
        lines.append("## News & Sentiment")
        lines.append(f"Overall sentiment: **{r.news.sentiment.label}** ({r.news.sentiment.weighted_score:+.2f})")
        lines.append("")
        lines.append(r.news.summary)
        lines.append("")
        if r.news.top_articles:
            lines.append("**Sources:**")
            lines += [f"- [{a.title}]({a.url}) — {a.source}" for a in r.news.top_articles[:6]]
            lines.append("")

    return "\n".join(lines).strip() + "\n"


def to_html(result: AnalysisResult) -> str:
    """Minimal, dependency-free HTML wrapper around the Markdown report."""
    md = to_markdown(result)
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>StockSense — {html.escape(result.ticker)}</title>"
        "<style>body{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:820px;"
        "margin:40px auto;padding:0 16px;color:#0f172a;line-height:1.5} "
        "pre{white-space:pre-wrap;font-family:inherit}</style></head>"
        f"<body><pre>{html.escape(md)}</pre></body></html>"
    )


def to_pdf(result: AnalysisResult) -> Optional[bytes]:
    """Best-effort PDF via fpdf2; returns None if it isn't installed or generation fails.

    Failures are logged rather than swallowed silently — a bare `except` here previously hid
    a real bug (an em dash in the report crashed the latin-1 core font) and the download
    button simply never appeared.
    """
    try:
        from fpdf import FPDF
    except ImportError:
        logger.info("fpdf2 not installed — PDF export unavailable")
        return None
    try:
        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Helvetica", size=11)
        for line in to_markdown(result).replace("#", "").splitlines():
            text = _pdf_safe(line[:110]).rstrip()
            # new_x/new_y return the cursor to the left margin; without them fpdf2 leaves it
            # at the right edge and the next call fails with "not enough horizontal space".
            pdf.multi_cell(0, 6, text or " ", new_x="LMARGIN", new_y="NEXT")
        return bytes(pdf.output())
    except Exception as exc:  # noqa: BLE001 - export must never break the page
        logger.warning("PDF export failed: %s: %s", type(exc).__name__, exc)
        return None
