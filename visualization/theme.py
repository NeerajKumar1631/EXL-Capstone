"""Shared visual theme: a registered Plotly template + palette constants.

Importing this module registers the `stocksense_light` template and makes it the
default, so every `go.Figure()` created afterwards inherits consistent fonts, grid,
background and colorway — without changing any chart function's signature.
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio

# ── Palette ───────────────────────────────────────────────
PRIMARY = "#2563eb"
ACCENT = "#0d9488"
POSITIVE = "#16a34a"
NEGATIVE = "#dc2626"
WARNING = "#d97706"
MUTED = "#64748b"
GRID = "#e5e7eb"
BG = "#ffffff"
TEXT = "#0f172a"

COLORWAY = [PRIMARY, ACCENT, "#f59e0b", "#8b5cf6", "#ec4899", "#14b8a6", "#ef4444", MUTED]

ACTION_COLORS = {"Buy": POSITIVE, "Hold": WARNING, "Sell": NEGATIVE}
SENTIMENT_COLORS = {"positive": POSITIVE, "negative": NEGATIVE, "neutral": MUTED}

_FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif"

_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        font=dict(family=_FONT, color=TEXT, size=13),
        paper_bgcolor=BG,
        plot_bgcolor=BG,
        colorway=COLORWAY,
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, showline=True),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, showline=True),
        hoverlabel=dict(bgcolor="white", font_size=12, font_family=_FONT),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        title=dict(font=dict(size=16, color=TEXT)),
        margin=dict(t=48, b=24, l=8, r=8),
    )
)

pio.templates["stocksense_light"] = _TEMPLATE
pio.templates.default = "stocksense_light"
