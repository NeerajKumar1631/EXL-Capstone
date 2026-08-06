"""Light CSS injection for the finance-pro look (cards, KPI tiles, badges, buttons).

Conservative: styles our own classes + a few safe, version-stable tweaks.
"""
from __future__ import annotations

import streamlit as st

_CSS = """
<style>
:root { --ss-primary:#2563eb; --ss-border:#e5e7eb; --ss-ink:#0f172a; --ss-muted:#64748b; }

.block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1200px; }
h1, h2, h3 { letter-spacing: -0.01em; color: var(--ss-ink); }
h1 { font-weight: 800; }

/* hero band on the landing */
.ss-hero {
  background: linear-gradient(135deg, #eef2ff 0%, #f0fdfa 100%);
  border: 1px solid var(--ss-border); border-radius: 16px;
  padding: 18px 22px; margin-bottom: 14px;
}
.ss-hero h3 { margin: 0 0 4px 0; }

/* generic card + tile helpers (used via st.markdown) */
.ss-card {
  background:#fff; border:1px solid var(--ss-border); border-radius:14px;
  padding:16px 18px; box-shadow:0 1px 2px rgba(16,24,40,0.04);
}
.ss-badge { display:inline-block; padding:3px 10px; border-radius:999px; font-size:.78rem; font-weight:600; color:#fff; }
.ss-muted { color:var(--ss-muted); font-size:.85rem; }

/* metric tiles */
div[data-testid="stMetric"] {
  background:#fff; border:1px solid var(--ss-border); border-radius:12px;
  padding:12px 14px; box-shadow:0 1px 2px rgba(16,24,40,0.03);
}
div[data-testid="stMetricLabel"] { color:var(--ss-muted); }

/* buttons: rounded, tile-like for the Explore grid */
.stButton > button {
  border-radius:10px; border:1px solid var(--ss-border); font-weight:600;
  transition: border-color .15s, box-shadow .15s, transform .05s;
  white-space: pre-line; text-align:left; padding:12px 14px;
}
.stButton > button:hover { border-color: var(--ss-primary); box-shadow:0 4px 14px rgba(37,99,235,.10); }
.stButton > button:active { transform: translateY(1px); }
.stButton > button[kind="primary"] { text-align:center; }

/* dataframes: softer border */
div[data-testid="stDataFrame"] { border:1px solid var(--ss-border); border-radius:12px; }

/* tabs / chat spacing */
div[data-testid="stChatMessage"] { background:#fff; border:1px solid var(--ss-border); border-radius:12px; }
</style>
"""


def inject() -> None:
    """Inject the stylesheet (cheap; safe to call every rerun)."""
    st.markdown(_CSS, unsafe_allow_html=True)
