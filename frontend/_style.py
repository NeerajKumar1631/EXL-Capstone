"""Application stylesheet: brand chrome, cards, tiles, badges, states.

Styles our own classes plus a small number of stable Streamlit test-ids. Kept in one
place so the visual language stays consistent across all eleven views.
"""
from __future__ import annotations

import streamlit as st

_CSS = """
<style>
:root {
  --ss-primary:#2563eb; --ss-primary-dark:#1d4ed8;
  --ss-ink:#0f172a; --ss-muted:#64748b; --ss-line:#e5e7eb;
  --ss-pos:#16a34a; --ss-neg:#dc2626; --ss-neutral:#94a3b8; --ss-warn:#d97706;
  --ss-surface:#ffffff; --ss-surface-2:#f8fafc;
}

/* ── Remove Streamlit's own chrome so this reads as a product, not a demo ── */
#MainMenu {display:none;}
footer {display:none;}
[data-testid="stToolbar"] {display:none;}
[data-testid="stDecoration"] {display:none;}
[data-testid="stAppDeployButton"] {display:none;}
[data-testid="stStatusWidget"] {display:none;}

.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1240px; }

h1, h2, h3 { letter-spacing:-0.015em; color:var(--ss-ink); }
h1 { font-weight:800; font-size:1.9rem; margin-bottom:.15rem; }
h2 { font-weight:700; font-size:1.3rem; }
h3 { font-weight:650; font-size:1.05rem; }

/* ── Sidebar brand ── */
.ss-brand { display:flex; align-items:center; gap:10px; padding:2px 0 6px 0; }
.ss-brand-mark {
  width:34px; height:34px; border-radius:9px; flex:0 0 34px;
  background:linear-gradient(135deg, var(--ss-primary) 0%, #0ea5e9 100%);
  display:flex; align-items:center; justify-content:center;
}
.ss-brand-name { font-weight:800; font-size:1.06rem; color:var(--ss-ink); line-height:1.1; }
.ss-brand-sub  { font-size:.72rem; color:var(--ss-muted); letter-spacing:.02em; }

/* ── Page header ── */
.ss-page-sub { color:var(--ss-muted); font-size:.92rem; margin:-2px 0 14px 0; }

/* ── Cards & tiles ── */
.ss-card {
  background:var(--ss-surface); border:1px solid var(--ss-line); border-radius:14px;
  padding:16px 18px; box-shadow:0 1px 2px rgba(16,24,40,.04);
}
.ss-muted { color:var(--ss-muted); font-size:.85rem; }

div[data-testid="stMetric"] {
  background:var(--ss-surface); border:1px solid var(--ss-line); border-radius:12px;
  padding:12px 14px; box-shadow:0 1px 2px rgba(16,24,40,.03);
}
div[data-testid="stMetricLabel"] { color:var(--ss-muted); font-weight:500; }

/* ── Verdict badge ── */
.ss-verdict {
  display:flex; align-items:baseline; justify-content:center; gap:14px;
  padding:16px 20px; border-radius:14px; color:#fff; font-weight:700;
}
.ss-verdict .ss-action { font-size:1.6rem; letter-spacing:.02em; }
.ss-verdict .ss-conf { font-size:.95rem; font-weight:600; opacity:.92; }

/* ── Sentiment dots (replaces emoji) ── */
.ss-dot {
  display:inline-block; width:9px; height:9px; border-radius:50%;
  margin-right:7px; vertical-align:middle;
}
.ss-dot-positive { background:var(--ss-pos); }
.ss-dot-negative { background:var(--ss-neg); }
.ss-dot-neutral  { background:var(--ss-neutral); }
.ss-tone { font-size:.86rem; color:var(--ss-muted); white-space:nowrap; }

/* ── Empty state ── */
.ss-empty {
  background:var(--ss-surface-2); border:1px dashed #cbd5e1; border-radius:16px;
  padding:30px 26px; text-align:center;
}
.ss-empty h3 { margin:0 0 6px 0; color:var(--ss-ink); }
.ss-empty p { margin:0 auto; max-width:520px; color:var(--ss-muted); font-size:.92rem; }

/* ── Loading skeleton ── */
.ss-skel {
  border-radius:12px; background:linear-gradient(90deg,#eef2f7 25%,#e2e8f0 37%,#eef2f7 63%);
  background-size:400% 100%; animation:ss-shimmer 1.3s ease-in-out infinite;
}
@keyframes ss-shimmer { 0%{background-position:100% 50%} 100%{background-position:0 50%} }

/* ── Buttons ── */
.stButton > button {
  border-radius:10px; border:1px solid var(--ss-line); font-weight:600;
  transition:border-color .15s, box-shadow .15s, transform .05s;
  white-space:pre-line; text-align:left; padding:12px 14px;
}
.stButton > button:hover { border-color:var(--ss-primary); box-shadow:0 4px 14px rgba(37,99,235,.10); }
.stButton > button:active { transform:translateY(1px); }
.stButton > button[kind="primary"] { text-align:center; }

/* ── Tables & chat ── */
div[data-testid="stDataFrame"] { border:1px solid var(--ss-line); border-radius:12px; }
div[data-testid="stChatMessage"] { background:var(--ss-surface); border:1px solid var(--ss-line); border-radius:12px; }
</style>
"""

# Brand mark: an upward trend line. Inline so there is no external asset to ship.
_LOGO_SVG = (
    '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" '
    'stroke="white" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">'
    '<polyline points="3,17 9,11 13,15 21,7"/><polyline points="15,7 21,7 21,13"/></svg>'
)

BRAND = (
    f'<div class="ss-brand"><div class="ss-brand-mark">{_LOGO_SVG}</div>'
    f'<div><div class="ss-brand-name">StockSense</div>'
    f'<div class="ss-brand-sub">FORECAST · SENTIMENT · RISK</div></div></div>'
)


def inject() -> None:
    """Inject the stylesheet (cheap; safe to call on every rerun)."""
    st.markdown(_CSS, unsafe_allow_html=True)
