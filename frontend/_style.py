"""Design system: tokens, chrome, dark sidebar, components, motion.

One file owns the whole visual language so all thirteen views stay consistent.

Structure of the sheet, in order:
  1. Tokens        — colors, radii, shadows, fonts, easing
  2. Chrome        — Streamlit menus/footer removed; canvas mesh background
  3. Sidebar       — dark navy panel: nav, inputs, buttons, captions all restyled for dark
  4. Typography    — Inter body, Space Grotesk display, tabular numerals for figures
  5. Components    — hero, KPI cards, section headers, verdict, chips, dots, empty states
  6. Widgets       — buttons, inputs, tables, charts, chat, expanders, alerts
  7. Motion        — entrance fade, hover lifts, status pulse, shimmer; all reduced-motion safe

Hover lifts are gated behind `@media (hover:hover)` so they never stick on touch screens.
Streamlit diffs the DOM between reruns, so entrance animation plays for new elements only.
"""
from __future__ import annotations

import streamlit as st

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

/* ════════ 1. TOKENS ════════ */
:root {
  --ss-primary:#2563eb; --ss-primary-2:#1d4ed8; --ss-sky:#0ea5e9; --ss-teal:#0d9488;
  --ss-ink:#0f172a; --ss-muted:#64748b; --ss-faint:#94a3b8;
  --ss-line:#e2e8f0; --ss-line-2:#cbd5e1;
  --ss-pos:#16a34a; --ss-pos-bg:#e8f7ee; --ss-pos-ink:#15803d;
  --ss-neg:#dc2626; --ss-neg-bg:#fdecec; --ss-neg-ink:#b91c1c;
  --ss-neutral:#94a3b8; --ss-neutral-bg:#eef2f7; --ss-neutral-ink:#475569;
  --ss-surface:#ffffff; --ss-surface-2:#f8fafc;

  /* dark sidebar palette */
  --sb-bg-a:#0e1a33; --sb-bg-b:#0a1122;
  --sb-ink:#e2e8f0; --sb-muted:#8b9bb8; --sb-faint:#5c6b8a;
  --sb-panel:#16233f; --sb-line:#243352;

  --ss-radius:14px; --ss-radius-lg:20px; --ss-radius-sm:10px;
  --ss-shadow-sm:0 1px 2px rgba(15,23,42,.05), 0 1px 1px rgba(15,23,42,.03);
  --ss-shadow-md:0 4px 12px rgba(15,23,42,.07), 0 2px 4px rgba(15,23,42,.05);
  --ss-shadow-lg:0 14px 34px rgba(15,23,42,.13), 0 5px 12px rgba(15,23,42,.06);
  --ss-font-body:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
  --ss-font-display:'Space Grotesk','Inter',sans-serif;
  --ss-ease:cubic-bezier(.22,.9,.35,1);
}

/* ════════ 2. CHROME & CANVAS ════════ */
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"],
[data-testid="stAppDeployButton"], [data-testid="stStatusWidget"] { display:none; }

/* The header is a fixed overlay. Keep it (it holds the sidebar-expand control when the
   sidebar is collapsed) but make it invisible: transparent instead of a white strip that
   cuts across the mesh background, and give content enough top padding to clear it. */
[data-testid="stHeader"] { background:transparent; }

.stApp {
  background:
    radial-gradient(1100px 520px at 88% -12%, rgba(37,99,235,.08), transparent 60%),
    radial-gradient(900px 480px at 12% 110%, rgba(13,148,136,.06), transparent 55%),
    #f5f7fb;
}
.block-container { padding-top:4rem; padding-bottom:4.5rem; max-width:1240px; }

/* ════════ 3. DARK SIDEBAR ════════ */
section[data-testid="stSidebar"] {
  background:linear-gradient(180deg, var(--sb-bg-a) 0%, var(--sb-bg-b) 100%);
  border-right:none; box-shadow:inset -1px 0 0 rgba(255,255,255,.04);
}
section[data-testid="stSidebar"] * { color:var(--sb-ink); }
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] *,
section[data-testid="stSidebar"] small { color:var(--sb-muted); }
section[data-testid="stSidebar"] hr { border-color:var(--sb-line); }

/* nav: quiet links, glowing active pill, uppercase section headers */
section[data-testid="stSidebar"] [data-testid="stSidebarNav"] a,
section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"] {
  border-radius:9px; margin:1px 6px; transition:background .15s;
}
section[data-testid="stSidebar"] a span { color:#b8c4da; font-weight:500; }
section[data-testid="stSidebar"] a:hover { background:rgba(59,118,240,.14); }
section[data-testid="stSidebar"] a[aria-current="page"] {
  background:linear-gradient(90deg, rgba(37,99,235,.32), rgba(14,165,233,.10));
  box-shadow:inset 2px 0 0 var(--ss-sky);
}
section[data-testid="stSidebar"] a[aria-current="page"] span { color:#fff; font-weight:600; }
section[data-testid="stSidebar"] [data-testid="stNavSectionHeader"] {
  color:var(--sb-faint); text-transform:uppercase; letter-spacing:.10em;
  font-size:.66rem; font-weight:700;
}

/* Streamlit overlays "Press Enter to apply" on top of the field — it collides with the
   placeholder and looks broken. The search box has its own help tooltip, so drop it. */
[data-testid="InputInstructions"] { display:none; }

/* sidebar widgets on dark. BaseWeb paints the field surface across several nested
   layers, so paint them ALL dark (belt and braces — transparent inner + styled wrapper
   left the white default showing through). The focus ring lives on :focus-within of the
   outer wrapper, overriding BaseWeb's own white glow. */
section[data-testid="stSidebar"] [data-baseweb="input"],
section[data-testid="stSidebar"] [data-baseweb="base-input"],
section[data-testid="stSidebar"] [data-baseweb="input"] > div,
section[data-testid="stSidebar"] [data-baseweb="base-input"] > div {
  background:var(--sb-panel) !important; border-color:var(--sb-line) !important;
  border-radius:10px;
}
section[data-testid="stSidebar"] [data-baseweb="input"] {
  border:1px solid var(--sb-line) !important;
  transition:border-color .15s, box-shadow .15s;
}
section[data-testid="stSidebar"] [data-baseweb="input"]:focus-within {
  border-color:var(--ss-sky) !important; box-shadow:0 0 0 3px rgba(14,165,233,.22);
}
section[data-testid="stSidebar"] input {
  background:var(--sb-panel) !important; color:var(--sb-ink) !important;
  border:none !important; box-shadow:none !important; outline:none !important;
}
section[data-testid="stSidebar"] input::placeholder { color:var(--sb-faint) !important; }
section[data-testid="stSidebar"] [data-baseweb="select"] > div,
section[data-testid="stSidebar"] [data-baseweb="select"] > div > div {
  background:var(--sb-panel) !important; border-color:var(--sb-line) !important;
  color:var(--sb-ink) !important;
}
section[data-testid="stSidebar"] [data-baseweb="select"] svg { fill:var(--sb-muted); }
section[data-testid="stSidebar"] .stButton > button {
  background:var(--sb-panel); color:var(--sb-ink);
  border:1px solid var(--sb-line); min-height:0; padding:9px 13px;
}
section[data-testid="stSidebar"] .stButton > button:hover { border-color:var(--ss-sky); }
section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
  background:linear-gradient(135deg, var(--ss-primary) 0%, var(--ss-sky) 120%);
  border:none; color:#fff; box-shadow:0 6px 18px rgba(37,99,235,.35);
}
section[data-testid="stSidebar"] div[data-testid="stExpander"] {
  background:rgba(255,255,255,.03); border:1px solid var(--sb-line) !important;
  border-radius:10px !important; box-shadow:none;
}

/* brand */
.ss-brand { display:flex; align-items:center; gap:11px; padding:6px 0 12px 0; }
.ss-brand-mark {
  width:38px; height:38px; border-radius:11px; flex:0 0 38px;
  background:linear-gradient(135deg, var(--ss-primary) 0%, var(--ss-sky) 100%);
  display:flex; align-items:center; justify-content:center;
  box-shadow:0 6px 16px rgba(14,165,233,.35);
}
.ss-brand-name { font-family:var(--ss-font-display); font-weight:700; font-size:1.1rem;
  color:#f1f5f9 !important; line-height:1.1; }
.ss-brand-sub { font-size:.64rem; color:var(--sb-faint) !important; letter-spacing:.14em; }

/* ════════ 4. TYPOGRAPHY ════════ */
html, body, .stApp { font-family:var(--ss-font-body); }
h1, h2, h3 { font-family:var(--ss-font-display); color:var(--ss-ink); letter-spacing:-0.02em; }
h1 { font-weight:700; font-size:2.05rem; margin-bottom:.1rem; }
h2 { font-weight:600; font-size:1.3rem; }
h3 { font-weight:600; font-size:1.02rem; }

.ss-eyebrow {
  color:var(--ss-primary); font-weight:700; font-size:.7rem;
  letter-spacing:.14em; text-transform:uppercase; margin-bottom:2px;
}
.ss-page-sub { color:var(--ss-muted); font-size:.94rem; margin:2px 0 20px 0; max-width:72ch; }

div[data-testid="stMetricValue"] {
  font-family:var(--ss-font-display); font-weight:600;
  font-variant-numeric:tabular-nums; letter-spacing:-0.01em;
}

/* ════════ 5. COMPONENTS ════════ */
/* hero */
.ss-hero {
  position:relative; overflow:hidden; border-radius:var(--ss-radius-lg);
  padding:34px 36px 30px; color:#fff;
  background:linear-gradient(118deg, #0d224f 0%, #123a86 52%, #0d5aa0 100%);
  box-shadow:var(--ss-shadow-lg);
}
.ss-hero::after {
  content:""; position:absolute; inset:0; pointer-events:none;
  background:
    radial-gradient(420px 220px at 85% 0%, rgba(56,189,248,.28), transparent 70%),
    radial-gradient(340px 200px at 8% 110%, rgba(13,148,136,.25), transparent 70%);
}
.ss-hero-eyebrow { position:relative; z-index:1; color:#7dd3fc; font-weight:700;
  font-size:.7rem; letter-spacing:.16em; text-transform:uppercase; }
.ss-hero-title { position:relative; z-index:1; font-family:var(--ss-font-display);
  font-size:1.85rem; font-weight:700; margin:6px 0 8px; letter-spacing:-0.02em; }
.ss-hero-sub { position:relative; z-index:1; color:#c3d5f4; font-size:.95rem;
  max-width:640px; line-height:1.55; }
.ss-hero-chips { position:relative; z-index:1; display:flex; flex-wrap:wrap; gap:10px; margin-top:18px; }
.ss-hero-chip {
  background:rgba(255,255,255,.11); border:1px solid rgba(255,255,255,.20);
  backdrop-filter:blur(4px); padding:6px 14px; border-radius:999px;
  font-size:.78rem; font-weight:600; color:#e6efff;
}

/* KPI cards — a self-aligning grid, replaces bare st.metric rows */
.ss-kpi-grid {
  display:grid; grid-template-columns:repeat(auto-fit, minmax(175px, 1fr));
  gap:14px; margin:8px 0 6px;
}
.ss-kpi {
  background:var(--ss-surface); border:1px solid var(--ss-line);
  border-radius:var(--ss-radius); padding:16px 17px 14px;
  box-shadow:var(--ss-shadow-sm); position:relative; overflow:hidden;
  transition:transform .22s var(--ss-ease), box-shadow .22s var(--ss-ease);
}
.ss-kpi::before {
  content:""; position:absolute; inset:0 0 auto 0; height:3px;
  background:linear-gradient(90deg, var(--ss-primary), var(--ss-sky));
  opacity:0; transition:opacity .22s;
}
@media (hover:hover) {
  .ss-kpi:hover { transform:translateY(-4px); box-shadow:var(--ss-shadow-lg); }
  .ss-kpi:hover::before { opacity:1; }
}
.ss-kpi-label { font-size:.72rem; color:var(--ss-muted); font-weight:600;
  letter-spacing:.06em; text-transform:uppercase; }
.ss-kpi-value { font-family:var(--ss-font-display); font-size:1.48rem; font-weight:700;
  color:var(--ss-ink); margin-top:5px; font-variant-numeric:tabular-nums; line-height:1.15; }
.ss-kpi-delta { display:inline-block; margin-top:7px; padding:2px 10px; border-radius:999px;
  font-size:.76rem; font-weight:600; font-variant-numeric:tabular-nums; }
.ss-kpi-delta.pos { background:var(--ss-pos-bg); color:var(--ss-pos-ink); }
.ss-kpi-delta.neg { background:var(--ss-neg-bg); color:var(--ss-neg-ink); }
.ss-kpi-delta.muted { background:var(--ss-neutral-bg); color:var(--ss-neutral-ink); }
.ss-kpi-sub { margin-top:7px; font-size:.74rem; color:var(--ss-faint); line-height:1.4; }

/* section header with accent bar */
.ss-section { display:flex; align-items:baseline; gap:11px; margin:30px 0 4px; }
.ss-section-bar { width:4px; height:17px; border-radius:2px; align-self:center;
  background:linear-gradient(180deg, var(--ss-primary), var(--ss-sky)); }
.ss-section-title { font-family:var(--ss-font-display); font-size:1.22rem;
  font-weight:600; color:var(--ss-ink); letter-spacing:-0.015em; }
.ss-section-cap { color:var(--ss-faint); font-size:.82rem; }

/* verdict */
.ss-verdict {
  display:flex; align-items:baseline; justify-content:center; gap:14px;
  padding:19px 22px; border-radius:16px; color:#fff; font-weight:700;
}
.ss-verdict .ss-action { font-family:var(--ss-font-display); font-size:1.75rem; letter-spacing:.02em; }
.ss-verdict .ss-conf { font-size:.95rem; font-weight:600; opacity:.92; font-variant-numeric:tabular-nums; }

/* sentiment dots & tone chips */
.ss-dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:7px; vertical-align:middle; }
.ss-dot-positive { background:var(--ss-pos); box-shadow:0 0 0 3px rgba(22,163,74,.14); }
.ss-dot-negative { background:var(--ss-neg); box-shadow:0 0 0 3px rgba(220,38,38,.14); }
.ss-dot-neutral  { background:var(--ss-neutral); box-shadow:0 0 0 3px rgba(148,163,184,.16); }
.ss-tone { font-size:.86rem; color:var(--ss-muted); white-space:nowrap; }

/* cards & empty state */
.ss-card,
div[data-testid="stVerticalBlockBorderWrapper"] {
  background:var(--ss-surface); border:1px solid var(--ss-line);
  border-radius:var(--ss-radius); box-shadow:var(--ss-shadow-sm);
  transition:transform .22s var(--ss-ease), box-shadow .22s var(--ss-ease), border-color .22s;
}
.ss-card { padding:16px 18px; }
@media (hover:hover) {
  div[data-testid="stVerticalBlockBorderWrapper"]:hover, .ss-card:hover {
    transform:translateY(-3px); box-shadow:var(--ss-shadow-lg); border-color:var(--ss-line-2);
  }
}
.ss-muted { color:var(--ss-muted); font-size:.85rem; }
.ss-empty {
  background:linear-gradient(150deg, #eef4ff 0%, #f2fbf9 100%);
  border:1px solid #dbe7fb; border-radius:18px;
  padding:36px 30px; text-align:center; box-shadow:var(--ss-shadow-sm);
}
.ss-empty h3 { margin:0 0 6px 0; }
.ss-empty p { margin:0 auto; max-width:540px; color:var(--ss-muted); font-size:.93rem; }

/* ════════ 6. WIDGETS ════════ */
div[data-testid="stMetric"] {
  background:var(--ss-surface); border:1px solid var(--ss-line);
  border-radius:var(--ss-radius); padding:14px 16px; box-shadow:var(--ss-shadow-sm);
  transition:transform .22s var(--ss-ease), box-shadow .22s var(--ss-ease);
}
@media (hover:hover) {
  div[data-testid="stMetric"]:hover { transform:translateY(-3px); box-shadow:var(--ss-shadow-md); }
}
div[data-testid="stMetricLabel"] { color:var(--ss-muted); font-weight:500; }

div[data-testid="stPlotlyChart"] {
  background:var(--ss-surface); border:1px solid var(--ss-line);
  border-radius:var(--ss-radius); padding:10px 8px 2px 2px; box-shadow:var(--ss-shadow-sm);
  transition:box-shadow .22s var(--ss-ease);
}
@media (hover:hover) { div[data-testid="stPlotlyChart"]:hover { box-shadow:var(--ss-shadow-md); } }

/* Ordinary buttons: compact, centered, quiet — sized like buttons, not cards. */
[data-testid="stMain"] .stButton > button, .stDownloadButton > button {
  border-radius:var(--ss-radius-sm); border:1px solid var(--ss-line);
  font-weight:600; background:var(--ss-surface); padding:9px 16px;
  transition:transform .16s var(--ss-ease), border-color .16s, box-shadow .16s, background .16s;
}
.stButton button p { margin:0; }   /* markdown labels: kill paragraph gaps inside buttons */
@media (hover:hover) {
  [data-testid="stMain"] .stButton > button:hover, .stDownloadButton > button:hover {
    border-color:var(--ss-primary); box-shadow:0 6px 18px rgba(37,99,235,.14); transform:translateY(-2px);
  }
}
[data-testid="stMain"] .stButton > button[kind="primary"] {
  color:#fff; border:none;
  background:linear-gradient(135deg, var(--ss-primary) 0%, var(--ss-primary-2) 100%);
  box-shadow:0 4px 14px rgba(37,99,235,.28);
}
@media (hover:hover) {
  [data-testid="stMain"] .stButton > button[kind="primary"]:hover {
    background:linear-gradient(135deg, #3b76f0 0%, var(--ss-primary) 100%);
    box-shadow:0 8px 22px rgba(37,99,235,.34);
  }
}

/* Explore index tiles ONLY (widgets keyed `idx_*` get a st-key-idx_* container class):
   proper cards — left-aligned, bold name, muted count line, gradient face, arrow on hover. */
[class*="st-key-idx_"] .stButton > button {
  width:100%; height:112px; padding:16px 38px 15px 20px;
  display:flex; align-items:stretch; text-align:left; white-space:pre-line;
  background:linear-gradient(180deg, #ffffff 0%, #fafcff 100%);
  border:1px solid var(--ss-line); border-radius:var(--ss-radius);
  box-shadow:var(--ss-shadow-sm); position:relative; overflow:hidden;
}
/* fixed-height card: name pinned to the top, count pinned to the bottom — rows stay
   perfectly level even when a long name wraps to two lines */
[class*="st-key-idx_"] .stButton > button [data-testid="stMarkdownContainer"] {
  display:flex; flex-direction:column; justify-content:space-between;
  height:100%; width:100%;
}
[class*="st-key-idx_"] .stButton > button p:first-child {
  font-family:var(--ss-font-display); font-weight:600; font-size:1rem; line-height:1.25;
  color:var(--ss-ink); margin:0; letter-spacing:-0.01em;
}
[class*="st-key-idx_"] .stButton > button p:last-child {
  color:var(--ss-muted); font-size:.8rem; font-weight:500; margin:0;
}
[class*="st-key-idx_"] .stButton > button::after {
  content:"→"; position:absolute; right:16px; top:16px;
  color:var(--ss-line-2); font-size:1rem; transition:color .18s, transform .18s;
}
@media (hover:hover) {
  [class*="st-key-idx_"] .stButton > button:hover {
    border-color:var(--ss-primary); transform:translateY(-3px);
    box-shadow:var(--ss-shadow-lg);
  }
  [class*="st-key-idx_"] .stButton > button:hover::after {
    color:var(--ss-primary); transform:translateX(3px);
  }
}

[data-testid="stMain"] div[data-testid="stTextInput"] input { border-radius:var(--ss-radius-sm); }
[data-testid="stMain"] div[data-testid="stTextInput"] input:focus {
  border-color:var(--ss-primary); box-shadow:0 0 0 3px rgba(37,99,235,.15);
}

div[data-testid="stDataFrame"] {
  border:1px solid var(--ss-line); border-radius:var(--ss-radius);
  overflow:hidden; box-shadow:var(--ss-shadow-sm);
}
div[data-testid="stChatMessage"] {
  background:var(--ss-surface); border:1px solid var(--ss-line);
  border-radius:var(--ss-radius); box-shadow:var(--ss-shadow-sm);
}
div[data-testid="stAlert"] { border-radius:var(--ss-radius-sm); }
[data-testid="stMain"] div[data-testid="stExpander"] {
  background:var(--ss-surface); border:1px solid var(--ss-line) !important;
  border-radius:var(--ss-radius-sm) !important; box-shadow:var(--ss-shadow-sm);
}
div[data-testid="stExpander"] summary { font-weight:600; }

::-webkit-scrollbar { width:9px; height:9px; }
::-webkit-scrollbar-thumb { background:var(--ss-line-2); border-radius:8px; }
::-webkit-scrollbar-thumb:hover { background:var(--ss-neutral); }
::-webkit-scrollbar-track { background:transparent; }

/* ════════ 7. MOTION ════════ */
@keyframes ss-fade-up { from {opacity:0; transform:translateY(7px);} to {opacity:1; transform:none;} }
.block-container [data-testid="stVerticalBlock"] > div { animation:ss-fade-up .34s var(--ss-ease) both; }

.ss-skel {
  border-radius:12px;
  background:linear-gradient(90deg,#eef2f7 25%,#e2e8f0 37%,#eef2f7 63%);
  background-size:400% 100%; animation:ss-shimmer 1.25s ease-in-out infinite;
}
@keyframes ss-shimmer { 0%{background-position:100% 50%} 100%{background-position:0 50%} }

[data-testid="stSpinner"] p { color:var(--ss-muted); }
div[data-testid="stExpander"]:has([data-testid="stSpinner"]) {
  border-left:3px solid var(--ss-primary) !important;
  animation:ss-pulse 1.6s ease-in-out infinite;
}
@keyframes ss-pulse { 0%,100% {box-shadow:var(--ss-shadow-sm);} 50% {box-shadow:0 0 0 4px rgba(37,99,235,.10);} }

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation:none !important; transition:none !important; }
}
</style>
"""

# Brand mark: an upward trend line. Inline so there is no external asset to ship.
_LOGO_SVG = (
    '<svg width="21" height="21" viewBox="0 0 24 24" fill="none" '
    'stroke="white" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round">'
    '<polyline points="3,17 9,11 13,15 21,7"/><polyline points="15,7 21,7 21,13"/></svg>'
)

BRAND = (
    f'<div class="ss-brand"><div class="ss-brand-mark">{_LOGO_SVG}</div>'
    f'<div><div class="ss-brand-name">StockSense</div>'
    f'<div class="ss-brand-sub">FORECAST · SENTIMENT · RISK</div></div></div>'
)


def inject() -> None:
    """Inject the stylesheet (cheap; safe to call every rerun)."""
    st.markdown(_CSS, unsafe_allow_html=True)
