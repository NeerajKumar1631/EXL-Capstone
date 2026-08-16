"""End-to-end integration + edge-case sweep across the whole v2/v3 surface.

Exercises: US + India analysis, invalid ticker, low-history guard, risk, screener,
compare (mixed region), report export, watchlist, and the chat agent (LLM + fallback).
Run:  PYTHONPATH=. .venv/bin/python scripts/integration_e2e.py

Runs against a **throwaway database and cache**, so a sweep never leaves rows in the real
`data_cache/stocksense.db` (it writes watchlist entries and daily sentiment readings). The
sandbox is removed on exit — including on failure, and on Ctrl-C.
"""
from __future__ import annotations

import atexit
import shutil
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# ── Sandbox: must be set before anything opens the database or the cache ──
from config.settings import settings  # noqa: E402

_SANDBOX = Path(tempfile.mkdtemp(prefix="stocksense_e2e_"))
settings.cache_dir = _SANDBOX
settings.db_path = _SANDBOX / "stocksense.db"
settings.ensure_dirs()


@atexit.register
def _cleanup() -> None:
    """Remove the sandbox however the script ends — success, exception or Ctrl-C."""
    shutil.rmtree(_SANDBOX, ignore_errors=True)


print(f"sandbox: {_SANDBOX} (removed on exit; your real database is untouched)")

results: list[tuple[str, bool, str]] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    results.append((name, bool(cond), detail))
    print(f"[{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))


# 1) US analysis (full pipeline)
from orchestration.pipeline import analyze

us = analyze("AAPL", use_llm=False)
check("US analyze (AAPL)", us.forecast is not None and us.recommendation is not None,
      f"{us.recommendation.action if us.recommendation else '?'}, risk={'yes' if us.risk else 'no'}")

# 2) India analysis (NSE data + ^NSEI benchmark)
inr = analyze("RELIANCE.NS", use_llm=False)
check("India analyze (RELIANCE.NS)", inr.forecast is not None,
      f"beta={inr.risk.beta if inr.risk else None}")

# 3) Invalid ticker degrades gracefully (no crash)
bad = analyze("ZZINVALIDZZ")
check("Invalid ticker graceful", bad.forecast is None and bool(bad.errors))

# 4) Low-history guard
from forecasting.forecaster import ForecastError, run_forecast

try:
    run_forecast(us.prices.head(80), "AAPL")   # < min_history_rows
    check("Low-history guard", False, "expected ForecastError")
except ForecastError:
    check("Low-history guard", True)

# 5) Screener (US, capped)
from screener.screener import screen

lb = screen("US", "dow30", limit=6)
check("Screener dow30", lb.scored >= 4 and lb.capped, f"scored {lb.scored}/{lb.requested}")

# 6) Compare (mixed US + India)
from compare.compare import compare

cmp = compare(["AAPL", "TCS.NS"])
ok_items = [i for i in cmp.items if i.ok]
check("Compare mixed region", len(ok_items) == 2, f"{[i.ticker for i in ok_items]}")

# 7) Report export
from report.export import to_html, to_markdown

md = to_markdown(us)
check("Report markdown", "Recommendation" in md and "not financial advice" in md.lower())
check("Report html", to_html(us).startswith("<!doctype html>"))

# 8) Watchlist round-trip
from database.db import add_watch, is_watched, list_watch, remove_watch

remove_watch("ZZ_E2E")
add_watch("ZZ_E2E", "US")
check("Watchlist add/list", is_watched("ZZ_E2E") and any(w["ticker"] == "ZZ_E2E" for w in list_watch()))
remove_watch("ZZ_E2E")
check("Watchlist remove", not is_watched("ZZ_E2E"))

# 9) Chat agent — LLM path (if available) and forced fallback
from chat.agent import ChatAgent

agent = ChatAgent()
ans, tools = agent.ask("How risky is AAPL?")
check("Chat answers", bool(ans) and "not financial advice" in ans.lower(),
      f"llm={agent.available}, tools={tools}")

agent._ready = False
fb_ans, fb_tools = agent.ask("compare AAPL and MSFT")
check("Chat fallback routes", fb_tools == ["compare_stocks"] and bool(fb_ans))

# ── Summary ───────────────────────────────────────────
passed = sum(1 for _, ok, _ in results if ok)
print(f"\n=== {passed}/{len(results)} checks passed ===")
raise SystemExit(0 if passed == len(results) else 1)
