"""Regression gate: load every view via AppTest and assert no exceptions.

Since the app moved to `st.navigation`, views are not standalone scripts — they render
inside `app.py`. We drive `app.py` once per view and route with `AppTest.switch_page`.

Each check also asserts the **expected title**, so a routing failure cannot make the
sweep pass by silently rendering the default page every time.

Injects a precomputed AnalysisResult into session_state so data-dependent views render
fully. Run:  PYTHONPATH=.:frontend .venv/bin/python scripts/apptest_all_pages.py
"""
from __future__ import annotations

import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "frontend"))

from streamlit.testing.v1 import AppTest  # noqa: E402

from orchestration.pipeline import analyze  # noqa: E402

APP = str(ROOT / "frontend" / "app.py")

# (page path relative to app.py, expected title). None = the default page.
VIEWS: list[tuple[str | None, str]] = [
    (None, "Apple Inc."),                              # Dashboard, with a result loaded
    ("views/forecast.py", "Forecast"),
    ("views/technical.py", "Technical Indicators"),
    ("views/news.py", "News"),
    ("views/sentiment.py", "News Sentiment"),
    ("views/recommendation.py", "Recommendation"),
    ("views/risk.py", "Risk & History"),
    ("views/screener.py", "Index Screener"),
    ("views/compare.py", "Compare Stocks"),
    ("views/watchlist.py", "Watchlist"),
    ("views/ask.py", "Ask the Analyst"),
    ("views/track_record.py", "Track Record"),
    ("views/history.py", "Analysis History"),
]


def _run(page: str | None, result) -> AppTest:
    at = AppTest.from_file(APP, default_timeout=90)
    at.session_state["results"] = {("AAPL", False): result}
    at.session_state["ticker"] = "AAPL"
    at.session_state["use_llm"] = False
    at.session_state["region"] = "US"
    if page:
        at.switch_page(page)
    at.run()
    return at


def main() -> int:
    result = analyze("AAPL", use_llm=False)
    print(f"seed result: forecast={result.forecast is not None} "
          f"reco={result.recommendation.action if result.recommendation else None}")

    failures = 0
    for page, expected_title in VIEWS:
        label = page or "views/dashboard.py (default)"
        at = _run(page, result)
        titles = [t.value for t in at.title]

        if at.exception:
            failures += 1
            print(f"FAIL {label} — exception")
            for e in at.exception:
                print("   ", repr(e)[:300])
        elif not any(expected_title in t for t in titles):
            failures += 1
            print(f"FAIL {label} — expected title {expected_title!r}, got {titles}")
        else:
            print(f"OK   {label}")

    print("\nALL VIEWS OK" if failures == 0 else f"\n{failures} VIEW(S) FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
