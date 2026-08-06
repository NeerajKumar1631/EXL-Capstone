"""Regression gate: load every Streamlit page via AppTest and assert no exceptions.

Injects a precomputed AnalysisResult into session_state so data-dependent pages render
fully. Run:  PYTHONPATH=.:frontend .venv/bin/python scripts/apptest_all_pages.py
"""
from __future__ import annotations

import glob
import pathlib
import sys
import warnings

warnings.filterwarnings("ignore")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "frontend"))

from streamlit.testing.v1 import AppTest  # noqa: E402

from orchestration.pipeline import analyze  # noqa: E402


def main() -> int:
    result = analyze("AAPL", use_llm=False)
    print(f"seed result: forecast={result.forecast is not None} reco={result.recommendation.action}")

    pages = [str(ROOT / "frontend" / "app.py")]
    pages += sorted(glob.glob(str(ROOT / "frontend" / "pages" / "*.py")))

    failures = 0
    for page in pages:
        rel = page.replace(str(ROOT) + "/", "")
        at = AppTest.from_file(page, default_timeout=90)
        at.session_state["results"] = {("AAPL", False): result}
        at.session_state["ticker"] = "AAPL"
        at.session_state["use_llm"] = False
        at.session_state["region"] = "US"
        at.run()
        if at.exception:
            failures += 1
            print(f"FAIL {rel}")
            for e in at.exception:
                print("   ", repr(e)[:300])
        else:
            print(f"OK   {rel}")

    print("\nALL PAGES OK" if failures == 0 else f"\n{failures} PAGE(S) FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
