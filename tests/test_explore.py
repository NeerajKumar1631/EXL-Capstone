import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "frontend"))

from streamlit.testing.v1 import AppTest  # noqa: E402


def _run_landing(region: str) -> AppTest:
    at = AppTest.from_file(str(ROOT / "frontend" / "app.py"), default_timeout=45)
    at.session_state["results"] = {}          # no result → Explore view
    at.session_state["ticker"] = "AAPL"
    at.session_state["use_llm"] = False
    at.session_state["region"] = region
    at.run()
    return at


def test_explore_renders_for_both_regions():
    for region in ("US", "INDIA"):
        at = _run_landing(region)
        assert not at.exception, f"{region}: {at.exception}"
        # index category tiles are rendered as buttons (≥4 indices per region)
        assert len(at.button) >= 4
