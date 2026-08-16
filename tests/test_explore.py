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


def test_navigation_routes_to_a_different_view():
    """Guards the AppTest sweep: if routing silently fell back to the default page,
    every view check would pass while testing only the dashboard."""
    at = AppTest.from_file(str(ROOT / "frontend" / "app.py"), default_timeout=45)
    at.session_state["results"] = {}
    at.session_state["region"] = "US"
    at.switch_page("views/history.py")
    at.run()
    assert not at.exception, at.exception
    assert any("Analysis History" in t.value for t in at.title), [t.value for t in at.title]
