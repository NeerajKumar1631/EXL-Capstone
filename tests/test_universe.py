import pytest

from config import universe
from config.universe import UniverseError
from data_ingestion.markets import benchmark_for, infer_region, normalize_ticker


# ── markets helpers ───────────────────────────────────
def test_normalize_ticker_india_suffix():
    assert normalize_ticker("reliance", "INDIA") == "RELIANCE.NS"
    assert normalize_ticker("TCS.NS", "INDIA") == "TCS.NS"      # idempotent
    assert normalize_ticker("500325.BO", "INDIA") == "500325.BO"  # keep BSE suffix
    assert normalize_ticker("aapl", "US") == "AAPL"              # US unchanged


def test_infer_region_and_benchmark():
    assert infer_region("TCS.NS") == "INDIA"
    assert infer_region("RELIANCE.BO") == "INDIA"
    assert infer_region("AAPL") == "US"
    assert benchmark_for("US") == "^GSPC"
    assert benchmark_for("INDIA") == "^NSEI"


# ── universe loading ──────────────────────────────────
def test_regions_present():
    assert set(universe.regions()) == {"US", "INDIA"}


@pytest.mark.parametrize("region", ["US", "INDIA"])
def test_every_index_loads_and_is_wellformed(region):
    idxs = universe.list_indices(region)
    assert idxs, f"no indices for {region}"
    for info in idxs:
        assert info.count == len(info.tickers) > 0
        # unique, non-empty, uppercase, correctly suffixed
        assert len(set(info.tickers)) == len(info.tickers), f"dupes in {info.key}"
        for t in info.tickers:
            assert t and t == t.upper().strip()
            if region == "INDIA":
                assert t.endswith((".NS", ".BO")), f"{t} missing NSE/BSE suffix"


def test_nifty50_is_reasonably_complete():
    info = universe.get_index("INDIA", "nifty50")
    assert info.full is True
    assert 45 <= info.count <= 51
    assert "RELIANCE.NS" in info.tickers


def test_subset_flag_is_honest():
    assert universe.get_index("US", "sp500").full is False   # curated subset
    assert universe.get_index("US", "dow30").full is True


def test_unknown_region_or_index_raises():
    with pytest.raises(UniverseError):
        universe.get_index("MARS", "x")
    with pytest.raises(UniverseError):
        universe.get_index("US", "no_such_index")
