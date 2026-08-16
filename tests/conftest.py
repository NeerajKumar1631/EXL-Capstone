import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session", autouse=True)
def isolated_storage(tmp_path_factory):
    """Point the database and disk cache at a throwaway directory for the whole run.

    Without this the suite writes into the real `data_cache/stocksense.db`: the watchlist
    tests add and remove rows there, and `test_pipeline_dag` analyzes a ticker named "TEST",
    which now records a sentiment reading. Tests must not leave data in a user's database.

    pytest manages the directory and removes old ones automatically (it keeps the last few
    runs), so nothing accumulates.
    """
    from config.settings import settings
    from database import db

    sandbox = tmp_path_factory.mktemp("stocksense_test_storage")
    saved = (settings.cache_dir, settings.db_path)
    settings.cache_dir = sandbox
    settings.db_path = sandbox / "stocksense.db"
    settings.ensure_dirs()
    db._session_factory.cache_clear()      # it caches an engine bound to the old path
    try:
        yield sandbox
    finally:
        settings.cache_dir, settings.db_path = saved
        db._session_factory.cache_clear()


@pytest.fixture
def synth_prices():
    """A deterministic synthetic OHLCV series with enough rows for feature windows."""
    n = 220
    idx = pd.date_range("2024-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = pd.DataFrame(
        {
            "Open": close * 0.995,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        },
        index=idx,
    )
    df.index.name = "Date"
    return df
