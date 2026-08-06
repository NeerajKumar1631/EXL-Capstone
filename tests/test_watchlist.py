"""Watchlist DB ops. Uses a sentinel ticker and cleans up so the real DB is unaffected."""
from database.db import add_watch, is_watched, list_watch, recent_runs, remove_watch

_SENTINEL = "ZZ_TEST_TICK"


def test_add_list_dedupe_remove():
    remove_watch(_SENTINEL)  # ensure clean start
    try:
        assert add_watch(_SENTINEL, "US") is True
        assert is_watched(_SENTINEL) is True
        assert any(w["ticker"] == _SENTINEL for w in list_watch())
        assert add_watch(_SENTINEL, "US") is False   # idempotent — already present
    finally:
        remove_watch(_SENTINEL)
    assert is_watched(_SENTINEL) is False
    remove_watch(_SENTINEL)                            # remove-missing is a no-op


def test_runs_table_still_readable():
    # adding the watchlist table must not break the existing runs table
    assert isinstance(recent_runs(1), list)
