"""Database sync to a Hugging Face Dataset — offline, with the Hub mocked.

The contract that matters: sync is opt-in, never blocks a request, and never raises. A
broken sync must degrade to "this instance keeps its own local data", not break an analysis.
"""
from unittest.mock import MagicMock, patch

import pytest

from config.settings import settings
from database import sync


@pytest.fixture(autouse=True)
def clean_module_state():
    """Reset the module's throttle/once-only state between tests."""
    sync._pulled = False
    sync._pending = False
    sync._last_push = 0.0
    yield
    sync._pulled = False
    sync._pending = False


@pytest.fixture
def configured(tmp_path):
    saved = (settings.hf_dataset_repo, settings.hf_token, settings.db_path)
    settings.hf_dataset_repo = "someone/stocksense-db"
    settings.hf_token = "hf_test"
    settings.db_path = tmp_path / "stocksense.db"
    settings.db_path.write_bytes(b"SQLite format 3\x00local")
    try:
        yield
    finally:
        (settings.hf_dataset_repo, settings.hf_token, settings.db_path) = saved


# ── opt-in ─────────────────────────────────────────────────
def test_disabled_by_default_touches_nothing():
    """No HF_DATASET_REPO (all local development) must mean no network at all."""
    saved = settings.hf_dataset_repo
    settings.hf_dataset_repo = ""
    try:
        with patch("huggingface_hub.hf_hub_download") as dl, \
             patch("huggingface_hub.HfApi") as api:
            assert sync.pull() is False
            sync.push()
        dl.assert_not_called()
        api.assert_not_called()
    finally:
        settings.hf_dataset_repo = saved


def test_token_without_repo_is_still_disabled():
    saved = (settings.hf_dataset_repo, settings.hf_token)
    settings.hf_dataset_repo, settings.hf_token = "", "hf_test"
    try:
        assert settings.has_db_sync is False
    finally:
        (settings.hf_dataset_repo, settings.hf_token) = saved


# ── pull ───────────────────────────────────────────────────
def test_pull_restores_the_database(configured, tmp_path):
    remote = tmp_path / "downloaded.db"
    remote.write_bytes(b"SQLite format 3\x00restored-from-hub")
    with patch("huggingface_hub.hf_hub_download", return_value=str(remote)):
        assert sync.pull() is True
    assert settings.db_path.read_bytes() == b"SQLite format 3\x00restored-from-hub"


def test_pull_runs_only_once_per_process(configured, tmp_path):
    """A second pull would clobber writes this instance already made."""
    remote = tmp_path / "d.db"
    remote.write_bytes(b"remote")
    with patch("huggingface_hub.hf_hub_download", return_value=str(remote)) as dl:
        assert sync.pull() is True
        assert sync.pull() is False
    assert dl.call_count == 1


def test_missing_dataset_starts_fresh_instead_of_raising(configured):
    """First ever deploy: nothing uploaded yet. That is normal, not an error."""
    with patch("huggingface_hub.hf_hub_download", side_effect=FileNotFoundError("404")):
        assert sync.pull() is False


# ── push ───────────────────────────────────────────────────
def test_push_uploads_the_database(configured):
    api = MagicMock()
    with patch("huggingface_hub.HfApi", return_value=api):
        sync._upload()
    api.create_repo.assert_called_once()
    assert api.create_repo.call_args.kwargs["private"] is True   # history stays private
    api.upload_file.assert_called_once()
    assert api.upload_file.call_args.kwargs["repo_id"] == "someone/stocksense-db"


def test_upload_failure_is_swallowed(configured):
    with patch("huggingface_hub.HfApi", side_effect=RuntimeError("hub down")):
        sync._upload()      # must not raise


def test_push_does_not_block_the_caller(configured):
    """Uploads run on a daemon thread; a request must never wait on the network."""
    started = []
    with patch("threading.Thread") as thread:
        sync.push()
        started.append(thread.call_args)
    assert started[0].kwargs["daemon"] is True


def test_repeated_pushes_are_coalesced(configured):
    """Bursts of writes must not queue a thread each."""
    with patch("threading.Thread") as thread:
        sync.push()
        sync.push()
        sync.push()
    assert thread.call_count == 1


def test_push_with_no_database_file_is_harmless(configured):
    settings.db_path.unlink()
    api = MagicMock()
    with patch("huggingface_hub.HfApi", return_value=api):
        sync._upload()
    api.upload_file.assert_not_called()
