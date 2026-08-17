"""Mirror the SQLite database to a private Hugging Face Dataset.

Free hosts (Hugging Face Spaces, Render, Railway…) give the container an **ephemeral
disk**: it is wiped on every restart, redeploy and idle-sleep. Without this module the
History, Track Record and Watchlist pages would reset each time — and Track Record is
worthless without accumulated history, since it grades predictions made on earlier days.

How it works:
- `pull()` runs once at startup and restores the last uploaded database.
- `push()` runs after a write and uploads the file back.

Design decisions worth knowing:
- **Throttled.** Uploading on every single write would hammer the Hub and slow the app, so
  pushes are coalesced: at most one every `_MIN_PUSH_INTERVAL` seconds, in a background
  thread so no user ever waits on the network.
- **Never fatal.** Every entry point swallows its exceptions and logs. A sync failure must
  degrade to "this instance keeps its own local data", never break an analysis.
- **Opt-in.** With `HF_DATASET_REPO` unset (the default, and all local development) every
  function is a no-op, so nothing touches the network.
- **Private by default.** The dataset is created with `private=True`; it holds the user's
  analysis history and should not be world-readable.
"""
from __future__ import annotations

import threading
import time
from pathlib import Path

from config.logging_config import get_logger
from config.settings import settings

logger = get_logger("database.sync")

_PATH_IN_REPO = "stocksense.db"
_MIN_PUSH_INTERVAL = 30.0        # seconds between uploads; writes in between are coalesced

_lock = threading.Lock()
_last_push = 0.0
_pending = False
_pulled = False


def _api():
    from huggingface_hub import HfApi

    return HfApi(token=settings.hf_token)


def pull() -> bool:
    """Restore the database from the Hub. Returns True if a file was downloaded.

    Safe to call repeatedly; only the first call per process does anything, because a later
    pull would clobber writes this instance has already made.
    """
    global _pulled
    if not settings.has_db_sync:
        return False
    with _lock:
        if _pulled:
            return False
        _pulled = True

    try:
        from huggingface_hub import hf_hub_download

        local = hf_hub_download(
            repo_id=settings.hf_dataset_repo, filename=_PATH_IN_REPO,
            repo_type="dataset", token=settings.hf_token,
        )
        dest = Path(settings.db_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(local).read_bytes())
        logger.info("restored database from %s (%d bytes)",
                    settings.hf_dataset_repo, dest.stat().st_size)
        return True
    except Exception as exc:  # noqa: BLE001 - a missing/unreachable dataset just means "start fresh"
        logger.info("no database restored from %s (starting fresh): %s",
                    settings.hf_dataset_repo, type(exc).__name__)
        return False


def _upload() -> None:
    path = Path(settings.db_path)
    if not path.exists():
        return
    try:
        api = _api()
        api.create_repo(repo_id=settings.hf_dataset_repo, repo_type="dataset",
                        private=True, exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(path), path_in_repo=_PATH_IN_REPO,
            repo_id=settings.hf_dataset_repo, repo_type="dataset",
            commit_message="Sync StockSense database",
        )
        logger.info("database synced to %s (%d bytes)",
                    settings.hf_dataset_repo, path.stat().st_size)
    except Exception as exc:  # noqa: BLE001 - never break the app over a sync
        logger.warning("database sync failed: %s: %s", type(exc).__name__, exc)


def _worker() -> None:
    """Upload, then keep uploading while more writes arrive, honouring the throttle."""
    global _last_push, _pending
    while True:
        with _lock:
            wait = max(0.0, _MIN_PUSH_INTERVAL - (time.monotonic() - _last_push))
        if wait:
            time.sleep(wait)
        with _lock:
            _pending = False
            _last_push = time.monotonic()
        _upload()
        with _lock:
            if not _pending:          # nothing new arrived while we were uploading
                return


def push() -> None:
    """Queue an upload of the database (throttled, off the request thread)."""
    global _pending
    if not settings.has_db_sync:
        return
    with _lock:
        if _pending:
            return                    # an upload is already queued; it will pick this up
        _pending = True
    threading.Thread(target=_worker, name="stocksense-db-sync", daemon=True).start()
