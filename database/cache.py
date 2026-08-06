"""Simple on-disk cache (parquet for DataFrames, JSON for objects) with TTL."""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from config.settings import settings


def _safe(key: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in key)


def _path(key: str, ext: str) -> Path:
    return settings.cache_dir / f"{_safe(key)}.{ext}"


def _fresh(path: Path, ttl_minutes: int) -> bool:
    if not path.exists():
        return False
    age_min = (time.time() - path.stat().st_mtime) / 60.0
    return age_min <= ttl_minutes


def read_df(key: str, ttl_minutes: Optional[int] = None) -> Optional[pd.DataFrame]:
    ttl = settings.cache_ttl_minutes if ttl_minutes is None else ttl_minutes
    path = _path(key, "parquet")
    if not _fresh(path, ttl):
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def write_df(key: str, df: pd.DataFrame) -> None:
    try:
        df.to_parquet(_path(key, "parquet"))
    except Exception:
        pass  # cache writes are best-effort


def read_json(key: str, ttl_minutes: Optional[int] = None) -> Optional[Any]:
    ttl = settings.cache_ttl_minutes if ttl_minutes is None else ttl_minutes
    path = _path(key, "json")
    if not _fresh(path, ttl):
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def write_json(key: str, obj: Any) -> None:
    try:
        _path(key, "json").write_text(json.dumps(obj, default=str))
    except Exception:
        pass
