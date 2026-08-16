"""Stock universe: bundled index constituents for US and India.

Data lives in `data/universe/{us,india}.json`. Bare symbols are normalized to
exchange-suffixed tickers (e.g. `.NS`) via `data_ingestion.markets.normalize_ticker`.
Large indices are honestly flagged `full=false` (curated subsets); refresh with
`scripts/refresh_universe.py`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache

from config.settings import PROJECT_ROOT
from data_ingestion.markets import REGIONS, normalize_ticker

_DIR = PROJECT_ROOT / "data" / "universe"


class UniverseError(ValueError):
    """Unknown region or index."""


@dataclass(frozen=True)
class IndexInfo:
    region: str
    key: str
    name: str
    full: bool
    count: int
    tickers: tuple[str, ...]     # normalized (exchange-suffixed)


@lru_cache(maxsize=None)
def _load(region: str) -> dict:
    region = region.upper()
    if region not in REGIONS:
        raise UniverseError(f"unknown region '{region}' (expected one of {REGIONS})")
    path = _DIR / f"{region.lower()}.json"
    if not path.exists():
        raise UniverseError(f"missing universe file: {path}")
    return json.loads(path.read_text())


def regions() -> list[str]:
    return list(REGIONS)


def index_keys(region: str) -> list[str]:
    return list(_load(region)["indices"].keys())


def get_index(region: str, key: str) -> IndexInfo:
    data = _load(region)
    indices = data["indices"]
    if key not in indices:
        raise UniverseError(f"unknown index '{key}' for region '{region}'")
    meta = indices[key]
    tickers = tuple(dict.fromkeys(  # de-dupe, preserve order
        normalize_ticker(s, region) for s in meta["tickers"] if s and s.strip()
    ))
    return IndexInfo(
        region=region.upper(), key=key, name=meta.get("name", key),
        full=bool(meta.get("full", False)), count=len(tickers), tickers=tickers,
    )


def list_indices(region: str) -> list[IndexInfo]:
    return [get_index(region, k) for k in index_keys(region)]


def all_indices() -> dict[str, list[IndexInfo]]:
    return {r: list_indices(r) for r in REGIONS}


@lru_cache(maxsize=None)
def searchable(region: str) -> tuple[str, ...]:
    """Every known ticker in a region, de-duplicated and sorted — for search suggestions."""
    seen: dict[str, None] = {}
    for info in list_indices(region):
        for t in info.tickers:
            seen.setdefault(t, None)
    return tuple(sorted(seen))
