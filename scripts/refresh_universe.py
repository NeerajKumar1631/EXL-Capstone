"""Best-effort refresh of index constituents into data/universe/*.json.

The bundled JSON files are the source of truth (curated, always-valid tickers). This
script tries to pull fuller lists (e.g. S&P 500 from Wikipedia via pandas.read_html) and
rewrite the JSON when a parser is available. It is intentionally optional and guarded —
if parsing deps are missing it prints guidance and changes nothing.

Usage:  PYTHONPATH=. .venv/bin/python scripts/refresh_universe.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
US_JSON = ROOT / "data" / "universe" / "us.json"


def refresh_sp500() -> bool:
    try:
        import pandas as pd

        tables = pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies")
        tickers = [str(t).replace(".", "-").strip() for t in tables[0]["Symbol"].tolist()]
        tickers = [t for t in tickers if t]
        if len(tickers) < 400:
            print(f"parsed only {len(tickers)} tickers — aborting to avoid a bad list")
            return False
        data = json.loads(US_JSON.read_text())
        data["indices"]["sp500"] = {
            "name": "S&P 500", "exchange": "US", "full": True, "tickers": tickers,
        }
        US_JSON.write_text(json.dumps(data, indent=2))
        print(f"updated sp500 with {len(tickers)} tickers")
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"could not refresh S&P 500 ({type(exc).__name__}: {exc}).")
        print("Bundled curated subset remains in use. Install lxml/html5lib to enable, or edit the JSON by hand.")
        return False


if __name__ == "__main__":
    ok = refresh_sp500()
    sys.exit(0 if ok else 1)
