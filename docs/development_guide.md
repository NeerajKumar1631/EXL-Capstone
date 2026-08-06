# Development Guide

## Setup
See the README. TL;DR: Python 3.13 venv → `uv pip install -r requirements.txt` → `brew install libomp`
(macOS) → copy `.env.example` to `.env`.

## Run
```bash
.venv/bin/streamlit run frontend/app.py      # dashboard
.venv/bin/python -m pytest tests/ -q          # tests
PYTHONPATH=. .venv/bin/python -c "from orchestration.pipeline import analyze; print(analyze('AAPL').recommendation.action)"
```

## Conventions
- Cross-module data uses the pydantic models in `orchestration/schemas.py`.
- Each capability is a domain module wrapped by a thin `Agent` (`agents/pipeline_agents.py`).
- Network/model calls: cache (`database/cache.py`), retry, and degrade — never crash the pipeline.
- Keep the honesty rules: predict returns, validate with `TimeSeriesSplit`, report skill vs. baseline,
  ground every LLM claim. See `CLAUDE.md`.

## Adding a new model
Add a `RegressorModel` factory in `forecasting/models.py` and include it in `build_gbm_models()`.
The forecaster picks up CV weights, holdout metrics, and horizons automatically.

## Adding a news source
Add a `_from_<source>()` in `data_ingestion/news.py` returning `list[Article]`, and wire it into
`fetch_news` (primary or fallback). Normalize to the `Article` schema; set `credibility` via
`config/sources.credibility_for`.

## Swapping the LLM
`llm/client.py` is provider-agnostic behind `generate_text` / `generate_json`. Change the SDK calls
there and update `settings.gemini_models`; nothing else changes.

## Extending the dashboard
Add a `frontend/pages/<n>_Name.py` that starts with `import _shared; _shared.setup("Name")` then
`r = _shared.require_result()`. Read from `r` (an `AnalysisResult`).

## Known environment quirks (macOS 26 + Homebrew Python)
`pip`/`mac_ver` bug (use `uv` + `_macver_patch.pth`), `libomp` for GBMs, broken `pyexpat` (no XML/RSS).
Documented in the README and `plan.md`.
