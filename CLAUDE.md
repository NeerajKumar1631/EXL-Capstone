# CLAUDE.md — working rules for StockSense AI

This file governs how any AI assistant (and human) works in this repo. Read it before making changes.

## Prime directive: keep the docs in sync

`plan.md` and `architecture.md` are **living documents**. They are not write-once.

- **Before** starting a task, check `plan.md` for the current phase and mark the task `[~]`.
- **After** finishing a task, mark it `[x]` in `plan.md`.
- **Whenever** you change a module contract, data flow, tech choice, or folder layout, update
  `architecture.md` in the **same change** — never let code and `architecture.md` diverge.
- If you discover a new risk or open question, add it to the relevant doc.
- Keep edits surgical: update the specific section, don't rewrite whole files.

## What we're building (one line)

A stock-analysis agent: ticker → ML forecast + news sentiment → grounded Buy/Hold/Sell with a
"why buy" thesis, in a Streamlit dashboard. **Not financial advice** — say so on every output.

## Non-negotiable engineering rules

1. **Reuse, don't rebuild.** If a maintained library/API provides a capability, wrap it. Only write
   custom logic when nothing suitable exists (orchestration, ensemble, ranker, fusion/recommendation).
2. **Forecast honesty.** Predict next-day **return** (not just price). Always validate with
   `TimeSeriesSplit` (no lookahead) and always report **directional accuracy** and **skill vs. the
   naive persistence baseline**. If nothing beats the baseline, say so.
3. **No hallucinated evidence.** The LLM may only cite numbers and article URLs it was given. Every
   recommendation factor is grounded; a post-check verifies cited URLs exist in the input set.
4. **Secrets stay in `.env`.** Never hardcode or commit API keys. `.env` is gitignored;
   `.env.example` documents the variables.
5. **Typed contracts.** Cross-module data uses the pydantic models in `orchestration/schemas.py`.
6. **Every agent/module** has logging, graceful error handling, type hints, and docstrings (PEP-8).

## Build order

Vertical slice first (get end-to-end running), then widen. Current phase & tasks live in `plan.md`.
Development order: Foundation → Vertical slice → Forecasting depth → News/retrieval depth →
Context/fusion → Dashboard → Hardening/docs.

## Environment

- Python **3.13** venv at `.venv/` (activate before running anything).
- Run the app: `streamlit run frontend/app.py`.
- Config via `.env` (see `.env.example`). Keys currently provisioned: `NEWS_API_KEY`, `GEMINI_API_KEY`.
- CPU-only; no GPU required (FinBERT + MiniLM run on CPU).

## Conventions

- Package-relative imports; keep modules small and single-purpose.
- Cache expensive I/O (prices, news, model artifacts) under `data_cache/` and `models_store/`
  (both gitignored).
- Prefer pure functions for data transforms; keep side effects (I/O, network) at the edges.
- Tests in `tests/`, run with `pytest`.
