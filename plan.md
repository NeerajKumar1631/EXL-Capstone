# StockSense AI — Implementation Plan

> **Status:** living document. Check off tasks as they land; add/adjust as reality dictates.
> Build order is a **vertical slice first** (get one end-to-end path running early to catch
> integration bugs), then widen to the full pipeline. Final deliverable = the **complete** flow.
> See `architecture.md` for design and `CLAUDE.md` for working rules.

## Goal

A runnable, end-to-end stock-analysis agent: ticker → forecast + news sentiment → grounded
Buy/Hold/Sell with a "why buy" thesis → Streamlit dashboard. US market first (best free data);
NSE later. LLM = Gemini 2.5 Flash. Not financial advice.

## Legend
`[ ]` todo `[~]` in progress `[x]` done

---

## Phase 0 — Foundation
- [x] Confirm design & decisions with user
- [x] Folder scaffold
- [x] `architecture.md`, `plan.md`, `CLAUDE.md`
- [x] `requirements.txt`, `.env` / `.env.example`, `.gitignore`
- [x] Python 3.13 venv + install deps (via `uv`; pip was broken by a macOS-26 `mac_ver` bug)
- [x] Smoke test: all 23 core imports OK
- [x] `config/settings.py` (pydantic-settings, reads `.env`), `config/logging_config.py`
- [x] `config/sources.py` (RSS list, credibility weights)
- [x] `orchestration/schemas.py` — all pydantic contracts
- [x] `utils/` — retry, timing helpers
- [x] `agents/base.py` — uniform Agent interface (run / log / error handling)

## Phase 1 — Vertical slice (end-to-end walking skeleton)
Goal: `analyze("AAPL")` returns a real recommendation and the dashboard renders.
- [ ] `data_ingestion/prices.py` — yfinance + cache + Stooq fallback
- [ ] `technical_analysis/features.py` + `indicators.py` — indicators, lags, returns, vol
- [ ] `forecasting/baseline.py` + `forecasting/gbm_models.py` (XGBoost only for the slice)
- [ ] `forecasting/evaluate.py` — RMSE/MAE/MAPE/R², directional accuracy, skill vs baseline
- [ ] `data_ingestion/news.py` — NewsAPI (RSS added in Phase 3)
- [ ] `retrieval/dedup.py` — RapidFuzz
- [ ] `sentiment/finbert.py` + `aggregate.py`
- [ ] `llm/client.py` (Gemini) + `llm/prompts.py` + `llm/summarizer.py`
- [ ] `recommendation/engine.py` — fusion + grounded reco + "why buy" thesis
- [ ] `orchestration/pipeline.py` — DAG wiring (parallel quant + news)
- [ ] `frontend/app.py` — minimal dashboard renders the full result
- [ ] **Milestone: run it end-to-end on AAPL, verify Gemini key works**

## Phase 2 — Forecasting depth  ✅ (built & tested on AAPL)
- [x] `forecasting/arima_model.py` (SARIMAX walk-forward on return series, guarded)
- [x] XGBoost + LightGBM + CatBoost in `forecasting/models.py`
- [x] `forecasting/ensemble.py` — inverse validation-RMSE weighted ensemble
- [x] Multi-horizon: next day / week / month (per-horizon cumulative targets)
- [x] `forecaster.py` — time-split, CV weights, holdout metrics, backtest arrays, beats_baseline
- [ ] Tests for forecasting metrics + no-lookahead validation (Phase 6)
- Note: models honestly do NOT reliably beat the naive baseline on daily returns (skill≈0,
  dirAcc≈50%). This is the correct, defensible finding — surfaced in the UI, not hidden.
- Note: skipped sklearn model persistence to `models_store/` — retrain is ~9s and results are
  cached at the Streamlit layer; persistence is a future optimization.

## Phase 3 — News & retrieval depth  ✅ (built & tested on AAPL)
- [x] `data_ingestion/news.py` — Event Registry (primary) + yfinance `.news` fallback (RSS dropped)
- [x] `retrieval/dedup.py` — RapidFuzz + URL dedup
- [x] `embeddings/encoder.py` — MiniLM (cached singleton)
- [x] `retrieval/ranker.py` — BM25 + semantic cosine fusion, top-k relevance
- [x] `sentiment/finbert.py` (512-token truncation, cached) + `sentiment/aggregate.py` (credibility-weighted)
- Verified: 13 on-topic Apple articles → deduped → ranked → FinBERT → aggregate positive 0.305.

## Phase 4 — Context, fusion & orchestration  ✅ (built & tested end-to-end)
- [x] `data_ingestion/context.py` — fundamentals (P/E, margins, growth) + macro proxies (VIX/DXY/US10Y)
- [x] `llm/client.py` — Gemini w/ model fallback chain + backoff + structured JSON; `llm/prompts.py`, `summarizer.py`
- [x] `recommendation/engine.py` — grounded LLM reco + "why buy" thesis + rule-based fallback + grounding post-check
- [x] `agents/pipeline_agents.py` — 9 thin agent wrappers; `orchestration/pipeline.py` — concurrent DAG, safe_run degradation
- Verified: `analyze("AAPL")` returns full result with LLM (Buy 0.62) AND with LLM off (rule-based Buy 0.37), 0 errors.

## Phase 5 — Full dashboard & polish  ✅ (built & verified via AppTest + live boot)
- [x] `visualization/charts.py` — Plotly (price+forecast, backtest, technical, sentiment gauge, importances)
- [x] `database/` — SQLite persistence of runs (`db.py`, `models.py`)
- [x] Streamlit multipage: Dashboard + Forecast, Technical, News, Sentiment, Recommendation, History
- [x] Session caching, live `st.status` progress, graceful error/warning surfaces
- [x] "Not financial advice" disclaimer in sidebar + on the recommendation page
- Verified: all 7 pages load with no exceptions; server boots (health=ok) at :8503.

## Phase 6 — Hardening & docs  ✅ (core complete)
- [x] `tests/` — 14 offline unit tests (leakage, metrics/skill, dedup, aggregation, ER-parse mock,
      grounding, rule-based fallback) — all pass
- [x] `docs/`: agent_workflow.md, database_schema.md, api_reference.md, development_guide.md
- [x] `README.md` — overview, install (+ macOS gotchas), env vars, run, examples, troubleshooting, mermaid
- [x] End-to-end verification: AAPL + MSFT (LLM & rule-based) + invalid ticker (graceful)
- [ ] Dockerfile + docker-compose (deferred — documented in README/plan)
- [ ] Postgres swap (deferred — SQLite works; one-file change behind `database/db.py`)
- [ ] (Optional) LangChain orchestration if tool-use is introduced (deferred)

---

## Open questions / risks
- **Gemini key format** (`AQ.Ab8…`) is non-standard — verify at the Phase 1 milestone; if it fails,
  user regenerates a standard `AIza…` key.
- **yfinance reliability** — mitigated by cache + Stooq fallback.
- **NewsAPI free tier** — limited history (~1 month) and request cap; RSS backfills coverage.
- **Wheel availability on 3.13** — verified fine; all 23 core libs import.

## macOS setup gotchas (document in README)
- **pip broken on macOS 26.x + Homebrew py3.13:** `platform.mac_ver()` returns `''`, crashing pip's
  truststore. Fixed with `.venv/.../site-packages/_macver_patch.py` + `aaa_macver_patch.pth` (falls back
  to `sw_vers`). We install with **`uv`** to sidestep pip entirely.
- **xgboost/lightgbm need OpenMP:** run `brew install libomp` or they fail to load `libomp.dylib`.
- **pyarrow mimalloc segfault (v2):** Streamlit converts DataFrames→Arrow in its script-runner thread;
  pyarrow's bundled mimalloc crashes in `mi_thread_init`. Fixed by `ARROW_DEFAULT_MEMORY_POOL=system`,
  set at startup in the venv `_macver_patch.py` (.pth) and by `run.sh`. Crash report:
  `mi_heap_main ← arrow::py::NdarrayToArrow ← pyarrow.array`.
- **langchain + xgboost segfault (v2):** import xgboost before `langchain-google-genai` (done in `chat/agent.py`).

---
---







# v2 Roadmap — Explore · Risk · Screener · Compare · Report · Chat · Polish

v1 above is **built, tested, and running**. v2 adds a US⇄India market/index browser, a Risk & History
module, an index Screener, Compare, Report export + Watchlist, a LangChain conversational analyst, and a
modern-light "finance-pro" visual polish. Locked decisions: **LangChain** chat agent (with a non-LLM
fallback); **modern-light finance-pro** theme.

## Guardrails (apply to every v2 phase)
- **Additive-only — never break v1.** `analyze()` stays as-is; `AnalysisResult` only *gains* an optional
  `risk` field; new session keys / DB table `watchlist` don't touch existing behavior or the `runs` table.
- **Test per phase + regression gate:** each phase ships edge-case unit tests AND must leave
  `pytest tests/ -q` green + an `AppTest` sweep of every page (existing 7 + new) exception-free.
- **Grounding & honesty preserved**; "not financial advice" on every output; graceful degradation
  (`safe_run`); chat falls back to a deterministic intent-parser if LangChain/Gemini is unavailable;
  screener never silently truncates (reports coverage).

## Phase v2.0 — Foundations (theme + universe + market helpers)  ✅
- [x] `.streamlit/config.toml` (light finance-pro theme) + `visualization/theme.py` (shared Plotly template
      registered + applied via import in `charts.py`) + `frontend/_style.py` (light CSS, injected in `_shared.setup`)
- [x] `config/universe.py` + `data/universe/{us,india}.json` — Dow30/Nasdaq100(subset)/S&P500(subset)/sectors
      & Nifty50/Sensex/NiftyBank/Nifty500(subset); `scripts/refresh_universe.py` (best-effort)
- [x] `data_ingestion/markets.py` — `normalize_ticker` (auto `.NS`), `infer_region`, `benchmark_for`
- [x] `config/settings.py` — additive: `default_region`, `screener_max_constituents`, `screener_concurrency`
- [x] `tests/test_universe.py` + `scripts/apptest_all_pages.py` (reusable regression gate)
- **Gate passed:** 22 tests pass (14 v1 + 8 new); all 7 pages load with theme; v1 intact.

## Phase v2.1 — Explore landing + market switcher (UI)  ✅
- [x] `_shared.py` region session key + sidebar market radio (US/India) + region-aware ticker normalization
- [x] `_shared.explore()` + `app.py` branch: Explore (index tiles + constituent picker + recent) when no
      result, else Dashboard. Manual ticker normalized by region on run.
- [x] `tests/test_explore.py` — Explore renders index tiles for both US and India, no exceptions
- **Gate passed:** 23 tests pass; AppTest sweep green; v1 dashboard path unchanged.

## Phase v2.2 — Risk & History module  ✅
- [x] `analytics/risk.py` — `compute_risk(prices, benchmark, name)`: vol, max drawdown (+dates), beta,
      52w range/position, historical VaR (95/99), Sharpe-like, biggest N moves, rolling-vol & drawdown series
- [x] schemas: **new** `RiskProfile`/`BigMove`; **added** `AnalysisResult.risk` (optional)
- [x] `RiskAgent` + benchmark fetch (Stage 1) + risk compute (Stage 2); `drawdown_chart`,
      `rolling_vol_chart`; `frontend/pages/7_Risk.py`
- [x] `tests/test_risk.py` (6): monotonic→0 DD, known −50% DD, flat→0 vol/None Sharpe, beta-vs-self≈1,
      None-benchmark→None beta, biggest-move ordering + VaR signs
- **Gate passed:** 29 tests; AAPL risk (vol 28.7%, β 1.09, DD −33.4%); all 8 pages load; v1 intact.

## Phase v2.3 — Screener / leaderboard  ✅
- [x] `screener/score.py` — `quick_score` (LLM-free: momentum + trend + low-vol composite, cached)
- [x] `screener/screener.py` — `screen(region, index, limit)` (concurrency-capped, partial-failure
      tolerant, coverage reported, honest cap flag)
- [x] schemas: **new** `ScoreCard`, `Leaderboard`; `frontend/pages/8_Screener.py` (leaderboard → analyze)
- [x] `tests/test_screener.py` (3): ranks desc + cap flag; tolerates failures + coverage; all-fail → empty
- **Gate passed:** 32 tests; live dow30 screen (6 scored, ranked, AMGN top); all 9 pages load; v1 intact.

## Phase v2.4 — Compare  ✅
- [x] `compare/compare.py` — `compare(tickers)` (concurrent analyze, dedupe, cap 3, invalid-flagged,
      rebased-to-100 overlay); schemas: **new** `CompareItem`/`ComparisonResult`; `compare_prices_chart`;
      `frontend/pages/9_Compare.py`
- [x] `tests/test_compare.py` (3): dedupe+order+rebase-to-100, cap at 3 + note, invalid-among-valid flagged
- **Gate passed:** 35 tests; all 10 pages load; v1 intact.

## Phase v2.5 — Report export + Watchlist  ✅
- [x] `report/export.py` — `to_markdown`/`to_html` (guarded for partial results) + optional `to_pdf` (fpdf2);
      `_shared.report_downloads` wired to Dashboard + Recommendation
- [x] `database/{models,db}.py` — **new** `watchlist` table + `add_watch/remove_watch/list_watch/is_watched`;
      sidebar watchlist expander + Dashboard add/remove button
- [x] `tests/test_report.py` (3) + `tests/test_watchlist.py` (2): all sections present, partial ok, HTML wrap;
      add/list/dedupe/remove + `runs` table intact
- **Gate passed:** 40 tests; all 10 pages load; v1 intact.

## Phase v2.6 — Conversational analyst (LangChain)  ✅
- [x] `langchain` + `langchain-google-genai` installed (uses our google-genai SDK)
- [x] `chat/tools.py` — 5 LangChain tools (analyze/risk/screen/compare/resolve), each returns grounded
      text and never raises; `chat/agent.py` — `ChatAgent` (langgraph tool-calling + memory +
      grounding/anti-injection system prompt) + **deterministic intent-parser fallback**; `pages/10_Ask.py`
- [x] `tests/test_chat_tools.py` (3) + `tests/test_chat_fallback.py` (6, LLM-free routing)
- [x] **Live-verified:** agent orchestrated analyze_stock + risk_history → grounded answer
- ⚠️ **Fixed a hard segfault:** `langchain-google-genai` + `xgboost` crash (OpenMP runtime conflict)
      unless xgboost is imported first — `chat/agent.py` preimports xgboost/lightgbm/catboost at module top.
- **Gate passed:** 49 tests; all 11 pages load; v1 intact.

## Phase v2.7 — Visual polish pass (finance-pro light)  ✅
- [x] Shared Plotly template on all charts; enriched CSS (hero band, cards, KPI tiles, tile-style buttons,
      dataframe/chat styling); Explore hero header
- **Gate passed:** 49 tests; all 11 pages load; server boots (health ok).

## Phase v2.8 — Integration + edge-case sweep + docs  ✅
- [x] `scripts/integration_e2e.py` — US + India analysis, invalid + low-history edge cases, screener,
      mixed-region compare, report export, watchlist, chat (LLM + fallback) — **12/12 checks pass**
- [x] Full `pytest` (49) + AppTest all 11 pages; **v1 regression** unchanged (`analyze('AAPL')` → Buy + risk)
- [x] Docs refresh: README (v2 features, pages, deps, OpenMP gotcha), architecture.md (§8b), docs/*, plan.md

## v2 verification
```bash
.venv/bin/python -m pytest tests/ -q                       # per phase — must stay green
PYTHONPATH=. .venv/bin/python -c "from orchestration.pipeline import analyze; r=analyze('AAPL', use_llm=False); assert r.forecast and r.recommendation; print('v1 OK')"   # regression
PYTHONPATH=. .venv/bin/python scripts/integration_e2e.py   # final cross-feature sweep
.venv/bin/streamlit run frontend/app.py                    # manual walkthrough
```
**Done when:** all phase tests + integration pass, v1 regression unchanged, every page renders in AppTest,
and each new feature works on ≥1 US and ≥1 India ticker with graceful invalid/low-history/quota handling.
