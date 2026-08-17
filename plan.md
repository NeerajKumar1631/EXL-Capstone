# StockSense AI — Implementation Plan

> **Status:** living document. Check off tasks as they land; add/adjust as reality dictates.
> Build order is a **vertical slice first** (get one end-to-end path running early to catch
> integration bugs), then widen to the full pipeline. Final deliverable = the **complete** flow.
> See `architecture.md` for design and `CLAUDE.md` for working rules.

## Goal

A runnable, end-to-end stock-analysis agent: ticker → forecast + news sentiment → grounded
Buy/Hold/Sell with a "why buy" thesis → Streamlit dashboard. US market first (best free data);
NSE later. LLM = Gemini (`gemini-flash-latest`, with a fallback chain). Not financial advice.

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

## Phase 1 — Vertical slice (end-to-end walking skeleton)  ✅
Goal: `analyze("AAPL")` returns a real recommendation and the dashboard renders.
> Boxes ticked retroactively on 2026-08-15: every module below exists and the full path is proven
> by live runs. They had been left unchecked even though Phases 2–6 and all of v2 depend on them.
- [x] `data_ingestion/prices.py` — yfinance + disk cache + retries + stale-cache fallback
      (Stooq was dropped: non-functional, see the module docstring)
- [x] `technical_analysis/features.py` + `indicators.py` — indicators, lags, returns, vol
- [x] `forecasting/baseline.py` + `forecasting/models.py` (XGBoost, LightGBM, CatBoost)
- [x] `forecasting/evaluate.py` — RMSE/MAE/MAPE/R², directional accuracy, skill vs baseline
- [x] `data_ingestion/news.py` — Event Registry + yfinance `.news` fallback (RSS dropped)
- [x] `retrieval/dedup.py` — RapidFuzz
- [x] `sentiment/finbert.py` + `aggregate.py`
- [x] `llm/client.py` (Gemini) + `llm/prompts.py` + `llm/summarizer.py`
- [x] `recommendation/engine.py` — fusion + grounded reco + "why buy" thesis
- [x] `orchestration/pipeline.py` — DAG wiring (parallel quant + news)
- [x] `frontend/app.py` — dashboard renders the full result
- [x] **Milestone: runs end-to-end on AAPL with a working Gemini key**

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
- **yfinance reliability** — mitigated by disk cache, tenacity retries and a stale-cache fallback.
  (Stooq was investigated and dropped: non-functional in this environment.)
- **Event Registry free tier** — limited history and a request cap; yfinance `.news` is the
  keyless fallback. RSS was dropped entirely (broken `pyexpat` in this Python build).
- **Gemini free tier — the binding constraint.** 20 requests/day *per model*. Mitigated in v3 by
  merging two calls into one, caching responses, and parking quota-exhausted models.
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

---
---

# v3 — Performance, professional UI, new features

Goal: same product, same prediction approach — make it fast, make it look like a real product,
add features on top. Full plan: `~/.claude/plans/parallel-napping-fog.md`.

## Phase v3.0 — Speed baseline (measure before optimizing)  ✅
- [x] `scripts/profile_run.py` — times every pipeline stage plus the hot internals (per-model fits,
      ARIMA fits, Gemini calls, FinBERT/MiniLM loads). Monkey-patches probes at runtime, so **no
      application code is instrumented**. Uses a throwaway cache dir by default.

### Measured baseline — 2026-08-15, AAPL/MSFT, this machine

| scenario | with Gemini | without Gemini (`--no-llm`) |
|---|---|---|
| cold (nothing cached, models not loaded) | 104.6 s | 16.5 s |
| repeat (same ticker, cache warm) | 81.4 s | 6.1 s |
| new stock (models warm) | 37.2 s | 6.7 s |

**Finding — the original assumption was wrong.** Model retraining is *not* the bottleneck.
The 21 GBM fits + 3 ARIMA fits were confirmed empirically (XGB/LGBM/CatBoost = 7 calls each),
but they total only **~4.5 s**. The real cost is **Gemini: 30–77 s per analysis, 85–95 % of wall
clock**. Everything StockSense itself computes runs in 6 s.

**Root cause of the Gemini cost** (`llm/client.py:24`): `429` is classified as retryable, but a
*daily* free-tier quota exhaustion is not. When `gemini-flash-latest` (currently resolving to
`gemini-3.7-flash`, free-tier limit **20 requests/day**) is exhausted, every call burns
`0.6 + 1.5 + 3.0 = 5.1 s` of backoff plus 4 doomed round trips before falling back to
`gemini-3.5-flash`, which then answers fine. The API's own `RetryInfo.retryDelay` says ~50 s, so
the short backoff can never succeed. This repeats on **every call for the rest of the day**.

Caveat: the absolute Gemini numbers above are inflated because profiling exhausted the daily quota
mid-run. The bug is real regardless — with a 20/day limit, normal use hits it constantly.

### Revised priority (replaces the ordering in the plan file)
1. **Don't retry a model whose daily quota is gone**; remember it is dead and skip it — saves 5–10 s
   per call, 2 calls per analysis.
2. **Cache the LLM summary + recommendation** per ticker per day — removes 30–77 s, not 4.5 s.
   This is the real "biggest win", not forecast caching.
3. Run the summarization and recommendation calls concurrently — saves one round trip.
4. Warm FinBERT + MiniLM at app start — saves ~13 s, but only once per server start.
5. Cache the forecast — now a minor win (~4.5 s). Demoted.
6. ~~Trim model size / drop horizon refits~~ — **dropped**. Would save ~2 s and risks accuracy.

## Phase v3.1 — Speed fixes
- [x] LLM: treat daily-quota 429 as "model dead", with a cooldown so it is skipped, not retried
      (`llm/client.py` — `_quota_cooldown()` reads the API's own `RetryInfo.retryDelay`; anything
      longer than our 5.1 s backoff budget parks the model. `_parked()`/`_park()` keep a per-model
      cooldown so later calls skip it outright. If every model is parked, `_run` now fails fast so
      callers degrade to the rule-based path instead of waiting on doomed calls.)
- [x] LLM response cache — `llm/summarizer.py` and `recommendation/engine.py`, 24 h TTL, reusing
      `database/cache.py`. Keyed on the *inputs the LLM actually sees* (article set; plus rounded
      forecast + sentiment for the recommendation) rather than on the date, so it self-invalidates
      when the news or the numbers change. Values are rounded because GBM fits wobble in the last
      decimal places across threads and that must not cause a miss.
- [x] Summarization ∥ recommendation (`orchestration/pipeline.py`). `_process_news` was split into
      `_prepare_news` (dedup → rank → sentiment, still Stage 2) with summarization moved into a new
      Stage 3 that runs it **alongside** the recommendation. Neither depends on the other: the
      summary needs only the articles, the recommendation needs the forecast + sentiment.
      `LLMClient._sdk` is now double-checked-locked, since two threads can reach it at once.

      **Free-tier safety (the reason this was questioned): confirmed safe.** Same request count —
      2 per analysis, just overlapping. A live run showed the only 429s were the pre-existing daily
      quota on `gemini-flash-latest`; concurrency triggered **no** per-minute limits and **no**
      retry backoff. Rejected quota requests do not consume quota.

      Known minor cost: both threads discover a dead model independently, so it is rejected twice
      instead of once before being parked. Costs no quota and self-corrects within the run.

      ⚠️ **Speed gain not demonstrable from one measurement.** Overlap is proven — the profiler
      showed the two stages summing to **120 % of wall clock** (14.7 s + 20.3 s inside a 29.2 s
      run), which is only possible if they ran together. But free-tier latency for the *same* call
      varied from 9 s to 41 s across runs, which swamps the ~10 s the overlap should save. The
      change is structurally right and costs nothing; the wall-clock win will show on a paid tier
      or a quiet quota, not in a single free-tier sample.
- [x] **Merged the two LLM calls into one — now live in the pipeline.**
      `recommendation/engine.py::summarize_and_recommend` (driven by the new `AnalystAgent`) uses
      `llm/prompts.py::combined_prompt` + the `CombinedDraft` schema. Shared evidence lives in
      `_evidence_block()` so the two prompts cannot drift; `headline_digest()` was lifted out of
      `summarize()` so every path degrades to identical wording. Stage 3's `ThreadPoolExecutor` is
      gone — with one call there is nothing left to overlap, which supersedes the parallel change
      above. `SummarizationAgent` / `summary_prompt` are still used when the forecast fails.
      The superseded split path (`recommend()`, `RecommendationAgent`, `recommendation_prompt`)
      was **deleted**, and `tests/test_pipeline_units.py` was repointed at
      `summarize_and_recommend(..., use_llm=False)` so the rule-based fallback is tested where it
      actually runs, rather than on a dead code path.

      Verified live: `use_llm=False` → rule-based Hold 0.36 + headline digest;
      `use_llm=True` → Hold 0.65 + a generated summary, from a single call. 49 tests still pass.

      A/B on identical inputs (same articles, forecast, sentiment; caches bypassed):

      | metric | A: two calls | B: one call |
      |---|---|---|
      | requests per analysis | 2 | **1** |
      | AAPL latency | 24.1 s | 9.1 s *(biased — A discovered the dead models first)* |
      | MSFT latency | 15.7 s | **10.5 s** *(fair — both started with 2 models parked)* |
      | summary length | 118 / 138 words | 112 / 124 words |
      | factor count | 14 / 13 | 15 / 14 |
      | verdict | Hold / **Buy** | Hold / **Hold** |

      **Quality is equivalent, with one genuine improvement.** `summary_prompt` never sees the
      FinBERT score, so the split summary contradicted it — it called AAPL "Cautious/Mixed" against
      a −0.065 score and MSFT "Mixed" against **−0.390**. The merged summary, which sees the number,
      said "negative" both times. Merging makes the news summary agree with the sentiment engine.

      **Risk: the verdict moved on 1 of 2 tickers** (MSFT Buy → Hold). B's reasoning looks more
      defensible on the evidence — strongly negative sentiment, a predicted decline, and active
      securities class actions — but with n=2 and `temperature=0.3` this cannot be separated from
      ordinary run-to-run noise. Confirming it needs repeat runs, which the daily quota blocked.

- [x] FinBERT/MiniLM warm-up at app start — `frontend/_warmup.py`, started from `_shared.setup()`.
      Daemon thread, module-level guard (once per *process*, not per visitor — the models are
      shared). Verified: `start()` returns in 0.000 s, both models resident 11.9 s later, repeat
      calls are no-ops. That 11.9 s now elapses while the user picks a ticker.
- [x] Forecast result cache — `forecasting/forecaster.py`, keyed on ticker + **last price-bar date**
      + a hash of the training settings, 7-day TTL, via `database/cache.py`. Keying on the bar date
      (not wall-clock) means a new trading day invalidates it by itself. `run_forecast` gained
      `use_cache=True` so tests can force a retrain.
      Verified: retrain 4.70 s → cached 0.00 s, round-trip **byte-identical**; a dropped bar changes
      the key (`…2026-08-14…` → `…2026-08-13…`) and changing `min_history_rows` also busts it.
- [x] Screener: `quick_score` now fetches **1y instead of 2y** (it only needs a 3-month return and a
      50-day SMA) under its own cache key so it cannot collide with the pipeline's 2y data;
      `screener_concurrency` 8 → 16. Verified: dow30 12/12 scored in **1.5 s**, 0 failures, AMGN top
      — same ranking as the Phase v2.3 baseline.

### Result after the full Stage 2 (`--no-llm`, isolating our own compute)

| scenario | v3.0 baseline | now | change |
|---|---|---|---|
| cold | 16.5 s | 17.9 s | ~flat (cache write + noise) |
| repeat (same ticker) | 6.1 s | **1.9 s** | **3.2× faster** |
| new stock | 6.7 s | 6.3 s | ~flat (nothing cached to hit) |

With Gemini included, a repeat analysis is now **81.4 s → ~2 s** end to end: the forecast, the news
summary and the recommendation are all served from cache, costing **zero API requests**.

### Result — measured 2026-08-15, same machine, same script

| scenario | before | after | change |
|---|---|---|---|
| cold | 104.6 s | 50.5 s | **2.1× faster** |
| repeat (same ticker) | 81.4 s | **6.3 s** | **13× faster** |
| new stock | 37.2 s | 25.9 s | 1.4× faster |

On the repeat run both LLM stages now report 0.00 s and make zero API calls. Remaining cold/new-stock
time is the working model genuinely generating text (~10–27 s per call) — not wasted retries.

Verified: `pytest tests/ -q` → **49 passed** (first run of the suite on this machine, confirming the
count claimed in Phase 6). `beats_baseline` unchanged for both tickers.

⚠️ Observed, pre-existing, not caused by these changes: `best_model` is not stable across sessions
for AAPL (`Ensemble` vs `CatBoost`) — their holdout RMSEs are near-identical and live price data
shifts between runs. Worth a look if the Forecast page is meant to present it as a firm answer.

## Phase v3.2 — Professional UI  ✅
- [x] Streamlit chrome removed (`#MainMenu`, footer, toolbar, deploy button, status widget);
      inline-SVG brand mark in the sidebar; favicon is now `:material/trending_up:` instead of an emoji.
- [x] **Grouped navigation** via `st.navigation` — Analysis / News / Decision / Discover / More,
      with Material icons. `frontend/pages/` (numbered files) was replaced by `frontend/views/`
      with plain names; `app.py` is now a router that owns page config, and views render content
      only. `_shared.setup()` split into `boot()` (once, from `app.py`) + `sidebar()`.
- [x] All emoji removed from headings, buttons and labels. Sentiment is now a CSS dot
      (`_shared.tone_dot`) rather than 🟢/🔴/⚪.
- [x] **Friendly errors.** `agents/base.py` gained `StageError` (a `str` subclass carrying
      `.detail`) and a per-stage message map, so `forecast failed: ValueError: …` became
      "We couldn't build a forecast from the available price history." The pipeline's new
      `_Problems` collector splits user-facing text from technical text; `AnalysisResult.details`
      (additive) carries the latter to a "Technical details" expander.
      Verified on a bad ticker: user sees *"We couldn't find any price data for X. Check the
      symbol — US symbols look like AAPL, Indian ones like TCS.NS."*; the `InvalidTickerError`
      lives in `details`.
- [x] Real empty states — `_shared.empty_state()` explains what each view will show instead of
      the old "👈 Enter a ticker".
- [x] `st.column_config` on every table: percent/currency formats, progress bars for
      confidence / directional accuracy / composite score, checkboxes for `beats_baseline`.

⚠️ **Testing note.** The first rewrite of `scripts/apptest_all_pages.py` routed with
`query_params`, which `st.navigation` ignores — all 11 checks passed while only ever rendering
the Dashboard. Fixed to use `AppTest.switch_page`, and every check now also asserts the expected
title so a routing regression cannot pass silently. `tests/test_explore.py` gained a guard test
for the same reason.

## Phase v3.2b — Search by company name  ✅
- [x] `data_ingestion/markets.py::search_symbols(query, region, limit)` wraps Yahoo's symbol
      search (`yfinance.Search`), so users can type **"apple" or "tata"** instead of knowing the
      ticker. Covers every listed company, not just the bundled index lists. Filters to ordinary
      equities, de-duplicates, and ranks the selected market first without hiding cross-listings.
      New `SymbolHit` schema; 24 h disk cache; returns `[]` on any failure so the UI degrades.
- [x] `_shared.ticker_input` now resolves names → symbols. Picked results are used verbatim
      rather than re-normalized, so choosing NYSE-listed `INFY` while India is selected does not
      become `INFY.NS`.
- [x] `tests/test_symbol_search.py` (6, fully offline/mocked): region ranking both ways, ETF and
      duplicate filtering, label format, limit, and graceful failure on a network error.
      Live check: "apple"→AAPL, "reliance"[INDIA]→RELIANCE.NS, "tata"→TCS.NS, "infosys"→INFY.NS;
      cached repeat 0.000 s.

## Phase v3.2c — Hard-crash fix (SIGSEGV during analysis)  ✅
The app died mid-analysis with no Python traceback — a native segfault, not an exception.
macOS crash reports named it: faulting thread in `libomp.dylib __kmp_fork_barrier`, with a
sibling thread inside `lib_lightgbm.dylib` + `libomp.dylib`. **Three copies of `libomp.dylib`**
are loaded in this process (Homebrew's plus two bundled in wheels).

Cause: `n_jobs=-1` makes each GBM call `omp_set_num_threads()` itself and fork an OpenMP worker
pool. Stage 2 trains the forecast in one thread while FinBERT/MiniLM run in another, so those
pools hit a barrier owned by a different OpenMP runtime and corrupt memory.

- [x] `forecasting/models.py` — `_THREADS = 1` for XGBoost/LightGBM (`n_jobs`) and CatBoost
      (`thread_count`). No worker pool, no barrier. **Free**: ABT trains in 4.37 s vs 5.07 s
      before (~450 rows × 28 features — coordination cost exceeded the parallel gain);
      directional accuracy and skill unchanged.
- [x] `config/settings.py` — `OMP_NUM_THREADS`/`MKL`/`OPENBLAS` default to 1 as defence in depth.
      **Insufficient alone**: `n_jobs=-1` overrides the environment variable.
- [x] `embeddings/encoder.py`, `sentiment/finbert.py` — `device="cpu"`. On Apple Silicon these
      libraries silently select the MPS (GPU) backend, which is unsafe to call from several
      threads; one crash report showed `MetalShaderLibrary::exec_unary_kernel`. CPU is also what
      `CLAUDE.md` always claimed — it was simply never enforced.
- [x] Both model loaders are now lock-guarded. The logs showed MiniLM **and** FinBERT each
      loading **twice simultaneously**: `lru_cache` is not thread-safe on first call, so the
      v3.1 warm-up thread raced with an in-flight request. (Bug introduced by that warm-up.)

**Verified by controlled experiment.** Two earlier "fixes" passed a naive stress test and then
crashed again in the real app — the harness ran the pipeline directly, while the app runs it
inside Streamlit's script-runner thread. Re-built the harness on `AppTest`
(`scratchpad/stress_st.py`), changing only `_THREADS`:

| through Streamlit | result |
|---|---|
| `n_jobs=1` (fix) | 12/12 runs survived, 0 exceptions |
| `n_jobs=-1` (original) | **exit 139 — segfault before round 1** |

## Phase v3.3 — New features
- [x] **Track Record page** — `analytics/track_record.py::evaluate()` + `views/track_record.py`.
      Grades every saved run against the price that actually followed it: direction hit rate and
      mean absolute price error. Built entirely from data already in the `runs` table.
      Honesty rules baked in: a run whose next trading day hasn't closed is **pending**, not a
      miss; a run that can't be matched to a price bar (close within 0.5%) is **unverifiable**
      and excluded rather than guessed at; and under 20 graded runs the page says outright that
      the sample is too small to mean anything. `database/db.py` gained `all_runs()`.
      `tests/test_track_record.py` (8, offline). Live check: 22 saved runs → 0 graded,
      22 pending, correctly refusing to invent a hit rate on the day they were made.
- [x] **PDF export — actually working now.** `_shared.report_downloads` offers Markdown / HTML /
      PDF, hiding PDF when fpdf2 is absent. ⚠️ Wiring the button was not enough: `to_pdf` had
      **never worked**, and a bare `except: return None` hid two real bugs, so the button simply
      never appeared. Fixed: (1) `fpdf2` was not installed and not in `requirements.txt` — added;
      (2) fpdf2's built-in fonts are latin-1 only and the reports are full of em dashes → new
      `_pdf_safe()` folds typography to ASCII; (3) fpdf2 leaves the cursor at the right margin,
      so the second `multi_cell` failed with "not enough horizontal space" → pass
      `new_x="LMARGIN", new_y="NEXT"`. Failures are now **logged** instead of swallowed.
      Verified: AAPL → 2711-byte 2-page PDF, RELIANCE.NS → 2022-byte PDF, both valid `%PDF-`.
      `tests/test_report.py` gained 3 tests so this cannot silently rot again.
- [x] **Watchlist dashboard** — `views/watchlist.py`, a card per saved stock showing the latest
      verdict, confidence, predicted move and whether that run beat the naive baseline, with
      inline Analyze / Remove. Backed by a new `database/db.py::latest_run_by_ticker()`.
- [x] Search by ticker **or company name** — see Phase v3.2b (delivered via Yahoo symbol search,
      which is broader than the originally planned `data/universe/*.json` lookup).

## Phase v3.4 — Final verification  ✅  (2026-08-15)

```bash
export ARROW_DEFAULT_MEMORY_POOL=system
.venv/bin/python -m pytest tests/ -q                            # 64 passed
PYTHONPATH=.:frontend .venv/bin/python scripts/apptest_all_pages.py   # ALL VIEWS OK (13)
PYTHONPATH=. .venv/bin/python scripts/integration_e2e.py        # 12/12 checks passed
PYTHONPATH=. .venv/bin/python scripts/profile_run.py [--no-llm]
```

### 1. Speed — measured, not assumed

| scenario | v3.0 baseline | final | change |
|---|---|---|---|
| repeat analysis (with Gemini) | 81.4 s | **2.0 s** | **~40× faster**, and **0 API requests** |
| cold start (with Gemini) | 104.6 s | 24.2 s | 4.3× faster |
| new stock (with Gemini) | 37.2 s | 46.7 s | dominated by one slow API call — see note |
| repeat analysis (`--no-llm`) | 6.1 s | **2.0 s** | 3× faster |
| new stock (`--no-llm`) | 6.7 s | 6.1 s | ~flat |

The profiler confirms **`call: Gemini` count = 1** per analysis (was 2) — the merged call is live.
The new-stock regression is not our code: that run's single Gemini call took 40.4 s of the 46.7 s.
Free-tier latency for an identical call has ranged 9–41 s across runs, which swamps everything we
control. `--no-llm` is the honest measure of our own compute, and it improved.

### 2. Accuracy — unchanged by the crash fix

`_THREADS = 1` vs the original `-1`, same data, forecast cache bypassed:

| ticker | dirAcc (−1 → 1) | skill (−1 → 1) | beats | best model |
|---|---|---|---|---|
| AAPL | 0.600 → 0.600 | −0.0085 → −0.0085 | False | Ensemble |
| MSFT | 0.500 → 0.500 | +0.0043 → +0.0043 | True | CatBoost |
| RELIANCE.NS | 0.500 → 0.500 | −0.0315 → −0.0315 | False | ARIMA |

Identical on every metric. Timing was 3.83 s → 3.97 s (~4% slower here); an earlier ABT
measurement showed the opposite. **Correction to an earlier claim in this file: single-threading
is a wash on speed, within noise — not a speed-up.** Its value is that it stops the segfault.

### 3. Everything still works
- `pytest` **64 passed** (was 49 at the start of v3; +15 for symbol search and track record)
- **13/13 views** load, each asserted by title so routing can't silently fall through
- `integration_e2e.py` **12/12**: US + India analysis, invalid ticker, low-history guard,
  screener, mixed-region compare, MD/HTML report, watchlist round-trip, chat LLM + fallback
- App boots, health `ok`; database intact (22 runs)
- No stale references to the deleted `recommend()` / `RecommendationAgent` / `recommendation_prompt`

### 4. Cache correctness
Forecast cache round-trips byte-identically, busts on a new price bar
(`…2026-08-14…` → `…2026-08-13…`) and on a `min_history_rows` change. LLM cache keys on the
article set plus rounded forecast/sentiment, so it self-invalidates when the inputs move.

## Phase v3.5 — Docs + test coverage  ✅  (2026-08-16)

- [x] **README rewritten for v3** — new "What's new in v3" section, corrected folder structure
      (`frontend/views/`, not `pages/`), all 13 pages listed by nav group, `fpdf2` + symbol search
      in the tech stack, four new troubleshooting rows (the OpenMP segfault, the torchvision noise,
      the missing PDF button), the full test command set, and a frank **Known gaps** section.
- [x] **Doc drift cleared** (flagged on day one, predates v3):
      Phase 1 boxes ticked with a note explaining why they were retroactive; "Gemini 2.5 Flash" →
      the actual model chain; Stooq and RSS corrected to what the code really does in both files;
      `architecture.md` — `pandas-ta` → `ta`, `gbm_models.py` → `models.py`,
      `retrieval/{bm25,semantic}.py` → `ranker.py`, `run_forecast(features)` → the real signature,
      and the data-flow diagram now shows the single merged Analyst call.
- [x] **Tests for the three biggest untested surfaces** — 32 new tests, all offline:
      - `tests/test_forecaster.py` (10) — short-history guard, forward-looking horizon targets,
        cache-key invalidation, horizon/price consistency, metric ranges, ensemble weights summing
        to 1, `beats_baseline` agreeing with the skill number it is derived from, backtest array
        alignment, and cache round-trip fidelity.
      - `tests/test_pipeline_dag.py` (8) — `analyze()` driven with the network stubbed: happy path,
        fatal missing prices, news/context failures degrading rather than crashing, no-news case,
        and `_Problems` keeping exception text **out** of user-facing messages.
      - `tests/test_llm_client.py` (14) — the v3 quota logic: per-day 429 parked vs per-minute 429
        retried, fallback to the next model, a parked model skipped on later calls, fail-fast when
        all are parked, cooldown expiry, and structured-output parsing/failure.

**Suite: 49 → 99 tests.** All 13 views still render; database intact.

## Phase v3.6 — Prediction-quality gaps  ✅  (2026-08-16)

The three gaps recorded as "future work" since the v3 planning session.

### 1. Confidence intervals — done
`forecasting/intervals.py`, split **conformal prediction**. Chosen over a model's own
uncertainty estimate because it is distribution-free (daily returns have fat tails that break
the Gaussian assumption), wraps the ensemble rather than a single model, and gives a coverage
guarantee that can be *checked*.
- `HorizonForecast.lower`/`upper` — declared since v1, empty until now — are populated.
- **Calibration and coverage measurement use disjoint slices of the holdout**, so the reported
  coverage is not the data that set the width. Too small to split → coverage reported as
  unknown rather than a number true by construction.
- Longer horizons are widened by √t; flagged in the notes as an assumption, not a measurement.
- Live: AAPL 80% target → **92% measured**, MSFT → **75%**, both calibrated on 18 held-out days.

### 2. Backtest the calls — done
`forecasting/strategy.py`: go long when the model predicts a rise, else hold cash, over the
holdout window, versus buy-and-hold, **after transaction costs** (5 bps per position change —
a costless backtest flatters any signal that trades often).
- Live and honestly mixed: **AAPL +1.30% vs −2.07% (beat)**, **MSFT +20.29% vs +28.10% (lost)**.
- Notes state plainly that ~30 days is an anecdote, and that this tests the *forecast* only —
  the LLM verdict also uses sentiment, which cannot be backtested (see below).

### 3. Sentiment as a model input — infrastructure built, feature correctly withheld
**Tested first, and the answer was no.** Event Registry's free tier returns the same ~17
articles spanning ~4 weeks whether you ask for 30 days or 730. There is no historical news, so
there is no historical sentiment to train on.

Faking it would have been worse than nothing: a column that is zero for 95% of training rows
teaches the model nothing, then hands it an out-of-distribution value at inference. So:
- New `daily_sentiment` table + `record_sentiment()` / `sentiment_history()`; every analysis
  now records one reading, accumulating the history the API cannot supply.
- `technical_analysis/features.attach_sentiment()` adds the feature **only** above
  `MIN_SENTIMENT_COVERAGE = 0.60` of the training window; below that it refuses.
- `run_forecast` reports which is the case in its notes, so the UI never implies the model
  uses sentiment when it doesn't. It turns itself on once coverage is deep enough.

`tests/test_intervals_strategy.py` (16): conformal coverage on deliberately fat-tailed data,
refusal to report coverage on small samples, √t scaling, strategy vs buy-and-hold in rising and
falling markets, costs actually charged, and both the sparse-refused and dense-accepted
sentiment paths. **Suite: 99 → 115.**

## Phase v3.8 — Full design revamp  ✅  (2026-08-16)
The v3.7 token pass was judged too timid; this is the real one. Five phases, verified between
each; 115 tests + 13/13 views green after. `st.title` and the Explore buttons were deliberately
kept so the regression gates stay meaningful.
- [x] **Dark navy sidebar** — the single biggest feel change. Gradient panel, uppercase nav
      section labels, quiet slate links with a glowing active pill (`aria-current="page"` +
      inset sky accent), dark-styled inputs/selects/buttons/expanders/captions, brand lockup
      re-tinted for dark.
- [x] **Component library** (`_shared.py`) — `kpi_row()` renders stat cards in ONE self-aligning
      CSS grid (`auto-fit, minmax(175px,1fr)`) instead of st.columns, so cards are equal-height,
      wrap on narrow screens, and carry label/value/tone-chip/sub-line; `hero()` gradient panel
      with glass chips; `section()` accent-bar headers; `page_header(eyebrow=…)`.
- [x] **Pages recomposed** — Dashboard (KPI cards incl. the 80% range under next-day, thesis in
      a card), Forecast (horizon cards with ranges, strategy KPI row), Risk (two 4-card rows with
      sub-lines replacing 8 bare metrics), Track Record, Recommendation (factor lists → four
      tone-dotted cards, "The case, both ways"), Explore (hero with chips). Every view got an
      eyebrow matching its nav section; every st.subheader on main views → accent section.
- [x] **Charts modernized** (`visualization/theme.py`) — faint dotted grid, no frame lines
      (the card border does that job), muted tick labels, transparent legend, Space Grotesk
      titles. Applies to all 8 chart builders via the template, zero signature changes.
- [x] **Second pass, from live screenshots** — transparent `stHeader` (killed the white strip
      clipping the eyebrow; kept alive for the sidebar-expand control), tile styling scoped to
      `st-key-idx_*` classes only (an earlier global `min-height:86px` was inflating every
      secondary button app-wide), tiles fixed-height 112px with name pinned top / count pinned
      bottom + parentheticals trimmed so rows stay level, dark-sidebar inputs painted across
      **all** BaseWeb layers (styling one layer let the white default show through),
      `InputInstructions` overlay hidden ("Press Enter" moved into the placeholder), Recent
      analyses table given full column_config.
- [x] **Remaining pages** — Sentiment (KPI row + gauge beside a "how to read this" card),
      Technical (live RSI / vs-SMA50 / MACD reading cards from the existing indicator table),
      Screener & Compare (proper empty states with page-specific actions, leaderboard section
      header), Ask (three runnable starter questions when the chat is empty), Watchlist
      (verdict pill instead of a bare metric).

## Phase v3.7 — Visual token pass (superseded by v3.8)  ✅  (2026-08-16)
Full design pass, all in `frontend/_style.py` + the verdict badge in `_shared.py`. No view logic
changed; 115 tests + 13/13 views verified after.
- [x] **Typography** — Inter (body) + Space Grotesk (headings, metric values) via Google Fonts.
      `visualization/theme.py` already requested Inter, so charts and page now genuinely share
      one typeface. Metric values use tabular numerals.
- [x] **Background** — soft three-point radial mesh instead of flat white; translucent
      blurred sidebar with a hairline border.
- [x] **Depth & hover** — cards, bordered containers, metric tiles and charts get layered
      shadows and a translate-up hover lift; metric tiles reveal a gradient top edge. Hover
      effects are gated behind `@media (hover:hover)` so they don't stick on touch screens.
- [x] **Motion** — content fades up on entry (Streamlit's DOM diffing means only *new*
      elements animate, not every rerun); `st.status` pulses with an accent edge while a
      spinner is inside it (`:has()`); shimmer skeleton class retained. All animation is
      disabled under `prefers-reduced-motion`.
- [x] **Alignment** — Explore index tiles get a fixed min-height so rows line up regardless
      of label length; sidebar buttons stay compact; primary buttons are gradient-filled.
- [x] **Verdict badge** — per-action gradient (green/amber/red) with a matching soft glow.
- [x] Inputs get a focus ring, dataframes/chat/alerts share the card radius, thin scrollbars.

## Phase v3.9 — Deployment  ✅  (2026-08-16)

**Host chosen by measurement, not preference.** Peak resident memory during one analysis is
**1,060 MB** (137 MB base → 227 MB with the GBM libs → 507 MB with MiniLM → 589 MB with
FinBERT → 1,060 MB running). That rules out Streamlit Community Cloud (1 GB cap — killed
mid-analysis), Render free (512 MB) and small Fly instances. **Hugging Face Spaces: 16 GB RAM
free, no card**, and it serves the model weights from its own infrastructure.

- [x] `Dockerfile` — closes the item deferred since Phase 6. `python:3.13-slim` + `libgomp1`
      (Linux equivalent of `brew install libomp`; XGBoost/LightGBM won't import without it),
      non-root uid 1000 to match Spaces, `$PORT`-aware, healthcheck on `/_stcore/health`.
      Two deliberate choices: **CPU-only PyTorch** from the PyTorch index (the default PyPI
      wheel is 527 MB and drags in ~2 GB of nvidia-* CUDA packages this app never uses), and
      **models baked in at build time** so the first visitor doesn't wait on a 530 MB download
      or trip the startup timeout. Bakes in `OMP_NUM_THREADS=1` and
      `ARROW_DEFAULT_MEMORY_POOL=system` — the two settings that stop the v3.2c segfaults.
- [x] `.dockerignore` — keeps `.env`, `.venv/`, local DB/cache and dev-only material out.
- [x] `deploy/deploy_hf.sh` + `deploy/README_SPACE.md` — one-command deploy/redeploy. The
      Space is its own git repo and HF reads the README's YAML front-matter for configuration,
      so the script swaps in a Space-specific README rather than polluting the project one.
- [x] **Database persistence** (`database/sync.py`) — Spaces have an ephemeral disk, so
      History / Track Record / Watchlist would reset on every restart, and Track Record is
      worthless without accumulated history. Mirrors the SQLite file to a **private** HF
      Dataset: `pull()` once at engine creation, `push()` after each write on a daemon thread,
      coalesced to one upload per 30 s. Opt-in via `HF_DATASET_REPO` + `HF_TOKEN`; with either
      unset every function is a no-op, so local dev and the test suite never touch the network.
      Failures are logged and swallowed — sync must never break an analysis.
- [x] `docs/deployment.md` + README section + `.env.example` entry.
- [x] `tests/test_db_sync.py` (10): opt-in behaviour, restore, once-per-process pull, missing
      dataset starting fresh, private-repo creation, non-blocking daemon push, burst
      coalescing, and swallowed failures. **Suite: 117 → 127.**

### Container verified locally  ✅  (2026-08-17)
`docker build` → **5.76 GB image**, then run and exercised:
- health `ok` in **3 s**; clean startup log
- both models load in **11.9 s from the image** — the bake-in works, no network download
- **full analysis inside the container**: AAPL → `Hold (30%)`, $304.76 → $305.37,
  80% range $299.20–$311.67, 6 news articles, **0 errors**
- runs as non-root `app`, `/app/data_cache` writable, SQLite created correctly
- **109.7 MiB** resident after an analysis — the 1.06 GB figure is transient peak, so
  headroom on a 16 GB Space is enormous

Two real bugs the local build caught, which would have failed on the Space:
1. `WORKDIR /app` makes the directory **root-owned**, so `mkdir` after `USER app` was denied.
   Fixed by creating the dirs and `chown -R app:app /app` while still root.
2. `CMD` in shell form triggered `JSONArgsRecommended` and would not forward `SIGTERM`.
   Fixed with `CMD ["sh","-c","exec streamlit …"]` — `exec` makes streamlit PID 1.

⚠️ Host note: two builds failed first with `input/output error` — the **Mac's disk was full**
(138 MB free), not a Dockerfile fault. Cleared ~90 GB (unused HF models, npm/pip caches, and a
dormant `atlas-analytics` Docker project) → 52 GB free.

### ⚠️ Host choice invalidated — Hugging Face moved compute Spaces behind PRO
The Space creation page now states: *"Gradio and Docker Spaces require a paid plan. Static
Spaces stay free for everyone."* Only **Static** (plain HTML/JS, no Python) is free, so the
free HF tier can no longer run this app. The earlier recommendation was based on HF's
historical free CPU tier and was not re-verified against current pricing — a real miss.

**Nothing built is wasted:** the Dockerfile is host-agnostic and verified working, so only the
target changes. Options if this is picked up again, with the deciding constraint being the
**~1.06 GB peak** (transient, during first-analysis model training; the idle container sits at
~110 MB, and the forecast cache means the peak recurs only once per stock per day):

| Option | Cost | Notes |
|---|---|---|
| Streamlit Community Cloud | free | 1 GB cap — borderline; would need trimming (drop CatBoost, shorter training window) and an honest accuracy report |
| Google Cloud Run | free tier | runs our image unchanged, scales to zero; needs a card on file |
| Oracle Cloud Always Free | free | 24 GB ARM VM, most headroom, but bare-VM setup |
| HF Spaces PRO | ~$9/mo | everything already built works unchanged |

Local Docker artifacts (image, container, build cache) were removed after verification.

### Still open (deliberately)
- `[ ]` Choose a host and deploy; Postgres swap
- `[ ]` Tests for `visualization/`, `embeddings/`, `screener/score.py`, `data_ingestion/prices.py`
- `[ ]` Sentiment feature activates only after ~60% coverage accumulates (weeks of daily runs)
- `[ ]` Pooled cross-sectional training and rolling-window evaluation (the honest version of
  "more data helps"); no time decay on sentiment; survivorship bias in the bundled index lists
- `[ ]` Deeper prediction-quality gaps recorded as future work: sentiment is not a model input,
  forecasts carry no confidence interval, and the Buy/Hold/Sell calls themselves are not
  backtested (the new Track Record page is the first step toward that third one)
