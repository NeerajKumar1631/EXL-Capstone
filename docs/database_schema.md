# Database Schema

SQLite at `data_cache/stocksense.db` (auto-created). ORM in `database/models.py`, access in
`database/db.py`. In addition, prices/news are cached as parquet/JSON under `data_cache/`.

## Table: `runs`

One row per completed analysis (persisted by `save_run`).

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | autoincrement |
| `created_at` | DATETIME | analysis timestamp |
| `ticker` | VARCHAR(20) | indexed |
| `company` | VARCHAR(120) | resolved name |
| `action` | VARCHAR(8) | Buy / Hold / Sell |
| `confidence` | FLOAT | 0–1 |
| `last_close` | FLOAT | most recent close |
| `next_day_price` | FLOAT | ensemble next-day price |
| `beats_baseline` | BOOLEAN | did the ensemble beat the naive baseline? |
| `best_model` | VARCHAR(40) | lowest holdout RMSE |
| `sentiment_label` | VARCHAR(12) | positive / negative / neutral |
| `sentiment_score` | FLOAT | credibility-weighted, −1..1 |
| `thesis` | TEXT | recommendation rationale |

Read recent rows via `database.db.recent_runs(limit)` (used by the **History** page).

## Table: `watchlist` (v2)

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | autoincrement |
| `ticker` | VARCHAR(20) | unique, indexed |
| `region` | VARCHAR(10) | US / INDIA |
| `added_at` | DATETIME | |

Managed via `add_watch` / `remove_watch` / `list_watch` / `is_watched` in `database/db.py`. This table
is additive — it does not touch the `runs` table.

## Table: `daily_sentiment` (v3)

One credibility-weighted sentiment reading per ticker per day.

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | autoincrement |
| `ticker` | VARCHAR(20) | indexed |
| `day` | VARCHAR(10) | `YYYY-MM-DD`, indexed; unique together with `ticker` |
| `score` | FLOAT | credibility-weighted, −1..1 |
| `label` | VARCHAR(12) | positive / negative / neutral |
| `n_articles` | INTEGER | how many articles fed the reading |
| `recorded_at` | DATETIME | |

**Why this table exists.** The news API serves only about four weeks of history, so a sentiment
*feature* cannot be trained on the past — it has to be accumulated going forward. Every analysis
writes one row (`record_sentiment`); `sentiment_history(ticker)` reads it back. The forecaster uses
it as a model input only once it covers ≥60% of the training window
(`technical_analysis/features.attach_sentiment`), and reports which case applies. Last write wins
for a given ticker+day. Additive — it does not touch `runs` or `watchlist`.

## Cache files (`data_cache/`)
- `prices_<TICKER>_<period>_<interval>.parquet` — OHLCV (TTL 60 min).
- `news_<TICKER>_<days>_<max>.json` — normalized articles (TTL 60 min).
- `forecast_<TICKER>_<last-bar-date>_<settings-hash>.json` — a full `ForecastResult` (TTL 7 days;
  the bar date in the key means a new trading day invalidates it by itself).
- `llm_analyst_<TICKER>_<inputs-hash>.json` — the merged summary + recommendation (TTL 24 h).
- `symsearch_<REGION>_<query>.json` — company-name search results (TTL 24 h).
