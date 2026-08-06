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

## Cache files (`data_cache/`)
- `prices_<TICKER>_<period>_<interval>.parquet` — OHLCV (TTL 60 min).
- `news_<TICKER>_<days>_<max>.json` — normalized articles (TTL 60 min).
