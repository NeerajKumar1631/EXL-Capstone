---
title: StockSense AI
emoji: 📈
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Stock forecasting with news sentiment and a grounded Buy/Hold/Sell call
---

# StockSense AI

Give it a ticker (or a company name). It forecasts the next move with real ML models, reads
and scores recent news, and fuses both into a grounded **Buy / Hold / Sell** call with a
plain-English thesis, confidence, risks and cited sources.

> ⚠️ **Not financial advice.** This is an educational decision-support tool. Markets are risky.

## What makes it honest

- Models predict **next-day returns**, not price levels, and are always graded against a naive
  *"tomorrow = today"* baseline. If the model can't beat it, the app says so instead of
  showing an impressive-looking but meaningless R².
- Every forecast carries an **80% prediction range** from split conformal prediction, and the
  interval's coverage is measured on data that wasn't used to set its width.
- A **strategy backtest** shows whether following the signal would actually have beaten
  buy-and-hold, after transaction costs — losses reported as prominently as wins.
- A **Track Record** page grades the app's own past predictions against what prices really did.
- Every recommendation factor cites a computed number or a real article; a post-check strips
  any URL the model invented.

## Configuration

Set these as **Secrets** in the Space settings:

| Secret | Required | Purpose |
|---|---|---|
| `GEMINI_API_KEY` | recommended | LLM reasoning. Without it the app falls back to a deterministic rule-based recommendation. |
| `NEWS_API_KEY` | optional | Event Registry (newsapi.ai) key. Without it, news comes from yfinance. |
| `HF_TOKEN` | optional | Needed only for database persistence (below). |
| `HF_DATASET_REPO` | optional | e.g. `your-name/stocksense-db`. Mirrors the SQLite database to a private Dataset so History, Track Record and Watchlist survive restarts — Spaces have an ephemeral disk. |

Source: [github.com/NeerajKumar1631/EXL-Capstone](https://github.com/NeerajKumar1631/EXL-Capstone)
