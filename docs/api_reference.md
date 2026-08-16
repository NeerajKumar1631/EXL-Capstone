# API Reference (internal)

The one entry point most callers need:

```python
from orchestration.pipeline import analyze

result = analyze("AAPL", use_llm=True)   # -> AnalysisResult
```

## `analyze(ticker, use_llm=True, top_k=None, progress=None) -> AnalysisResult`
Runs the full DAG. `use_llm=False` forces the rule-based recommendation. `progress` is an optional
`callable(str)` for UI status updates.

## Key functions

| Function | Signature | Returns |
|---|---|---|
| `data_ingestion.prices.fetch_prices` | `(ticker, period=None, interval=None, use_cache=True)` | OHLCV `DataFrame` |
| `data_ingestion.news.fetch_news` | `(ticker, company, days=None, max_articles=None)` | `list[Article]` |
| `data_ingestion.context.fetch_context` | `(ticker)` | `MarketContext` |
| `data_ingestion.markets.search_symbols` | `(query, region="US", limit=8)` | `list[SymbolHit]` — search by ticker **or company name** |
| `analytics.track_record.evaluate` | `(limit=500)` | `TrackRecord` — past predictions graded against real prices |
| `technical_analysis.features.training_frame` | `(prices)` | `(X, y, feats, cols)` |
| `forecasting.forecaster.run_forecast` | `(prices, ticker, use_cache=True)` | `ForecastResult` (cached per ticker + last bar date) |
| `retrieval.dedup.deduplicate` | `(articles, threshold=88)` | `list[Article]` |
| `retrieval.ranker.rank` | `(query, articles, top_k, alpha=0.6)` | top-k `list[Article]` |
| `sentiment.finbert.score` | `(articles)` | scored `list[Article]` |
| `sentiment.aggregate.aggregate` | `(articles)` | `SentimentSummary` |
| `llm.summarizer.summarize` | `(company, ticker, articles, use_llm=True)` | `str` |
| `llm.summarizer.headline_digest` | `(company, articles)` | `str` — deterministic, no LLM |
| `recommendation.engine.summarize_and_recommend` | `(company, ticker, forecast, sentiment, context, articles, use_llm=True)` | `(str, Recommendation)` — one Gemini call returning both |

## Schemas (`orchestration/schemas.py`)
`Article`, `SentimentSummary`, `NewsResult`, `ModelMetrics`, `HorizonForecast`, `ModelForecast`,
`ForecastResult`, `MarketContext`, `Recommendation`, `AnalysisResult`. All pydantic (except DataFrames).

## LLM client (`llm/client.py`)
`get_llm().generate_text(prompt)` and `generate_json(prompt, schema)`. Tries `settings.gemini_models`
in order, retries transient errors, raises `LLMUnavailable` when all fail (callers fall back).
