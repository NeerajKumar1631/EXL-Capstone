# Interview Questions — StockSense AI

Questions you're likely to be asked about this project, with answers grounded in what the code
actually does. Every number here is measured, not estimated.

**How to use this:** don't memorise the answers. Understand the *reasoning*, because the good
follow-up questions probe why you chose one thing over another. The strongest answers in here
are the ones where the honest answer is "it doesn't work well, and here's how we prove it".

---

## Contents

1. [The project in one minute](#1-the-project-in-one-minute)
2. [Machine learning](#2-machine-learning)
3. [Honesty and evaluation](#3-honesty-and-evaluation)
4. [NLP and sentiment](#4-nlp-and-sentiment)
5. [LLM and prompt engineering](#5-llm-and-prompt-engineering)
6. [System design](#6-system-design)
7. [Performance](#7-performance)
8. [Debugging war stories](#8-debugging-war-stories)
9. [Testing](#9-testing)
10. [Limitations — expect these](#10-limitations--expect-these)
11. [Rapid-fire](#11-rapid-fire)

---

## 1. The project in one minute

**Q: Explain your project.**

You give it a stock symbol. It does three things at once:

1. **Forecasts** the next price move using machine learning trained on past price patterns
2. **Reads the news** about that company and scores whether it's positive or negative
3. **Combines both** into a Buy / Hold / Sell recommendation, written in plain English, where
   every reason cites either a real number or a real article

It's a Streamlit web app with 13 pages, backed by a Python pipeline. Roughly 5,000 lines of
code and 127 tests.

**The thing that makes it different:** most stock predictors show impressive accuracy numbers
that don't survive scrutiny. This one grades itself against a "do nothing" baseline and tells
you plainly when it isn't beating it — which, for daily price direction, is most of the time.

**Q: Why did you build it this way?**

The guiding rule was *reuse, don't rebuild*. Anywhere a maintained library already solved
something, we wrapped it rather than writing our own. Custom code is limited to the parts
nobody else can do for us: the pipeline that ties it together, the ensemble and its evaluation,
the news ranking, and the logic that fuses signals into a recommendation.

---

## 2. Machine learning

**Q: Which models did you use, and why those?**

Four, then blended:

| Model | Why |
|---|---|
| **XGBoost, LightGBM, CatBoost** | Gradient-boosted trees. Best-in-class for small tabular data, which is what technical indicators produce. |
| **ARIMA (SARIMAX)** | A classical statistical model on the return series. Included as a different *kind* of model — if the tree models all fail the same way, ARIMA might not. |
| **Ensemble** | Weighted average of the above, weights proportional to `1 / validation error`, so better models count more. |

Diversity is the point. Three tree models that make the same mistake give false confidence.

**Q: Why predict *returns* instead of *prices*? This is a common trap.**

Because predicting price levels makes a useless model look brilliant.

Stock prices are *non-stationary* — they trend. If a stock is around ₹1000, a model that just
predicts "tomorrow ≈ today" gets an R² of 0.99 and looks amazing. It has learned nothing.

Returns — `log(today ÷ yesterday)` — are roughly stationary. They hover around zero and don't
trend. A model has to find real signal to do well, and there's nowhere to hide.

We display the price by converting back: `predicted_price = last_close × exp(predicted_return)`.
So the user sees a price, but the model is judged on the honest quantity.

**Q: What features does the model use?**

About 28, all designed to be roughly stationary:

- **Lagged returns** — the last 1, 2, 3, 5 and 10 days
- **Rolling statistics** — mean and standard deviation over 5, 10, 20 days
- **Volatility** — 20-day realised, annualised
- **Volume features** — change and ratio to its own 20-day average
- **Intraday range** — (high − low) ÷ close
- **Technical indicators as ratios** — RSI, MACD, Bollinger, ATR, and moving averages

**Follow-up you should expect: why ratios?** Because a raw 50-day moving average of ₹1000 is
just the price again — non-stationary, and meaningless once the stock moves to ₹2000. We use
`close ÷ SMA50` instead, which stays around 1.0 regardless of the price level and actually
says something: "the price is 3% above its average."

**Q: How do you validate? Why not normal cross-validation?**

`TimeSeriesSplit`, never plain k-fold.

Normal cross-validation shuffles data randomly. With time series that means **training on the
future to predict the past** — a leak that produces fantastic scores and a worthless model.

`TimeSeriesSplit` always trains on earlier data and tests on later data. We also hold out the
final 30 days completely, untouched during training, for the final score.

**Q: How do you prevent look-ahead bias?**

Three ways:

1. The target is `log_ret.shift(-1)` — tomorrow's return sits on today's row, so today's
   features can only be things known today
2. `TimeSeriesSplit` for validation, no shuffling anywhere
3. **A test that enforces it.** `tests/test_features.py` recomputes the shift independently and
   asserts the last row's target is `NaN`, because tomorrow's return doesn't exist yet.

That third one matters. Anyone can say they avoided leakage; a test proves it.

---

## 3. Honesty and evaluation

This section is where the project is strongest. Lead with it.

**Q: What's your model's accuracy?**

Around **50% directional accuracy — a coin flip** — and I can show you that it usually does
**not** beat a naive baseline.

That's not a failure to hide; it's the finding. Daily stock direction is close to random, which
matches decades of published research. A project claiming 95% accuracy on daily direction has
almost certainly leaked future data.

The app displays this prominently. When the model doesn't beat the baseline, the Dashboard
shows a warning and the recommendation logic **automatically down-weights the forecast** and
leans on news and fundamentals instead.

**Q: What is the "naive baseline" and why does it matter?**

The simplest possible forecast: *"tomorrow's price = today's price"*, i.e. a predicted return
of zero.

It matters because it's shockingly hard to beat, and because without it, error metrics are
meaningless. An RMSE of 0.02 sounds fine — but if the baseline gets 0.019, your model is worse
than doing nothing.

We report **skill score** = `(baseline_error − model_error) ÷ baseline_error`. Above zero means
you added value. Below zero means you'd have done better predicting nothing.

**Q: You show a prediction range. How is that calculated?**

**Split conformal prediction.** We take the model's errors on data it never trained on, find
the 80th percentile of those errors, and the range is `prediction ± that value`.

Two reasons for conformal over the usual approach:

- It's **distribution-free** — it doesn't assume errors follow a bell curve. Stock returns have
  "fat tails" (extreme moves happen far more often than a normal distribution predicts), which
  breaks the standard assumption.
- Its coverage can be **checked**, and we check it.

**The part I'd highlight:** we calibrate the width on one slice of held-out data and measure
coverage on a *different* slice. Otherwise you're grading your homework with the answer key —
the coverage would come out right by construction and mean nothing. When there's too little
data to split, the app reports coverage as *unknown* rather than quoting a fake number.

Measured live: AAPL got 92% actual coverage against an 80% target, MSFT 75%.

**Q: Does following the model actually make money?**

Sometimes, and we measure it rather than assuming.

There's a backtest: go long when the model predicts a rise, hold cash otherwise, over the
held-out window, **charging transaction costs** (5 basis points per trade — a costless backtest
flatters any strategy that trades a lot).

Real measured results:
- **AAPL: +1.30% vs buy-and-hold −2.07%** → beat it
- **MSFT: +20.29% vs buy-and-hold +28.10%** → **lost to it**

The app shows the loss as prominently as the win. And it states plainly that a ~30-day window
is an anecdote, not evidence.

---

## 4. NLP and sentiment

**Q: How does sentiment analysis work here?**

A four-step pipeline:

1. **Fetch** — Event Registry API, falling back to Yahoo Finance if there's no key
2. **De-duplicate** — the same story gets syndicated everywhere. `RapidFuzz` compares titles
   with fuzzy matching; we also normalise URLs to catch tracking-parameter duplicates
3. **Rank by relevance** — combine two signals:
   - **BM25** — keyword matching (the algorithm behind classic search engines)
   - **Semantic similarity** — MiniLM converts text to vectors; similar meaning = closer
     vectors. Catches "iPhone maker" matching "Apple" where keywords wouldn't.

   Fused 60% semantic / 40% keyword, keeping the top 8.
4. **Score** — FinBERT, a BERT model fine-tuned on financial text

**Q: Why FinBERT and not general sentiment analysis?**

Financial language is different. "Shares plunged after the company **beat** estimates" — a
general model sees "beat" as positive and "plunged" as negative and gets confused. FinBERT was
trained on financial text and understands the domain conventions.

**Q: What's the credibility weighting?**

Not all sources deserve equal weight. Reuters and Bloomberg get 1.0; a promotional blog gets
0.6. The overall score is a weighted average, so one hype piece can't outvote Reuters.

**Q: Is sentiment used in your ML model?**

**No — and refusing to fake it was a deliberate decision I'd defend.**

To train on sentiment, you need *historical* daily sentiment going back through your training
window. I tested whether the news API could provide that: asking for 730 days returns the same
~17 articles covering about four weeks as asking for 30 days. **There is no history available.**

I could have added a sentiment column anyway. It would have been empty for ~95% of training
rows, so the model would learn nothing from it, and then at prediction time it would receive a
real value unlike anything it saw in training. That's worse than not having the feature — it
looks impressive and actively degrades the model.

What we built instead: the app **records a sentiment reading every time it runs**, accumulating
the history the API won't give us. The feature switches itself on automatically once coverage
reaches 60% of the training window. Until then the app states plainly that sentiment shapes the
recommendation but not the forecast.

**This is a strong answer** because it shows you tested an assumption, got an inconvenient
result, and chose the honest engineering path.

---

## 5. LLM and prompt engineering

**Q: How do you stop the LLM hallucinating fake news or numbers?**

Three layers:

1. **Give it everything, let it invent nothing.** The prompt contains the exact computed
   numbers and the full article list. It's instructed to use only those.
2. **Structured output.** The response is validated against a Pydantic schema, so we get typed
   fields, not free text we have to parse.
3. **A verification step that actually checks.** After the response comes back, `_ground()`
   extracts every URL the model cited and confirms it exists in the articles we supplied. Any
   invented URL is **deleted from the output**.

Layer 3 is the one worth emphasising — instructions alone are a hope, not a guarantee.

**Q: You merged two LLM calls into one. Why?**

Originally: one call for the news summary, another for the recommendation. Merging them halved
API usage — which matters enormously on a free tier of 20 requests/day.

But the better reason emerged from testing. The summary prompt **never saw the sentiment
score**. So the two could contradict each other — measured: the summary called MSFT's news
"Mixed" while FinBERT scored it **−0.39**, clearly negative. Two pages of the same app
disagreeing.

The merged prompt sees both, and agreed with the score in every test.

**I also measured the risk honestly:** on 2 tickers tested, the verdict changed on one (MSFT
went Buy → Hold). With temperature 0.3 the model isn't deterministic, so I couldn't separate a
real effect from noise without more runs than the daily quota allowed — and I documented that
uncertainty rather than claiming the merge was strictly better.

**Q: What happens when the LLM is unavailable?**

It never breaks. There's a deterministic fallback that combines the forecast and sentiment
with fixed weights:

- If the model beats the baseline: 50% forecast, 50% sentiment
- If it doesn't: **25% forecast, 75% sentiment** — trust the coin-flip model less

Plus a model fallback chain: if the primary Gemini model is exhausted, it tries the next.

**Q: Tell me about the quota bug you found.**

Analyses were taking 81 seconds. Profiling showed Gemini was 85–95% of that, so I looked closer.

Google returns HTTP 429 for two very different situations: a short per-minute rate limit, and a
per-**day** quota exhaustion. Our code treated both as "retry shortly", so on every single call
it retried a permanently-dead model four times — burning ~5 seconds of sleep — before falling
through to a model that worked.

The fix reads Google's own `retryDelay` field. If the wait exceeds our retry budget, we "park"
that model and skip it entirely on later calls.

**Q: How did you make it faster overall?**

Repeat analysis went **81 seconds → 2 seconds**, using zero API requests:

| Change | Effect |
|---|---|
| Cache LLM responses | Biggest win — keyed on the inputs, so it self-invalidates when news or numbers change |
| Merge two calls into one | Halves quota use |
| Fix the quota retry bug | ~5s per call |
| Cache the forecast | Keyed on ticker + **last price bar date**, so a new trading day invalidates it automatically |
| Pre-load models at startup | ~13s off the first analysis |

**The key insight: I profiled before optimising.** My plan assumed model retraining was the
bottleneck. Measurement proved it was only 4.5 seconds — the LLM was the real cost. Without
profiling I'd have optimised the wrong thing entirely.

---

## 6. System design

**Q: Walk me through the architecture.**

One entry point: `analyze(ticker)`. It runs in three stages:

```
Stage 1 (parallel):  prices ‖ news ‖ company info ‖ market index
Stage 2 (parallel):  forecast ‖ (de-dup → rank → sentiment) ‖ risk metrics
Stage 3:             one LLM call → news summary + recommendation
```

Stages 1 and 2 use threads because they're waiting on network and are independent.

Every step is wrapped in a uniform `Agent` class with `safe_run()`, which **never raises**. If
the news API dies, that becomes a warning and the analysis continues on price data alone. Only
missing price data is fatal — without it there's nothing to forecast.

**Q: How do modules communicate?**

Through Pydantic models in `orchestration/schemas.py` — `Article`, `ForecastResult`,
`Recommendation`, `AnalysisResult` and others. Every boundary is typed and validated, so a
module can be tested or replaced in isolation.

**Q: Your project calls things "agents" but you said they're not LLM agents. Explain.**

Honest naming matters. The brief listed ten "agents", but architecturally only the
recommendation step involves genuine LLM reasoning. The rest are deterministic modules wrapping
libraries — yfinance, RapidFuzz, FinBERT, XGBoost.

We kept the "agent" name because each exposes the same `run()` interface with logging, timing
and error handling. But `architecture.md` states plainly which ones actually use an LLM. Calling
a `yfinance` wrapper an "AI agent" would be dishonest.

The **Ask** page is a genuine agent though — LangChain/LangGraph, tool-calling, it decides which
tools to invoke and in what order.

**Q: Why SQLite and not PostgreSQL?**

It's a single-user analytical app. SQLite needs no server, no configuration, and is a single
file. All database access sits behind `database/db.py`, so switching to Postgres means changing
one connection string — the rest of the app never touches SQL.

Choosing the simpler tool and isolating the decision is the point.

---

## 7. Performance

**Q: What's the caching strategy?**

Four layers, each keyed so it invalidates itself correctly:

| What | Key | Lifetime |
|---|---|---|
| Prices | ticker + period + interval | 60 min |
| News | ticker + lookback + max | 60 min |
| **Forecast** | ticker + **last price bar date** + settings hash | 7 days |
| **LLM response** | the article set + rounded forecast and sentiment | 24 h |

**The two interesting ones:**

The forecast key uses the *date of the last price bar*, not the clock. A new trading day
produces a different key automatically — there's no way to serve yesterday's forecast as
today's. The settings hash means changing a training parameter also busts it.

The LLM key uses the *inputs the model saw*. If the news or the numbers change, the key changes
and we regenerate. Values are rounded, because gradient-boosting produces tiny floating-point
differences between runs that would otherwise cause constant cache misses.

**Q: How do you know the cache is correct?**

Tests. The forecast cache is verified to round-trip byte-identically, to produce a different key
when one price bar is removed, and to invalidate when a setting changes.

---

## 8. Debugging war stories

These make excellent interview material — they show real debugging, not tutorial-following.

**Q: Tell me about a difficult bug.**

The app crashed mid-analysis with **no Python error at all** — the process just died. No
traceback means it's not a Python exception; it's a crash in native code.

macOS writes crash reports, so I read one. It pointed to `libomp.dylib`, the OpenMP threading
library, inside a thread synchronisation barrier.

The cause: three separate copies of OpenMP were loaded in one process — one from Homebrew for
XGBoost, and others bundled inside Python packages. Our pipeline trains models in one thread
while FinBERT runs in another. Each library starts its own worker threads, they hit a
synchronisation barrier belonging to a *different* copy of the library, and memory gets
corrupted.

**What I'd emphasise:** my first two fixes failed.

1. I set `OMP_NUM_THREADS=1`. It crashed again — that environment variable is **overridden** by
   the `n_jobs=-1` parameter passed to the models.
2. The next crash pointed somewhere different: PyTorch's Metal/GPU backend. The models were
   silently running on the GPU, which isn't safe to call from multiple threads. Fixing that was
   correct — the docs claimed CPU-only and the code never enforced it — but it crashed again.

The real problem was my **test harness**, not my fix. I was calling the pipeline directly,
while the real app runs it inside Streamlit's script-runner thread. Once I rebuilt the test to
drive the actual app, I could run a controlled experiment: change one variable, `n_jobs`.

- `n_jobs=1` → 12 out of 12 runs survived
- `n_jobs=-1` → segfault before run 1

That's causality, not correlation. And the fix cost nothing — accuracy was identical on all
three test stocks, because with only ~450 training rows, coordinating threads costs more than
it saves.

**Q: Any bug that taught you something about testing?**

Two, both about tests that pass while testing nothing.

**The routing bug.** After restructuring the app's navigation, I wrote a test that loaded all 13
pages — all passed. But I'd used the wrong mechanism to switch pages, so **every check silently
rendered the same default page**. Thirteen green ticks testing one page. I fixed the routing and
added an assertion on each page's *title*, so it can't happen again.

**The PDF export.** I reported PDF export as working. It never had. A bare `except: return None`
was hiding two real bugs — the library wasn't installed, and the reports contain em-dashes that
the PDF font can't encode. The button silently never appeared. Now failures are logged, and
there are tests asserting a real PDF comes out.

The lesson both times: **a test that can't fail is worse than no test**, because it buys false
confidence.

---

## 9. Testing

**Q: What do you test?**

127 tests, all offline — no API keys or internet needed. They run against a temporary database
so they never touch real data.

The ones worth mentioning:

- **No look-ahead** — independently recomputes the target shift and asserts the last row is empty
- **Metric correctness** — a perfect prediction must score skill > 0.99; a sign-flipped one < 0
- **`beats_baseline` consistency** — asserts the headline honesty flag can't disagree with the
  metric it's derived from
- **Conformal coverage** — deliberately feeds fat-tailed (non-normal) data and checks coverage
  still lands near target
- **Grounding** — asserts a fabricated URL is stripped from the output
- **Graceful degradation** — kills the news API mid-pipeline and asserts the analysis still
  completes, and that raw exception text never reaches the user
- **Quota logic** — per-day 429 gets parked, per-minute 429 gets retried

**Q: How do you test a Streamlit UI?**

`AppTest`, Streamlit's official harness. It runs the app headlessly and lets you inspect
elements and click buttons. Our sweep drives all 13 views and asserts each renders its expected
title without exceptions.

---

## 10. Limitations — expect these

Being asked about weaknesses is an opportunity. Naming them yourself reads as rigour; being
caught out doesn't.

**Q: What are the limitations?**

Stated plainly in the README under *Known gaps*:

1. **Sentiment isn't a model input yet** — no historical news exists to train on; the app
   accumulates its own and enables the feature automatically once there's enough
2. **One model per stock, ~450 rows, one test window** — pooling across many stocks and rolling
   the evaluation would be more rigorous. *This*, not "more rows", is the real version of the
   "more data would help" argument
3. **The strategy backtest is ~30 days** — an illustration, not evidence
4. **Sentiment has no time decay** — a two-week-old article counts as much as this morning's
5. **Survivorship bias** — the index lists are today's members, so any historical test
   implicitly assumes you knew which companies would survive
6. **Not production-hardened** — no login, no rate limiting, SQLite, no CI

**Q: If you had another month, what would you do?**

In order of value:

1. **Sentiment as a real model feature** — the infrastructure is built and waiting on data
2. **Pool training across stocks** — one model on 50 stocks × 450 rows instead of 50 separate
   small models
3. **Rolling-window evaluation** — many test windows, not one, so results aren't a lucky sample
4. **Backtest the LLM's verdict itself** — currently only the price forecast is backtested

---

## 11. Rapid-fire

**Why log returns?** They're additive over time and roughly symmetric, which suits models better
than percentages.

**Why an ensemble?** Different models make different mistakes. Averaging cancels some of them.

**How are ensemble weights set?** Inversely proportional to each model's validation error, so
better models count more. Weights sum to 1 — and there's a test asserting that.

**Why BM25 *and* semantic search?** Keywords catch exact matches, embeddings catch meaning.
"iPhone maker" should match Apple; only the semantic half gets that.

**What's `TimeSeriesSplit`?** Cross-validation that respects time order — always train on the
past, test on the future.

**Why is the API key called `NEWS_API_KEY` but it isn't newsapi.org?** A legacy naming mistake.
It's an Event Registry key. Documented in `.env.example` and `config/settings.py` so nobody
loses an hour to it.

**Biggest performance win?** Caching LLM responses — 81 seconds to 2.

**Biggest correctness win?** Grading every model against a naive baseline and reporting when it
loses.

**What would you do differently?** Profile before planning. My optimisation plan assumed model
training was the bottleneck; it was 4.5 seconds out of 81.
