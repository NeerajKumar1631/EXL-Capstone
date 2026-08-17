# Local Setup Guide

How to get StockSense AI running on your own machine, from nothing.

Written so you can follow it without knowing the project. Every step says **what** to do and
**why**, and the troubleshooting section covers every error we actually hit while building it.

**Time needed:** about 15 minutes, most of it waiting for downloads.

---

## What you need first

| Requirement | Why | How to check |
|---|---|---|
| **Python 3.11 or newer** (3.13 recommended) | The code uses modern type syntax like `list[str] \| None` | `python3 --version` |
| **~3 GB free disk** | PyTorch and the ML libraries are large | `df -h .` |
| **Internet** | Prices, news, and a one-time model download | — |
| **Git** | To clone the repo | `git --version` |

You do **not** need a GPU. Everything runs on CPU by design.

You do **not** need API keys to start — the app works without them and tells you what it's
falling back to. Keys make it better; see [step 5](#5-add-api-keys-optional-but-recommended).

---

## 1. Get the code

```bash
git clone https://github.com/NeerajKumar1631/EXL-Capstone.git
cd EXL-Capstone
```

## 2. Create a virtual environment

A virtual environment keeps this project's libraries separate from everything else on your
machine, so nothing else breaks.

```bash
python3 -m venv .venv
```

This creates a `.venv/` folder. You don't need to "activate" it if you use the full path
`.venv/bin/python` in the commands below (which is what we do — it avoids a whole class of
"wrong Python" confusion).

## 3. Install the libraries

```bash
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

This downloads about 2 GB and takes 5–10 minutes. PyTorch is the big one.

> **On macOS, if `pip` crashes** — see [Troubleshooting](#pip-crashes-on-macos) below. Some
> macOS + Homebrew Python combinations have a broken `pip`; the fix is to use `uv` instead.

## 4. macOS and Linux only: install the OpenMP runtime

XGBoost and LightGBM need a library called OpenMP. **They will fail to import without it.**

```bash
# macOS
brew install libomp

# Ubuntu / Debian
sudo apt-get install libgomp1
```

Windows users: nothing to do, it's bundled.

## 5. Add API keys (optional but recommended)

```bash
cp .env.example .env
```

Then open `.env` in any editor and fill in what you have:

| Key | Where to get it | What happens without it |
|---|---|---|
| `GEMINI_API_KEY` | [aistudio.google.com/apikey](https://aistudio.google.com/apikey) — free | The app still gives a Buy/Hold/Sell call, but from a simple rule instead of AI reasoning. The **Ask** chat page falls back to keyword matching. |
| `NEWS_API_KEY` | [eventregistry.org](https://eventregistry.org) — free | News comes from Yahoo Finance instead. Fewer articles, shorter text. |
| `HF_TOKEN` | [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) | Nothing — it only speeds up the model download. |
| `HF_DATASET_REPO` | — | Nothing locally. Only used when deploying to a server with a disk that gets wiped. |

**The app runs fine with an empty `.env`.** It will just tell you it's using fallbacks.

> ⚠️ Never commit `.env`. It's already in `.gitignore`.
> Note the variable is `NEWS_API_KEY` and it's an **Event Registry** key — *not* newsapi.org.

## 6. Run it

```bash
./run.sh
```

Open **http://localhost:8501**.

> If you get `permission denied`, run `chmod +x run.sh` once, or use `bash run.sh`.

**Always use `./run.sh`, not `streamlit run` directly.** The script sets two environment
variables the app needs on macOS to avoid crashing — see
[Why run.sh exists](#why-runsh-exists-and-not-plain-streamlit-run).

### First run will be slow

The first analysis downloads two AI models (~530 MB total): FinBERT for sentiment and MiniLM
for matching articles to a company. **This happens once.** After that they're cached on disk,
and the app pre-loads them in the background when it starts.

Expect roughly:
- First ever analysis: **60–90 seconds**
- A new stock after that: **20–40 seconds**
- The same stock again the same day: **about 2 seconds** (everything is cached)

---

## 7. Try it out

1. In the sidebar, type `AAPL` — or just type `apple`, the search finds the ticker for you —
   then press **Enter**
2. Click **Run analysis**
3. Look around the pages in the left menu

Good first stops:
- **Dashboard** — the verdict and headline numbers
- **Forecast** — every model graded against a "tomorrow = today" baseline, plus whether
  following the signal would actually have made money
- **Track Record** — how the app's own past predictions actually turned out

To try Indian stocks, switch **Market** to *India* in the sidebar and search `reliance` or `tata`.

---

## Running the tests

```bash
export ARROW_DEFAULT_MEMORY_POOL=system
.venv/bin/python -m pytest tests/ -q
```

Expect **127 passing**. These run fully offline — no API keys, no internet needed. They use a
temporary database, so they never touch your real data.

Two deeper checks:

```bash
# Every page renders without error (13 views)
PYTHONPATH=.:frontend .venv/bin/python scripts/apptest_all_pages.py

# Live end-to-end sweep — uses real network and a little API quota
PYTHONPATH=. .venv/bin/python scripts/integration_e2e.py
```

---

## Optional: run it in Docker instead

If you'd rather not install Python libraries at all:

```bash
docker build -t stocksense .
docker run --rm -p 7860:7860 -e GEMINI_API_KEY=your_key stocksense
```

Then open **http://localhost:7860**.

The build takes 10–15 minutes because it bakes the AI models into the image, and produces a
~5.8 GB image. It handles the OpenMP library and all environment settings for you.

---

## Troubleshooting

### `pip` crashes on macOS

Some macOS versions return an empty value from `platform.mac_ver()`, which crashes pip's
certificate handling. Use [`uv`](https://github.com/astral-sh/uv), a faster pip replacement:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv pip install --python .venv/bin/python -r requirements.txt
```

### `libxgboost.dylib` or `lib_lightgbm.dylib` won't load

The OpenMP runtime is missing. Go back to [step 4](#4-macos-and-linux-only-install-the-openmp-runtime).

### "Python quit unexpectedly" during an analysis

You're running `streamlit run` directly instead of `./run.sh`. See below.

### Hundreds of `No module named 'torchvision'` messages

**Harmless.** Streamlit's file-watcher inspects every loaded module, and the `transformers`
library loads its image-processing code when inspected — which wants `torchvision`, which we
don't install because this app only handles text. `./run.sh` turns the watcher off so you
don't see it. Use `STOCKSENSE_DEV=1 ./run.sh` if you want auto-reload while editing code.

### "No price data for TICKER"

The symbol is wrong. US stocks look like `AAPL`; Indian stocks need a suffix, like `TCS.NS`.
Or just type the company name and let the search find it.

### The recommendation says "rule-based"

No Gemini key, an invalid one, or you've used up the free daily quota (20 requests per day per
model). The app rotates through backup models automatically and falls back to a rule-based
answer when they're all exhausted. Add or rotate a key in `.env` and restart.

### The PDF download button is missing

`fpdf2` isn't installed: `.venv/bin/python -m pip install fpdf2`. The button hides itself
rather than showing a broken link.

---

## Why `run.sh` exists (and not plain `streamlit run`)

`run.sh` sets two environment variables that this app genuinely needs. Both come from real
crashes we debugged, and both are documented in the script itself:

**`ARROW_DEFAULT_MEMORY_POOL=system`** — Streamlit converts tables to Apache Arrow format on a
background thread. The memory allocator bundled with `pyarrow` crashes when used that way on
macOS, taking the whole app down.

**`--server.fileWatcherType=none`** — stops the harmless-but-noisy `torchvision` messages
described above.

There's a third protection built into the code itself (`config/settings.py`): the app forces
single-threaded maths. Three separate copies of the OpenMP library end up loaded in one
process, and if the forecasting models and PyTorch each start their own worker threads, they
collide and crash the app with a segmentation fault. Single-threaded is just as fast here,
because the training data is small.

---

## Project layout, briefly

```
config/              settings (reads .env), logging, index lists
data_ingestion/      prices (yfinance), news, company info, symbol search
technical_analysis/  indicators and model features
forecasting/         the ML models, evaluation, prediction ranges, strategy backtest
retrieval/           removes duplicate articles, ranks by relevance
sentiment/           FinBERT scoring and aggregation
llm/                 Gemini client and prompts
recommendation/      combines everything into a Buy/Hold/Sell call
analytics/           risk metrics, track record
orchestration/       schemas.py (data contracts) + pipeline.py (runs it all)
frontend/            app.py (router) + views/ (13 pages)
tests/               127 tests
```

The single entry point is `orchestration/pipeline.py::analyze(ticker)`. Everything else feeds
into it. If you want to understand the code, start there.

More detail: [`architecture.md`](../architecture.md) ·
[`docs/api_reference.md`](api_reference.md) · [`docs/agent_workflow.md`](agent_workflow.md)
