# Deployment

## Why Hugging Face Spaces

The app's peak memory was measured, not guessed:

| Stage | Resident memory |
|---|---|
| Python + pandas / numpy / Streamlit | 137 MB |
| + XGBoost, LightGBM, CatBoost | 227 MB |
| + MiniLM | 507 MB |
| + FinBERT | 589 MB |
| **+ one full analysis** | **1,060 MB** |

That rules out the obvious free host: **Streamlit Community Cloud caps at 1 GB**, so the
container would be killed mid-analysis. Render's free tier (512 MB) and Fly's smallest
instances are further out of reach.

**Hugging Face Spaces gives 16 GB RAM and 2 vCPU free**, needs no credit card, and serves the
FinBERT/MiniLM weights from its own infrastructure. It is the natural home for this app.

---

## One-time setup

### 1. Create the Space

At [huggingface.co/new-space](https://huggingface.co/new-space):

- **Space SDK:** `Docker` → *Blank*  (not the Streamlit SDK — this repo ships its own Dockerfile)
- **Hardware:** `CPU basic` (free)
- **Visibility:** your choice

### 2. Get a write token

[huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → **New token** →
role **write**. Then:

```bash
export HF_TOKEN=hf_xxxxxxxxxxxx
```

### 3. Deploy

```bash
./deploy/deploy_hf.sh <your-username>/<your-space-name>
```

The script copies the app into a checkout of the Space repo, swaps in
`deploy/README_SPACE.md` (Hugging Face reads its YAML front-matter to configure the Space —
which is why the project README can't be used directly), commits and pushes. The Space
rebuilds automatically.

**The first build takes 10–15 minutes**, because it bakes FinBERT and MiniLM into the image.
That is deliberate: downloading ~530 MB on the first request instead would make the first
visitor wait minutes, and can exceed the startup timeout.

### 4. Set the secrets

Space → **Settings** → **Variables and secrets**:

| Secret | Required? | Effect if missing |
|---|---|---|
| `GEMINI_API_KEY` | recommended | Recommendations fall back to the deterministic rule-based path; the Ask page uses keyword routing instead of the LLM agent. |
| `NEWS_API_KEY` | optional | News comes from yfinance instead of Event Registry — fewer articles, shorter text. |
| `HF_TOKEN` | optional | Needed only for database persistence (below). |
| `HF_DATASET_REPO` | optional | e.g. `your-name/stocksense-db`. Enables persistence. |

Changing secrets restarts the Space; no rebuild.

### 5. Redeploying

Re-run the same command. It skips the push when nothing changed.

```bash
./deploy/deploy_hf.sh <your-username>/<your-space-name>
```

---

## Database persistence (recommended)

Spaces have an **ephemeral disk**: it is wiped on every restart, redeploy and idle-sleep.
Without persistence the History, Track Record and Watchlist pages start empty each time —
and **Track Record is meaningless without it**, since it grades predictions made on earlier
days against what prices later did.

`database/sync.py` mirrors the SQLite file to a private Hugging Face Dataset:

- `pull()` runs once when the database engine is created, restoring the last upload.
- `push()` runs after each write, on a background thread, coalesced to at most one upload
  every 30 seconds so bursts of writes don't hammer the Hub.

To enable it, set **both** `HF_TOKEN` and `HF_DATASET_REPO` (e.g. `your-name/stocksense-db`).
The dataset is created automatically on first push, **private**. With either unset, every sync
function is a no-op and nothing touches the network — which is exactly how local development
and the test suite run.

A sync failure is logged and swallowed by design: the app keeps working with local-only data.

---

## Running the container anywhere else

The image is host-agnostic — it listens on `$PORT` (7860 by default) and takes configuration
from environment variables.

```bash
docker build -t stocksense .
docker run --rm -p 7860:7860 \
  -e GEMINI_API_KEY=... \
  -e NEWS_API_KEY=... \
  stocksense
```

Then open <http://localhost:7860>.

Two environment settings the image fixes for you, both learned the hard way:

- `OMP_NUM_THREADS=1` — three copies of `libomp` end up in one process; letting the GBMs fork
  OpenMP worker pools while PyTorch runs in another thread segfaults the app. Accuracy is
  identical either way.
- `ARROW_DEFAULT_MEMORY_POOL=system` — pyarrow's bundled mimalloc crashes inside Streamlit's
  script-runner thread.

`libgomp1` is installed in the image; it is the Linux equivalent of `brew install libomp` and
XGBoost/LightGBM fail to import without it.

---

## Cost and limits to expect

- **Free CPU Spaces sleep after ~48 h idle** and cold-start in a minute or so. The models are
  in the image, so no re-download.
- **Gemini free tier is 20 requests/day per model.** One analysis costs 1 request (the
  summary and recommendation were merged into a single call); one Ask question costs 2–3.
  Repeat analyses of an already-analyzed stock cost **zero** — they are served from cache.
  The client rotates through a fallback chain and parks models that hit their daily quota.
- **Event Registry free tier** returns ~4 weeks of history, which is why the sentiment feature
  accumulates its own daily readings rather than training on past news.
