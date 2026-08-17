# StockSense AI — production image.
#
# Targets Hugging Face Spaces (Docker SDK) but runs anywhere: Render, Railway, Fly, a VPS,
# or locally. Listens on $PORT (7860 by default, which is what HF Spaces expects).
#
# Two things this image does deliberately:
#
#  1. **CPU-only PyTorch.** The default `torch` wheel bundles ~2 GB of CUDA libraries that
#     this app never uses — it runs FinBERT and MiniLM on CPU by design. Installing from
#     PyTorch's CPU index first means the later `pip install -r requirements.txt` sees torch
#     as already satisfied and keeps the small build.
#
#  2. **Models baked in at build time.** FinBERT (~440 MB) and MiniLM (~90 MB) are downloaded
#     during the build instead of on first request, so the first visitor doesn't wait several
#     minutes and a cold start can't time out.

FROM python:3.13-slim

# libgomp1 is the OpenMP runtime XGBoost/LightGBM link against — without it they fail to
# load at import. The macOS equivalent of `brew install libomp`.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Non-root user. HF Spaces runs as uid 1000; matching it keeps the caches writable.
RUN useradd -m -u 1000 app
ENV HOME=/home/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # Models live in the image, not a mounted volume.
    HF_HOME=/home/app/.cache/huggingface \
    # See config/settings.py — three OpenMP runtimes in one process segfault when the GBMs
    # train while PyTorch runs in another thread. Belt and braces alongside the code default.
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    # pyarrow's bundled mimalloc crashes in Streamlit's script-runner thread.
    ARROW_DEFAULT_MEMORY_POOL=system

WORKDIR /app

# Dependencies first so code edits don't invalidate the (slow) install layer.
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

# Writable scratch for the SQLite database and the price/news cache. Created (and /app
# handed over) while still root — `WORKDIR` makes /app root-owned, so a `mkdir` here after
# `USER app` would be denied. On a host with an ephemeral disk this is wiped on restart:
# set HF_DATASET_REPO + HF_TOKEN and `database/sync.py` mirrors the database to a private
# HF Dataset and restores it on boot.
RUN mkdir -p /app/data_cache /app/models_store && chown -R app:app /app

# Bake the models in. Runs as `app` so the cache lands in a directory the runtime user owns.
USER app
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
from transformers import pipeline; \
SentenceTransformer('all-MiniLM-L6-v2', device='cpu'); \
pipeline('sentiment-analysis', model='ProsusAI/finbert', device='cpu'); \
print('models cached')"

COPY --chown=app:app . .

EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT:-7860}/_stcore/health" || exit 1

# JSON form so Docker doesn't wrap this in an implicit shell; `exec` then replaces sh with
# streamlit, making it PID 1 so it receives SIGTERM directly on `docker stop`.
CMD ["sh", "-c", "exec streamlit run frontend/app.py \
    --server.port=${PORT:-7860} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --server.fileWatcherType=none \
    --browser.gatherUsageStats=false"]
