"""Central configuration, loaded from environment / .env via pydantic-settings."""
from __future__ import annotations

import os
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# ── OpenMP: must be set before xgboost/lightgbm/catboost/torch ever load ──
# Two OpenMP runtimes live in this process — Homebrew's libomp (the GBMs) and the copy
# bundled inside PyTorch. When the forecast trains in one thread while MiniLM/FinBERT run
# in another, their worker threads collide and the process dies with SIGSEGV inside
# `__kmp_fork_barrier` (confirmed from a macOS crash report).
#
# Pinning to one thread removes the worker pool, and therefore the barrier. It is not a
# performance sacrifice: on ~450-row training data, thread coordination costs more than it
# saves — measured 5.07s -> 4.02s, with identical directional accuracy and skill.
#
# This lives here, not in run.sh, so tests and scripts get it too. Every heavy library in
# this codebase is imported lazily inside functions, so `config.settings` always wins the race.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """Typed application settings. Reads from environment and the project .env file."""

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ── Secrets / API keys ────────────────────────────────
    # NEWS_API_KEY holds an Event Registry (newsapi.ai) key, NOT a newsapi.org key.
    news_api_key: str = Field(default="", alias="NEWS_API_KEY")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-flash-latest", alias="GEMINI_MODEL")
    hf_token: str = Field(default="", alias="HF_TOKEN")
    # Optional: sync the SQLite database to a private HF Dataset so History / Track Record /
    # Watchlist survive restarts on hosts with an ephemeral disk (e.g. HF Spaces).
    # Format "user/dataset-name". Empty (the default) keeps everything purely local.
    hf_dataset_repo: str = Field(default="", alias="HF_DATASET_REPO")

    # Gemini fallback chain tried in order if the primary is retired/quota-limited.
    gemini_model_fallbacks: tuple[str, ...] = ("gemini-3.5-flash", "gemini-3-flash-preview")

    # ── Data / forecasting tunables ───────────────────────
    price_period: str = "2y"          # yfinance history window for training
    price_interval: str = "1d"
    min_history_rows: int = 150       # minimum labelled rows required to train
    news_lookback_days: int = 14
    news_max_articles: int = 40       # cap fetched before dedup/rank
    news_top_k: int = 8               # articles sent to the LLM
    forecast_test_size: int = 30      # backtest window (days) for charts/metrics
    event_registry_url: str = "https://eventregistry.org/api/v1/article/getArticles"

    # ── v2: markets & screener ────────────────────────────
    default_region: str = "US"        # "US" or "INDIA"
    screener_max_constituents: int = 60   # cap per index (huge indices are time-boxed)
    screener_concurrency: int = 16        # parallel workers for batch scoring (I/O-bound fetches)

    # ── Paths ─────────────────────────────────────────────
    cache_dir: Path = PROJECT_ROOT / "data_cache"
    models_dir: Path = PROJECT_ROOT / "models_store"
    db_path: Path = PROJECT_ROOT / "data_cache" / "stocksense.db"
    cache_ttl_minutes: int = 60       # re-fetch prices/news older than this

    def ensure_dirs(self) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    @property
    def gemini_models(self) -> list[str]:
        """Primary model followed by fallbacks, de-duplicated, order preserved."""
        seen: dict[str, None] = {}
        for m in (self.gemini_model, *self.gemini_model_fallbacks):
            if m:
                seen.setdefault(m, None)
        return list(seen)

    @property
    def has_gemini(self) -> bool:
        return bool(self.gemini_api_key)

    @property
    def has_news_api(self) -> bool:
        return bool(self.news_api_key)

    @property
    def has_db_sync(self) -> bool:
        """True when the database should be mirrored to a Hugging Face Dataset."""
        return bool(self.hf_dataset_repo and self.hf_token)


settings = Settings()
settings.ensure_dirs()
