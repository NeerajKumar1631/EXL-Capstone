"""Stage-by-stage timing for `analyze()` — the Stage 1 speed baseline.

This changes no application code. It monkey-patches timing probes onto the agent
wrapper and a few hot internals, runs the real pipeline, and prints where the
seconds actually went.

Run:
    PYTHONPATH=. .venv/bin/python scripts/profile_run.py
    PYTHONPATH=. .venv/bin/python scripts/profile_run.py --ticker MSFT --no-llm

Three scenarios run in one process (this is the point — the second and third
benefit from models already being in memory):

  cold    nothing cached, models not yet loaded   -> the worst case, a first ever run
  repeat  the same ticker again                   -> what a returning user waits for
  new     a different ticker, models now warm     -> a genuinely new stock

By default a throwaway cache directory is used, so your real `data_cache/` is left
untouched. Pass --use-real-cache to profile against it instead.

Note: with `use_llm` on this makes real Gemini calls (2 per run), which consumes
free-tier quota. Pass --no-llm to skip them.
"""
from __future__ import annotations

import argparse
import pathlib
import shutil
import sys
import tempfile
import threading
import time
from collections import defaultdict

_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Recording ─────────────────────────────────────────────────────────
# Stages run in threads, so every write is locked.
_lock = threading.Lock()
_seconds: dict[str, float] = defaultdict(float)
_calls: dict[str, int] = defaultdict(int)


def _record(label: str, elapsed: float) -> None:
    with _lock:
        _seconds[label] += elapsed
        _calls[label] += 1


def _reset() -> None:
    with _lock:
        _seconds.clear()
        _calls.clear()


def _snapshot() -> list[tuple[str, int, float]]:
    with _lock:
        return sorted(
            ((label, _calls[label], secs) for label, secs in _seconds.items()),
            key=lambda row: -row[2],
        )


def _probe(label: str, fn):
    """Wrap a plain function so its runtime is recorded under `label`."""

    def inner(*args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(*args, **kwargs)
        finally:
            _record(label, time.perf_counter() - start)

    return inner


def _probe_method(label_of, fn):
    """Wrap a method; `label_of(self)` builds the label so we can use self.name."""

    def inner(self, *args, **kwargs):
        start = time.perf_counter()
        try:
            return fn(self, *args, **kwargs)
        finally:
            _record(label_of(self), time.perf_counter() - start)

    return inner


def install_probes() -> None:
    """Patch the pipeline's stages and its slowest internals to record timings."""
    import embeddings.encoder as encoder
    import orchestration.pipeline as pipeline
    import sentiment.finbert as finbert
    from agents.base import Agent
    from forecasting import arima_model
    from forecasting.models import RegressorModel
    from llm.client import LLMClient

    # Every pipeline stage (all 10 agents inherit this single entry point).
    Agent.run = _probe_method(lambda self: f"stage: {self.name}", Agent.run)

    # Not agents, but they happen inside analyze().
    pipeline._safe_fetch_prices = _probe("stage: benchmark_prices", pipeline._safe_fetch_prices)
    pipeline.resolve_company_name = _probe("stage: resolve_company", pipeline.resolve_company_name)

    # The suspected bottleneck: how many model fits, and how long each kind takes.
    RegressorModel.fit = _probe_method(lambda self: f"  fit: {self.name}", RegressorModel.fit)
    arima_model._fit = _probe("  fit: ARIMA", arima_model._fit)

    # Network + model-load costs hiding inside the stages above.
    LLMClient._run = _probe_method(lambda self: "  call: Gemini", LLMClient._run)
    finbert.get_finbert = _probe("  load: FinBERT", finbert.get_finbert)
    encoder.get_encoder = _probe("  load: MiniLM", encoder.get_encoder)


# ── Reporting ─────────────────────────────────────────────────────────
_WIDTH = 66


def _report(title: str, wall: float, rows: list[tuple[str, int, float]], result) -> None:
    print()
    print(title)
    print("=" * _WIDTH)
    print(f"{'step':<36}{'calls':>7}{'seconds':>11}{'% wall':>10}")
    print("-" * _WIDTH)
    for label, calls, secs in rows:
        print(f"{label:<36}{calls:>7}{secs:>11.2f}{secs / wall * 100:>9.0f}%")
    print("-" * _WIDTH)
    print(f"{'TOTAL (wall clock)':<36}{'':>7}{wall:>11.2f}")

    if result is not None:
        fc = result.forecast
        verdict = result.recommendation.action if result.recommendation else "—"
        if fc:
            print(f"\nresult: {verdict} · beats_baseline={fc.beats_baseline} · best={fc.best_model}")
        else:
            print(f"\nresult: {verdict} · no forecast produced")
        for err in result.errors:
            print(f"  ERROR   {err}")
        for warn in result.warnings:
            print(f"  warning {warn}")


def _run_once(ticker: str, use_llm: bool):
    from orchestration.pipeline import analyze

    _reset()
    start = time.perf_counter()
    try:
        result = analyze(ticker, use_llm=use_llm)
    except Exception as exc:  # noqa: BLE001 - profiling must not die on a bad ticker
        wall = time.perf_counter() - start
        print(f"\n!! analyze({ticker}) raised {type(exc).__name__}: {exc}")
        return wall, _snapshot(), None
    return time.perf_counter() - start, _snapshot(), result


def main() -> int:
    ap = argparse.ArgumentParser(description="Time each stage of an analyze() run.")
    ap.add_argument("--ticker", default="AAPL", help="ticker for the cold + repeat runs")
    ap.add_argument("--other", default="MSFT", help="ticker for the 'new stock' run")
    ap.add_argument("--no-llm", action="store_true", help="skip Gemini (saves quota)")
    ap.add_argument("--use-real-cache", action="store_true",
                    help="profile against data_cache/ instead of a throwaway directory")
    args = ap.parse_args()

    use_llm = not args.no_llm

    from config.settings import settings

    tmp_dir = None
    if not args.use_real_cache:
        tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="stocksense_profile_"))
        settings.cache_dir = tmp_dir
        settings.ensure_dirs()

    install_probes()

    print("=" * _WIDTH)
    print("StockSense — speed baseline")
    print("=" * _WIDTH)
    print(f"tickers      {args.ticker} (cold, repeat) · {args.other} (new stock)")
    print(f"Gemini       {'on — this uses free-tier quota' if use_llm else 'off'}")
    print(f"cache dir    {settings.cache_dir}{'' if args.use_real_cache else '  (throwaway)'}")
    print("\nStages run concurrently, so the percentages add up to more than 100%.")
    print("Read them as 'share of total wall time', not as slices of a pie.")

    try:
        cold_wall, cold_rows, cold_res = _run_once(args.ticker, use_llm)
        _report(f"1. COLD — {args.ticker}, nothing cached, models not yet loaded",
                cold_wall, cold_rows, cold_res)

        repeat_wall, repeat_rows, repeat_res = _run_once(args.ticker, use_llm)
        _report(f"2. REPEAT — {args.ticker} again, cache warm, models in memory",
                repeat_wall, repeat_rows, repeat_res)

        new_wall, new_rows, new_res = _run_once(args.other, use_llm)
        _report(f"3. NEW STOCK — {args.other}, its own cache cold, models warm",
                new_wall, new_rows, new_res)

        print()
        print("=" * _WIDTH)
        print("SUMMARY")
        print("=" * _WIDTH)
        print(f"{'cold (' + args.ticker + ')':<36}{cold_wall:>11.2f}s")
        print(f"{'repeat (' + args.ticker + ')':<36}{repeat_wall:>11.2f}s")
        print(f"{'new stock (' + args.other + ')':<36}{new_wall:>11.2f}s")
        print()
        print("The gap between 'cold' and 'repeat' is what model loading costs (Stage 2.2).")
        print("Whatever 'repeat' still spends on stage: forecast is what caching removes (Stage 2.1).")
    finally:
        if tmp_dir is not None:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
