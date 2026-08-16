#!/usr/bin/env bash
# Launch StockSense AI with the required macOS environment fixes.
#
#   - ARROW_DEFAULT_MEMORY_POOL=system : pyarrow's bundled mimalloc segfaults in
#     mi_thread_init when Streamlit converts a DataFrame to Arrow in its script-runner
#     thread (st.dataframe). The system allocator avoids it.
#
#   - fileWatcherType=none : Streamlit's watcher walks every loaded module to decide what
#     to watch. `transformers` uses lazy imports, so merely inspecting it forces its vision
#     modules to import, which need torchvision (not installed) — dumping ~100 harmless
#     tracebacks per run. Nothing breaks; FinBERT is text-only. Turning the watcher off
#     keeps the log clean. Set STOCKSENSE_DEV=1 to restore hot-reload while editing code.
set -e
cd "$(dirname "$0")"
export ARROW_DEFAULT_MEMORY_POOL="${ARROW_DEFAULT_MEMORY_POOL:-system}"

WATCHER=()
if [ "${STOCKSENSE_DEV:-0}" != "1" ]; then
  WATCHER=(--server.fileWatcherType none)
fi

# "$@" last so any flag you pass overrides the defaults above.
exec .venv/bin/streamlit run frontend/app.py "${WATCHER[@]}" "$@"
