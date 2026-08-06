#!/usr/bin/env bash
# Launch StockSense AI with the required macOS environment fixes.
#   - ARROW_DEFAULT_MEMORY_POOL=system : pyarrow's bundled mimalloc segfaults in
#     mi_thread_init when Streamlit converts a DataFrame to Arrow in its script-runner
#     thread (st.dataframe). The system allocator avoids it.
set -e
cd "$(dirname "$0")"
export ARROW_DEFAULT_MEMORY_POOL="${ARROW_DEFAULT_MEMORY_POOL:-system}"
exec .venv/bin/streamlit run frontend/app.py "$@"
