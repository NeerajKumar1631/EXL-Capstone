#!/usr/bin/env bash
# Push StockSense to a Hugging Face Space.
#
#   ./deploy/deploy_hf.sh <username>/<space-name>
#
# The Space is its own git repo. This copies the app into a clean checkout of it, swaps in
# the Space README (Hugging Face reads its YAML front-matter to configure the Space — that
# is why the project README can't be used directly), commits and pushes.
#
# Re-run it any time to redeploy; the Space rebuilds automatically on push.
#
# Auth: needs a Hugging Face token with **write** access.
#   export HF_TOKEN=hf_xxx          (or run `huggingface-cli login` first)
set -euo pipefail

SPACE="${1:-}"
if [[ -z "$SPACE" || "$SPACE" != */* ]]; then
  echo "usage: $0 <username>/<space-name>" >&2
  echo "  e.g. $0 neerajkumar/stocksense-ai" >&2
  exit 1
fi

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TOKEN="${HF_TOKEN:-}"
if [[ -z "$TOKEN" && -f "$ROOT/.env" ]]; then           # fall back to the local .env
  TOKEN="$(grep -E '^HF_TOKEN=' "$ROOT/.env" | cut -d= -f2- | tr -d '"'"'"' \r\n' || true)"
fi
if [[ -z "$TOKEN" ]]; then
  echo "error: no HF_TOKEN. Create one at https://huggingface.co/settings/tokens (write access)," >&2
  echo "       then: export HF_TOKEN=hf_xxx" >&2
  exit 1
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "==> cloning Space $SPACE"
if ! git clone --depth 1 "https://oauth2:${TOKEN}@huggingface.co/spaces/${SPACE}" "$WORK/space" 2>/dev/null; then
  echo "error: could not clone https://huggingface.co/spaces/${SPACE}" >&2
  echo "       Create the Space first (SDK: Docker), then re-run this script." >&2
  exit 1
fi

echo "==> copying application files"
# Mirror the repo, minus local state, secrets and dev-only material. Keeping tests/ and
# scripts/ out matches .dockerignore and keeps the Space small.
rsync -a --delete \
  --exclude '.git/' \
  --exclude '.venv/' \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '.env' \
  --exclude 'data_cache/' \
  --exclude 'models_store/' \
  --exclude '*.db' \
  --exclude '.pytest_cache/' \
  --exclude '.claude/' \
  --exclude 'tests/' \
  --exclude 'scripts/' \
  --exclude 'deploy/' \
  --exclude 'project-images.pdf' \
  --exclude 'README.md' \
  "$ROOT/" "$WORK/space/"

# The Space README carries the front-matter HF needs; the project README does not.
cp "$ROOT/deploy/README_SPACE.md" "$WORK/space/README.md"

cd "$WORK/space"
git add -A
if git diff --cached --quiet; then
  echo "==> nothing changed; Space is already up to date"
  exit 0
fi

git -c user.email="deploy@stocksense.local" -c user.name="StockSense Deploy" \
    commit -q -m "Deploy StockSense AI ($(date -u +%Y-%m-%dT%H:%MZ))"
echo "==> pushing"
git push -q origin main

echo
echo "Deployed. Build progress: https://huggingface.co/spaces/${SPACE}"
echo "First build takes ~10-15 min (it bakes FinBERT + MiniLM into the image)."
echo
echo "Remember to set these Secrets in Space settings:"
echo "  GEMINI_API_KEY   (LLM reasoning; falls back to rule-based without it)"
echo "  NEWS_API_KEY     (optional - Event Registry)"
echo "  HF_TOKEN         (optional - only for database persistence)"
echo "  HF_DATASET_REPO  (optional - e.g. ${SPACE%%/*}/stocksense-db)"
