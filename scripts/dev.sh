#!/usr/bin/env bash
# Starts AI Image Studio for development: the API on 127.0.0.1:8000 and the app
# on http://localhost:3000, both with live reload. On a Mac the browser opens by
# itself. Press Ctrl+C to stop both.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# shellcheck source=lib/env.sh
source "$ROOT/scripts/lib/env.sh"

IS_MAC=""
if [[ "$(uname)" == "Darwin" ]]; then
  IS_MAC=1
fi

# Already running (e.g. started by the Mac app): just show it.
if ais_is_running 3; then
  echo "AI Image Studio is already running at $AIS_URL (stop it with scripts/stop.sh to develop)."
  if [[ -n "$IS_MAC" ]]; then
    open "$AIS_URL"
  fi
  exit 0
fi

ais_load_tools

if [[ ! -f "$ROOT/backend/.env" || ! -d "$ROOT/frontend/node_modules" ]]; then
  echo "Run ./scripts/setup.sh first."
  exit 1
fi

cd "$ROOT/backend"
# Apply any new database changes that came with updated code.
uv run --quiet alembic upgrade head

uv run --quiet uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload &
API_PID=$!

cd "$ROOT/frontend"
# Listen on this computer only (127.0.0.1), never on the network.
npm run dev &
WEB_PID=$!

cleanup() {
  trap - INT TERM EXIT
  # Signal the whole process group, so npm's and uvicorn's child processes stop too.
  kill 0 2>/dev/null || true
}
trap cleanup INT TERM EXIT

# Open the browser once the studio answers (the first page takes a few seconds).
if [[ -n "$IS_MAC" ]]; then
  (
    for _ in {1..120}; do
      if ais_is_running 60; then
        open "$AIS_URL"
        break
      fi
      sleep 1
    done
  ) &
fi

echo
echo "  AI Image Studio is starting…"
echo "  It opens at $AIS_URL"
echo "  Keep this window open. Press Ctrl+C to stop."
echo

# Stop both when either one exits. (A loop, because macOS's bash has no `wait -n`.)
while kill -0 "$API_PID" 2>/dev/null && kill -0 "$WEB_PID" 2>/dev/null; do
  sleep 1
done
