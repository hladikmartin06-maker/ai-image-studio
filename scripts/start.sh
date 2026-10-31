#!/usr/bin/env bash
# Starts AI Image Studio in the background, if it is not running yet, and waits
# until it answers on http://localhost:3000. The Mac app and install.sh use it.
# It runs the optimised build of the app and rebuilds it after an update.
# Stop it with scripts/stop.sh. For development with live reload, use dev.sh.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_DIR="$ROOT/.run"

# shellcheck source=lib/env.sh
source "$ROOT/scripts/lib/env.sh"
LOG_DIR="$(ais_log_dir "$ROOT")"
mkdir -p "$RUN_DIR" "$LOG_DIR"

if ais_is_running 3; then
  echo "AI Image Studio is already running at $AIS_URL"
  exit 0
fi

# Another program on the studio's ports (for example another local studio).
for port in 3000 8000; do
  if ais_port_busy "$port"; then
    echo "Port $port is used by another program. Stop it (for example another local studio) and try again."
    exit 1
  fi
done

ais_load_tools

if [[ ! -f "$ROOT/backend/.env" || ! -x "$ROOT/backend/.venv/bin/uvicorn" || ! -d "$ROOT/frontend/node_modules" ]]; then
  echo "AI Image Studio is not set up yet. Run scripts/setup.sh first."
  exit 1
fi

# Build the app when there is no build for the current version of the code.
cd "$ROOT/frontend"
REVISION="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || echo unknown)"
if [[ ! -f .next/BUILD_ID || "$(cat .next/ais-build-revision 2>/dev/null || true)" != "$REVISION" ]]; then
  echo "Preparing the app (the first start after an update takes about a minute)…"
  if ! NEXT_TELEMETRY_DISABLED=1 npm run build > "$LOG_DIR/build.log" 2>&1; then
    echo "The app could not be built. Details: $LOG_DIR/build.log"
    exit 1
  fi
  echo "$REVISION" > .next/ais-build-revision
fi

# Database changes that came with the update.
cd "$ROOT/backend"
if ! .venv/bin/alembic upgrade head >> "$LOG_DIR/api.log" 2>&1; then
  echo "The database could not be reached or updated. Is PostgreSQL running? Details: $LOG_DIR/api.log"
  exit 1
fi

# Both servers listen on this computer only (127.0.0.1), never on the network.
nohup .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 >> "$LOG_DIR/api.log" 2>&1 &
echo $! > "$RUN_DIR/api.pid"

cd "$ROOT/frontend"
NEXT_TELEMETRY_DISABLED=1 nohup node_modules/.bin/next start --hostname 127.0.0.1 --port 3000 >> "$LOG_DIR/web.log" 2>&1 &
echo $! > "$RUN_DIR/web.pid"

for _ in {1..60}; do
  if ais_is_running 2; then
    echo "AI Image Studio is running at $AIS_URL"
    exit 0
  fi
  if ! kill -0 "$(cat "$RUN_DIR/api.pid")" 2>/dev/null || ! kill -0 "$(cat "$RUN_DIR/web.pid")" 2>/dev/null; then
    break
  fi
  sleep 1
done

echo "AI Image Studio did not start. Details: $LOG_DIR"
"$ROOT/scripts/stop.sh" >/dev/null 2>&1 || true
exit 1
