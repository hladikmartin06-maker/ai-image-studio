#!/usr/bin/env bash
# Stops AI Image Studio when it was started in the background (scripts/start.sh
# or the Mac app), and waits until it has fully stopped, so a restart right
# after never finds the ports still taken. Safe to run when it is not running.

set -uo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_DIR="$ROOT/.run"

# A process and everything it started, children first.
tree() {
  local pid="$1" child
  for child in $(pgrep -P "$pid" 2>/dev/null); do
    tree "$child"
  done
  echo "$pid"
}

pids=""
for name in web api; do
  pid_file="$RUN_DIR/$name.pid"
  if [[ -f "$pid_file" ]]; then
    pid="$(cat "$pid_file")"
    if kill -0 "$pid" 2>/dev/null; then
      pids="$pids $(tree "$pid" | tr '\n' ' ')"
    fi
    rm -f "$pid_file"
  fi
done

if [[ -z "${pids// /}" ]]; then
  echo "AI Image Studio was not running in the background."
  exit 0
fi

# shellcheck disable=SC2086  # one pid per word
kill $pids 2>/dev/null || true

# Wait up to 10 seconds for a clean exit, then force the rest.
for _ in {1..20}; do
  alive=""
  for pid in $pids; do
    if kill -0 "$pid" 2>/dev/null; then
      alive="$alive $pid"
    fi
  done
  if [[ -z "$alive" ]]; then
    echo "AI Image Studio stopped."
    exit 0
  fi
  sleep 0.5
done

# shellcheck disable=SC2086
kill -9 $alive 2>/dev/null || true
echo "AI Image Studio stopped."
