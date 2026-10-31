# Shared by the scripts (sourced, not run). Written for macOS's bash 3.2.

AIS_APP_NAME="AI Image Studio"
AIS_APP_ID="ai-image-studio"
AIS_URL="http://localhost:3000"

# Makes Homebrew's tools available and starts PostgreSQL on a Mac.
ais_load_tools() {
  if [[ "$(uname)" != "Darwin" ]]; then
    return 0
  fi

  # Apps opened from the Dock or Finder start with a minimal PATH.
  if ! command -v brew >/dev/null 2>&1; then
    local candidate
    for candidate in /opt/homebrew/bin/brew /usr/local/bin/brew; do
      if [[ -x "$candidate" ]]; then
        eval "$("$candidate" shellenv)"
        break
      fi
    done
  fi
  if ! command -v brew >/dev/null 2>&1; then
    return 0
  fi

  # Homebrew's PostgreSQL 16 is not on the PATH by default.
  local pg_prefix
  pg_prefix="$(brew --prefix postgresql@16 2>/dev/null || true)"
  if [[ -n "$pg_prefix" && -d "$pg_prefix/bin" ]]; then
    export PATH="$pg_prefix/bin:$PATH"
  fi

  if command -v pg_isready >/dev/null 2>&1 && ! pg_isready -h localhost -q 2>/dev/null; then
    echo "Starting PostgreSQL…"
    brew services start postgresql@16 >/dev/null

    for _ in {1..20}; do
      if pg_isready -h localhost -q 2>/dev/null; then
        break
      fi
      sleep 0.5
    done
  fi
  return 0
}

# True when this studio (not some other program) answers on port 3000.
ais_is_running() {
  curl -fs -m "${1:-3}" "$AIS_URL/api/health" 2>/dev/null | grep -q "\"app\":\"$AIS_APP_ID\""
}

# True when something listens on a local TCP port.
ais_port_busy() {
  (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null
}

ais_log_dir() {
  if [[ "$(uname)" == "Darwin" ]]; then
    echo "$HOME/Library/Logs/$AIS_APP_NAME"
  else
    echo "$1/.run/logs"
  fi
}
