#!/usr/bin/env bash
# One-time setup for AI Image Studio. Safe to run again at any time.
# scripts/install.sh runs it with AIS_INSTALLER=1, which skips the question.
#
# On a Mac it uses Homebrew to install what is missing (it asks first):
#   node, uv (Python + packages) and PostgreSQL 16.
# Then it creates the database, writes backend/.env with a fresh secret,
# installs the app's dependencies and creates the database tables.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BACKEND="$ROOT/backend"
FRONTEND="$ROOT/frontend"
DB_NAME="ai_image_studio"

bold() { printf "\n\033[1m%s\033[0m\n" "$1"; }
ok() { printf "  \033[32m✓\033[0m %s\n" "$1"; }
fail() { printf "\n  \033[31m✗ %s\033[0m\n\n" "$1"; exit 1; }

# --- 1. Tools --------------------------------------------------------------------

bold "1/5  Checking tools"

if [[ "$(uname)" == "Darwin" ]]; then
  if ! command -v brew >/dev/null 2>&1; then
    fail "Homebrew is not installed. Install it from https://brew.sh (one command), then run this script again."
  fi
  # Homebrew's PostgreSQL 16 is not on the PATH by default.
  PG_PREFIX="$(brew --prefix postgresql@16 2>/dev/null || true)"
  if [[ -n "$PG_PREFIX" && -d "$PG_PREFIX/bin" ]]; then
    export PATH="$PG_PREFIX/bin:$PATH"
  fi

  # A plain string, not an array: macOS ships bash 3.2, which mishandles empty
  # arrays under `set -u`.
  missing=""
  command -v node >/dev/null 2>&1 || missing="$missing node"
  command -v uv >/dev/null 2>&1 || missing="$missing uv"
  command -v psql >/dev/null 2>&1 || missing="$missing postgresql@16"
  if [[ -n "$missing" ]]; then
    echo "  Missing:$missing"
    if [[ "${AIS_INSTALLER:-}" == "1" ]]; then
      answer="Y"  # install.sh: running the one command already said yes
    else
      read -r -p "  Install them with Homebrew now? [Y/n] " answer
    fi
    if [[ "${answer:-Y}" =~ ^[Nn] ]]; then
      fail "Install$missing and run this script again."
    fi
    # shellcheck disable=SC2086  # word splitting is intended: one formula per word
    brew install $missing
    PG_PREFIX="$(brew --prefix postgresql@16)"
    export PATH="$PG_PREFIX/bin:$PATH"
  fi
fi

command -v node >/dev/null 2>&1 || fail "Node.js is not installed (https://nodejs.org)."
command -v uv >/dev/null 2>&1 || fail "uv is not installed (https://docs.astral.sh/uv/)."
command -v psql >/dev/null 2>&1 || fail "PostgreSQL is not installed."
# Next.js 16 needs Node.js 20.9 or newer.
node_ok() {
  node -e 'const [a, b] = process.versions.node.split(".").map(Number); process.exit(a > 20 || (a === 20 && b >= 9) ? 0 : 1)'
}
if ! node_ok; then
  if [[ "$(uname)" == "Darwin" ]] && command -v brew >/dev/null 2>&1; then
    echo "  Node.js $(node -v) is too old; updating it with Homebrew…"
    brew upgrade node >/dev/null 2>&1 || brew install node
    hash -r
  fi
  node_ok || fail "Node.js 20.9 or newer is needed (found $(node -v))."
fi
ok "node $(node -v), $(uv --version), $(psql --version | head -1)"

# --- 2. Database -----------------------------------------------------------------

bold "2/5  Database"

if ! pg_isready -h localhost -q 2>/dev/null; then
  if [[ "$(uname)" == "Darwin" ]]; then
    echo "  Starting PostgreSQL…"
    brew services start postgresql@16 >/dev/null
    for _ in {1..20}; do pg_isready -h localhost -q 2>/dev/null && break; sleep 0.5; done
  fi
fi
pg_isready -h localhost -q 2>/dev/null || fail "PostgreSQL is not running on localhost:5432."
ok "PostgreSQL is running"

if [[ ! -f "$BACKEND/.env" ]]; then
  SECRET="$(openssl rand -hex 32)"
  sed "s|^APP_SECRET_KEY=.*|APP_SECRET_KEY=$SECRET|" "$BACKEND/.env.example" > "$BACKEND/.env"
  chmod 600 "$BACKEND/.env"
  ok "Created backend/.env with a new APP_SECRET_KEY"
else
  ok "backend/.env already exists (left unchanged)"
fi

if psql -h localhost -d postgres -Atc "SELECT 1 FROM pg_database WHERE datname='$DB_NAME'" 2>/dev/null | grep -q 1; then
  ok "Database $DB_NAME exists"
elif createdb -h localhost "$DB_NAME" 2>/dev/null; then
  ok "Created database $DB_NAME"
else
  echo "  Could not create the database automatically."
  echo "  If your PostgreSQL needs a user and password, set DATABASE_URL in backend/.env"
  echo "  and create the database yourself, then run this script again."
  fail "Database setup incomplete."
fi

# --- 3. Backend ------------------------------------------------------------------

bold "3/5  Backend (Python)"
cd "$BACKEND"
uv sync --quiet
ok "Python packages installed"

# --- 4. Tables -------------------------------------------------------------------

bold "4/5  Database tables"
uv run --quiet alembic upgrade head
ok "Database is up to date"

# --- 5. Frontend -----------------------------------------------------------------

bold "5/5  Frontend (Node.js)"
cd "$FRONTEND"
npm install --silent --no-fund --no-audit
ok "Node packages installed"

if [[ "${AIS_INSTALLER:-}" == "1" ]]; then
  bold "Setup complete."
else
  bold "Setup complete. Start the studio with:"
  echo "  ./scripts/start.sh    (in the background, like the Mac app)"
  echo "  ./scripts/dev.sh      (for development, with live reload)"
  echo
fi
