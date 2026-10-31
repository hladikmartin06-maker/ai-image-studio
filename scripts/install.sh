#!/usr/bin/env bash
# Installs AI Image Studio the first time, updates it after that, and starts it.
#
# Run it with the single command from README.md:
#   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/hladikmartin06-maker/ai-image-studio/main/scripts/install.sh)"
#
# On a Mac it installs Homebrew if needed, downloads the studio into ~/ai-image-studio,
# installs Node.js, uv and PostgreSQL 16, prepares the database, builds the app,
# starts it in the background and installs "AI Image Studio.app", which opens it in
# its own Chrome window. Running it again updates everything and is always safe.

set -euo pipefail

# Where the code comes from. The studio is the whole repository.
REPO_URL="${AIS_REPO_URL:-https://github.com/hladikmartin06-maker/ai-image-studio.git}"
BRANCH="${AIS_BRANCH:-main}"
INSTALL_DIR="${AIS_DIR:-$HOME/ai-image-studio}"
APP_DIR="$INSTALL_DIR"

bold() { printf "\n\033[1m%s\033[0m\n" "$1"; }
ok() { printf "  \033[32m✓\033[0m %s\n" "$1"; }
fail() { printf "\n  \033[31m✗ %s\033[0m\n\n" "$1"; exit 1; }

bold "AI Image Studio"

if [[ "$(uname)" == "Darwin" ]]; then
  # --- Homebrew --------------------------------------------------------------
  BREW=""
  for candidate in /opt/homebrew/bin/brew /usr/local/bin/brew; do
    if [[ -x "$candidate" ]]; then
      BREW="$candidate"
      break
    fi
  done
  if [[ -z "$BREW" ]]; then
    bold "Installing Homebrew (it asks for your Mac password)"
    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
    for candidate in /opt/homebrew/bin/brew /usr/local/bin/brew; do
      if [[ -x "$candidate" ]]; then
        BREW="$candidate"
        break
      fi
    done
  fi
  if [[ -z "$BREW" ]]; then
    fail "Homebrew did not install. Install it from https://brew.sh, then run this command again."
  fi

  # Make brew available now and in every new Terminal window.
  eval "$("$BREW" shellenv)"
  SHELLENV_LINE="eval \"\$($BREW shellenv)\""
  if ! grep -qsF "$SHELLENV_LINE" "$HOME/.zprofile"; then
    echo "$SHELLENV_LINE" >> "$HOME/.zprofile"
  fi
  ok "Homebrew is ready"

  # git comes with Apple's Command Line Tools, which Homebrew installs.
  if ! xcode-select -p >/dev/null 2>&1; then
    xcode-select --install >/dev/null 2>&1 || true
    fail "Finish installing the Command Line Tools in the window that opened, then run this command again."
  fi
fi

# --- Download or update ---------------------------------------------------------

command -v git >/dev/null 2>&1 || fail "git is not installed."

if [[ -d "$INSTALL_DIR/.git" ]]; then
  if git -C "$INSTALL_DIR" pull --ff-only --quiet; then
    ok "Updated $INSTALL_DIR"
  else
    echo "  Could not update (files were changed by hand?). Starting the version you have."
  fi
elif [[ -e "$INSTALL_DIR" ]]; then
  fail "$INSTALL_DIR already exists but is not a copy of the studio. Rename that folder and run this command again."
else
  bold "Downloading the studio into $INSTALL_DIR"
  git clone --quiet --depth 1 --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
  ok "Downloaded"
fi

[[ -x "$APP_DIR/scripts/setup.sh" ]] || fail "The download in $APP_DIR is incomplete. Check the branch $BRANCH."
cd "$APP_DIR"

# --- Tools, database and packages -----------------------------------------------

AIS_INSTALLER=1 ./scripts/setup.sh

# --- Start in the background ------------------------------------------------------

# Restart so an update is built and used.
./scripts/stop.sh >/dev/null 2>&1 || true
bold "Starting AI Image Studio"
./scripts/start.sh

# --- The Mac app ------------------------------------------------------------------

if [[ "$(uname)" == "Darwin" ]]; then
  APP_PATH="$(./scripts/mac-app.sh)"
  ok "Installed the app: $APP_PATH"
  bold "Done. AI Image Studio is opening in its own window."
  echo "  Next time, open it from Launchpad or the Applications folder."
  echo "  Tip: drag it to the Dock to keep it there."
  echo "  It keeps running in the background until you restart the Mac."
  echo "  To stop it now: $APP_DIR/scripts/stop.sh"
  echo "  Logs: ~/Library/Logs/AI Image Studio"
  echo
  open "$APP_PATH"
else
  bold "Done. Open http://localhost:3000"
  echo "  To stop it: $APP_DIR/scripts/stop.sh"
  echo
fi
