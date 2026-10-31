# Body of the "AI Image Studio.app" launcher. scripts/mac-app.sh puts a
# shebang and APP_DIR=<path to ai-image-studio> above this text.
#
# Clicking the app starts the studio in the background when it is not running,
# then opens it in its own Chrome window (or the default browser).

URL="http://localhost:3000"
LOG_DIR="$HOME/Library/Logs/AI Image Studio"
mkdir -p "$LOG_DIR"

notify() {
  osascript -e "display notification \"$1\" with title \"AI Image Studio\"" >/dev/null 2>&1 || true
}

running() {
  curl -fs -m 2 "$URL/api/health" 2>/dev/null | grep -q '"app":"ai-image-studio"'
}

if ! running; then
  notify "Starting… this takes a few seconds."
  if ! OUTPUT="$("$APP_DIR/scripts/start.sh" 2>&1)"; then
    echo "$OUTPUT" >> "$LOG_DIR/launcher.log"
    REASON="$(echo "$OUTPUT" | tail -1 | sed 's/"/\\"/g')"
    osascript -e "display dialog \"AI Image Studio could not start.\n\n$REASON\n\nThe details are in your Library/Logs/AI Image Studio folder.\" with title \"AI Image Studio\" buttons {\"OK\"} default button \"OK\" with icon caution" >/dev/null 2>&1 || true
    exit 1
  fi
  echo "$OUTPUT" >> "$LOG_DIR/launcher.log"
fi

if open -Ra "Google Chrome" >/dev/null 2>&1; then
  # --app opens a window without tabs or an address bar, like an installed app.
  open -na "Google Chrome" --args --app="$URL"
else
  open "$URL"
fi
