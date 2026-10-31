#!/usr/bin/env bash
# Creates "AI Image Studio.app" in the Applications folder (macOS). Clicking it
# starts the studio in the background and opens it in its own Chrome window.
# install.sh runs this; running it again replaces the app.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
APP_NAME="AI Image Studio"

if [[ "$(uname)" != "Darwin" && "${AIS_FORCE_BUNDLE:-}" != "1" ]]; then
  echo "The app is only for macOS. Start the studio with scripts/start.sh."
  exit 0
fi

# /Applications when it is writable (normal for an admin account), otherwise ~/Applications.
if [[ -n "${AIS_APP_DEST:-}" ]]; then
  DEST="$AIS_APP_DEST"
elif [[ -w /Applications ]]; then
  DEST="/Applications"
else
  DEST="$HOME/Applications"
fi
mkdir -p "$DEST"
APP="$DEST/$APP_NAME.app"

rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cat > "$APP/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>
  <string>AI Image Studio</string>
  <key>CFBundleDisplayName</key>
  <string>AI Image Studio</string>
  <key>CFBundleIdentifier</key>
  <string>local.ai-image-studio.launcher</string>
  <key>CFBundleExecutable</key>
  <string>AI Image Studio</string>
  <key>CFBundleIconFile</key>
  <string>AppIcon</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>1.0</string>
  <key>CFBundleVersion</key>
  <string>1</string>
  <key>LSMinimumSystemVersion</key>
  <string>11.0</string>
  <key>NSHighResolutionCapable</key>
  <true/>
</dict>
</plist>
PLIST

LAUNCHER="$APP/Contents/MacOS/$APP_NAME"
{
  echo '#!/bin/bash'
  printf 'APP_DIR=%q\n' "$ROOT"
  cat "$ROOT/scripts/assets/launcher.sh"
} > "$LAUNCHER"
chmod +x "$LAUNCHER"

# The icon: macOS builds .icns files with sips and iconutil.
if command -v sips >/dev/null 2>&1 && command -v iconutil >/dev/null 2>&1; then
  WORK="$(mktemp -d)"
  ICONSET="$WORK/AppIcon.iconset"
  mkdir -p "$ICONSET"
  for size in 16 32 128 256 512; do
    sips -z "$size" "$size" "$ROOT/scripts/assets/AppIcon.png" --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
    double=$((size * 2))
    sips -z "$double" "$double" "$ROOT/scripts/assets/AppIcon.png" --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
  done
  iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/AppIcon.icns" || echo "  (The app icon could not be created; the app still works.)"
  rm -rf "$WORK"
fi
touch "$APP"

echo "$APP"
