"""The studio agent that runs inside GPU containers, and how it is started there.

The agent source is sent with the workload (gzip + base64 in an environment variable), so the
container needs nothing but Python 3 and the agent always matches this version of the app.
"""

from __future__ import annotations

import base64
import gzip
import secrets
from pathlib import Path

# Training containers (ai-toolkit image).
AGENT_PORT = 8765
# ComfyUI containers: RunPod's ComfyUI image uses 8188 (ComfyUI), 8080, 8888 and 22 itself;
# only this port is exposed.
COMFY_AGENT_PORT = 8189
WORKSPACE = "/workspace"
SOURCE = Path(__file__).with_name("studio_agent.py")

# Training: the container's start command (bash -c) runs the agent from the variable.
BOOT = (
    'python3 -c "import base64,gzip,os;'
    "exec(gzip.decompress(base64.b64decode(os.environ['STUDIO_AGENT_CODE'])))\""
)

# ComfyUI pods (RunPod's runpod/comfyui image, which already holds ComfyUI and PyTorch): the
# start command (/bin/bash -c) writes the agent to a file, starts it in the background and
# hands over to the image's own /start.sh, which copies ComfyUI into /workspace on the first
# start and runs it on port 8188. Nothing is installed. ComfyUI is told to listen on
# 127.0.0.1 only (its args file); the image's Jupyter and FileBrowser get random passwords;
# none of their ports is exposed, and the agent's secrets are not passed on to ComfyUI.
# Custom ComfyUI nodes (comfy/custom_nodes.py: pinned files, checked by SHA-256 on the Mac),
# a gzipped tar in STUDIO_NODES_CODE, unpacked into custom_nodes before ComfyUI starts: into
# the image's ComfyUI (the RunPod start script copies it onto the volume on the first start;
# Modal runs it from there) and into the volume's copy when it already exists (a pod started
# again). Without the variable nothing changes.
_NODES = r"""if [ -n "${STUDIO_NODES_CODE:-}" ]; then
  for D in "${STUDIO_COMFY_BAKED:-/opt/comfyui-baked}" "$W/runpod-slim/ComfyUI"; do
    if [ -d "$D" ]; then
      mkdir -p "$D/custom_nodes"
      printf '%s' "$STUDIO_NODES_CODE" | base64 -d | tar -xzf - -C "$D/custom_nodes" || true
    fi
  done
fi
"""

COMFY_BOOT = r"""set -e
W="${STUDIO_WORKSPACE:-/workspace}"
AGENT="$(mktemp -t studio-agent.XXXXXX)"
STUDIO_BOOT_LOG="$(mktemp -t studio-boot.XXXXXX)"
export STUDIO_BOOT_LOG
printf '%s' "$STUDIO_AGENT_CODE" | base64 -d | gzip -dc > "$AGENT"
mkdir -p "$W/runpod-slim"
ARGS="$W/runpod-slim/comfyui_args.txt"
[ -f "$ARGS" ] || echo '# Add your custom ComfyUI arguments here (one per line)' > "$ARGS"
grep -qxF -- '--listen 127.0.0.1' "$ARGS" || echo '--listen 127.0.0.1' >> "$ARGS"
JUPYTER_PASSWORD="$(od -An -tx1 -N16 /dev/urandom | tr -d ' \n')"
FILEBROWSER_PASSWORD="$(od -An -tx1 -N16 /dev/urandom | tr -d ' \n')"
export JUPYTER_PASSWORD FILEBROWSER_PASSWORD
# The agent comes back if it ever stops (its jobs are then reported as gone).
nohup bash -c 'while :; do python3 "$0"; sleep 3; done' "$AGENT" > "$AGENT.log" 2>&1 &
{NODES}unset STUDIO_AGENT_CODE STUDIO_AGENT_TOKEN HF_TOKEN STUDIO_NODES_CODE
exec > >(tee -a "$STUDIO_BOOT_LOG") 2>&1
exec "${STUDIO_START_SCRIPT:-/start.sh}"
""".replace("{NODES}", _NODES)

# Modal sandboxes (the same image, its ENTRYPOINT cleared): the image's /start.sh is not used
# there. It starts SSH, FileBrowser (its database on the Volume) and Jupyter, copies ComfyUI
# onto the Volume and builds a venv there, under `set -e` — any step the sandbox does not
# allow ends the container before ComfyUI starts. Here ComfyUI runs straight from the image
# (/opt/comfyui-baked, with the image's own Python packages); only its models folder lives on
# the Volume, in the same place as on RunPod ($W/runpod-slim/ComfyUI/models), so downloads
# and LoRAs are kept for the next start. Nothing is installed or downloaded at start.
COMFY_BOOT_MODAL = r"""set -e
cd /
W="${STUDIO_WORKSPACE:-/workspace}"
AGENT="$(mktemp -t studio-agent.XXXXXX)"
STUDIO_BOOT_LOG="$(mktemp -t studio-boot.XXXXXX)"
export STUDIO_BOOT_LOG
printf '%s' "$STUDIO_AGENT_CODE" | base64 -d | gzip -dc > "$AGENT"
# The agent comes back if it ever stops (its jobs are then reported as gone).
nohup bash -c 'while :; do python3 "$0"; sleep 3; done' "$AGENT" > "$AGENT.log" 2>&1 &
{NODES}unset STUDIO_AGENT_CODE STUDIO_AGENT_TOKEN HF_TOKEN STUDIO_NODES_CODE
exec > >(tee -a "$STUDIO_BOOT_LOG") 2>&1
B="${STUDIO_COMFY_BAKED:-/opt/comfyui-baked}"
C="$W/runpod-slim/ComfyUI"
echo "Preparing ComfyUI: models folder on the Modal Volume ($C/models)"
mkdir -p "$C/models"
# The image's model folders (placeholders); files already on the Volume are kept.
cp -rn "$B/models/." "$C/models/" 2>/dev/null || true
rm -rf "$B/models"
ln -s "$C/models" "$B/models"
cd "$B"
PY="$(command -v python3.12 || command -v python3)"
echo "Starting ComfyUI on 127.0.0.1:${STUDIO_COMFY_PORT:-8188}"
exec "$PY" main.py --listen 127.0.0.1 --port "${STUDIO_COMFY_PORT:-8188}"
""".replace("{NODES}", _NODES)


def packed_source() -> str:
    return base64.b64encode(gzip.compress(SOURCE.read_bytes(), 9, mtime=0)).decode()


def new_token() -> str:
    return secrets.token_urlsafe(32)


def agent_env(
    token: str, extra: dict[str, str] | None = None, port: int = AGENT_PORT
) -> dict[str, str]:
    env = {
        "STUDIO_AGENT_CODE": packed_source(),
        "STUDIO_AGENT_TOKEN": token,
        "STUDIO_AGENT_PORT": str(port),
        "STUDIO_WORKSPACE": WORKSPACE,
    }
    env.update(extra or {})
    return env
