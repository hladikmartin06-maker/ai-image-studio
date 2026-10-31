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
unset STUDIO_AGENT_CODE STUDIO_AGENT_TOKEN HF_TOKEN
exec > >(tee -a "$STUDIO_BOOT_LOG") 2>&1
exec "${STUDIO_START_SCRIPT:-/start.sh}"
"""


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
