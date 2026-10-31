"""The studio agent that runs inside GPU containers, and how it is started there.

The agent source is sent with the workload (compressed in an environment variable), so the
container needs nothing but Python 3 and the agent always matches this version of the app.
"""

from __future__ import annotations

import base64
import secrets
import zlib
from pathlib import Path

AGENT_PORT = 8765
WORKSPACE = "/workspace"
SOURCE = Path(__file__).with_name("studio_agent.py")

# The container's start command (bash -c): run the agent from the environment variable.
BOOT = (
    'python3 -c "import base64,os,zlib;'
    "exec(zlib.decompress(base64.b64decode(os.environ['STUDIO_AGENT_CODE'])))\""
)


def packed_source() -> str:
    return base64.b64encode(zlib.compress(SOURCE.read_bytes(), 9)).decode()


def new_token() -> str:
    return secrets.token_urlsafe(32)


def agent_env(token: str, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {
        "STUDIO_AGENT_CODE": packed_source(),
        "STUDIO_AGENT_TOKEN": token,
        "STUDIO_AGENT_PORT": str(AGENT_PORT),
        "STUDIO_WORKSPACE": WORKSPACE,
    }
    env.update(extra or {})
    return env
