"""Hugging Face checks for the user's token: whose token it is, and whether it can download
the gated Krea 2 repositories (the license must be accepted once on huggingface.co).

Endpoints, as huggingface_hub 2.4.0 calls them (hf_api.py `whoami`, `auth_check`;
utils/_http.py `hf_raise_for_status`):

GET /api/whoami-v2                     Bearer token -> 200 {name, ...}; 401 invalid token.
                                       Strictly rate-limited, so results are cached.
GET /api/models/{repo_id}/auth-check   200 = the token can read the repository;
                                       403 with header X-Error-Code: GatedRepo = gated and
                                       not accepted yet (or the request is still pending);
                                       401 = invalid token; 404 = not found.

Hugging Face has no API to accept a license for a user: it is the user's own agreement, made
with one click on the repository page. The studio links there and checks the result.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Literal

import httpx

from app.config import get_settings
from app.lora.presets import ARCHITECTURES

Access = Literal["granted", "needs_license", "invalid_token", "not_found", "unknown"]

_CACHE: dict[str, tuple[float, Any]] = {}
_CACHE_SECONDS = 600


def gated_repos() -> list[dict[str, str]]:
    """The gated repositories LoRA training downloads, one per base model."""
    out: list[dict[str, str]] = []
    for arch in ARCHITECTURES.values():
        for key, variant in arch["variants"].items():
            if variant.get("gated"):
                repo = str(variant["name_or_path"])
                out.append(
                    {
                        "repo": repo,
                        "variant": key,
                        "name": str(variant["name"]),
                        "url": f"https://huggingface.co/{repo}",
                    }
                )
    return out


class HubClient:
    # Tests replace this with a mock transport.
    transport: httpx.AsyncBaseTransport | None = None

    def __init__(self, token: str):
        self.token = token

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=get_settings().huggingface_url.rstrip("/"),
            headers={"Authorization": f"Bearer {self.token}"},
            timeout=httpx.Timeout(15, connect=8),
            transport=self.transport,
        )

    async def whoami(self) -> tuple[Literal["valid", "invalid", "unknown"], str | None]:
        try:
            async with self._client() as client:
                r = await client.get("/api/whoami-v2")
        except httpx.HTTPError:
            return "unknown", None
        if r.status_code == 401:
            return "invalid", None
        if r.status_code != 200:
            return "unknown", None
        try:
            name = r.json().get("name")
        except ValueError:
            name = None
        return "valid", str(name) if name else None

    async def access(self, repo: str) -> Access:
        try:
            async with self._client() as client:
                r = await client.get(f"/api/models/{repo}/auth-check")
        except httpx.HTTPError:
            return "unknown"
        if r.status_code == 200:
            return "granted"
        if r.headers.get("X-Error-Code") == "GatedRepo":
            return "needs_license"
        if r.status_code == 401:
            return "invalid_token"
        if r.status_code == 404:
            return "not_found"
        return "unknown"


def _key(*parts: str) -> str:
    return hashlib.sha256("\x00".join(parts).encode()).hexdigest()


def _cached(key: str) -> Any | None:
    item = _CACHE.get(key)
    if item and time.monotonic() - item[0] < _CACHE_SECONDS:
        return item[1]
    _CACHE.pop(key, None)
    return None


def clear_cache() -> None:
    _CACHE.clear()


async def repo_access(token: str, repo: str, refresh: bool = False) -> Access:
    """Access to one repository. Only a definite answer is cached ("unknown" is retried)."""
    key = _key("access", token, repo)
    if not refresh and (hit := _cached(key)) is not None:
        return hit
    result = await HubClient(token).access(repo)
    if result != "unknown":
        _CACHE[key] = (time.monotonic(), result)
    return result


async def status(token: str, refresh: bool = False) -> dict[str, Any]:
    """What the integration card shows: the token's account and each gated repository."""
    key = _key("whoami", token)
    who = None if refresh else _cached(key)
    if who is None:
        who = await HubClient(token).whoami()
        if who[0] != "unknown":
            _CACHE[key] = (time.monotonic(), who)
    token_state, account = who
    repos = []
    for item in gated_repos():
        access: Access = (
            "invalid_token"
            if token_state == "invalid"
            else await repo_access(token, item["repo"], refresh)
        )
        repos.append({**item, "access": access})
    ready = token_state != "invalid" and all(r["access"] == "granted" for r in repos)
    return {"token": token_state, "account": account, "repos": repos, "ready": ready}
