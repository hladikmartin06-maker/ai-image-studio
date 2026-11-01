"""A fake WaveSpeed + Atlas Cloud API (httpx.MockTransport) with the documented shapes.

WaveSpeed: POST /api/v3/{model}, GET /api/v3/predictions/{id}/result,
           POST /api/v3/media/upload/binary, GET /api/v3/balance
Atlas:     POST /api/v1/model/generateImage, GET /api/v1/model/prediction/{id},
           POST /api/v1/model/uploadMedia
Images are served from https://cdn.fake/<name>.png.
"""

from __future__ import annotations

import io
import itertools
import json
from dataclasses import dataclass, field
from typing import Any

import httpx
from PIL import Image as PILImage

VALID_KEY = "test-key-1234567890"


def png_bytes(width: int = 64, height: int = 48, color=(120, 80, 200)) -> bytes:
    out = io.BytesIO()
    PILImage.new("RGB", (width, height), color).save(out, format="PNG")
    return out.getvalue()


@dataclass
class FakeProviders:
    submits: list[dict[str, Any]] = field(default_factory=list)
    uploads: list[str] = field(default_factory=list)
    polls: int = 0
    predictions: dict[str, dict[str, Any]] = field(default_factory=dict)
    # Behaviour switches for error tests.
    submit_status: int = 200
    submit_body: dict[str, Any] | None = None
    fail_message: str | None = None
    polls_until_done: int = 1
    nsfw: bool = False
    _ids: itertools.count = field(default_factory=lambda: itertools.count(1))

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def _auth_ok(self, request: httpx.Request) -> bool:
        return request.headers.get("authorization") == f"Bearer {VALID_KEY}"

    def _new_prediction(self, provider: str, model: str, body: dict[str, Any]) -> str:
        pid = f"{provider}-{next(self._ids)}"
        count = int(body.get("n") or 1)
        self.predictions[pid] = {"model": model, "count": count, "polls": 0}
        self.submits.append({"provider": provider, "model": model, "body": body})
        return pid

    def _result(self, pid: str) -> dict[str, Any]:
        self.polls += 1
        pred = self.predictions[pid]
        pred["polls"] += 1
        if pred["polls"] < self.polls_until_done:
            return {"id": pid, "status": "processing", "outputs": []}
        if self.fail_message is not None:
            return {"id": pid, "status": "failed", "outputs": [], "error": self.fail_message}
        outputs = [f"https://cdn.fake/{pid}-{i}.png" for i in range(pred["count"])]
        return {
            "id": pid,
            "status": "completed",
            "outputs": outputs,
            "has_nsfw_contents": [self.nsfw] * len(outputs),
        }

    def handle(self, request: httpx.Request) -> httpx.Response:
        url = request.url
        if url.host == "cdn.fake":
            return httpx.Response(
                200, content=png_bytes(96, 72), headers={"content-type": "image/png"}
            )
        if not self._auth_ok(request):
            return httpx.Response(401, json={"code": 401, "message": "Unauthorized"})
        path = url.path

        # --- WaveSpeed
        if url.host == "api.wavespeed.fake":
            if path == "/api/v3/balance":
                return httpx.Response(200, json={"code": 200, "data": {"balance": 12.5}})
            if path == "/api/v3/media/upload/binary":
                name = f"https://cdn.fake/upload-{len(self.uploads) + 1}.png"
                self.uploads.append(name)
                return httpx.Response(200, json={"code": 200, "data": {"download_url": name}})
            if path.startswith("/api/v3/predictions/") and path.endswith("/result"):
                pid = path.split("/")[4]
                return httpx.Response(200, json={"code": 200, "data": self._result(pid)})
            if request.method == "POST" and path.startswith("/api/v3/"):
                if self.submit_status != 200:
                    return httpx.Response(
                        self.submit_status, json=self.submit_body or {"message": "error"}
                    )
                body = json.loads(request.content)
                pid = self._new_prediction("wavespeed", path[len("/api/v3/") :], body)
                return httpx.Response(
                    200,
                    json={
                        "code": 200,
                        "message": "success",
                        "data": {"id": pid, "status": "created"},
                    },
                )

        # --- Atlas
        if url.host == "api.atlas.fake":
            if path == "/api/v1/model/uploadMedia":
                name = f"https://cdn.fake/atlas-upload-{len(self.uploads) + 1}.png"
                self.uploads.append(name)
                return httpx.Response(200, json={"code": 200, "data": {"download_url": name}})
            if path.startswith("/api/v1/model/prediction/"):
                pid = path.rsplit("/", 1)[-1]
                if pid not in self.predictions:
                    return httpx.Response(404, json={"code": 404, "message": "not found"})
                return httpx.Response(200, json={"code": 200, "data": self._result(pid)})
            if path == "/api/v1/model/generateImage":
                if self.submit_status != 200:
                    return httpx.Response(
                        self.submit_status, json=self.submit_body or {"message": "error"}
                    )
                body = json.loads(request.content)
                pid = self._new_prediction("atlas", body["model"], body)
                return httpx.Response(200, json={"code": 200, "data": {"id": pid}})
        return httpx.Response(404, json={"message": "unknown route"})


@dataclass
class FakeHub:
    """Hugging Face Hub: GET /api/whoami-v2 and GET /api/models/{repo}/auth-check, with the
    status codes and X-Error-Code header huggingface_hub reads."""

    invalid_tokens: set[str] = field(default_factory=set)
    # Tokens that have not accepted the license of these repositories yet.
    no_license: dict[str, set[str]] = field(default_factory=dict)
    calls: list[str] = field(default_factory=list)

    def transport(self) -> httpx.MockTransport:
        def handler(request: httpx.Request) -> httpx.Response:
            token = request.headers.get("authorization", "").removeprefix("Bearer ")
            path = request.url.path
            self.calls.append(path)
            if token in self.invalid_tokens:
                return httpx.Response(
                    401,
                    headers={"X-Error-Message": "Invalid credentials in Authorization header"},
                    json={"error": "Invalid credentials in Authorization header"},
                )
            if path == "/api/whoami-v2":
                return httpx.Response(200, json={"type": "user", "name": "studio-user"})
            if path.startswith("/api/models/") and path.endswith("/auth-check"):
                repo = path.removeprefix("/api/models/").removesuffix("/auth-check")
                if repo in self.no_license.get(token, set()):
                    return httpx.Response(
                        403,
                        headers={
                            "X-Error-Code": "GatedRepo",
                            "X-Error-Message": f"Access to model {repo} is restricted. You "
                            "must have access to it and be authenticated to access it.",
                        },
                        json={"error": "Access to model is restricted."},
                    )
                return httpx.Response(200, json={})
            return httpx.Response(404, headers={"X-Error-Code": "RepoNotFound"})

        return httpx.MockTransport(handler)
