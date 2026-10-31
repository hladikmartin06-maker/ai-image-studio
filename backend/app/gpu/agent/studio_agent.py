"""AI Image Studio agent: a small HTTP service that runs inside the user's GPU container
(RunPod pod or Modal sandbox) so the studio on the Mac can move files, run training, download
models and reach ComfyUI there. One file, Python standard library only (3.8+), nothing to
install. Started by the studio either from a file the container's start command writes
(ComfyUI pods, see gpu/agent COMFY_BOOT) or with:

    python3 -c "import base64,gzip,os;exec(gzip.decompress(
        base64.b64decode(os.environ['STUDIO_AGENT_CODE'])))"   (one line, training)

Environment:
  STUDIO_AGENT_TOKEN  required; every request needs "Authorization: Bearer <token>"
  STUDIO_AGENT_PORT   default 8765
  STUDIO_WORKSPACE    default /workspace (all files live below it)
  STUDIO_AITK_DIR     ai-toolkit folder for training (default /app/ai-toolkit)
  STUDIO_COMFY_DIR    ComfyUI folder (default $STUDIO_WORKSPACE/runpod-slim/ComfyUI, where
                      RunPod's ComfyUI image keeps it)
  STUDIO_COMFY_PORT   ComfyUI's local port (default 8188; never exposed)
  STUDIO_BOOT_LOG     the container's start log (ComfyUI's output), served on /boot-log
  HF_TOKEN            optional, for gated Hugging Face downloads (sent to huggingface.co only)

API (JSON unless noted):
  GET    /health                     agent, GPU (nvidia-smi), CPU load, RAM, disk, jobs
  GET    /boot-log?offset=<n>        plain text of the container's start log from that offset
  GET    /files?dir=<rel>            files below a folder (recursive)
  PUT    /files/<rel>?offset=<n>     write a chunk (offset 0 starts the file)
  GET    /files/<rel>                download; honours "Range: bytes=a-b"
  DELETE /files/<rel>
  POST   /jobs                       {id, kind: train|models|download, args}
                                     models: {models: [{url, folder, filename}]}
                                     download: {url, folder, filename}
                                     (downloads: https://huggingface.co only, into
                                     ComfyUI/models/<folder>/, as .part until complete)
  GET    /jobs/<id>                  status, exit code, progress (step/total, files), loss
  GET    /jobs/<id>/logs?offset=<n>  plain text from that byte offset
  POST   /jobs/<id>/cancel
  *      /comfy/<path>               proxied to the local ComfyUI (127.0.0.1:8188)
"""

import contextlib
import hmac
import http.client
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

VERSION = "2"
TOKEN = os.environ.get("STUDIO_AGENT_TOKEN", "")
PORT = int(os.environ.get("STUDIO_AGENT_PORT", "8765"))
WORKSPACE = os.path.realpath(os.environ.get("STUDIO_WORKSPACE", "/workspace"))
AITK_DIR = os.environ.get("STUDIO_AITK_DIR", "/app/ai-toolkit")
COMFY_DIR = os.environ.get("STUDIO_COMFY_DIR", os.path.join(WORKSPACE, "runpod-slim", "ComfyUI"))
COMFY_PORT = int(os.environ.get("STUDIO_COMFY_PORT", "8188"))
BOOT_LOG = os.environ.get("STUDIO_BOOT_LOG", "")
CHUNK = 1024 * 1024

STEP_RE = re.compile(r"(\d+)/(\d+) \[")
LOSS_RE = re.compile(r"loss[:=]\s*([0-9]*\.?[0-9]+(?:e[-+]?\d+)?)", re.IGNORECASE)

JOBS = {}
JOBS_LOCK = threading.Lock()


def safe_path(rel):
    """An absolute path inside the workspace, or None."""
    rel = urllib.parse.unquote(rel or "").lstrip("/")
    path = os.path.realpath(os.path.join(WORKSPACE, rel))
    if path != WORKSPACE and not path.startswith(WORKSPACE + os.sep):
        return None
    return path


def gpu_info():
    fields = "name,utilization.gpu,memory.used,memory.total,temperature.gpu,driver_version"
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue

        def num(v):
            try:
                return float(v)
            except ValueError:
                return None

        gpus.append(
            {
                "name": parts[0],
                "utilization": num(parts[1]),
                "memory_used_mb": num(parts[2]),
                "memory_total_mb": num(parts[3]),
                "temperature": num(parts[4]),
                "driver": parts[5],
            }
        )
    return gpus


def memory_info():
    info = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                key, _, value = line.partition(":")
                if key in ("MemTotal", "MemAvailable"):
                    info[key] = int(value.strip().split()[0]) * 1024
    except OSError:
        pass
    return {"ram_total": info.get("MemTotal"), "ram_available": info.get("MemAvailable")}


class Job:
    def __init__(self, job_id, kind, command, cwd, env):
        self.id = job_id
        self.kind = kind
        self.log_path = os.path.join(WORKSPACE, ".studio", "logs", job_id + ".log")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.status = "running"
        self.exit_code = None
        self.started_at = time.time()
        self.ended_at = None
        self.step = None
        self.total = None
        self.loss = []
        self._parsed = 0
        self.cancelled = False
        with open(self.log_path, "ab") as log:
            self.proc = subprocess.Popen(
                command,
                cwd=cwd,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        threading.Thread(target=self._wait, daemon=True).start()

    def _wait(self):
        code = self.proc.wait()
        self.exit_code = code
        self.ended_at = time.time()
        self.parse()
        if self.cancelled:
            self.status = "cancelled"
        else:
            self.status = "completed" if code == 0 else "failed"

    def parse(self):
        """Reads new log output: tqdm "step/total [" and "loss: x" (ai-toolkit)."""
        try:
            with open(self.log_path, "rb") as f:
                f.seek(self._parsed)
                data = f.read()
        except OSError:
            return
        self._parsed += len(data)
        text = data.decode("utf-8", "replace").replace("\r", "\n")
        for line in text.splitlines():
            m = STEP_RE.search(line)
            if not m:
                continue
            self.step, self.total = int(m.group(1)), int(m.group(2))
            loss = LOSS_RE.search(line)
            if loss and self.kind == "train":
                value = float(loss.group(1))
                if not self.loss or self.loss[-1][0] != self.step:
                    self.loss.append([self.step, value])
        if len(self.loss) > 2000:
            self.loss = self.loss[-2000:]

    def cancel(self):
        if self.status != "running":
            return
        self.cancelled = True
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except OSError:
            return
        for _ in range(30):
            if self.proc.poll() is not None:
                return
            time.sleep(0.5)
        with contextlib.suppress(OSError):
            os.killpg(self.proc.pid, signal.SIGKILL)

    def json(self):
        if self.status == "running":
            self.parse()
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "exit_code": self.exit_code,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "step": self.step,
            "total": self.total,
            "loss": self.loss[-500:],
            "log_size": os.path.getsize(self.log_path) if os.path.exists(self.log_path) else 0,
        }


# --- model downloads (Hugging Face only) -----------------------------------------------------

HF_HOST = "huggingface.co"
# Hugging Face serves file contents from its own CDN hosts (e.g. cas-bridge.xethub.hf.co,
# cdn-lfs.huggingface.co); redirects anywhere else are refused.
HF_REDIRECT_SUFFIXES = (".huggingface.co", ".hf.co")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
FOLDER_RE = re.compile(r"[a-z0-9_]{1,40}")
NAME_RE = re.compile(r"[\w.\- ]{1,200}")


class DownloadError(Exception):
    def __init__(self, code, message):
        Exception.__init__(self, message)
        self.code = code  # denied | not_found | no_space | not_allowed | failed | cancelled
        self.message = message


def hf_url(url):
    """Only https://huggingface.co/... is downloaded."""
    url = str(url or "")
    parts = urllib.parse.urlsplit(url)
    if (
        parts.scheme != "https"
        or (parts.hostname or "").lower() != HF_HOST
        or parts.username
        or parts.password
        or parts.port not in (None, 443)
    ):
        raise ValueError("only https://huggingface.co downloads are allowed")
    return url


def _hf_host(host):
    host = (host or "").lower()
    return host == HF_HOST or host.endswith(HF_REDIRECT_SUFFIXES)


class _HFRedirect(urllib.request.HTTPRedirectHandler):
    """Follows Hugging Face's redirects to its CDN only. The token goes to huggingface.co
    only (it is an unredirected header, re-added on same-host redirects)."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        parts = urllib.parse.urlsplit(newurl)
        if parts.scheme != "https" or not _hf_host(parts.hostname):
            raise DownloadError("not_allowed", "redirect outside Hugging Face refused")
        new = urllib.request.HTTPRedirectHandler.redirect_request(
            self, req, fp, code, msg, headers, newurl
        )
        if new is not None and HF_TOKEN and (parts.hostname or "").lower() == HF_HOST:
            new.add_unredirected_header("Authorization", "Bearer " + HF_TOKEN)
        return new


def _opener():
    return urllib.request.build_opener(_HFRedirect())


def model_targets(models):
    """[(url, absolute destination, name)] inside ComfyUI/models/<folder>/."""
    out = []
    for m in models:
        url = hf_url(m.get("url"))
        folder = str(m.get("folder", ""))
        name = str(m.get("filename", ""))
        if not FOLDER_RE.fullmatch(folder) or not NAME_RE.fullmatch(name) or name in (".", ".."):
            raise ValueError("bad model folder or file name")
        out.append((url, os.path.join(COMFY_DIR, "models", folder, name), name))
    if not out:
        raise ValueError("no files to download")
    return out


class DownloadJob:
    """Downloads files one after another in a thread: each is written as <name>.part
    (resumed after a dropped connection) and renamed when complete. Files that are already
    there are skipped. Progress per file is in json()["files"]."""

    RETRIES = 5
    RETRY_PAUSE = 2.0  # seconds, times the attempt

    def __init__(self, job_id, kind, targets):
        self.id = job_id
        self.kind = kind
        self.log_path = os.path.join(WORKSPACE, ".studio", "logs", job_id + ".log")
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self.status = "running"
        self.exit_code = None
        self.error = None
        self.started_at = time.time()
        self.ended_at = None
        self.cancelled = False
        self.targets = targets
        self.files = [
            {"name": name, "size": None, "done": 0, "status": "waiting"}
            for _url, _dest, name in targets
        ]
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _log(self, text):
        with open(self.log_path, "a") as log:
            log.write(text + "\n")

    def _run(self):
        try:
            if not os.path.isdir(COMFY_DIR):
                raise DownloadError("failed", "ComfyUI is not set up yet (" + COMFY_DIR + ")")
            total = len(self.targets)
            for index, (url, dest, name) in enumerate(self.targets):
                state = self.files[index]
                self._log("[studio] file %d/%d %s" % (index + 1, total, name))
                if os.path.isfile(dest) and os.path.getsize(dest) > 0:
                    size = os.path.getsize(dest)
                    state.update(size=size, done=size, status="skipped")
                    self._log("[studio] already there: " + name)
                    continue
                state["status"] = "downloading"
                self._fetch(url, dest, state)
                state["status"] = "done"
                self._log("[studio] saved " + dest)
            with contextlib.suppress(AttributeError, OSError):
                os.sync()  # flush to the volume (Modal commits it in the background)
            self.exit_code = 0
            self.status = "completed"
            self._log("[studio] models ready")
        except DownloadError as exc:
            self.error = {"code": exc.code, "message": exc.message}
            self.exit_code = 1
            self.status = "cancelled" if exc.code == "cancelled" else "failed"
            self._log("[studio] download failed: " + exc.message)
        except Exception as exc:  # noqa: BLE001 - reported to the studio, never raised
            self.error = {"code": "failed", "message": type(exc).__name__ + ": " + str(exc)}
            self.exit_code = 1
            self.status = "failed"
            self._log("[studio] download failed: " + self.error["message"])
        self.ended_at = time.time()

    def _fetch(self, url, dest, state):
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        part = dest + ".part"
        opener = _opener()
        attempt = 0
        while True:
            if self.cancelled:
                raise DownloadError("cancelled", "cancelled")
            have = os.path.getsize(part) if os.path.exists(part) else 0
            request = urllib.request.Request(
                url, headers={"User-Agent": "ai-image-studio/" + VERSION}
            )
            if HF_TOKEN and (urllib.parse.urlsplit(url).hostname or "").lower() == HF_HOST:
                request.add_unredirected_header("Authorization", "Bearer " + HF_TOKEN)
            if have:
                request.add_header("Range", "bytes=%d-" % have)
            try:
                response = opener.open(request, timeout=60)
            except urllib.error.HTTPError as exc:
                if exc.code == 416 and have:
                    whole = re.search(r"/(\d+)\s*$", exc.headers.get("Content-Range") or "")
                    if whole and int(whole.group(1)) == have:
                        state["size"] = have
                        break  # the .part file is already complete
                    os.remove(part)  # not this file: start again
                    attempt = self._retry(attempt, "HTTP 416")
                    continue
                if exc.code in (401, 403):
                    raise DownloadError(
                        "denied", "Hugging Face refused the download (HTTP %d)" % exc.code
                    ) from exc
                if exc.code == 404:
                    raise DownloadError("not_found", "file not found on Hugging Face") from exc
                attempt = self._retry(attempt, "HTTP %d" % exc.code)
                continue
            except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
                attempt = self._retry(attempt, type(exc).__name__ + ": " + str(exc)[:200])
                continue
            try:
                self._write(response, part, have, state)
            except DownloadError:
                raise
            except (OSError, http.client.HTTPException) as exc:
                if getattr(exc, "errno", None) == 28:
                    raise DownloadError("no_space", "No space left on device") from exc
                attempt = self._retry(attempt, type(exc).__name__ + ": " + str(exc)[:200])
                continue
            finally:
                response.close()
            if state.get("size") is None or os.path.getsize(part) >= state["size"]:
                break
            attempt = self._retry(attempt, "connection closed early")
        os.replace(part, dest)
        state["done"] = os.path.getsize(dest)
        state["size"] = state["done"]

    def _retry(self, attempt, reason):
        attempt += 1
        if attempt > self.RETRIES:
            raise DownloadError("failed", reason)
        self._log("[studio] retrying (%s)" % reason)
        until = time.time() + self.RETRY_PAUSE * attempt
        while time.time() < until:
            if self.cancelled:
                raise DownloadError("cancelled", "cancelled")
            time.sleep(0.2)
        return attempt

    def _write(self, response, part, have, state):
        if have and response.status != 206:
            have = 0  # the server sent the whole file again
        length = response.headers.get("Content-Length")
        rng = re.search(r"/(\d+)\s*$", response.headers.get("Content-Range") or "")
        if rng:
            state["size"] = int(rng.group(1))
        elif length is not None:
            state["size"] = have + int(length)
        state["done"] = have
        if state.get("size"):
            free = shutil.disk_usage(os.path.dirname(part)).free
            if state["size"] - have > free:
                raise DownloadError(
                    "no_space",
                    "No space left on device (needs %.1f GB, %.1f GB free)"
                    % ((state["size"] - have) / 1e9, free / 1e9),
                )
        with open(part, "ab" if have else "wb") as f:
            while True:
                if self.cancelled:
                    raise DownloadError("cancelled", "cancelled")
                data = response.read(CHUNK)
                if not data:
                    break
                f.write(data)
                state["done"] += len(data)

    def cancel(self):
        self.cancelled = True

    def json(self):
        done = sum(f["done"] for f in self.files)
        sizes = [f["size"] for f in self.files]
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "exit_code": self.exit_code,
            "error": self.error,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "step": None,
            "total": None,
            "loss": [],
            "files": [dict(f) for f in self.files],
            "bytes_done": done,
            "bytes_total": sum(sizes) if all(s is not None for s in sizes) else None,
            "log_size": os.path.getsize(self.log_path) if os.path.exists(self.log_path) else 0,
        }


def start_job(body):
    job_id = re.sub(r"[^a-zA-Z0-9_-]", "", str(body.get("id", "")))[:64]
    kind = body.get("kind")
    args = body.get("args") or {}
    if not job_id:
        raise ValueError("id is required")
    with JOBS_LOCK:
        existing = JOBS.get(job_id)
        if existing is not None and existing.status == "running":
            return existing
        if kind in ("models", "download"):
            models = args.get("models") if kind == "models" else [args]
            job = DownloadJob(job_id, kind, model_targets(models or []))
        elif kind == "train":
            config = safe_path(args.get("config", ""))
            if not config or not os.path.isfile(config):
                raise ValueError("config file not found")
            if not os.path.isfile(os.path.join(AITK_DIR, "run.py")):
                raise ValueError("ai-toolkit was not found in " + AITK_DIR)
            env = dict(os.environ)
            for key in ("STUDIO_AGENT_TOKEN", "STUDIO_AGENT_CODE"):
                env.pop(key, None)
            env["PYTHONUNBUFFERED"] = "1"
            command = [os.environ.get("STUDIO_PYTHON", "python3"), "run.py", config]
            job = Job(job_id, kind, command, AITK_DIR, env)
        else:
            raise ValueError("unknown job kind")
        JOBS[job_id] = job
    return job


# A start log can show generated passwords (e.g. the image's SSH root password).
SECRET_LINE = re.compile(r"(password[^:\n=]*[:=]\s*)\S+", re.IGNORECASE)


def read_boot_log(offset):
    if not BOOT_LOG:
        return b"", 0
    try:
        with open(BOOT_LOG, "rb") as f:
            f.seek(max(0, offset))
            data = f.read(512 * 1024)
    except OSError:
        return b"", max(0, offset)
    text = SECRET_LINE.sub(lambda m: m.group(1) + "********", data.decode("utf-8", "replace"))
    return text.encode(), max(0, offset) + len(data)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "StudioAgent/" + VERSION

    def log_message(self, fmt, *args):  # keep request lines (they hold no secrets) short
        sys.stderr.write("[agent] " + (fmt % args) + "\n")

    # --- helpers
    def _authorized(self):
        header = self.headers.get("Authorization", "")
        return bool(TOKEN) and hmac.compare_digest(header, "Bearer " + TOKEN)

    def _json(self, status, data):
        raw = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _body(self, limit=10 * 1024 * 1024):
        length = int(self.headers.get("Content-Length") or 0)
        if length > limit:
            raise ValueError("body too large")
        return self.rfile.read(length) if length else b""

    def _drain(self):
        """Reads an unread request body so the kept-alive connection stays in sync."""
        remaining = int(self.headers.get("Content-Length") or 0)
        while remaining > 0:
            data = self.rfile.read(min(CHUNK, remaining))
            if not data:
                break
            remaining -= len(data)

    def _route(self, method):
        if not self._authorized():
            self._drain()
            self._json(401, {"error": "unauthorized"})
            return
        parsed = urllib.parse.urlsplit(self.path)
        path, query = parsed.path, urllib.parse.parse_qs(parsed.query)
        try:
            if path.startswith("/comfy/") or path == "/comfy":
                return self._proxy(method, path[len("/comfy") :] or "/", parsed.query)
            if method == "GET" and path == "/health":
                return self._health()
            if method == "GET" and path == "/boot-log":
                data, next_offset = read_boot_log(int(query.get("offset", ["0"])[0]))
                return self._text(data, next_offset)
            if path == "/files" and method == "GET":
                return self._list(query.get("dir", [""])[0])
            if path.startswith("/files/"):
                rel = path[len("/files/") :]
                if method == "PUT":
                    return self._put(rel, int(query.get("offset", ["0"])[0]))
                if method == "GET":
                    return self._get(rel)
                if method == "DELETE":
                    return self._delete(rel)
            if path == "/jobs" and method == "POST":
                job = start_job(json.loads(self._body() or b"{}"))
                return self._json(200, job.json())
            m = re.fullmatch(r"/jobs/([a-zA-Z0-9_-]+)(/logs|/cancel)?", path)
            if m:
                job = JOBS.get(m.group(1))
                if job is None:
                    return self._json(404, {"error": "unknown job"})
                if m.group(2) == "/logs" and method == "GET":
                    return self._logs(job, int(query.get("offset", ["0"])[0]))
                if m.group(2) == "/cancel" and method == "POST":
                    threading.Thread(target=job.cancel, daemon=True).start()
                    return self._json(200, {"ok": True})
                if not m.group(2) and method == "GET":
                    return self._json(200, job.json())
            self._drain()
            self._json(404, {"error": "not found"})
        except ValueError as exc:
            self._json(400, {"error": str(exc)})
        except OSError as exc:
            self._json(500, {"error": type(exc).__name__ + ": " + str(exc)[:200]})

    def do_GET(self):  # noqa: N802
        self._route("GET")

    def do_POST(self):  # noqa: N802
        self._route("POST")

    def do_PUT(self):  # noqa: N802
        self._route("PUT")

    def do_DELETE(self):  # noqa: N802
        self._route("DELETE")

    # --- endpoints
    def _health(self):
        usage = shutil.disk_usage(WORKSPACE) if os.path.isdir(WORKSPACE) else None
        try:
            load = os.getloadavg()[0]
        except OSError:
            load = None
        self._json(
            200,
            {
                "ok": True,
                "agent_version": VERSION,
                "workspace": WORKSPACE,
                "gpus": gpu_info(),
                "cpu_count": os.cpu_count(),
                "load_1m": load,
                "disk_free": usage.free if usage else None,
                "disk_total": usage.total if usage else None,
                "aitoolkit": os.path.isfile(os.path.join(AITK_DIR, "run.py")),
                "comfyui_dir": os.path.isdir(COMFY_DIR),
                "jobs": {k: j.status for k, j in JOBS.items()},
                **memory_info(),
            },
        )

    def _list(self, rel):
        base = safe_path(rel)
        if base is None:
            return self._json(400, {"error": "bad path"})
        files = []
        if os.path.isdir(base):
            for root, _dirs, names in os.walk(base):
                for name in names:
                    full = os.path.join(root, name)
                    st = os.stat(full)
                    files.append(
                        {
                            "path": os.path.relpath(full, WORKSPACE),
                            "size": st.st_size,
                            "mtime": st.st_mtime,
                        }
                    )
                    if len(files) >= 5000:
                        break
        self._json(200, {"files": files})

    def _put(self, rel, offset):
        path = safe_path(rel)
        if path is None or path == WORKSPACE:
            self._drain()
            return self._json(400, {"error": "bad path"})
        os.makedirs(os.path.dirname(path), exist_ok=True)
        length = int(self.headers.get("Content-Length") or 0)
        if offset == 0:
            mode = "wb"
        else:
            if not os.path.exists(path) or os.path.getsize(path) != offset:
                self._drain()
                size = os.path.getsize(path) if os.path.exists(path) else 0
                return self._json(409, {"error": "offset mismatch", "size": size})
            mode = "ab"
        remaining = length
        with open(path, mode) as f:
            while remaining > 0:
                data = self.rfile.read(min(CHUNK, remaining))
                if not data:
                    break
                f.write(data)
                remaining -= len(data)
        self._json(200, {"size": os.path.getsize(path)})

    def _get(self, rel):
        path = safe_path(rel)
        if path is None or not os.path.isfile(path):
            return self._json(404, {"error": "not found"})
        size = os.path.getsize(path)
        start, end = 0, size - 1
        rng = self.headers.get("Range", "")
        m = re.fullmatch(r"bytes=(\d+)-(\d*)", rng.strip())
        if m:
            start = int(m.group(1))
            end = min(int(m.group(2)) if m.group(2) else size - 1, size - 1)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", "bytes */" + str(size))
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self.send_response(206)
            self.send_header("Content-Range", "bytes %d-%d/%d" % (start, end, size))
        else:
            self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            remaining = end - start + 1
            while remaining > 0:
                data = f.read(min(CHUNK, remaining))
                if not data:
                    break
                self.wfile.write(data)
                remaining -= len(data)

    def _delete(self, rel):
        path = safe_path(rel)
        if path is None or path == WORKSPACE:
            return self._json(400, {"error": "bad path"})
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.exists(path):
            os.remove(path)
        self._json(200, {"ok": True})

    def _logs(self, job, offset):
        try:
            with open(job.log_path, "rb") as f:
                f.seek(max(0, offset))
                data = f.read(512 * 1024)
        except OSError:
            data = b""
        self._text(data, max(0, offset) + len(data))

    def _text(self, data, next_offset):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Next-Offset", str(next_offset))
        self.end_headers()
        self.wfile.write(data)

    def _proxy(self, method, path, query):
        url = "http://127.0.0.1:%d%s%s" % (COMFY_PORT, path, ("?" + query) if query else "")
        body = self._body(200 * 1024 * 1024) if method in ("POST", "PUT") else None
        headers = {}
        if self.headers.get("Content-Type"):
            headers["Content-Type"] = self.headers["Content-Type"]
        request = urllib.request.Request(url, data=body, method=method, headers=headers)
        try:
            response = urllib.request.urlopen(request, timeout=120)
            status = response.status
        except urllib.error.HTTPError as exc:
            response = exc
            status = exc.code
        except (urllib.error.URLError, OSError) as exc:
            return self._json(502, {"error": "ComfyUI is not reachable: " + str(exc)[:200]})
        data = response.read()
        self.send_response(status)
        self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    if not TOKEN:
        sys.stderr.write("STUDIO_AGENT_TOKEN is not set\n")
        sys.exit(2)
    os.makedirs(WORKSPACE, exist_ok=True)
    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    server.daemon_threads = True
    sys.stderr.write("[agent] listening on %d, workspace %s\n" % (PORT, WORKSPACE))
    server.serve_forever()


if __name__ == "__main__":
    main()
