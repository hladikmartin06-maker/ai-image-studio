"""AI Image Studio agent: a small HTTP service that runs inside the user's GPU container
(RunPod pod or Modal sandbox) so the studio on the Mac can move files, run training and
reach ComfyUI there. Python standard library only (3.8+), started by the studio with:

    python3 -c "import base64,os,zlib;exec(zlib.decompress(
        base64.b64decode(os.environ['STUDIO_AGENT_CODE'])))"   (one line)

Environment:
  STUDIO_AGENT_TOKEN  required; every request needs "Authorization: Bearer <token>"
  STUDIO_AGENT_PORT   default 8765
  STUDIO_WORKSPACE    default /workspace (all files live below it)
  STUDIO_AITK_DIR     ai-toolkit folder for training (default /app/ai-toolkit)
  STUDIO_COMFY_DIR    ComfyUI folder (default $STUDIO_WORKSPACE/ComfyUI)
  HF_TOKEN            optional, for gated Hugging Face downloads

API (JSON unless noted):
  GET    /health                     agent, GPU (nvidia-smi), CPU load, RAM, disk, jobs
  GET    /files?dir=<rel>            files below a folder (recursive)
  PUT    /files/<rel>?offset=<n>     write a chunk (offset 0 starts the file)
  GET    /files/<rel>                download; honours "Range: bytes=a-b"
  DELETE /files/<rel>
  POST   /jobs                       {id, kind: train|comfyui|download, args}
  GET    /jobs/<id>                  status, exit code, progress (step/total), loss points
  GET    /jobs/<id>/logs?offset=<n>  plain text from that byte offset
  POST   /jobs/<id>/cancel
  *      /comfy/<path>               proxied to the local ComfyUI (127.0.0.1:8188)
"""

import contextlib
import hmac
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

VERSION = "1"
TOKEN = os.environ.get("STUDIO_AGENT_TOKEN", "")
PORT = int(os.environ.get("STUDIO_AGENT_PORT", "8765"))
WORKSPACE = os.path.realpath(os.environ.get("STUDIO_WORKSPACE", "/workspace"))
AITK_DIR = os.environ.get("STUDIO_AITK_DIR", "/app/ai-toolkit")
COMFY_DIR = os.environ.get("STUDIO_COMFY_DIR", os.path.join(WORKSPACE, "ComfyUI"))
COMFY_PORT = int(os.environ.get("STUDIO_COMFY_PORT", "8188"))
COMFY_REPO = "https://github.com/comfyanonymous/ComfyUI.git"
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


def _shell_quote(value):
    return "'" + str(value).replace("'", "'\"'\"'") + "'"


def comfyui_script(models):
    """Installs ComfyUI once (into the workspace, so a persistent volume keeps it), fetches
    the requested model files that are missing, then runs it on 127.0.0.1 only."""
    lines = [
        "set -e",
        "cd " + _shell_quote(WORKSPACE),
        "if [ ! -f " + _shell_quote(os.path.join(COMFY_DIR, "main.py")) + " ]; then",
        "  echo '[studio] installing ComfyUI'",
        "  git clone --depth 1 " + COMFY_REPO + " " + _shell_quote(COMFY_DIR),
        "fi",
        "cd " + _shell_quote(COMFY_DIR),
        "export PIP_BREAK_SYSTEM_PACKAGES=1",
        "if [ ! -f .studio-installed ]; then",
        "  python3 -m pip install -r requirements.txt && touch .studio-installed",
        "fi",
    ]
    for m in models:
        folder = re.sub(r"[^a-z_]", "", str(m.get("folder", "")))
        name = os.path.basename(str(m.get("filename", "")))
        url = str(m.get("url", ""))
        if not folder or not name or not url.startswith("https://"):
            continue
        dest = os.path.join(COMFY_DIR, "models", folder, name)
        auth = ""
        if "huggingface.co" in url:
            auth = '${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} '
        lines += [
            "if [ ! -s " + _shell_quote(dest) + " ]; then",
            "  echo '[studio] downloading " + name + "'",
            "  mkdir -p " + _shell_quote(os.path.dirname(dest)),
            "  curl -fL --retry 5 "
            + auth
            + "-o "
            + _shell_quote(dest + ".part")
            + " "
            + _shell_quote(url)
            + " && mv "
            + _shell_quote(dest + ".part")
            + " "
            + _shell_quote(dest),
            "fi",
        ]
    lines += [
        "echo '[studio] starting ComfyUI'",
        "exec python3 main.py --listen 127.0.0.1 --port " + str(COMFY_PORT),
    ]
    return "\n".join(lines)


def start_job(body):
    job_id = re.sub(r"[^a-zA-Z0-9_-]", "", str(body.get("id", "")))[:64]
    kind = body.get("kind")
    args = body.get("args") or {}
    if not job_id:
        raise ValueError("id is required")
    env = dict(os.environ)
    env.pop("STUDIO_AGENT_TOKEN", None)
    env.pop("STUDIO_AGENT_CODE", None)
    env["PYTHONUNBUFFERED"] = "1"
    if kind == "train":
        config = safe_path(args.get("config", ""))
        if not config or not os.path.isfile(config):
            raise ValueError("config file not found")
        command = [os.environ.get("STUDIO_PYTHON", "python3"), "run.py", config]
        cwd = AITK_DIR
        if not os.path.isfile(os.path.join(AITK_DIR, "run.py")):
            raise ValueError("ai-toolkit was not found in " + AITK_DIR)
    elif kind == "comfyui":
        command = ["bash", "-c", comfyui_script(args.get("models") or [])]
        cwd = WORKSPACE
    elif kind == "download":
        dest = safe_path(args.get("dest", ""))
        url = str(args.get("url", ""))
        if not dest or not url.startswith("https://"):
            raise ValueError("an https url and a destination are required")
        auth = ""
        if "huggingface.co" in url:
            auth = '${HF_TOKEN:+-H "Authorization: Bearer $HF_TOKEN"} '
        script = (
            "set -e; mkdir -p "
            + _shell_quote(os.path.dirname(dest))
            + "; curl -fL --retry 5 "
            + auth
            + "-o "
            + _shell_quote(dest + ".part")
            + " "
            + _shell_quote(url)
            + "; mv "
            + _shell_quote(dest + ".part")
            + " "
            + _shell_quote(dest)
        )
        command = ["bash", "-c", script]
        cwd = WORKSPACE
    else:
        raise ValueError("unknown job kind")
    with JOBS_LOCK:
        existing = JOBS.get(job_id)
        if existing is not None and existing.status == "running":
            return existing
        job = Job(job_id, kind, command, cwd, env)
        JOBS[job_id] = job
    return job


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
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Next-Offset", str(max(0, offset) + len(data)))
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
