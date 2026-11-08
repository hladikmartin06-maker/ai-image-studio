"""A stand-in for ComfyUI's HTTP API (tests and local trying-out only; never production).

It reproduces the documented request/response shapes of the endpoints the studio uses:
/system_stats, /object_info[/{node}], /models/{folder}, /prompt, /queue, /history/{id},
/view, /interrupt. Prompts are "executed" after a few polls and return a generated test
pattern, clearly not a model output. Standard library only, so the studio agent can also
start it as "ComfyUI" in tests:

    python3 tests/fake_comfy_server.py --listen 127.0.0.1 --port 8188
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import struct
import sys
import threading
import uuid
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

KREA_FILES = {
    "diffusion_models": ["krea2_turbo_fp8_scaled.safetensors", "krea2_raw_bf16.safetensors"],
    "text_encoders": ["qwen3vl_4b_fp8_scaled.safetensors"],
    "vae": ["qwen_image_vae.safetensors"],
    "loras": ["krea2_darkbrush.safetensors"],
    "checkpoints": [],
}
CLIP_TYPES = ["stable_diffusion", "flux", "qwen_image", "krea2", "wan", "lumina2"]
NODES = [
    "UNETLoader",
    "CLIPLoader",
    "VAELoader",
    "CLIPTextEncode",
    "ConditioningZeroOut",
    "EmptyLatentImage",
    "KSampler",
    "VAEDecode",
    "SaveImage",
    "LoraLoaderModelOnly",
    "CheckpointLoaderSimple",
    # Krea 2 image to image (built-in nodes).
    "ImageScale",
    "RepeatLatentBatch",
    # Advanced (Krea 2 edit graph; SAM masks for Face only and Body).
    "EmptySD3LatentImage",
    "MaskComposite",
    "PreviewImage",
    "InvertMask",
    "SetLatentNoiseMask",
    # Built-in nodes of the Enhance photo workflow (PGUpscaler; ComfyUI >= 0.30).
    "LoadImage",
    "ImageScaleToTotalPixels",
    "ModelSamplingAuraFlow",
    "VAEEncode",
    "SAM3_Detect",
    "GrowMask",
    "MaskToImage",
    "ImageBlur",
    "ImageToMask",
    "ImageCompositeMasked",
    "ImageScaleBy",
    "SeedVR2Preprocess",
    "VAEEncodeTiled",
    "SeedVR2Conditioning",
    "VAEDecodeTiled",
    "SeedVR2PostProcessing",
]
# The Enhance photo (PGUpscaler) model files, by ComfyUI models folder.
ENHANCE_FILES = {
    "diffusion_models": [
        "z_image_turbo_int8_convrot.safetensors",
        "seedvr2_7b_int8_convrot.safetensors",
    ],
    "text_encoders": ["qwen_3_4b_fp8_mixed.safetensors"],
    "vae": ["ae.safetensors", "seedvr2_ema_vae_fp16.safetensors"],
    "checkpoints": ["sam3.1_multiplex_fp16.safetensors"],
}


def png(width: int, height: int, color: tuple[int, int, int]) -> bytes:
    """A minimal valid PNG (solid color) without Pillow."""
    row = b"\x00" + bytes(color) * width
    raw = row * height

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 6))
        + chunk(b"IEND", b"")
    )


class FakeComfy:
    def __init__(
        self,
        files: dict[str, list[str]] | None = None,
        polls: int = 2,
        models_dir: str = "",
        scan: bool = False,
        custom_nodes: list[str] | None = None,
    ):
        self.files = {k: list(v) for k, v in (files or KREA_FILES).items()}
        # Nodes of a custom node pack (Advanced: comfyui-krea2edit).
        self.custom_nodes = list(custom_nodes or [])
        # Like ComfyUI, also list the files in <models_dir>/models/loras (sub folders too).
        self.models_dir = models_dir
        # Started by the agent: every model folder is read from disk (downloads show up).
        self.scan = scan
        self.polls_until_done = polls
        self.prompts: dict[str, dict] = {}
        self.order: list[str] = []
        # The most prompts this ComfyUI ever had waiting or running at once.
        self.max_active = 0
        self.interrupted: list[str] = []
        self.fail_with: str | None = None
        self.require_token: str | None = None
        # Photos uploaded through POST /upload/image: name -> bytes.
        self.uploads: dict[str, bytes] = {}
        self.lock = threading.Lock()

    def listed(self, folder: str) -> list[str]:
        root = os.path.join(self.models_dir, "models", folder)
        if self.scan and folder != "loras":
            if not os.path.isdir(root):
                return []
            return sorted(n for n in os.listdir(root) if n.endswith(".safetensors"))
        return self.files[folder]

    def loras(self) -> list[str]:
        names = list(self.files["loras"])
        root = os.path.join(self.models_dir, "models", "loras") if self.models_dir else ""
        if root and os.path.isdir(root):
            for folder, _dirs, files in os.walk(root):
                for name in files:
                    if name.endswith(".safetensors"):
                        names.append(os.path.relpath(os.path.join(folder, name), root))
        return names

    # --- node definitions
    def object_info(self) -> dict:
        def combo(values):
            return [values]

        info = {name: {"input": {"required": {}}, "python_module": "nodes"} for name in NODES}
        info["UNETLoader"]["input"]["required"] = {
            "unet_name": combo(self.listed("diffusion_models")),
            "weight_dtype": combo(["default", "fp8_e4m3fn"]),
        }
        info["CLIPLoader"]["input"]["required"] = {
            "clip_name": combo(self.listed("text_encoders")),
            "type": combo(CLIP_TYPES),
        }
        info["VAELoader"]["input"]["required"] = {"vae_name": combo(self.listed("vae"))}
        info["LoraLoaderModelOnly"]["input"]["required"] = {
            "lora_name": combo(self.loras()),
            "strength_model": ["FLOAT", {"default": 1.0}],
        }
        info["CheckpointLoaderSimple"]["input"]["required"] = {
            "ckpt_name": combo(self.listed("checkpoints"))
        }
        info["FaceDetailer"] = {"input": {"required": {}}, "python_module": "custom_nodes.impact"}
        for name in self.custom_nodes:
            info[name] = {"input": {"required": {}}, "python_module": "custom_nodes.krea2edit"}
        return info

    # --- request handling (shared by the HTTP server and the httpx MockTransport)
    def handle(self, method: str, path: str, query: dict, body: bytes, headers: dict):
        if self.require_token and headers.get("authorization") != f"Bearer {self.require_token}":
            return 401, "application/json", b'{"error": "unauthorized"}'
        if method == "GET" and path == "/system_stats":
            return self._json(
                {
                    "system": {
                        "os": "posix",
                        "ram_total": 64 * 2**30,
                        "ram_free": 40 * 2**30,
                        "comfyui_version": "0.43.0",
                        "python_version": "3.12.3 (main)",
                        "pytorch_version": "2.9.0+cu128",
                        "argv": ["main.py"],
                    },
                    "devices": [
                        {
                            "name": "cuda:0 NVIDIA L40S : cudaMallocAsync",
                            "type": "cuda",
                            "index": 0,
                            "vram_total": 48 * 2**30,
                            "vram_free": 45 * 2**30,
                        }
                    ],
                }
            )
        if method == "GET" and path == "/object_info":
            return self._json(self.object_info())
        if method == "GET" and path.startswith("/object_info/"):
            node = path.split("/", 2)[2]
            info = self.object_info()
            return self._json({node: info[node]} if node in info else {})
        if method == "GET" and path.startswith("/models/"):
            folder = path.split("/", 2)[2]
            if folder not in self.files:
                return 404, "text/plain", b""
            return self._json(self.listed(folder))
        if method == "POST" and path == "/upload/image":
            # multipart/form-data with the file in "image"; ComfyUI answers with its name.
            match = re.search(rb'name="image"; filename="([^"]+)"\r\n[^\r]*\r\n\r\n', body)
            if not match:
                return self._json({"error": "no image"}, 400)
            name = match.group(1).decode()
            data = body[match.end() :].rsplit(b"\r\n--", 1)[0]
            with self.lock:
                self.uploads[name] = data
            return self._json({"name": name, "subfolder": "", "type": "input"})
        if method == "POST" and path == "/prompt":
            data = json.loads(body or b"{}")
            prompt = data.get("prompt") or {}
            errors = {}
            known = self.object_info()
            for node_id, node in prompt.items():
                kind = node.get("class_type")
                if kind not in known:
                    errors[node_id] = {
                        "class_type": kind,
                        "errors": [{"message": "missing", "details": f"Node {kind} not found"}],
                    }
                if kind == "LoadImage" and node["inputs"].get("image") not in self.uploads:
                    errors[node_id] = {
                        "class_type": kind,
                        "errors": [
                            {"details": f"Invalid image file: {node['inputs'].get('image')}"}
                        ],
                    }
                for name in ("unet_name", "vae_name", "clip_name", "ckpt_name"):
                    value = node.get("inputs", {}).get(name)
                    folder = {
                        "unet_name": "diffusion_models",
                        "vae_name": "vae",
                        "clip_name": "text_encoders",
                        "ckpt_name": "checkpoints",
                    }[name]
                    if value is not None and value not in self.listed(folder):
                        errors[node_id] = {
                            "class_type": kind,
                            "errors": [{"details": f"Value not in list: {name}: '{value}'"}],
                        }
                if kind == "LoraLoaderModelOnly":
                    lora = node["inputs"].get("lora_name")
                    if lora not in self.loras():
                        errors[node_id] = {
                            "class_type": kind,
                            "errors": [{"details": f"Value not in list: lora_name: '{lora}'"}],
                        }
            if errors:
                return self._json(
                    {
                        "error": {"message": "Prompt outputs failed validation"},
                        "node_errors": errors,
                    },
                    400,
                )
            prompt_id = data.get("prompt_id") or str(uuid.uuid4())
            with self.lock:
                self.prompts[prompt_id] = {"prompt": prompt, "polls": 0, "done": False}
                self.order.append(prompt_id)
                active = [p for p in self.order if not self.prompts[p]["done"]]
                self.max_active = max(self.max_active, len(active))
            return self._json(
                {"prompt_id": prompt_id, "number": len(self.order), "node_errors": {}}
            )
        if method == "GET" and path == "/queue":
            running, pending = [], []
            with self.lock:
                active = [p for p in self.order if not self.prompts[p]["done"]]
            for index, pid in enumerate(active):
                (running if index == 0 else pending).append([index, pid, {}, {}, []])
            return self._json({"queue_running": running, "queue_pending": pending})
        if method == "GET" and path.startswith("/history/"):
            pid = path.split("/", 2)[2]
            with self.lock:
                entry = self.prompts.get(pid)
                if entry is None:
                    return self._json({})
                active = [p for p in self.order if not self.prompts[p]["done"]]
                if active and active[0] == pid:
                    entry["polls"] += 1
                    if entry["polls"] >= self.polls_until_done:
                        entry["done"] = True
                if not entry["done"]:
                    return self._json({})
            if pid in self.interrupted:
                return self._json(
                    {
                        pid: {
                            "outputs": {},
                            "status": {
                                "status_str": "error",
                                "completed": False,
                                "messages": [["execution_interrupted", {"prompt_id": pid}]],
                            },
                        }
                    }
                )
            if self.fail_with:
                return self._json(
                    {
                        pid: {
                            "outputs": {},
                            "status": {
                                "status_str": "error",
                                "completed": False,
                                "messages": [
                                    [
                                        "execution_error",
                                        {
                                            "node_type": "KSampler",
                                            "exception_message": self.fail_with,
                                        },
                                    ]
                                ],
                            },
                        }
                    }
                )
            batch = 1
            for node in entry["prompt"].values():
                if node.get("class_type") == "EmptyLatentImage":
                    batch = int(node["inputs"].get("batch_size", 1))
                if node.get("class_type") == "RepeatLatentBatch":
                    batch = int(node["inputs"].get("amount", 1))
            images = [
                {
                    "filename": f"ai-image-studio_{pid[:8]}_{i}.png",
                    "subfolder": "",
                    "type": "output",
                }
                for i in range(batch)
            ]
            return self._json(
                {
                    pid: {
                        "prompt": [0, pid, entry["prompt"], {}, []],
                        "outputs": {"9": {"images": images}},
                        "status": {"status_str": "success", "completed": True, "messages": []},
                    }
                }
            )
        if method == "GET" and path == "/view":
            name = (query.get("filename") or [""])[0]
            # A colour of its own for every file name, so two results are never the same file
            # (the studio stores identical files once).
            r, g, b = hashlib.sha256(name.encode()).digest()[:3]
            return 200, "image/png", png(96, 128, (r, g, b))
        if method == "POST" and path == "/interrupt":
            data = json.loads(body or b"{}")
            pid = data.get("prompt_id")
            with self.lock:
                if pid in self.prompts:
                    self.interrupted.append(pid)
                    self.prompts[pid]["done"] = True
            return 200, "text/plain", b""
        if method == "POST" and path == "/queue":
            data = json.loads(body or b"{}")
            with self.lock:
                for pid in data.get("delete", []):
                    if pid in self.prompts and pid != self.order[0]:
                        self.prompts[pid]["done"] = True
                        self.interrupted.append(pid)
            return 200, "text/plain", b""
        return 404, "text/plain", b"not found"

    @staticmethod
    def _json(data, status: int = 200):
        return status, "application/json", json.dumps(data).encode()

    def transport(self):
        import httpx

        def handler(request: httpx.Request) -> httpx.Response:
            status, ctype, payload = self.handle(
                request.method,
                request.url.path,
                parse_qs(request.url.query.decode()),
                request.content,
                {k.lower(): v for k, v in request.headers.items()},
            )
            return httpx.Response(status, content=payload, headers={"content-type": ctype})

        return httpx.MockTransport(handler)


def serve(port: int, state: FakeComfy | None = None) -> ThreadingHTTPServer:
    fake = state or FakeComfy()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):  # quiet
            return

        def _do(self, method: str) -> None:
            parts = urlsplit(self.path)
            length = int(self.headers.get("Content-Length") or 0)
            body = self.rfile.read(length) if length else b""
            status, ctype, payload = fake.handle(
                method,
                parts.path,
                parse_qs(parts.query),
                body,
                {k.lower(): v for k, v in self.headers.items()},
            )
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):  # noqa: N802
            self._do("GET")

        def do_POST(self):  # noqa: N802
            self._do("POST")

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    return server


if __name__ == "__main__":
    port = 8188
    args = sys.argv[1:]
    here = os.path.dirname(os.path.abspath(__file__))
    if "--port" in args:
        port = int(args[args.index("--port") + 1])
    sys.stdout.write(f"fake ComfyUI on {port}\n")
    sys.stdout.flush()
    # Started by the agent as ComfyUI/main.py: also list LoRAs copied into ./models/loras.
    serve(port, FakeComfy(models_dir=here, scan=True)).serve_forever()
