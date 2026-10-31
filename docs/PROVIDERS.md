# Providers and models

The studio talks to two providers. Every endpoint, model id and parameter below comes from
the providers' public documentation and model pages. Anything not confirmed there is either
left out or marked as a conservative choice. The code is `backend/app/catalog.py` (models)
and `backend/app/providers/` (HTTP).

> Status: written from the official pages as found on 2026-10-31. The build sandbox could not
> reach wavespeed.ai / atlascloud.ai directly, so the full parameter tables were read through
> search results of those pages. **Nothing has been run against the real APIs yet**: the first
> real generation per model is the final check. If a provider rejects a parameter, the error
> appears in the job card and in History (with the exact request that was sent).

## WaveSpeed (`https://api.wavespeed.ai`)

| What | Request | Response |
|---|---|---|
| Submit | `POST /api/v3/{model_id}` (JSON body) | `{code, message, data: {id, status, …}}` |
| Result | `GET /api/v3/predictions/{id}/result` | `data: {status, outputs[], error, has_nsfw_contents[]}` |
| Upload | `POST /api/v3/media/upload/binary` (multipart `file`) | `data.download_url` |
| Balance | `GET /api/v3/balance` | `data.balance` (USD) |

Auth: `Authorization: Bearer <key>`. Status values: created, processing, completed, failed.
Docs: https://wavespeed.ai/docs — predictions, media upload, balance.

## Atlas Cloud (`https://api.atlascloud.ai`)

| What | Request | Response |
|---|---|---|
| Submit | `POST /api/v1/model/generateImage` (JSON `{model, …}`) | `data.id` |
| Result | `GET /api/v1/model/prediction/{id}` | `data: {status, outputs[], error}` |
| Upload | `POST /api/v1/model/uploadMedia` (multipart `file`) | `data.download_url` |

Auth: `Authorization: Bearer <key>`. Status values: processing, completed, failed.
Atlas documents no balance endpoint, so the app shows no balance for it.
"Test connection" looks up a prediction id that cannot exist (free): 401/403 = key rejected.
Docs: https://www.atlascloud.ai/docs — predictions, image models.

## Models

Images per generation: 1–4. Where an endpoint documents no count parameter, the studio sends
one request per image (each is billed by the provider; the UI says so).

### Nano Banana Pro (SFW) — Google Gemini 3 Pro Image

| Provider | Endpoint | Parameters sent |
|---|---|---|
| WaveSpeed | `google/nano-banana-pro/text-to-image` | prompt, aspect_ratio, resolution (1k/2k/4k), output_format (png/jpeg) |
| WaveSpeed | `google/nano-banana-pro/edit` | prompt, images, aspect_ratio (optional), resolution, output_format |
| Atlas | `google/nano-banana-pro/text-to-image` | prompt, aspect_ratio, resolution, output_format |
| Atlas | `google/nano-banana-pro/edit` | prompt, images, aspect_ratio (optional), resolution |

Aspect ratios: 1:1, 3:2, 2:3, 3:4, 4:3, 4:5, 5:4, 9:16, 16:9, 21:9 (documented on WaveSpeed).
Conservative: up to 4 reference images (the providers' maximum was not confirmed).
Atlas `output_format` on edit was not documented → not sent.

### Seedream (SFW) — ByteDance Seedream 4.5

| Provider | Endpoint | Parameters sent |
|---|---|---|
| WaveSpeed | `bytedance/seedream-v4.5` | prompt, size `W*H` (2K or 4K preset) |
| WaveSpeed | `bytedance/seedream-v4.5/edit` | prompt, images (1–10), size (2K preset) or none |
| Atlas | `bytedance/seedream-v4.5` | prompt, size `W*H` (2K preset) |
| Atlas | `bytedance/seedream-v4.5/edit` | prompt, images |

Sizes are ByteDance's 2K presets (min 2560×1440 pixels) and 4K presets within 4096 per side.
WaveSpeed documents 512–8192 per side for text-to-image and 1024–4096 for edit. Atlas lists
fixed presets; only the 2K presets are offered there. Atlas edit documents no size field.

### Wan 2.7 (SFW + NSFW) — Alibaba Wan 2.7 Image

| Provider | Endpoint | Parameters sent |
|---|---|---|
| WaveSpeed | `alibaba/wan-2.7/text-to-image` | prompt, size `W*H` (1K/2K), seed (−1 = random), thinking_mode |
| WaveSpeed | `alibaba/wan-2.7/image-edit` | prompt, images (1–3), size or none, seed |
| Atlas | `alibaba/wan-2.7/text-to-image` | prompt, size (`1K`/`2K`), seed, thinking_mode |
| Atlas | `alibaba/wan-2.7/image-edit` | prompt, images, size (`1K`/`2K`), seed |

### Wan 2.7 Pro (SFW + NSFW) — Alibaba Wan 2.7 Image Pro

| Provider | Endpoint | Parameters sent |
|---|---|---|
| WaveSpeed | `alibaba/wan-2.7/text-to-image-pro` | prompt, size `W*H` (1K/2K/4K), seed, thinking_mode |
| WaveSpeed | `alibaba/wan-2.7/image-edit-pro` | prompt, images (1–3), size or none, seed |
| Atlas | `alibaba/wan-2.7-pro/text-to-image` | prompt, size (`1K`/`2K`/`4K`), seed, thinking_mode, n (images) |
| Atlas | `alibaba/wan-2.7-pro/image-edit` | prompt, images, size (`1K`/`2K`), seed |

Not documented for these endpoints and therefore never sent or shown: negative prompt,
guidance scale, steps, safety-checker switches. Mature content on the Wan models follows the
providers' own policies; images the provider flags (`has_nsfw_contents`) are blurred when
"Blur mature content" is on.

## Reference images and AI models

Reference images reach a provider only through an editing endpoint above, and never more than
that endpoint's limit (WaveSpeed: Nano Banana Pro 4, Seedream 10, Wan 2.7 / Pro 3; Atlas:
Nano Banana Pro 4, Seedream 4, Wan 2.7 / Pro 3 — where a provider documents no maximum, the
catalog uses a conservative one). A saved AI model adds its images first (primary first) only up to
the room left after the user's own references; with references switched off, or on an endpoint
without editing, nothing is uploaded. Each file is uploaded to a provider once and its URL is
reused for 24 hours: neither provider documents how long an uploaded file stays available, so
the studio uploads it again after that instead of risking an expired URL.

The description of an AI model is plain text appended to the prompt (`Character: adult …`, plus
"Keep the same person as in the reference images …" when images are attached). It is not a
provider parameter; History shows the prompt exactly as sent. **Not run against the real
WaveSpeed / Atlas APIs** — how strongly each model keeps the identity is up to the model.

## Errors

HTTP status and provider messages are normalized (`providers/base.py`): 401/403 invalid key,
402 or "insufficient/balance/credit" insufficient credits, 429 rate limit, 400/422 invalid
parameters (or content policy), 404 unsupported model, 5xx provider error, network errors.
Polling retries transient errors; a submit is never repeated.

# AI assistant providers (chat)

Separate from the image providers: own keys (`ai_integrations` table, AES-256-GCM), own code
(`backend/app/ai/`). Model lists are never hard-coded as truth: when a key is connected or
tested, the app calls the provider's **models endpoint** and caches the result. Each cost
tier (low / balanced / high) picks the first *preferred* id that the key really offers, else
a name-based match from the live list; the user can choose any listed model per tier.

| Provider | API | Models endpoint | Chat (streaming) | Preferred ids (low / balanced / high) |
|---|---|---|---|---|
| Anthropic Claude | official `anthropic` Python SDK | `client.models.list()` | `client.messages.stream()` | `claude-haiku-5-5` / `claude-sonnet-5-5` / `claude-opus-5-5` |
| OpenAI | `https://api.openai.com/v1` | `GET /models` | `POST /chat/completions` (`stream`, `stream_options.include_usage`, `max_completion_tokens`, `reasoning_effort`) | `gpt-5.6-luna` / `gpt-5.6-terra` / `gpt-5.6-sol` (fallbacks `gpt-5-nano`, `gpt-5-mini`, `gpt-5`) |
| xAI Grok | `https://api.x.ai/v1` (OpenAI-compatible) | `GET /models` | `POST /chat/completions` | `grok-4.20-non-reasoning` / same / `grok-4.20-reasoning` |
| Google Gemini | `https://generativelanguage.googleapis.com/v1beta` | `GET /models` | `POST /models/{id}:streamGenerateContent?alt=sse` | `gemini-3.5-flash-lite` / `gemini-3.8-flash` / `gemini-3.1-pro-preview` (fallbacks 3.1 / 2.5) |
| OpenAI-compatible | user-supplied base URL (presets: OpenRouter, Groq, DeepSeek, Mistral, Together, Ollama, LM Studio) | `GET /models` | `POST /chat/completions` | from the live list by name |

Status (2026-10-31): Anthropic ids and request shapes come from Anthropic's own reference
(Claude API skill, models cached 2026-10-06). OpenAI, xAI and Gemini ids were found through
web search of the providers' and third-party pages — **unverified**; the live models list
decides. **Not run against any real chat API in this sandbox** (outbound access blocked);
verified with fake servers that reproduce the documented wire formats.

Cost controls in the app: local routing (no extra call), cheapest tier by default
(`ai_mode = low`), output caps per tier (900 / 1400 / 2400 tokens), Claude `effort` low/medium
and automatic prompt caching, OpenAI `reasoning_effort: low`, Gemini 3 `thinkingLevel: low`,
studio context only when it changed, a token budget for replayed history, older turns
folded into a ≤120-word summary by the low tier, free local answers for greetings/thanks,
and a 24 h cache for identical requests. Anthropic Opus 5.5 / Sonnet 5.5 requests carry the
server-side refusal fallback (`fallbacks: "default"`, beta `server-side-fallback-2026-07-01`).
No prices are shown: only the providers' reported token counts (≈ when estimated).

# GPU compute providers

Separate again: `gpu_integrations` (encrypted JSON credentials), `gpu_workloads` (history),
`backend/app/gpu/`. Starting anything that bills requires `confirm_billing: true`, sent only
after the user confirmed a dialog that says so.

## RunPod — API key

| What | Request |
|---|---|
| List / create pods | `GET` / `POST https://rest.runpod.io/v1/pods` (`name, imageName, gpuTypeIds, gpuCount, cloudType, containerDiskInGb, volumeInGb, ports, env`) |
| Start / stop / terminate | `POST /v1/pods/{id}/start`, `POST /v1/pods/{id}/stop`, `DELETE /v1/pods/{id}` |
| Serverless endpoints | `GET /v1/endpoints`, `PATCH /v1/endpoints/{id}` (`workersMin, workersMax`), `DELETE`; health `GET https://api.runpod.ai/v2/{id}/health` |
| GPU catalog | GraphQL `https://api.runpod.io/graphql`: `gpuTypes { id displayName memoryInGb securePrice communityPrice }` |
| Balance | GraphQL `myself { clientBalance currentSpendPerHr }` (shown only when returned; otherwise spend = sum of running pods' `costPerHr`) |

HTTP ports are reachable at `https://{podId}-{port}.proxy.runpod.net`. Field names of the REST
responses are read defensively. Unverified against the live API (sandbox).

## Modal — token id + token secret

Through the official `modal` SDK (`modal.Client.from_credentials`, documented for managing
Modal on behalf of users). Workloads are **Sandboxes** in the app `ai-image-studio-gpu`,
tagged `created-by=ai-image-studio`: `Sandbox.create(..., gpu="H100:2", image=Image.from_registry(...),
timeout, encrypted_ports)`, `Sandbox.list(app_id, tags)`, `Sandbox.from_id(...).terminate()`,
`sandbox.tunnels()` for the HTTPS URLs of exposed ports. Sandboxes cannot be paused (only
terminated) and Modal has no balance or price API, so neither is shown. GPU names from the
Modal GPU guide: T4, L4, A10, L40S, A100-40GB, A100-80GB, H100, H200, B200. SDK signatures
were checked against `modal` 1.6.3; not run against Modal's servers here.

# ComfyUI and Krea 2

Verified on 2026-10-31 from the sources' own files (GitHub; Hugging Face pages could not be
opened from the build sandbox):

- **Krea 2** (github.com/krea-ai/krea-2, released 2026-06-22): RAW = undistilled pretrained
  checkpoint for fine-tuning and LoRA training (52 steps, CFG 3.5, generation up to 1k);
  Turbo = 8-step distilled, CFG off, 1k–2k. “LoRAs trained on RAW will work well on Turbo.”
  HF repos `krea/Krea-2-Raw`, `krea/Krea-2-Turbo` (gated: license acceptance + token).
  Community license; commercial license via Krea.
- **ComfyUI** (comfyanonymous/ComfyUI, version 0.43.0 at the time): native Krea 2 support —
  `supported_models.Krea2`, `CLIPLoader` type `krea2` (Qwen3-VL-4B), sampling shift 1.15.
- **Comfy-Org template** `image_krea2_turbo_t2i` (min ComfyUI 0.24.1): `UNETLoader`
  `krea2_turbo_fp8_scaled.safetensors` (12.24 GB), `CLIPLoader`
  `qwen3vl_4b_fp8_scaled.safetensors` (4.88 GB, type krea2), `VAELoader`
  `qwen_image_vae.safetensors` (242 MB), `LoraLoaderModelOnly`, `CLIPTextEncode`,
  `ConditioningZeroOut` (negative), `EmptyLatentImage`, `KSampler` 8 steps / CFG 1 / euler /
  simple, `VAEDecode`, `SaveImage`. Files from `huggingface.co/Comfy-Org/Krea-2`. Int8 file:
  `krea2_turbo_int8_convrot.safetensors`.
- **API used**: `/system_stats`, `/object_info`, `/models/{folder}`, `POST /prompt` (client
  `prompt_id` accepted), `/queue`, `/history/{id}`, `/view`, `POST /interrupt {prompt_id}`,
  `POST /queue {delete}`, `/ws` progress.

Not verified: the ComfyUI file name of the RAW checkpoint (detected by name; added by URL),
the sampler of RAW inside ComfyUI (studio default euler/simple), VRAM needs (Krea publishes
none). Only the API format of workflows can be run; UI-format files are kept for export.
**Not run against a real ComfyUI with Krea 2 weights** — verified with a fake ComfyUI that
reproduces the API.

## ComfyUI on a rented GPU (setup)

Goal: ComfyUI ready in about 1–2 minutes; nothing is installed when the GPU starts; models
are installed afterwards with one click. Code: `backend/app/comfy/deploy.py`,
`backend/app/gpu/agent/` (`COMFY_BOOT`, `studio_agent.py`).

- **Image**: RunPod's official ComfyUI image, which already holds ComfyUI and PyTorch —
  source `github.com/runpod-workers/comfyui-base` (main, commit 33dfcd6 of 2026-08-27:
  `Dockerfile`, `start.sh`, `docker-bake.hcl`, `README.md`, `README_CUDA13.md`), tags on
  Docker Hub (`hub.docker.com/v2/repositories/runpod/comfyui/tags`, checked 2026-10-31):
  - `runpod/comfyui:cuda12.8` (= `latest`; ComfyUI v0.30.0, PyTorch 2.10.0+cu128, Ubuntu
    24.04, Python 3.12) with `allowedCudaVersions` `["12.8", "12.9"]`;
  - `runpod/comfyui:cuda13.0` (PyTorch 2.10.0+cu130; "Blackwell / RTX 5090+") with
    `["13.0", "13.1", "13.2", "13.3"]` — used for Blackwell GPUs (the CUDA 12.8 template "does
    not support CUDA 13 (Blackwell / RTX 5090)", its README) and when RunPod has no free host
    with a 12.8/12.9 driver for the chosen GPU.
  - ComfyUI v0.30.0 ≥ 0.24.1, the minimum of Comfy-Org's Krea 2 template.
- **What the image's `/start.sh` does** (read from its source): SSH (random root password in
  the log unless `PUBLIC_KEY`), FileBrowser on 8080 (`FILEBROWSER_PASSWORD`), JupyterLab on
  8888 (`JUPYTER_PASSWORD`), on the first start copies `/opt/comfyui-baked` to
  `/workspace/runpod-slim/ComfyUI` and creates a venv with the image's packages (no network),
  then runs `python main.py --listen 0.0.0.0 --port 8188 --enable-cors-header` plus the lines
  of `/workspace/runpod-slim/comfyui_args.txt`. ENTRYPOINT `["/start.sh"]`; `python3` is 3.12.
- **Start command** (`CreatePodRequest`): `dockerEntrypoint ["/bin/bash", "-c"]`,
  `dockerStartCmd [COMFY_BOOT]`: writes the agent (gzip + base64 in `STUDIO_AGENT_CODE`) to a
  temp file, starts it in the background (restarted if it stops), adds `--listen 127.0.0.1`
  to `comfyui_args.txt` (argparse: the last `--listen` wins), sets random Jupyter and
  FileBrowser passwords, removes the agent's secrets from the environment and ends with
  `exec /start.sh`; its output also goes to a start log the agent serves on `/boot-log`
  (password lines hidden).
- **Disk**: pod volume (`volumeInGb`, `volumeMountPath /workspace`) — ComfyUI and the models
  stay when the pod is stopped. Network volumes are not used.
- **Network**: only the agent's port `8189/http`, reached at
  `https://{podId}-8189.proxy.runpod.net`, every request with the per-deployment Bearer
  token; ComfyUI (8188), FileBrowser, Jupyter and SSH ports are not exposed.
- **Ready check**: every ~10 s the studio asks the agent (`/health`), then ComfyUI through the
  agent (`/comfy/system_stats`, `/object_info`). "Ready" only when ComfyUI itself answers. No
  answer within 30 minutes (Modal: 45) stops the GPU.
- **Models**: "Download" asks the agent to fetch the text encoder, VAE and diffusion model from
  `huggingface.co/Comfy-Org/Krea-2` directly on the GPU, never through the Mac. Only
  `https://huggingface.co/…` is accepted (redirects only to Hugging Face's CDN hosts
  `*.huggingface.co` / `*.hf.co`; the token is sent to huggingface.co only); each file is
  written as `.part` (resumed after a dropped connection) and renamed when complete; free disk
  space is checked; files already there are skipped; progress per file. Saved into
  `/workspace/runpod-slim/ComfyUI/models/<folder>`; started only once ComfyUI answers. ComfyUI
  lists new files without a restart (its `folder_paths` file list is refreshed when a folder
  changes); the studio checks `/object_info` again after the download.
- **CUDA list**: RunPod's published API (`runpod/docs`, `api-reference/openapi.json`, checked
  2026-10-31) lists `allowedCudaVersions` values up to "13.0" only. "13.1"–"13.3" are sent as
  the owner asked; if RunPod refuses them (HTTP 400 naming the CUDA field), the request is sent
  again with `["13.0"]` (a refused request creates no pod). Training pods do the same.
- **No free GPU**: RunPod's wording "There are no longer any instances available …" /
  "does not have the resources" (from RunPod's own pod tests in comfyui-base,
  `tests/runpod_smoke/pod.py`) is shown as "No GPU of this type is free right now"; then the
  next image is tried. GPU lists offer NVIDIA GPUs with 24 GB or more, and only those RunPod
  prices in the chosen cloud.
- **Modal**: `modal.Image.from_registry("runpod/comfyui:cuda12.8").entrypoint([])`
  (`Image.entrypoint`, modal 1.6.3), the same command and agent, a named Volume at
  `/workspace` (sandbox mounts use background commits, `allow_background_commits=True` in the
  SDK). The first start pulls the image and copies ComfyUI onto the Volume; later starts reuse
  it.
- Connections set up by older studio versions (which installed ComfyUI themselves) are not
  started again: the studio asks to delete them and rent a new GPU.
- **Not run on a real RunPod or Modal GPU** and not against huggingface.co: verified from the
  sources above and with a fake RunPod that runs the real start command and agent with a fake
  `start.sh` (first-start copy, args file) and a fake ComfyUI, and a local file server for the
  downloader (resume, rename, skip, token only to huggingface.co).

# LoRA training (ai-toolkit)

- **Trainer**: ostris/ai-toolkit, arch `krea2` (`extensions_built_in/diffusion_models/krea2`):
  single-stream MMDiT (features 6144, 28 layers), Qwen3-VL-4B, Qwen-Image VAE; RAW and Turbo
  share the architecture; Turbo training uses `assistant_lora_path`
  `ostris/krea2_turbo_training_adapter/krea2_turbo_training_adapter_v1.safetensors`; LoRA keys
  use ComfyUI's `diffusion_model.` prefix. Config fields from `toolkit/config_modules.py`
  (network linear/alpha, optimizer adamw8bit/adamw/adafactor/prodigy, lr_scheduler,
  timestep_type, diff_output_preservation, quantize, low_vram, save/sample). JSON configs
  are accepted (`toolkit/config.py`). Run: `python run.py <config>`.
- **Image**: `ostris/aitoolkit:latest` (docker-compose of the repo; needs an NVIDIA driver for
  CUDA 13). Not available in ai-toolkit for Krea 2: text encoder training, validation loss,
  a training seed. Regularization images are not offered by the studio.
- **GPU side**: the studio agent (`backend/app/gpu/agent`) in the container; training pods get
  `dockerEntrypoint`/`dockerStartCmd` (`BOOT`), port 8765/http and a volume at /workspace,
  reached at `https://{podId}-8765.proxy.runpod.net`, with `allowedCudaVersions` 13.0–13.3
  (13.0 alone if RunPod refuses the others); Modal sandboxes get `encrypted_ports`, a Secret for
  tokens and a named Volume. ComfyUI deployments: see "ComfyUI on a rented GPU" above.

Presets are the studio's starting points, not published recommendations. Progress and loss
come from ai-toolkit's own output. **Not run on a real RunPod or Modal GPU** — verified with a
fake RunPod that starts the real agent locally and a fake `run.py`.

# Hugging Face (token and Krea 2 license)

Integrations › Hugging Face stores the user's token (encrypted, `user_secrets`) and checks it;
LoRA training checks the license again before any GPU is rented. Code:
`backend/app/services/huggingface.py`.

| What | Request | Meaning |
|---|---|---|
| Token owner | `GET https://huggingface.co/api/whoami-v2`, `Authorization: Bearer <token>` | 200 `{name, …}`; 401 = invalid token (never saved). Strictly rate-limited, so cached 10 min |
| Repository access | `GET https://huggingface.co/api/models/{repo}/auth-check` | 200 = can download; 403 with header `X-Error-Code: GatedRepo` = license not accepted (or request pending); 401 = invalid token; 404 = not found |

Repositories checked: `krea/Krea-2-Raw` and `krea/Krea-2-Turbo` (both gated, from
`lora/presets.py`). There is **no API to accept a license for a user**: it is the user's own
agreement, made with "Agree and access repository" on the repository page, so the studio links
there and checks again.

Status (2026-10-31): endpoints, headers and status codes read from the source of
`huggingface_hub` 2.4.0 (`hf_api.py` `whoami` / `auth_check`, `utils/_http.py`
`hf_raise_for_status`). huggingface.co could not be reached from the build sandbox: **not run
against the real Hub** — verified with a fake Hub (`tests/fakes.py` `FakeHub`, and
`tests/fake_server.py` for trying the app). When the Hub cannot be reached, the token is still
saved and training is not blocked (the trainer then reports a download error itself).

# Assistant photos

A chat message can carry up to 4 photos (uploaded like reference images). They are sent
downscaled to 1024 px JPEG in the provider's own image format (Anthropic image blocks, OpenAI
`image_url` data URLs, Gemini `inline_data`) — the same code path as AI captions in LoRA
training — and only with the message they belong to; later turns replay a short note instead
of the image. Models without vision answer with the provider's error.
