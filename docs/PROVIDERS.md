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
| List prices | `GET /api/v3/models` | `data: [{model_id, name, base_price, …}]` (`base_price` USD) |
| Price estimate | `POST /api/v3/model/pricing` `{model_id, inputs}` | `data: {model_id, unit_price}` (USD) |

Auth: `Authorization: Bearer <key>`. Status values: created, processing, completed, failed.
Docs: https://wavespeed.ai/docs — predictions, media upload, balance.

### Prices (shown before every hosted run)

Sources: https://wavespeed.ai/docs/docs-common-api/models (List Models: `base_price`, float,
USD), https://wavespeed.ai/docs/docs-common-api/pricing (Pricing API: body `model_id` +
`inputs`, response `data.unit_price`), https://wavespeed.ai/docs/how-pricing-works (the price
depends on the inputs, e.g. resolution; check it before running).

> Status (2026-11-02): **unverified**. The sandbox could not open wavespeed.ai (blocked by
> its network policy); the shapes above come from search results of those pages, and neither
> call has run against the real API. So both are read tolerantly: a missing or non-numeric
> price counts as unknown, never as 0. "How to Reduce Costs" mentions `/api/v3/model/price`
> (singular); the studio uses `/model/pricing` from the API reference.

How the studio uses them (`services/pricing.py`):
- Model cards: `base_price` of each catalog endpoint on the chosen provider ("≈ $X / image",
  cached 10 minutes per user).
- Clicking Generate (or Repeat) first runs every check of a real submit, then asks the
  pricing endpoint with the exact inputs the run would send (references as placeholders), one
  call per request size. The confirmation shows the total, the price per image and the source;
  the run starts only after the user confirms. Without an estimate, the list price × images is
  shown and labelled as the list price. Nothing is generated or charged by these calls.

## Atlas Cloud (`https://api.atlascloud.ai`)

| What | Request | Response |
|---|---|---|
| Submit | `POST /api/v1/model/generateImage` (JSON `{model, …}`) | `data.id` |
| Result | `GET /api/v1/model/prediction/{id}` | `data: {status, outputs[], error}` |
| Upload | `POST /api/v1/model/uploadMedia` (multipart `file`) | `data.download_url` |

Auth: `Authorization: Bearer <key>`. Status values: processing, completed, failed.
Atlas documents no balance endpoint, so the app shows no balance for it. It has no price API
either: its docs point to a pricing page (https://www.atlascloud.ai/docs/models/price; image
models are billed per generation). The studio shows "Price unknown" with that link, and the
run still needs the user's confirmation.
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
"Blur mature content" is on. On top of that, Generate has an "18+ content" switch (off by
default; `adult` in the request): off, explicit prompts are refused on Wan as well
(`adult_off`); on, they are allowed (never with minor-related terms) and every image of the
run is marked 18+. The switch is not a provider parameter and is never sent; it is disabled on
the SFW-only models.

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

## RunPod — API key (REST API v2)

RunPod's REST API v1 (`rest.runpod.io/v1`) is deprecated and retired on 2026-11-15
(`runpod/docs` › `api-reference-v2/migrate-from-v1.mdx`); the studio uses v2 at
`https://api.runpod.io/v2` (OpenAPI `api-reference-v2/openapi.json`, docs commit f2481a7 of
2026-10-30; also how `runpod/runpodctl` 5b85fc4 calls it). Code: `backend/app/gpu/runpod.py`.

| What | Request |
|---|---|
| GPU catalog | `GET /v2/catalog/gpus?include=AVAILABILITY&product=POD&cloud=SECURE|COMMUNITY` → per GPU: `id, name, manufacturer, memory, secure, community, price {secure, community}` (USD per GPU per hour), `availability` (NONE/LOW/MEDIUM/HIGH), `dataCenters`, `cudaVersions [{version, available}]`; one GPU: `GET /v2/catalog/gpus/{id}?…` |
| List / create pods | `GET /v2/pods` (`{pods, pagination}`), `POST /v2/pods` (`CreatePodRequest`: `name, image, gpu {id, count, allowedCudaVersions}, cloud, disk, ports, env, entrypoint, cmd, mounts {persistent {size, path}}`; closed schema) |
| One pod | `GET /v2/pods/{id}`: `status` PROVISIONING/STARTING/RUNNING/EXITED/ERROR/TERMINATED, `cost` (the pod's USD/h), `gpu`, `cudaVersion`, `ports` |
| Start / stop / terminate | `POST /v2/pods/{id}/action {"action": "start"|"stop"}`, `DELETE /v2/pods/{id}` |
| Serverless endpoints | `GET /v2/serverless`, `PATCH /v2/serverless/{id} {workers: {min, max}}`, `DELETE`; health `GET https://api.runpod.ai/v2/{id}/health` |
| Balance | GraphQL `https://api.runpod.io/graphql` `myself { clientBalance currentSpendPerHr }` (shown only when returned) |

- **Errors** are RFC 9457 objects `{title, status, detail, errors}`. A capacity refusal is a
  400 or 409 and creates no pod (runpodctl's `pods create` moves to the next GPU type on
  exactly those); a 422 is a malformed body; a 5xx or no answer may have created a pod and is
  never retried.
- **Why "everything was out of stock"** (2026-11-01): `gpu.allowedCudaVersions` "matching
  is exact, so a version no machine reports yields a capacity error rather than a fallback —
  discover valid values per GPU type via `GET /v2/catalog/gpus?include=AVAILABILITY&product=
  POD` (`cudaVersions`)". The studio now asks only for CUDA versions the catalog lists as
  free for that GPU and cloud, sends no data center, vCPU or RAM minimums and no network
  volume, and uses small disks (ComfyUI: 30 GB container disk, 80 GB pod volume).
- **GPU lists** show NVIDIA GPUs with 24 GB or more: those with a free machine whose driver
  fits the task (ComfyUI: CUDA 12.8/12.9 or 13.x; training: 13.x), Secure Cloud first;
  Community Cloud (its own price) only when Secure has none; fully booked GPUs are listed but
  cannot be picked. The list refreshes every minute.
- **Price confirmation**: the confirmed hourly price is sent with the request; if the
  catalog price differs at start, nothing is created (`price_changed`) and the owner confirms
  again. Storage (docs `accounts-billing/billing.mdx`): container disk and volume 0.10 USD per
  GB per month while running, volume 0.20 while stopped; network volumes 0.07 (not used).

HTTP ports are reachable at `https://{podId}-{port}.proxy.runpod.net`. api.runpod.io and the
proxy are behind Cloudflare: requests carry the studio's own User-Agent. **Not run against the
live API** (the build sandbox cannot reach api.runpod.io); verified with a fake that follows
the v2 schema (closed create body, catalog with cudaVersions, capacity 409).

## Modal — token id + token secret

Through the official `modal` SDK (`modal.Client.from_credentials`, documented for managing
Modal on behalf of users). Workloads are **Sandboxes** in the app `ai-image-studio-gpu`,
tagged `created-by=ai-image-studio` (and `studio-connection=<id>` for ComfyUI):
`Sandbox.create(*cmd, gpu=…, cpu=4, memory=32768, image=Image.from_registry(…).entrypoint([]),
volumes={"/workspace": Volume "ais-comfyui"}, encrypted_ports=[8189], secrets=[…], timeout)`,
`Sandbox.list(app_id, tags)`, `Sandbox.from_id(...).terminate()`, `sandbox.tunnels()` for the
HTTPS URLs. Sandboxes cannot be paused: every stop ends one (the Volume stays). GPU names as in
modal-labs/modal-examples: T4, L4, A10G, L40S, A100-40GB, A100-80GB, H100, H200, B200. SDK
signatures checked against `modal` 1.6.3; not run against Modal's servers here.

- **Created in the background**: `Sandbox.create` builds the image first (the first time it
  pulls RunPod's ComfyUI image), which can take many minutes, so the request answers at once
  (stage "creating") and the studio shows the progress; after a restart of the app the
  sandbox is found again by its tag and never created twice. Start limit 45 minutes.
- **Prices (live)**: `modal.Workspace.from_context(client=…).billing.rates()` — "current
  pricing rates for the given workspace", a mapping of rate name → `Decimal` (the CLI's
  `modal billing rates`). Source: the `modal` SDK 1.6.3 itself (`modal/_workspace.py`,
  `_WorkspaceBillingManager.rates`, RPC `WorkspaceBillingRates`). The SDK does **not** document
  the names of the rates or their unit, and modal.com could not be opened from the sandbox, so
  `parse_rates` reads them defensively: names are matched loosely (GPU names with or without
  "gpu"/"nvidia", memory and punctuation; "A10" = A10G; "cpu"/"core", "memory"/"ram"), the unit
  comes from the name when it says "sec"/"hour", otherwise from the unit that fits the list
  prices below, and a value outside ⅓–3× of the list price is ignored. Cached 10 minutes per
  token (failures: retried after 2 minutes); any failure falls back to the list prices.
  **Not yet run against Modal's servers**: the first real GPU list shows whether the prices
  say "from Modal's API" (no "≈") or fall back.
  With live rates: the GPU list shows them without "≈", the running cost uses them
  (GPU + the sandbox's CPU and memory), and a ComfyUI start is refused when the GPU's rate
  changed since the owner confirmed it (`price_changed`, checked before the background
  creation).
- **Prices (fallback)**: a fixed table of Modal's list prices (USD per hour, billed per
  second), the owner's last known values from https://modal.com/pricing (2026-11-01; the page
  could not be opened from the build sandbox, **not re-checked**): T4 0.59, L4 0.80, A10G 1.10,
  L40S 1.95, A100-40GB 2.10, A100-80GB 2.50, H100 3.95, B200 6.25 (H200: none known, none
  shown); CPU 0.0472 per core-hour and memory 0.008 per GiB-hour on top (the sandbox's 4 cores
  and 32 GiB ≈ 0.4448 per hour). Used per GPU when no rate was read, always labelled an
  estimate with a link to the pricing page; Volume storage is not in it.

## Sessions and spending (both clouds)

Before a GPU starts, the confirmation shows the hourly price, the most the session can cost
(hourly × session length, capped by the spending limit) and the storage prices. While it runs
the studio shows "spent so far" and "time left": billable seconds since the start request ×
the pod's own rate (RunPod: `cost` of the pod; Modal: its live rate, else the list price) — always labelled an
estimate (the cloud's own billing counts). The GPU is stopped at the session's time limit
(default 2 h), at the spending limit, or after inactivity.

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

- **RAW files** (checked 2026-11-01, `huggingface.co/api/models/Comfy-Org/Krea-2/tree/main`,
  not gated; the download links answer with a redirect to Hugging Face's CDN):
  `diffusion_models/krea2_raw_fp8_scaled.safetensors` (13,141,730,784 bytes; the one-click
  default), `krea2_raw_int8_convrot` (13,492,686,496) and `krea2_raw_bf16` (26,283,332,608).
  RAW renders at 1K with 52 steps and CFG 3.5 (Krea's GitHub) and a negative prompt.
- **Image to image** (Generate › Krea 2 › "Start from a photo", Turbo and RAW): the empty
  latent is replaced by the user's photo — `LoadImage` (uploaded with `POST /upload/image`
  before each run) → `ImageScale` (lanczos, center crop) to the photo's aspect ratio at the
  tier's area (multiples of 16) → `VAEEncode` → `RepeatLatentBatch` (the image count) →
  `KSampler` with `denoise` 0.1–0.95 (default 0.55); LoRA and AI model as usual. All
  built-in nodes (checked in ComfyUI v0.35.0 `nodes.py`). Krea publishes no image-to-image
  template or settings; the defaults and the wording of the strength levels are the
  studio's. Only the built-in workflow takes a photo.

- **Content**: Krea 2 renders 18+ content (the owner's own Turbo runs, 2026-11-01), so it is
  listed as SFW + NSFW. The studio allows it only with the "18+ content" switch on
  (adults only, fictional characters); minor-related terms are always refused. Krea 2's
  community license (`LICENSE.pdf`) could not be read from the build sandbox (its CDN is
  blocked): check its terms yourself.

Not verified: the sampler of RAW inside ComfyUI (studio default euler/simple; Comfy-Org has
no RAW template), image-to-image quality and good denoise values for Krea 2, VRAM needs (Krea
publishes none). Only the API format of workflows can be run; UI-format files are kept for export.
**Not run against a real ComfyUI with Krea 2 weights** — verified with a fake ComfyUI that
reproduces the API.

## ComfyUI on a rented GPU (setup)

Goal: ComfyUI ready in about 1–2 minutes; nothing is installed when the GPU starts; models
are installed afterwards with one click. Code: `backend/app/comfy/deploy.py`,
`backend/app/gpu/agent/` (`COMFY_BOOT`, `studio_agent.py`).

- **Image**: RunPod's official ComfyUI image, which already holds ComfyUI and PyTorch —
  source `github.com/runpod-workers/comfyui-base` (main, commit 33dfcd6 of 2026-08-27:
  `Dockerfile`, `start.sh`, `docker-bake.hcl`, `README.md`, `README_CUDA13.md`), tags on
  Docker Hub (`hub.docker.com/v2/repositories/runpod/comfyui/tags`, checked 2026-10-31 and
  2026-11-01). Pinned since 2026-11-01 to release 1.4.1 with ComfyUI v0.35.0 (pushed
  2026-10-19), the image the "PG - Free Upscaler" template was made for (its setup.sh):
  - `runpod/comfyui:1.4.1-comfyuiv0.35.0-cuda12.8` with `allowedCudaVersions`
    `["12.8", "12.9"]`;
  - `runpod/comfyui:1.4.1-comfyuiv0.35.0-cuda13.0` ("Blackwell / RTX 5090+") with
    `["13.0", "13.1", "13.2", "13.3"]` — used for Blackwell GPUs (the CUDA 12.8 template "does
    not support CUDA 13 (Blackwell / RTX 5090)", its README) and when RunPod has no free host
    with a 12.8/12.9 driver for the chosen GPU.
  - ComfyUI v0.35.0 ≥ 0.24.1, the minimum of Comfy-Org's Krea 2 template. The 1.4.1
    `start.sh` uses the same paths and args file as before (checked in its source).
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
- **Image and CUDA from the catalog**: the CUDA 12.8 image with the free 12.8/12.9 versions;
  the CUDA 13 image with the free 13.x versions when the GPU is offered with CUDA 13 only (or
  is a Blackwell GPU). When a machine is taken at create time (capacity 400/409), the next
  image is tried — same GPU and cloud, so the same price. "No GPU of this type is free right
  now" when nothing fits.
- **Modal**: `modal.Image.from_registry("runpod/comfyui:1.4.1-comfyuiv0.35.0-cuda12.8").entrypoint([])`
  (`Image.entrypoint`, modal 1.6.3), the agent, a named Volume at `/workspace` (sandbox mounts
  use background commits, `allow_background_commits=True` in the SDK), `workdir="/"`.
  Changed 2026-11-01 after every Modal start ended with "The GPU stopped before ComfyUI was
  ready": the image's WORKDIR is `/workspace/runpod-slim` (its Dockerfile), which the empty
  Volume mounted at `/workspace` hides, and the image's `/start.sh` (SSH, FileBrowser with its
  database on the Volume, Jupyter, copying ComfyUI and building a venv on the Volume, all under
  `set -e`) does more than a sandbox needs. Modal now runs `COMFY_BOOT_MODAL`: the agent, then
  ComfyUI straight from `/opt/comfyui-baked` with the image's Python (`python3.12 main.py
  --listen 127.0.0.1 --port 8188`), its `models` folder a link to
  `/workspace/runpod-slim/ComfyUI/models` on the Volume (the same path as on RunPod, so the
  agent's downloads and LoRAs work unchanged and stay for the next start). Which of these
  stopped the real sandbox is **not confirmed** (Modal could not be reached from the build
  sandbox); when a sandbox ends, the studio now shows its exit code and last lines
  (`sandbox.logs.tail`, Modal's own system lines included, lines with secrets left out).
- Connections set up by older studio versions (which installed ComfyUI themselves) are not
  started again: the studio asks to delete them and rent a new GPU.
- **Not run on a real RunPod or Modal GPU** and not against huggingface.co: verified from the
  sources above and with a fake RunPod that runs the real start command and agent with a fake
  `start.sh` (first-start copy, args file) and a fake ComfyUI, and a local file server for the
  downloader (resume, rename, skip, token only to huggingface.co).

# Enhance photo (PGUpscaler)

Generate › "Enhance photo" runs the workflow of the free RunPod template **"PG - Free
Upscaler"** (`console.runpod.io/hub/template/okugktg6fs`, video "Celý ComfyUI Workflow, ktorý
používam ÚPLNE ZADARMO" by Tomáš Ižvolt). Code: `backend/app/comfy/enhance.py`. Checked
2026-11-01 from:

- the template page (its `__NEXT_DATA__`: name, image, disks, readme) and its start command,
  which downloads `setup.sh` from the public Hugging Face repository
  `huggingface.co/AIPRIVATEGROUP/pgupscaler` (the owner also pasted it);
- the repository's file tree (`/api/models/AIPRIVATEGROUP/pgupscaler/tree/main`, sizes),
  `workflows/PGUpscaler - EN.json` and `- SK.json` (nodes, links, widget values, the strength
  picker's values, the instructions note) and `README.md` (settings). The SK file the owner
  sent on 2026-11-01 is the same graph as the repository's (compared node by node);
- ComfyUI's own source at v0.30.0 and v0.35.0: every node of the workflow is built in
  (`SAM3_Detect`, `SeedVR2Preprocess`, `SeedVR2Conditioning`, `SeedVR2PostProcessing`,
  `ModelSamplingAuraFlow`, `ImageScaleToTotalPixels`, `ImageCompositeMasked`, …). The
  template's custom nodes (Pixaroma, rgthree) only draw its controls and previews.

What the studio does (not the template's `setup.sh`, which installs custom nodes and changes
the image): the same graph in API format, on the same ComfyUI connection as Krea 2.

- **Files** (6, ≈ 22.7 GB; from `…/resolve/main/models/<folder>/<file>`, through the agent,
  huggingface.co only): `diffusion_models/z_image_turbo_int8_convrot.safetensors`,
  `text_encoders/qwen_3_4b_fp8_mixed.safetensors`, `vae/ae.safetensors`,
  `checkpoints/sam3.1_multiplex_fp16.safetensors`,
  `diffusion_models/seedvr2_7b_int8_convrot.safetensors`,
  `vae/seedvr2_ema_vae_fp16.safetensors`. "Installed" only when ComfyUI lists all six.
- **Graph**: LoadImage → ImageScaleToTotalPixels (2 MP, lanczos) → Z-Image Turbo img2img
  (UNETLoader, CLIPLoader type `lumina2`, ModelSamplingAuraFlow shift 3, the template's skin
  prompt, ConditioningZeroOut negative, KSampler 5 steps, CFG 1, dpmpp_2m_sde / beta,
  denoise = strength 0.15 / 0.2 / 0.25 / 0.3 / 0.35 / 0.4, the values of the template's
  picker) → optional face only (CheckpointLoaderSimple SAM 3.1, "face" prompt, SAM3_Detect
  threshold 0.5, GrowMask 16, blur 31 / σ 10, ImageCompositeMasked)
  → ImageScaleBy 1.5 / 2 / 3 → SeedVR2 7B (tiled VAE 512 / 128, KSampler 1 step euler /
  simple) → SeedVR2PostProcessing color correction `lab` → SaveImage.
- **Prompts** (the template's two editable texts, English, checked like prompts): the
  prompt of the skin fix (node "✍️ Popis pleti"), shown prefilled with the template's text
  and editable — the note suggests adding "freckles" and, from strength 0.35, what the photo
  shows; empty = the template's text — and, with face only, what SAM 3.1 looks for (node
  "🎯 Čo opraviť", default "face", e.g. "face, neck, hand"). The run's prompt is stored as
  its prompt (cards, History, "load settings").
- **Before / after**: the template shows an Image Comparer (slide) and the original and the
  result side by side; the studio's image viewer does both from the stored photo and the
  result (no extra ComfyUI nodes).
- **Photo**: an upload, an AI model photo or a generated image of the same user, uploaded to
  that ComfyUI with `POST /upload/image` for the run (again on "Repeat").
- **GPU**: 24 GB or more like every GPU list; 3× is "very slow" and the README recommends
  48 GB, so the studio warns below that.
- **Not run on a real GPU** and not against huggingface.co (its CDN is blocked in the build
  sandbox, so the files were not downloaded): verified with the fake ComfyUI, which checks
  the nodes, the file names per loader and the uploaded photo, and with the real agent.

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

- **Faster runs** (Advanced, checked 2026-11-01 in ai-toolkit's `main`:
  `jobs/process/BaseSDTrainProcess.py`, `extensions_built_in/sd_trainer/SDTrainer.py`,
  `toolkit/config_modules.py`):
  - "Test images only at the end": `train.skip_first_sample: true` (no baseline before
    step 1) and `sample.sample_every: 0` (no step matches); ai-toolkit still samples once
    after the final step unless `disable_sampling`. The images do not change the LoRA.
  - "Protect the base model" off: no `diff_output_preservation` — SDTrainer then skips the
    prior prediction and the preservation prediction it runs every step, so steps are
    faster, but prompts with the class word may drift towards the character.
  - "Save memory while computing" off: `train.gradient_checkpointing: false` (ai-toolkit's
    default is true). Same training math, faster, much more VRAM; the studio then asks for an
    80 GB GPU — an estimate, not measured for Krea 2 — and refuses a smaller one.

- **Continue training** (LoRA › Runs › "Continue training", or Training › "Start from"):
  `network.pretrained_lora_path` (ai-toolkit `toolkit/config_modules.py` NetworkConfig;
  `BaseSDTrainProcess.get_latest_save_path` loads it when the run has no checkpoint of its
  own, without its step metadata, so the new run counts its own steps). The earlier
  version's file is uploaded with the dataset as `runs/<run>/start_lora.safetensors`; its
  rank and alpha are kept (the server overrides the request) so the weights fit; the dataset
  is all current photos. The studio suggests 1000 steps and the gentle learning speed
  (its own choice). The result is a new version; the earlier one is not changed. Not run on
  a real GPU.

Presets are the studio's starting points, not published recommendations. Progress and loss
come from ai-toolkit's own output (the run page no longer draws a loss chart). **Not run on a real RunPod or Modal GPU** — verified with a
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
Hugging Face's own reason for a refusal (`X-Error-Message` header, as huggingface_hub reads it,
else the JSON `error`) is shown next to the token or the repository; fine-grained tokens need
"Read access to contents of all public gated repos you can access".

# Assistant photos

A chat message can carry up to 4 photos (uploaded like reference images). They are sent
downscaled to 1024 px JPEG in the provider's own image format (Anthropic image blocks, OpenAI
`image_url` data URLs, Gemini `inline_data`) — the same code path as AI captions in LoRA
training — and only with the message they belong to; later turns replay a short note instead
of the image. Models without vision answer with the provider's error.
