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

### Wan 3.0 and Wan 3.0 Prime (video, SFW) — Alibaba Wan 3.0

Verified 2026-11-02 from WaveSpeed's API docs (the session's network allowed wavespeed.ai):
https://wavespeed.ai/docs/docs-api/alibaba/alibaba-wan-3.0-text-to-video,
…-wan-3.0-image-to-video, …-wan-3.0-prime-text-to-video and …-wan-3.0-prime-image-to-video.

| Provider | Endpoint | Parameters sent |
|---|---|---|
| WaveSpeed | `alibaba/wan-3.0/text-to-video` | prompt, resolution (480p/720p/1080p, default 720p), aspect_ratio (16:9, 9:16, 1:1, 4:3, 3:4; default 16:9), duration (2–30 s, default 5), generate_audio (default true), enable_prompt_expansion (default false), seed (−1 = random) |
| WaveSpeed | `alibaba/wan-3.0/image-to-video` | the same, plus image (first frame, required) and last_image (optional); aspect_ratio optional (else it follows the photo) |
| WaveSpeed | `alibaba/wan-3.0-prime/text-to-video` | as Wan 3.0 text-to-video |
| WaveSpeed | `alibaba/wan-3.0-prime/image-to-video` | as Wan 3.0 image-to-video |

Results come back like images (`data.outputs[]`, URLs); the studio stores the file as it is
(MP4, its size read from the file's `tkhd` box, up to 1 GB) and shows it with a player. No
negative prompt is documented, so none is sent. Prices on those pages (per second of billed
duration, rounded up; the studio shows WaveSpeed's own estimate before a run): Wan 3.0 480p
$0.05, 720p $0.10, 1080p $0.20; Prime 480p $0.075, 720p $0.15, 1080p $0.30. Treated as SFW
(explicit prompts refused); WaveSpeed lists a separate "spicy" image-to-video model, not used.
Atlas Cloud lists Wan 3.0 too (`alibaba/wan-3.0/image-to-video` via
`/api/v1/model/generateVideo`, from search results only): **not added** — its pages
(www.atlascloud.ai) are still blocked from the sandbox, so nothing there is verified.
**Not yet run against the real API.**

## Reference images and AI models

Reference images reach a provider only through an editing endpoint above, and never more than
that endpoint's limit (WaveSpeed: Nano Banana Pro 4, Seedream 10, Wan 2.7 / Pro 3; Atlas:
Nano Banana Pro 4, Seedream 4, Wan 2.7 / Pro 3 — where a provider documents no maximum, the
catalog uses a conservative one). A saved AI model adds its images first (primary first) only up to
the room left after the user's own references; with references switched off, or on an endpoint
without editing, nothing is uploaded. Each file is uploaded to a provider once and its URL is
reused for 24 hours: neither provider documents how long an uploaded file stays available, so
the studio uploads it again after that instead of risking an expired URL.

Photos up to 20 MB can be uploaded to the studio (references, AI model photos, photos for
Krea 2 and Enhance photo) and are stored as they are. Neither WaveSpeed nor Atlas Cloud
documents a maximum upload size, so a file sent to them is at most 10 MB (the studio's earlier
limit, known to work): a larger photo goes as a high-quality JPEG (quality 92, scaled down only
as far as needed; `references.for_provider`). ComfyUI gets the original.

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
- **No free GPU**: `Sandbox.create` can fail with "Insufficient capacity to create sandbox"
  (`ResourceExhaustedError`; seen by the owner on 2026-11-02 and described as transient
  capacity exhaustion in https://github.com/dryingpaint/biomodelbench-private-eval/pull/44 —
  not in Modal's docs). Nothing exists or is billed then. The studio maps it to "unavailable"
  (not "internal problem") and asks again after 20, 40, then every 60 seconds for up to 20
  minutes (ComfyUI: the connection shows "Waiting for a free GPU"; LoRA training: the
  version stays queued), then gives up with "Modal had no free <GPU>… pick another GPU". The
  session time starts only when the sandbox exists.
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

# Advanced generation (Krea 2 + SAM 3.1 + checks)

Generate › "Pokročilé" (`backend/app/advanced/`, since 2026-11-04, the owner's request). It
uses Krea 2 RAW / Turbo (the files, loaders and sampler settings of `krea2.VARIANTS`, read
only), SAM 3.1 (`checkpoints/sam3.1_multiplex_fp16.safetensors`, the Enhance photo file from
`huggingface.co/AIPRIVATEGROUP/pgupscaler`, installable alone with one click), since
2026-11-06 Krea 2 AnyPaint (below), SFace on this Mac and the user's own AI assistant. No
upscaler (since 2026-11-06, the owner's request: Enhance photo is a separate tool). No Flux,
no Qwen-Image / Qwen-Image-Edit.

- **Krea 2 AnyPaint** (the owner's plan, phase 1, approved 2026-11-06; `comfy/anypaint.py`,
  `comfy/custom_nodes.py`): every region edit (hair, mouth, eyes, face, expression, clothing,
  background, body, breasts, hips) is drawn by it instead of the masked redraw with
  DifferentialDiffusion described below.
  - Model card `huggingface.co/yijunwang2/krea2-anypaint`, revision `1a9fb37a`: a functional
    adapter for arbitrary-mask inpainting, outpainting and editing, rank/alpha 32/32, trained
    against Krea 2 Raw and intended for Krea 2 Turbo; recommended 8 steps, guidance 0, LoRA
    scale 1.0, 384 px reference with VLM encoding, K/V cache, a 32 px boundary band redrawn
    to blend; known pixels kept at every flow step; "no hidden post-generation source
    composite". White mask pixels are redrawn, black kept. Weights: Krea 2 Community License;
    pipeline code Apache 2.0. Its responsible-use text forbids non-consensual intimate
    imagery and impersonation; adult content is neither allowed nor forbidden by it in so
    many words (Krea 2's own acceptable use policy was not read).
  - File `krea2_anypaint_rank32.safetensors`, 228,587,752 bytes, SHA-256
    `3a7d09f6b27f8ead160d340f2f59c11f4ee635c4a1ee87ffd8b1b9f9ba412f7a` (Hugging Face's LFS
    hash), into `models/loras` with one click; the GPU agent renames a download into place
    only when its SHA-256 matches (agent version 3; a mismatch deletes it and the job fails
    with "checksum").
  - ComfyUI nodes `github.com/alexw5702-afk/krea2-anypaint` (MIT, v0.1.0, commit
    `675be5a91eadbf8b8997b21c0e8e1848310b9571` of 2026-08-04; files and SHA-256:
    `__init__.py` `4c0f96b0…`, `nodes.py` `08559583…`, `anypaint_helpers.py` `b1ed902f…`,
    `krea2_registered_core.py` `0da66ac8…`, full hashes in `comfy/custom_nodes.py`). They
    import einops, numpy and Pillow (ComfyUI v0.35.0's own requirements) and ComfyUI
    internals that v0.35.0 has (`comfy/text_encoders/krea2.py` KREA2_TEMPLATE,
    `optimized_attention_masked`, `apply_rope`, `timestep_embedding`, `node_helpers`;
    checked in its source). Krea2AnyPaintPrepare (source, left/top/right/bottom, 
    reference_max_edge, boundary_redraw_px, generated_mask → semantic_reference, known_image,
    generated_mask, keep_mask, canvas size), Krea2AnyPaintEncode (clip, prompt, vae, the
    three images/masks, vlm_reference → positive, latent with its noise mask),
    Krea2AnyPaintModelPatch (model, kv_cache). The Mac downloads the pinned files from
    `raw.githubusercontent.com`, checks each by SHA-256 (a file that does not match is never
    sent) and passes them, gzipped, with the GPU's start command, which unpacks them into
    `/opt/comfyui-baked/custom_nodes` and the volume's `ComfyUI/custom_nodes` before ComfyUI
    starts (RunPod's start script copies the baked folder onto the volume on the first start;
    Modal runs ComfyUI from it). Nothing is installed with pip. An exception to "built-in
    nodes only", approved by the owner. Without them a region edit is refused ("rent the GPU
    again") — never run another way; every step's label names its method.
  - Graph: UNETLoader → LoraLoaderModelOnly (AnyPaint 1.0) → LoraLoaderModelOnly (the
    character) → ModelPatch → KSampler (8 steps, CFG 1, euler, simple); positive and latent
    from Encode, negative the Turbo zero-out. The result is pasted back 40 px past the mask
    (AnyPaint's band plus 8), so every pixel further away is the photo's own; the
    background's around the people's exact outline (feather 4), so they stay pixel by pixel.
  - The owner's numbers, **estimates** (not from AnyPaint's authors, who validated a full
    redraw, denoise 1.0, and AnyPaint alone): the figure (SAM 3.1 "body", "breasts", "hips
    and buttocks") denoise 0.85, the character LoRA at 0.65 of its strength, the mask grown
    48 px past the outline; the background denoise 1.0; "Unify the light" (off by default)
    0.2 over the whole photo afterwards (Turbo at 0.3 drew hatching and cracks, 2026-11-05;
    it also changes the person); the face in its crop at 0.65 with the LoRA at 1.0. Not yet
    run on a GPU.
- **Nodes** (built in, checked in ComfyUI v0.35.0's `nodes.py` and `comfy_extras`, the
  pinned rented image): text to image = UNETLoader, CLIPLoader (`krea2`),
  LoraLoaderModelOnly, CLIPTextEncode, ConditioningZeroOut (Turbo) or a negative (RAW),
  EmptyLatentImage, KSampler, VAEDecode, SaveImage. Edit = LoadImage → ImageScale (the
  photo's aspect ratio, `crop: disabled`) → VAEEncode; per region: CheckpointLoaderSimple
  (SAM 3.1) + CLIPTextEncode with its own text model (the region's words, e.g. "hair",
  "mouth and lips") → `SAM3_Detect` (threshold 0.4, 2 refine iterations, masks merged) →
  `GrowMask` 16 px → `MaskToImage` / `ImageBlur` 16 / `ImageToMask` (soft edge) →
  `SetLatentNoiseMask` with that soft mask and `DifferentialDiffusion` on the model (since
  2026-11-05; `comfy_extras/nodes_differential_diffusion.py`, input `model`, its optional
  `strength` left at 1.0; marked experimental, loaded by v0.35.0's `nodes.py`): the region is
  redrawn fully and its edge gradually, so no seam → KSampler at the region's strength →
  VAEDecode → `ImageCompositeMasked` with the same soft mask onto the scaled photo, so every
  pixel outside the region is the photo's own. Without a region: the same image to image on
  the whole photo (no mask, no DifferentialDiffusion).
- **Strengths** (denoise; the studio's own values, Krea publishes none): hair 0.85, mouth
  0.7, eyes 0.6, face 0.55 (0.75 when the person becomes the LoRA's character), expression
  0.6, clothing 0.85, background 0.9, body (the person) 0.45, whole photo 0.5; with a new
  colour hair 0.85 (0.75 left the old colour's strands as light streaks, three times on
  2026-11-05; 0.95 drew a hood over a hand and moved a hand to the face) and clothing 0.75; each correction adds 0.1 and a new seed (0.2–0.95). The
  user can set one strength instead.
- **The owner's tests on a GPU (2026-11-05, Krea 2 Turbo, 10 requests)**: every one was an
  edit with a character LoRA, "Whole photo" and analysis off — one image to image at 0.5.
  That keeps the photo's hair colour (blonde stayed blonde), its body and half its face, and
  the colour tint and face crop never ran. Two photos were small (548 × 640, 1016 × 1410)
  and were stretched to 2K, so the results were soft. The face check compared with the
  woman in the photo, not with the LoRA. What changed because of it is below.
- **Colour changes** (hair, clothing; since 2026-11-04 after the owner's first tests): Krea 2
  Turbo runs at CFG 1, so a negative prompt cannot push the old colour away and a masked
  redraw tends to keep it. The region is therefore tinted first: `EmptyImage` (`color` =
  0xRRGGBB, checked in v0.35.0 `nodes.py`) → `ImageBlend` (`comfy_extras/
  nodes_post_processing.py`, modes normal, multiply, screen, overlay, soft_light,
  difference): "multiply" for a darker colour (luma < 0.5) — at 0.85 since 2026-11-05, at
  1.0 since the owner's car photo the same night (0.85 left the blonde strands as lighter
  streaks, hair 42/28/23 with highlights at 54; 1.0 gave 32/21/17 and 43) — and "screen" at
  0.55 for a lighter one, +0.1 per correction up to 1.0 — tried on the owner's
  blonde photo: "normal" 0.55 washed the strands to a flat grey, multiply kept every strand
  and its light. A second blend "normal" at 0.5 towards the flat colour (tried 2026-11-05
  against blonde streaks) turned platinum hair into a flat brown patch that Krea 2 at 0.75
  did not redraw into strands, and was removed the same day → `ImageCompositeMasked` inside what SAM 3.1 found (not the grown rim, so no
  coloured halo on the skin) → VAEEncode. The
  colour is the plan's `color` (the assistant gives a hex value; without one the studio's
  own palette of hair tones, English and Slovak words next to "hair"/"vlasy",
  `compiler.HAIR_COLOURS`; in a sentence about the hair a colour up to six words after
  "to"/"into"/"na"/"do" wins — "her platinum blonde hair color to deep dark brown" had
  tinted the hair platinum before 2026-11-05), shown and editable in the plan (or removed). The colour is also
  put first in the edit's prompt with "one even natural colour from roots to ends without
  highlights" ("very dark brown, almost black hair, …" — the palette's dark brown, #281B14,
  is the LoRA characters' near-black brown the owner asked for), and the hair mask grows 32 px
  instead of 16, so no rim of the old colour stays. The eyes and brows are taken out of the
  hair (SAM 3.1 "eyes and eyebrows", grown 8 px, subtracted from the grown mask and from the
  tinted area): a strand over the face had pulled an eye into the whole-photo redraw and
  deformed it (owner's test, 2026-11-05). "Dark brown" is #281B14 since the same day — at
  #3A281D the hair came out at its tone (measured 54/37/27) and the owner wanted it darker.
- **A person becomes the LoRA's character** (an edit with a LoRA, switch on by default,
  since 2026-11-05; tuned the same night on the owner's A40 with his three photos — a car at
  night, lying on a bed, a tilted tongue-out selfie — and LoRA Taila 2, whose photos are 64 %
  alike in SFace): (1) the face first, on the photo itself, in its enlarged crop with the
  LoRA at 0.65 and only the request's sentences about the face (`compiler.region_prompt`);
  (2) the other edits (hair …), each with its own sentences. Measured: the face first reached 59–65 %
  on the car and the bed (41 % when it came after a masked redraw of the whole person); one
  pass at 0.75–0.85 and two at 0.6 gave the same, but only gentle passes kept the selfie's
  tongue-out expression (0.85 lost it; at 0.75 with the whole text, hair was drawn over the
  face). The person was not redrawn from then until 2026-11-06 (see "The figure"): with
  only the face left out, at 0.7 it lost an expression, drew a foot into a pillow and once a
  new close-up; at 0.6 an arm on the bed became a hand propping the head; at 0.45 it changed
  nothing visible. A light pass over the whole photo afterwards (0.3,
  tried and removed the same night) looked like one skin tone and light at small size, but
  up close Krea 2 Turbo — distilled for 8 steps from noise — drew hatching into the hair,
  cracks into the forehead and white dashes; the hair step before it was clean. A body edit and its
  face exclusion (SAM 3.1 "face" grown 8 px → `MaskComposite` "subtract", clamped 0–1; an
  empty mask when SAM 3.1 finds nothing, `comfy_extras/nodes_sam3.py` v0.35.0) remain for
  requests without a LoRA. The face is measured with SFace against the LoRA's training
  photos and is a check of its own ("face_match"), always — also with the checks off. It
  passes at 0.85 × how alike the LoRA's photos are to each other, never below SFace's
  threshold 0.363. A face short of it is redrawn again at the same 0.65 with a new seed, each
  try from the same image (tries built on each other added up artifacts on the selfie), at
  least two tries, and the most alike face is kept. The prompts leave out the request's
  "keep/preserve" sentences: "preserve … the exact mouth shape, lip position" opened the
  car photo's closed mouth and a correction added a tongue; without them the mouth stayed
  closed on two seeds and the selfie kept its tongue on two seeds — the photo keeps the
  expression, the words only drew mouths. Even so, at 0.6–0.65 some seeds opened the car
  photo's closed mouth (teeth, once a tongue). The face's mouth is therefore redrawn at
  half the strength: SAM 3.1 "mouth and lips" (grown 8 px) × `SolidMask` 0.5
  (`MaskComposite` "multiply") subtracted from the face's mask, which DifferentialDiffusion
  reads as a per-pixel strength. Three seeds each: the car's mouth stayed closed every time
  (54–61 % alike), the selfie kept its tongue every time.
  Next morning (2026-11-06, five photos) one seed still drew a tongue hanging over the chin
  even with the mouth left out entirely: Taila 2's dataset has one photo with the tongue
  out, so the LoRA draws it now and then. The face is therefore tried up to three times
  before the other edits, each from the same photo with a new seed; a try is not used when
  the area below the original's lips (YuNet's mouth corners, 0.8 face widths wide, down to
  0.45 face heights) turned clearly redder in more than 0.5 % of its pixels
  (`quality.new_mouth`: red − mean(green, blue) up by more than 25). Measured on the car:
  tries with a new tongue 1.1–2.1 %, tries with the closed mouth kept 0.0 % on six seeds.
  Of the remaining tries the most alike to the LoRA's photos is kept.
  The half-strength mouth was removed the same morning: on a photo whose face changed more
  (sitting on the floor) it left the old full lips on the new face, misplaced; with the
  mouth redrawn fully the lips looked natural.
  The redness measure was removed the same morning too: on more photos the LoRA's fuller,
  pinker lips raised it as much as a tongue (0–1.9 % without a tongue, 1.0–1.9 % with one;
  lower boxes missed the tongue), so it threw away good faces. The tries pick by likeness
  only. A LoRA trained with an expression in its photos (Taila 2: one tongue-out photo) can
  draw it now and then; that is fixed by leaving such a photo out of its dataset.
  The face is now the last step: the hair step after it changed the face in a mirror selfie
  and a photo on the floor (SFace 47 % → 28 % and 16 %); hair first and the face last gave
  58 %, 63 % and 66 % (car) with clean faces. The hair (redrawn at 0.85) keeps them: without
  "preserve the exact original wavy hairstyle, bangs" the selfie's waves became straight. YuNet misplaces
  the face of an extreme expression (the selfie: the box beside the mouth), so its score is
  low whatever the face looks like; the gentle tries then still keep the expression.
- **The person in one redraw** (2026-11-06, replaced the same day by AnyPaint steps; kept as
  a record — the owner's request: breasts, waist, hips
  like the LoRA's, the whole photo consistent; tried the same day on his A40 with his six
  photos — a car, a bed, a tongue-out selfie, a mirror selfie, a woman on the floor, a lift
  with a mirror — and five new ones made for the test). First a figure step and a hair step
  ran one after the other; the hair step then left torn patches of the old top beside the new
  hair (the selfie), smeared a face and drew a second hand on a cheek, and the figure step
  (person minus face, head, hair, hands and feet, 0.75) once drew a second hand on a chest.
  Now `graphs.person` redraws the whole person once with the LoRA: SAM 3.1 "person" plus
  "reflection of a person" (`MaskComposite` "add"; "person" left out the woman in the lift's
  mirror, the photo's subject; "reflection of a person" finds the person in ordinary photos
  too), grown 40 px so the outline can change, pasted back whole (`ImageCompositeMasked` with
  that soft area, not with the strengths, so nothing ghosts). DifferentialDiffusion reads the
  noise mask as a per-pixel strength, so the parts get shares of 0.85 (`SolidMask` ×,
  `MaskComposite` subtract/add, blurred with radius 31 because a jump in strength is drawn as
  a hard line): figure 0.82; the hair the text changes 1.0, reaching 24 px past what SAM 3.1
  found; clothing, hands and feet 0.5 (at full strength grey sweatpants came out beige, pink
  lace shorts a white thong and a second hand appeared; at 0.35 a hand stayed paler than the
  new arm); the face 0.4 (kept out entirely, its edge cut the new bangs with a hard line and
  left pieces of the old ones on the forehead); when only the hair's colour changes, a ring
  64 px around the face at most 0.55 (with the hair beside a turned face at full strength
  Krea 2 drew a second, frontal face into the hair next to the old one — the mirror selfie);
  not when the text restyles the hair (`compiler.restyles`: the ring kept a bob from becoming
  the ponytail asked for). Less than full strength on the hair left
  it blurred, because a near-black tint leaves no strands to keep, and still did not keep a
  woman's curls (0.9 and 0.65 tried): with this LoRA, whose photos all have straight hair, a
  new colour can straighten curls. The tint fades out over 12 px past what SAM 3.1 found
  (8 px further): with a hard edge the line between tinted and untinted strands stayed where
  the redraw is gentle. SAM 3.1 returns only its best match per text in ComfyUI v0.35.0
  unless the text gives a count (`comfy/text_encoders/sam3_clip.py`, `_parse_prompts`:
  "text:N"; `nodes_sam3.py` keeps the N best above the threshold), so the parts are "hair:2"
  (kept on the people, so a woman and her reflection both change and a stranger does not),
  "face:2", "clothing:4", "hands:4", "feet:4". ImageBlur takes a sigma of 0.1–10 only (a
  radius of 31 with sigma 15.5 was refused), so the sigma is clamped and the tests check
  every graph against the node limits read from the GPU.
- **Hair without a LoRA**: the hair step redraws a ring 72 px around the hair at 0.45 of its
  strength and pastes it back with the hair, so what lay under the old strands is drawn to
  match; the face (minus hair over it) and the eyes and brows are left alone. Tried on a new
  photo ("change her hair color to bright red"): red hair, the rest unchanged.
- **Choosing the face try** (2026-11-06): only scores SFace can stand behind (≥ 0.363)
  choose — on the tongue-out selfie every try scored 0.02–0.15, the choice was chance and
  took the one try that had closed the mouth; then the first try is kept. A later try must be
  0.08 more alike than the first: the most alike face tends to the LoRA's neutral look
  (a café photo's "keep her smile" lost the smile to a try 0.03 more alike).
- **Clothing**: SAM 3.1 looks for every piece of the garment the request names ("dress:3":
  an arm across a red dress split it and half of it stayed red); its new colour
  (after "to" in its own clause) tints it and is said first, at 0.85 (at 0.75 and with "her
  red dress" first a red dress tinted black came out dark red). A new colour may come with a
  new cut (the red dress came out as a black wrap dress).
- **Prompts per step without analysis**: each step gets the request's sentences about it; the
  figure and the clothing keep only their clauses (between commas and "and") — "her hair to
  dark brown and her red dress to a black dress" gives the clothing step only the dress; a
  colour belongs to its own clause. A step no sentence speaks of gets only who it is about
  ("lmxx2 woman", else "the woman"): the face crop had got "her red dress to a black dress"
  and the figure "a hotel bathroom with marble walls". The text alone may name what to change
  (hair, face, body, background, clothing, mouth, eyes, expression; "close her mouth", "make
  her smile"): no region has to be chosen then (it was refused before). With the LoRA's
  character the mouth, eyes and expression are part of the face step.
- **Text to image with a LoRA** (2026-11-06): eight hard prompts (a street from afar, a gym
  mirror selfie, a beach from above, a café profile, from behind, a laughing close-up, a bed
  with a laptop, a car at night) gave faces 42–66 % alike at LoRA strength 1.0 and 44–69 %
  at 1.3, with a fuller figure closer to the LoRA's and no artifacts, so the Advanced panel's
  LoRA strength starts at 1.3 for text to image (1.0 for edits; the user's choice wins). A face short of the
  goal is redrawn like an edit's LoRA face (its crop, 0.65, up to three tries) and the most
  alike image is kept, the generated one included; no face (from behind) = as generated.
  Krea 2 Turbo sometimes ignores parts of a prompt (two of three 18+ prompts came out
  dressed; "looking back over her shoulder" came out from behind).
- **The main face**: YuNet's largest face among those scored at least 0.9 × its best (the
  lift photo's frontal reflection 0.93 against the larger cut-off profile 0.76), for the
  face crop and the likeness (`faces.main_face`).
- **Without AI analysis** the regions the text asks to change are added to the chosen ones
  (since 2026-11-05): per sentence a word for the hair, face or body with a change word
  ("change", "match", "zmeň" …) or a hair colour, unless the sentence keeps something
  ("preserve", "maintain", "zachovaj" …). Order: whole, body, background, clothing, face,
  expression, eyes, mouth, hair.
- **Face regions** (mouth, eyes, face, expression): a mouth is a few latent pixels in a whole
  photo, which is why closed lips, teeth, the tongue and expressions came out badly. The face
  is found on this Mac (YuNet, the face-match detector; nothing is sent) and a square of 1.8×
  the face box (at least 320 px, so at most ~3.2× enlarged) is cropped (`ImageCrop`, x/y/
  width/height, `comfy_extras/nodes_images.py`, marked deprecated in v0.35.0 but registered),
  scaled to 1024 px, masked and redrawn there, scaled back and pasted at the same place with
  `ImageCompositeMasked` inside the feathered mask only — the head's pose and every other
  pixel stay the photo's. A close-up (the face already ≥ ~790 px) or no face found: the whole
  photo as before. A closed mouth is also said first in the prompt ("closed mouth, lips
  gently together, …") because Turbo ignores negative prompts; the plan states head pose and
  gaze and adds a check that the head is turned and tilted as in the original.
- **Plan**: the assistant (its Balanced model as chosen in Integrations) answers JSON — what to
  keep, what to change, the prompt (no contradictions: closed lips removes "open mouth",
  teeth, tongue), up to 6 edits and 10 checks. It is checked like every prompt
  (`services/safety.py`; 18+ only with the switch, minors always refused), shown and editable
  before the run. Tokens are recorded (purpose "advanced"); no prices are shown.
- **Checks** (estimates, never guarantees): the assistant answers per requirement on the
  result (and the original for an edit), passed or not with a note, and lists artifacts;
  SFace identity = the median similarity to the original photo (edit) or the LoRA's photos
  (text to image), shown against SFace's match threshold, faces under 64 px not compared;
  changed area = share of pixels that differ by more than 24 levels. The best attempt has the
  fewest failed checks, then the most passed, then the higher identity.
- **Failures**: a failed step, a refused or unreadable answer of the assistant, or a missing
  file / node fails the job with that message — nothing else is tried instead. LoRA
  compatibility is `krea2.lora_compatibility` (strict, as in Krea 2).
- **Not run on a real GPU**, with a real SAM 3.1 or against a real assistant: verified with
  the fake ComfyUI (nodes, files, the uploaded photo, the graphs sent) and the fake AI
  answers. How well SAM 3.1 finds a region and how well Krea 2 redraws a masked latent
  (hair colour, closed mouth, expression, the person with a LoRA at 0.45 without its face)
  is untested —
  likewise the tint modes and strengths, DifferentialDiffusion's edge, the face crop and
  the enlarging of small photos; the owner's manual tests on a GPU decide the default
  values. The face crop was checked with the real YuNet detector on test photos (lena,
  obama, messi5); the tint modes only as pixel blends on a photo, not through Krea 2.

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

- **Defaults from ai-toolkit itself** (checked 2026-11-03 in its `main`): its training form
  sets `train.timestep_type: "linear"` for Krea 2 RAW and Turbo, and 9 sample steps with
  guidance 1 for Turbo (`extensions_built_in/diffusion_models/ui.tsx`, entries "krea2" and
  "krea2:turbo"); differential output preservation is off by default
  (`ui/src/app/jobs/new/jobConfig.ts`: `diff_output_preservation: false`, steps 3000, lr 1e-4,
  rank/alpha 32, adamw8bit, caption dropout 0.05). The studio used `sigmoid` and preservation
  on before 2026-11-03; both now follow ai-toolkit (preservation stays a switch). The owner's
  runs (2026-11-03) also showed that captions naming the hair colour on every photo gave a
  weaker likeness than the AI captions, which leave identity out: the dataset check warns
  when more than half of the captions name it.

- **Faster runs** (Advanced, checked 2026-11-01 in ai-toolkit's `main`:
  `jobs/process/BaseSDTrainProcess.py`, `extensions_built_in/sd_trainer/SDTrainer.py`,
  `toolkit/config_modules.py`):
  - "Test images only at the end": `sample.sample_every: 0` (no step matches); ai-toolkit
    still samples once after the final step unless `disable_sampling`. The images do not
    change the LoRA.
  - Every run sets `train.skip_first_sample: true` (since 2026-11-03): no "baseline" images
    of the untrained model before step 1 — ai-toolkit made them again after every restart
    (out-of-memory retry), so one run showed the same step-0 images twice.
  - Sample files (checked 2026-11-03 in `main`, `toolkit/config_modules.py`
    GenerateImageConfig): `samples/<ms>__<step:09d>_<index>.jpg`, written in
    `samples/.tmp` and moved into place, plus a 300 px thumbnail
    `samples/.thumbs/<file>.jpg` for every image. The studio copies only the images directly
    in `samples/` that match the name pattern (the thumbnails showed up as extra images) and
    shows the newest step's set, one image per prompt.
  - "Protect the base model" off: no `diff_output_preservation` — SDTrainer then skips the
    prior prediction and the preservation prediction it runs every step, so steps are
    faster, but prompts with the class word may drift towards the character.
  - "Save memory while computing" off: `train.gradient_checkpointing: false` (ai-toolkit's
    default is true). Same training math, faster, much more VRAM. The owner's continued run
    on an 80 GB GPU (2026-11-02) stopped with "OOM during training step 3 times in a row,
    aborting training", so the studio now asks for 96 GB or more (still an estimate) and
    refuses a smaller GPU.
  - **Out of GPU memory** (torch "out of memory" / `OutOfMemoryError`, ai-toolkit "OOM during
    training step"): the run is not lost. The studio writes a config with more memory saving
    and starts the training again on the same GPU (already running; models and cached
    latents are there), at most twice, one change at a time, the cheapest for quality
    first (since 2026-11-03; before, quantization went on together with gradient
    checkpointing and an 80 GB run dropped to ~40 GB): 1) gradient checkpointing,
    2) batch 1 with the same effective batch (gradient accumulation × old batch),
    3) `model.quantize`/`quantize_te`, 4) `model.low_vram`. The version's settings show the
    change and the run shows a
    note; when nothing is left to save it fails with "The GPU ran out of memory…" (lower
    resolution or a bigger GPU). A checkpoint saved before the stop is where ai-toolkit
    resumes.

- **GPUs with 80 GB or more** (the owner's rule, 2026-11-03; `full_precision_gb` in
  `lora/presets.py`): the run uses the whole GPU — `model.quantize`, `quantize_te` and
  `low_vram` off (full precision), whatever the form says; the GPU's memory is the
  provider's figure from the GPU picker (`gpu_memory_gb`). Gradient checkpointing stays as
  chosen (on by default: off ran out of memory on 80 GB). How much memory a run then uses
  is not published (Krea 2 / ai-toolkit); the studio's estimate is ~48 GB plus activations.

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

AI captions (`lora/captions.py`): output budget 1024 tokens, because the token limit of newer
models also covers their reasoning (OpenAI `max_completion_tokens`, Anthropic `max_tokens` with
adaptive thinking, Gemini `maxOutputTokens`); with the old 200 a caption could stop
mid-sentence. When a provider still reports the limit (`finish_reason: length`, `max_tokens`,
`MAX_TOKENS`), the image is asked once more with 4096; if it is cut off again, only the
finished phrases are kept (up to the last comma or full stop), or the image gets a clear error.

# Face match (on this Mac)

`backend/app/services/faces.py`, verified 2026-11-02:
- Detector: `face_detection_yunet_2023mar.onnx` (232,589 bytes, SHA-256
  8f2383e4…2552fa4, MIT) — huggingface.co/opencv/face_detection_yunet at commit 3cc26e7f.
- Recognizer: `face_recognition_sface_2021dec.onnx` (38,696,353 bytes, SHA-256
  0ba9fbfa…7c34e79, Apache-2.0) — huggingface.co/opencv/face_recognition_sface at commit
  3d708243.
- Second source: media.githubusercontent.com/media/opencv/opencv_zoo/main/models/… (same
  files; the sandbox downloaded both from there and the hashes matched Hugging Face's LFS ids).
- OpenCV's `FaceDetectorYN` and `FaceRecognizerSF` (opencv-python-headless; 5.0.0.93 tested)
  as in the zoo's demo: the largest face, `alignCrop`, `feature`; the score is the cosine
  similarity, 0.363 or more = the same person (the zoo's `sface.py`). Checked here with real
  photos: the same face 0.91–0.96, a different person 0.13.

- **LoRA checks with the same models** (since 2026-11-03): "Check faces" on a LoRA's dataset
  (`POST /api/loras/{id}/dataset/faces`) gives every photo in use its median similarity to the
  other photos' faces; below 0.363 it is marked (another person, or a face from the side,
  which SFace compares less well). A run's validation images get a likeness per step
  (`GET /api/loras/{id}/versions/{v}/likeness`): each image's median similarity to the
  training photos (median, so one odd photo cannot decide it), the mean per step, and the
  best step; it is kept in the version's progress until new images arrive. The validation
  prompts (when the user gives none) are three portraits from the front with a neutral
  expression in different places (since 2026-11-04; before: a window portrait, a full-body
  street shot and a smile, which the face model compares less reliably), so runs with the
  default prompts can be compared with each other; other prompts give other numbers.
- **What the numbers mean** (measured 2026-11-03 with these models on the example photos of
  github.com/ageitgey/face_recognition): two different photos of one real person 0.69–0.74
  (Obama 0.744, Rose Leslie 0.700, Kit Harington 0.692), different people −0.12…0.26, a
  downscaled copy of the same photo 0.95 — so a raw score of 0.9 is not reachable with
  different photos. Next to the run's scores the studio shows the photos' own **baseline**,
  measured the same way (each photo's median similarity to the others, median over the
  photos), as a reference only; the scores are never scaled or converted (the owner's rule,
  2026-11-03: no derived ratings).
- Faces narrower than 64 px (e.g. full-body shots; messi5.jpg's 30 px face) are not compared:
  SFace aligns every face to 112 px and such a score is unreliable. "Check faces" also reports
  each face's width as a share of the photo and counts the photos where it is at least 0.2.

# A LoRA dataset from an AI model (hosted)

LoRA › Photos › "Create a dataset from an AI model" (`routes/loras.py`, since 2026-11-04):
the same hosted request as Generate with an AI model — its photos as references on the
model's editing endpoint (Nano Banana Pro, Seedream, Wan 2.7 Pro; only parameters the
catalog documents: 3:4 and the endpoint's 2K size where offered), its description per the
usual rule — once per shot of the studio's own list (`DATASET_SHOTS`: 24 shots, repeated for
36; with 18+ on Wan 2.7 Pro, three full-body shots without clothes). The price shown first is
the provider's estimate for one photo × the number of photos (Atlas: unknown, linked);
nothing starts without that confirmation (`confirmed`), and every request is checked before
the first one is sent. Not run against the real WaveSpeed / Atlas APIs (fake providers only).

# Spending this month

`backend/app/services/spending.py`: only numbers the providers gave. Hosted runs: the price the
user confirmed before the run (`Generation.cost_estimate`, WaveSpeed's own estimate; Atlas has
none, so its runs are only counted). GPUs: hours of the workloads the studio started × the
hourly price they started at (`GPUWorkload.meta.cost_per_hour`: RunPod's pod `cost`, Modal's
rates or list price). Modal also shows its real cost from `Workspace.billing.report(start,
resolution="h")` (SDK 1.6.3; the whole workspace, up to the last full hour) — **not yet run
against Modal's servers**. Older runs (before this) have no stored price.
