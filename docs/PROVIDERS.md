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
