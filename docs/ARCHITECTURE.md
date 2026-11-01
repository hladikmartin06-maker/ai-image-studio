# Architecture

```
Browser (Chrome --app window)
   │  http://localhost:3000   (same origin; httpOnly session cookie)
   ▼
Next.js 16 (127.0.0.1:3000) ── /api/* rewrite ──▶ FastAPI (127.0.0.1:8000)
   React 19 + TanStack Query                         │
   server-sent events /api/events ◀──────────────────┤ in-process event broker
                                                     │ job worker (asyncio)
                                                     ├──▶ WaveSpeed / Atlas Cloud (HTTPS)
                                                     ├──▶ PostgreSQL 16 (metadata)
                                                     └──▶ backend/storage/<user>/ (files)
```

## Backend (`backend/app`)

| Path | Role |
|---|---|
| `main.py` | FastAPI app, error format `{error: {code, message}}`, CSRF dependency, worker lifespan |
| `config.py` | Settings from env / `backend/.env` |
| `db.py`, `models.py` | SQLAlchemy 2 (async, psycopg 3), tables below |
| `security.py` | scrypt passwords, session tokens (SHA-256 stored), AES-256-GCM for API keys |
| `deps.py` | current user from the session cookie, `x-studio-client` CSRF header check |
| `catalog.py` | the four models, one `Variant` per provider endpoint, validation, request bodies |
| `providers/` | `base.py` contract + error normalization, `wavespeed.py`, `atlas.py` |
| `services/generation.py` | validates a request against the catalog and creates the job |
| `services/worker.py` | runs jobs: upload references, submit, poll, download, store, publish |
| `services/storage.py` | originals byte-for-byte + 640 px WebP thumbnails, path-safe |
| `services/references.py` | reference validation, SHA-256 de-duplication, provider upload reuse (24 h), clean-up |
| `services/personas.py` | AI models: profile schema, prompt text, adult-only checks, which references go with a request |
| `services/events.py` | per-user publish/subscribe for live updates |
| `services/safety.py` | prompt checks before anything is sent |
| `services/preferences.py` | the Settings page document |
| `routes/` | auth/account, integrations, reference images, AI models (`personas.py`), generations/history, images, misc, ai, gpu |
| `ai/registry.py` | chat providers, cost tiers, model resolution from the live model lists |
| `ai/providers/` | `anthropic_provider.py` (official SDK), `openai_compat.py` (OpenAI, xAI, any compatible API), `gemini.py` |
| `ai/router.py` | local complexity routing → tier, output cap, effort |
| `ai/context.py` | cacheable system prompt, studio context, history window, summaries |
| `ai/service.py` | one assistant turn: local/cached/streamed answer, attached photos (downscaled, sent only with their message), usage, background summary |
| `gpu/` | `base.py` contract, `runpod.py` (REST + GraphQL), `modal_provider.py` (SDK), `registry.py`, `service.py` (credentials, launch + record, cost) |
| `gpu/agent/` | `studio_agent.py`: stdlib HTTP agent that runs inside a GPU container (files with offsets/ranges, jobs: train / models / download — downloads from huggingface.co only, logs, the container's start log, ComfyUI proxy), sent compressed in the environment; `COMFY_BOOT` starts it in RunPod's ComfyUI image before the image's own `/start.sh` |
| `gpu/agent_client.py` | client for the agent (chunked resumable upload, ranged resumable download with SHA-256) |
| `comfy/` | `client.py` (ComfyUI HTTP API), `krea2.py` (verified facts, file detection, sizes, workflow, LoRA compatibility), `workflows.py` (import, secret scrubbing, bindings, LoRA insertion), `service.py` (connections, inspection, LoRA placement), `generation.py`, `runner.py` (in the job worker), `deploy.py` (ComfyUI on RunPod/Modal) |
| `lora/` | `presets.py` (per-architecture options and presets), `aitoolkit.py` (job file), `dataset.py` (analysis, dHash, packing with crop), `captions.py` (vision captions via the assistant provider), `safetensors_meta.py`, `runner.py` (training state machine), `serial.py` |
| `services/gpu_runner.py` | background loop that advances training runs and deployments (state in the database) |
| `services/uploads.py` | chunked uploads (8 MB pieces) for LoRA and model files |
| `services/huggingface.py` | Hugging Face token check (`whoami-v2`) and access to the gated Krea 2 repositories (`auth-check`), cached; used by Integrations › Hugging Face and before training rents a GPU |

### Database

| Table | Content |
|---|---|
| `users` | email, name, password hash |
| `auth_sessions` | token hash, user agent, expiry |
| `user_settings` | preferences JSON |
| `integrations` | provider, encrypted key, last 4 characters, status, last error |
| `reference_images` | uploaded files (SHA-256, one row per distinct file), per-provider upload URLs and their upload time |
| `personas` / `persona_images` | AI models (saved fictional adult characters): name, description, private notes, appearance JSON, primary image; their reference images (links to `reference_images`, so a duplicate shares files) |
| `generations` | the history entry: model, provider, endpoint, prompt (and `final_prompt` as sent with an AI model's description), settings, request body, AI model used (`persona_id`, `persona_meta`), status, error, timestamps, idempotency key |
| `provider_tasks` | one provider request each: status, provider id, outputs, error |
| `images` | file paths, size, dimensions, favorite (kept in the data, no longer shown), in-library flag, NSFW flag |
| `ai_integrations` | chat provider, encrypted key, base URL (compatible APIs), cached model list, tier choices |
| `chat_conversations` / `chat_messages` | conversations (title, rolling summary) and messages with the exact text sent, model, tier, tokens |
| `ai_usage_events` | every chat/summary call and every free (local/cached) answer |
| `gpu_integrations` / `gpu_workloads` | encrypted GPU credentials; workloads started from the studio |
| `user_secrets` | other encrypted credentials (Hugging Face token) |
| `comfy_connections` | ComfyUI servers: kind, URL, encrypted auth or agent token, inspection result, deployment state |
| `comfy_workflows` | imported workflows (API format runs, UI format export only), bindings |
| `loras` / `lora_dataset_items` / `lora_versions` | LoRA projects, dataset (links to reference images + caption, split, crop, dHash), training runs and imported files (config, ai-toolkit config, GPU, progress, artifacts, deployments) |

Images are files on disk; the database stores only relative paths. Every query filters by
`user_id`.

### Generation job lifecycle

```
POST /api/generations ─▶ queued ─▶ processing ─▶ completed (≥1 image; errors kept as note)
       (validated,         │                └──▶ failed (no image)
        idempotent)        └──▶ cancelled (only while queued)

provider task:  pending ─▶ submitting ─▶ processing ─▶ completed | failed
```

- The browser sends a `client_request_id` per click; a repeat returns the same job.
- A task is sent at most once. After a restart, tasks with a provider id are polled again;
  a task stuck in `submitting` (the provider may have accepted it) is failed as
  `interrupted` instead of being resent.
- Polling: 1.5 s growing to 6 s, transient errors retried, 15 min timeout.
- Outputs are downloaded at once (provider URLs expire), stored, thumbnailed, and each
  change is pushed to the browser over server-sent events. The browser falls back to
  polling active jobs every 4 s when the stream is unavailable.

### AI models in a generation

```
POST /api/generations  {…, persona: {id, reference_image_ids, use_references, use_description}}
  edit endpoint for this model + provider?  no ──▶ description only (text-to-image)
  room = endpoint's max references − the user's own references
  AI model images (primary first, in the user's order) up to that room; they go first
  description on? ──▶ final_prompt = prompt + "Character: adult …" (+ "keep the same person …"
                      when references are attached); prompt rules run on the final text
  history keeps the AI model's name and the references it contributed; "regenerate" repeats
  the request exactly as sent, even after the AI model was edited
```

A reference file is stored once per user (SHA-256) and uploaded to a provider once; the
upload URL is reused for 24 h (how long providers keep uploads is not documented), then the
file is uploaded again. Deleting an AI model or one of its images deletes the file unless
another AI model links it or a running generation needs it.

### ComfyUI and LoRA training

```
Generate (ComfyUI · Krea 2) ─▶ POST /api/comfy/generate ─▶ Generation(provider "comfyui")
  worker: place the LoRA file on the connection (folder copy / agent upload, once)
          prompt_id chosen and saved, then POST /prompt (a run that may have been accepted
          is looked up after a restart, never resent) ─▶ /queue + /history (+ /ws progress)
          ─▶ /view ─▶ library (model, LoRA, prompt, seed, workflow kept in comfy_meta)

LoRA run: queued ─▶ create pod/sandbox (ai-toolkit image + agent, confirm_billing)
          preparing (agent health) ─▶ uploading (dataset copy + captions + config.json)
          running (agent job "train": python run.py; step/loss parsed, logs, samples)
          downloading (final + checkpoints, ranged, SHA-256) ─▶ completed ─▶ GPU removed
```

ComfyUI on a GPU: POST /api/comfy/deployments (confirm_billing) ─▶ pod / sandbox with
          runpod/comfyui:1.4.1-comfyuiv0.35.0 (cuda12.8, else cuda13.0) and COMFY_BOOT
          (agent, then /start.sh)
          starting (agent /health) ─▶ booting (ComfyUI through the agent) ─▶ ready
          ─▶ POST …/models/install: agent job "models" downloads from huggingface.co
          (Krea 2, or variant "pgupscaler" for Enhance photo)

Enhance photo: POST /api/comfy/enhance ─▶ Generation (model pgupscaler, mode image-edit)
          worker: upload the photo (POST /upload/image) ─▶ PGUpscaler workflow (comfy/
          enhance.py, built-in nodes only) ─▶ /prompt ─▶ result saved like Krea 2
          (no answer within 30 min, Modal 45: the GPU is stopped)

ComfyUI deployed by the studio is never exposed: only the agent port is, and every request
needs the agent token. Files in a container are copied to this Mac before the workload is
removed; RunPod keeps a pod volume across stop/start, Modal a named Volume (ComfyUI's models
folder; on Modal ComfyUI runs from the image, `COMFY_BOOT_MODAL`).

### Assistant turn

```
POST /api/ai/conversations/{id}/messages  (text, studio context, quick action)
  local answer?  ──yes──▶ stream it (0 tokens)
  route locally: simple|complex × mode → tier → model of the connected provider
  context: system prompt (stable, cached) + summary + recent turns within a token budget
           + studio context only when it changed
  identical request answered in the last 24 h? ──yes──▶ cached answer (0 tokens)
  stream from the provider → SSE start / delta… / done (usage) — Stop keeps the partial text
  history outgrew the window → low tier folds older turns into the summary (background)
```

The Generate page publishes its draft to `lib/studio-bridge.ts`; prompt and settings blocks
in answers can be applied back to it ("Use in studio", "Apply settings").

## Frontend (`frontend/src`)

| Path | Role |
|---|---|
| `proxy.ts` | refuses requests with a foreign `Host` (DNS rebinding protection) |
| `app/welcome` | sign in / create account (only with `AUTO_LOGIN=false`; by default the shell signs in automatically through `POST /api/auth/auto`) |
| `app/(studio)/…` | dashboard, generate (online models or Krea 2 with the ComfyUI choice inside), ai-models, loras (list + training history; `[id]` photos, captions, training, runs/versions, tests, details), library (generated images or AI model photos), library/history, assistant, integrations (images, AI, GPU, Hugging Face), settings |
| `components/comfy`, `components/lora`, `components/studio` | ComfyUI connection bar (This Mac / Rent a GPU), manage window, workflows, Krea 2 panel on Generate; dataset, captions, simplified training form, runs, comparison; shared status, GPU picker |
| `components/personas` | AI model picker and Generate section, profile form, reference collection, "Add to AI model" |
| `components/assistant` | chat panel, safe Markdown with prompt/settings action cards, floating drawer |
| `components/integrations` | image, AI and GPU provider management; Hugging Face token and Krea 2 license check |
| `components/ui` | design-system components (button, select, popover, modal, confirm, switch, …) |
| `components/layout` | shell, sidebar, user menu, page header |
| `components/images` | justified grid, tile, viewer, browser (filters + bulk), AI model photos, actions |
| `components/generate` | model picker, prompt editor, references, settings controls, job card |
| `lib/api.ts`, `lib/queries.ts` | fetch wrapper and TanStack Query hooks (cache, optimistic updates) |
| `lib/events.ts` | live updates (SSE → query cache) |
| `lib/i18n` | English and Slovak |

### Design system

Premium Noir: deep black / anthracite / graphite surfaces, white and neutral-gray text,
violet accent only for primary actions, selection and progress. Tokens are CSS variables in
`app/globals.css` with a full light theme. System font stack (SF Pro on macOS), 10–18 px
radii, quiet borders, short ease-out transitions, honouring "reduce motion". Grid density
changes the justified-row height. Every list has skeleton, empty and error states.
