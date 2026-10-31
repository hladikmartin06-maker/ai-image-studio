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
| `ai/service.py` | one assistant turn: local/cached/streamed answer, usage, background summary |
| `gpu/` | `base.py` contract, `runpod.py` (REST + GraphQL), `modal_provider.py` (SDK), `registry.py` |

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
| `images` | file paths, size, dimensions, favorite, in-library flag, NSFW flag |
| `ai_integrations` | chat provider, encrypted key, base URL (compatible APIs), cached model list, tier choices |
| `chat_conversations` / `chat_messages` | conversations (title, rolling summary) and messages with the exact text sent, model, tier, tokens |
| `ai_usage_events` | every chat/summary call and every free (local/cached) answer |
| `gpu_integrations` / `gpu_workloads` | encrypted GPU credentials; workloads started from the studio |

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
| `app/welcome` | sign in / create account |
| `app/(studio)/…` | dashboard, generate, ai-models (list, `[id]` detail), assistant, library, library/history, favorites, compute (GPU), integrations (image / AI / GPU tabs), settings |
| `components/personas` | AI model picker and Generate section, profile form, reference collection, "Add to AI model" |
| `components/assistant` | chat panel, safe Markdown with prompt/settings action cards, floating drawer |
| `components/integrations` | image, AI and GPU provider management |
| `components/ui` | design-system components (button, select, popover, modal, confirm, switch, …) |
| `components/layout` | shell, sidebar, user menu, page header |
| `components/images` | justified grid, tile, viewer, browser (filters + bulk), actions |
| `components/generate` | model picker, prompt editor, references, settings controls, job card |
| `lib/api.ts`, `lib/queries.ts` | fetch wrapper and TanStack Query hooks (cache, optimistic favorites) |
| `lib/events.ts` | live updates (SSE → query cache) |
| `lib/i18n` | English and Slovak |

### Design system

Premium Noir: deep black / anthracite / graphite surfaces, white and neutral-gray text,
violet accent only for primary actions, selection and progress. Tokens are CSS variables in
`app/globals.css` with a full light theme. System font stack (SF Pro on macOS), 10–18 px
radii, quiet borders, short ease-out transitions, honouring "reduce motion". Grid density
changes the justified-row height. Every list has skeleton, empty and error states.
