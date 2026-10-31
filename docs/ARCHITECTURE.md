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
| `services/events.py` | per-user publish/subscribe for live updates |
| `services/safety.py` | prompt checks before anything is sent |
| `services/preferences.py` | the Settings page document |
| `routes/` | auth/account, integrations, reference images, generations/history, images, misc |

### Database

| Table | Content |
|---|---|
| `users` | email, name, password hash |
| `auth_sessions` | token hash, user agent, expiry |
| `user_settings` | preferences JSON |
| `integrations` | provider, encrypted key, last 4 characters, status, last error |
| `reference_images` | uploaded files, per-provider upload URLs (uploaded once) |
| `generations` | the history entry: model, provider, endpoint, prompt, settings, request body, status, error, timestamps, idempotency key |
| `provider_tasks` | one provider request each: status, provider id, outputs, error |
| `images` | file paths, size, dimensions, favorite, in-library flag, NSFW flag |

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

## Frontend (`frontend/src`)

| Path | Role |
|---|---|
| `proxy.ts` | refuses requests with a foreign `Host` (DNS rebinding protection) |
| `app/welcome` | sign in / create account |
| `app/(studio)/…` | dashboard, generate, library, library/history, favorites, integrations, settings |
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
