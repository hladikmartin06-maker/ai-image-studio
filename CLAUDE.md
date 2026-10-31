# AI Image Studio — instructions for Claude Code

A local Mac app for AI image generation: FastAPI + PostgreSQL backend, Next.js frontend,
opened in its own Chrome window. Architecture: `docs/ARCHITECTURE.md`. Provider facts and
what is verified: `docs/PROVIDERS.md`. User guide: `README.md`.

Home of the code: GitHub `hladikmartin06-maker/ai-image-studio`, branch `main` (the app is
the whole repository). The scripts follow the pattern of `ai-model-studio/scripts/` in
`biznismarko9-source/tiqr-manager` (branch `claude/optimistic-euler-fvf8rc`); keep them the
same way. Never push to `biznismarko9-source` repositories.

## How the app runs (do not change without the owner)

- Runs locally on the owner's Mac: backend Python FastAPI + PostgreSQL 16, frontend Next.js.
  Both servers listen only on 127.0.0.1 — backend port 8000, web port 3000 — never on the
  network. The browser talks only to Next.js; `/api/*` is proxied to the backend.
- Install and update with one Terminal command (the launch command below). `scripts/install.sh`
  installs Homebrew, Node.js, uv and PostgreSQL 16, downloads the project into
  `~/ai-image-studio`, prepares the database, builds the app, starts it in the background and
  opens it. Running it again = update (safe).
- `scripts/mac-app.sh` creates `AI Image Studio.app` in Applications (own icon,
  `scripts/assets/AppIcon.png`). Clicking it starts the servers when they are not running and
  opens Google Chrome with `--app=http://localhost:3000` (window without tabs or address bar);
  without Chrome it opens the default browser.
- Scripts: `install.sh`, `setup.sh`, `start.sh`, `stop.sh`, `mac-app.sh`, `dev.sh`,
  `assets/launcher.sh` (body of the app), `lib/env.sh` (shared). They must run on macOS's
  bash 3.2 and pass shellcheck. Logs go to `~/Library/Logs/AI Image Studio`.
- `GET /api/health` returns `"app":"ai-image-studio"`; scripts use it to tell this studio
  apart from another program on port 3000.

## Product rules

- Image generation only. Exactly four models: Nano Banana Pro (SFW), Seedream (SFW),
  Wan 2.7 (SFW + NSFW), Wan 2.7 Pro (SFW + NSFW). No other models or modes.
- Providers: WaveSpeed and Atlas Cloud only, through `backend/app/providers/` (one adapter
  each, shared contract in `base.py`). The model catalog is `backend/app/catalog.py`: one
  `Variant` per provider endpoint. A UI control exists only when the provider documents that
  parameter for that endpoint; `normalize_settings` drops everything else.
- Never invent endpoints, model ids, parameters, responses, prices or balances. Verify from
  the provider's docs, record the source and status in `docs/PROVIDERS.md`, and say clearly
  what is unverified.
- Never send a provider request twice when it may have been accepted (it could be charged
  twice): a task interrupted while "submitting" is failed as `interrupted`, not resent.
- API keys only in the backend: encrypted with AES-256-GCM (key derived from
  `APP_SECRET_KEY` in `backend/.env`), never returned to the browser, never in code, logs,
  docs or generation history. The browser shows only the masked last four characters.
- Every query on user data filters by `user_id`. State-changing requests need the
  `x-studio-client: web` header (CSRF guard).
- No fake results or fabricated statistics; empty, loading and error states everywhere.
- Prompts with sexual content and minor-related terms are always refused; explicit prompts
  are refused on the SFW-only models (`services/safety.py`).
- AI Models (`personas`, page `/ai-models`) are saved fictional adult characters, not image
  models. Creating one needs the adult confirmation; profile text with minor-related terms is
  refused; age ≥ 18. Their description is appended to the prompt only when the user keeps it
  on (History shows `final_prompt`). Their reference images are sent only through an editing
  endpoint and within its `max_reference_images`, primary first, after the user's own.
- Three separate integration families, never mixed: image providers (`providers/`,
  `integrations`), AI chat providers (`ai/`, `ai_integrations`) and GPU compute
  (`gpu/`, `gpu_integrations`). A new chat or GPU provider = one registry entry + adapter.
- Assistant: chat model ids come from each provider's live models endpoint; tier defaults
  are preferences only. Claude goes through the official `anthropic` SDK. Keep the cost
  controls (local routing, tier caps, caching, summaries, context only when changed, local
  and cached answers). Never show prices that no API returns; show reported tokens.
- GPU: anything that starts billing needs `confirm_billing: true` from a UI confirmation.

## Code rules

- Inspect existing code first; keep the structure (routes / services / providers, Alembic
  migrations, `frontend/src/lib/api.ts` + `lib/queries.ts`, components in
  `src/components/ui`). No unnecessary dependencies; no unrelated changes.
- Design system: tokens in `frontend/src/app/globals.css` (dark default + light). Reuse the
  components in `src/components/ui/index.tsx`.
- Every UI string in both `src/lib/i18n/en.ts` and `sk.ts` (the type enforces it).
- New DB change = a new numbered Alembic migration with a downgrade.
- Add or update tests for every change (`backend/tests`, fake providers in `tests/fakes.py`).
- Before reporting done, run:
  `cd backend && uv run pytest -q && uv run ruff check . && uv run ruff format --check . && uv run alembic check`
  and `cd frontend && npx tsc --noEmit -p . && npx eslint src && npx next build`,
  and `cd scripts && uvx --from shellcheck-py shellcheck -x -s bash *.sh lib/env.sh assets/launcher.sh`.
  Cloud sandbox: `service postgresql start` first. A local fake provider for trying the whole
  app: `backend/tests/fake_server.py` (instructions in its docstring) — never in production.
- Commit messages in English, no model names.

## Owner

- Answer in Slovak, short and simple.
- No progress messages while working; deliver only the result.
- Commit and push to `main` right away when the work is done and all checks pass (the owner
  approved this permanently).
- Say plainly what was not tested against the real WaveSpeed / Atlas API.
- End every reply with the launch command:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/hladikmartin06-maker/ai-image-studio/main/scripts/install.sh)"
```
