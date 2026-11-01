# AI Image Studio — instructions for Claude Code

A local Mac app for AI image generation: FastAPI + PostgreSQL backend, Next.js frontend,
opened in its own Chrome window. Architecture: `docs/ARCHITECTURE.md`. Provider facts and
what is verified: `docs/PROVIDERS.md`. User guide: `README.md`.

Home of the code: GitHub `hladikmartin06-maker/ai-image-studio`, branch `main` (the app is
the whole repository). The scripts follow the pattern of `ai-model-studio/scripts/` in
`biznismarko9-source/tiqr-manager` (branch `claude/optimistic-euler-fvf8rc`); keep them the
same way. That branch and folder no longer exist on GitHub (checked 2026-10-31), so the
scripts in this repository are now the reference. Never push to `biznismarko9-source`
repositories.

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

- Image generation only. Hosted models: Nano Banana Pro (SFW), Seedream (SFW), Wan 2.7 and
  Wan 2.7 Pro (SFW + NSFW). Plus Krea 2 (RAW, Turbo) on the user's own ComfyUI
  (`backend/app/comfy/krea2.py` holds the verified facts). No other models or modes.
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
- No sign-in screen for now (the owner's choice): `AUTO_LOGIN` (default true) opens the
  first account on this Mac through `POST /api/auth/auto`, or creates one; password, sessions
  and sign out are hidden. `AUTO_LOGIN=false` in `backend/.env` brings sign-in back. Keep the
  localhost-only servers, the Host check (`proxy.ts`) and the client header, which protect it.
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
- ComfyUI: only the API format of workflows runs; secrets are scrubbed from imported and
  exported workflows. LoRA compatibility is strict (`krea2.lora_compatibility`): other
  architectures never load, RAW LoRAs work on RAW and Turbo, Turbo LoRAs only on Turbo; a
  missing model file or node is an error, never a silent substitute.
- LoRA training (ai-toolkit on RunPod/Modal through the studio agent, `gpu/agent`): billing
  needs `confirm_billing`; artifacts are copied to the Mac before the GPU is removed; progress
  and loss only from the trainer's output; LoRAs only for fictional adults.
- Three separate integration families, never mixed: image providers (`providers/`,
  `integrations`), AI chat providers (`ai/`, `ai_integrations`) and GPU compute
  (`gpu/`, `gpu_integrations`). A new chat or GPU provider = one registry entry + adapter.
- Assistant: chat model ids come from each provider's live models endpoint; tier defaults
  are preferences only. Claude goes through the official `anthropic` SDK. Keep the cost
  controls (local routing, tier caps, caching, summaries, context only when changed, local
  and cached answers). Never show prices that no API returns; show reported tokens.
- GPU: anything that starts billing needs `confirm_billing: true` from a UI confirmation.
- Simple for beginners (the owner's rule): short menu — Dashboard, Generate, AI Models, LoRA,
  Image Library, Assistant, Integrations, Settings. No Favorites. ComfyUI is chosen inside
  Generate (Krea 2) with only two options: "This Mac" and "Rent a GPU". LoRA is one page
  (projects, photos, captions, training, versions, tests, training history). GPU providers are
  connected in Integrations and picked only where they are used (Krea 2 on Generate, LoRA
  training) — no separate GPU page. Advanced options stay collapsed; no fields beginners do
  not need (e.g. trainer image, run notes). Training always removes the GPU afterwards.
- GPU lists show only NVIDIA GPUs with 24 GB or more (and at least what the task needs),
  smallest first (`MIN_GPU_GB` in `components/studio/common.tsx`).
- Krea 2 on Generate goes in steps: 1 choose and connect the GPU (progress, "Try again"),
  2 install the model with one click once ComfyUI answers, 3 AI model, LoRA, prompt and
  generate.
- Rented ComfyUI GPUs (the owner's design, `comfy/deploy.py`, sources in `docs/PROVIDERS.md`):
  ready in about 1–2 minutes, nothing installed at start. RunPod's image `runpod/comfyui:cuda12.8`
  (`allowedCudaVersions` 12.8, 12.9), `runpod/comfyui:cuda13.0` (13.0–13.3, only the ones
  RunPod's API spec accepts — it refuses the whole request otherwise) only for GPUs that
  offer CUDA 13 only. Start command `/bin/bash -c`: write the agent file (base64 env),
  start it in the background, `exec /start.sh`. Volume at /workspace. Only the agent's port
  8189/http is exposed (Bearer token per deployment); ComfyUI's 8188 stays private and listens
  on 127.0.0.1. Poll ~10 s; "Ready" only when ComfyUI answers through the agent; no answer in
  30 minutes (Modal 45) stops the GPU. Models: the agent downloads them on the GPU from
  huggingface.co only, as .part then renamed, into `/workspace/runpod-slim/ComfyUI/models/…`,
  only after ComfyUI answers. Modal: the same image with `.entrypoint([])`, command, agent
  and a Volume at /workspace.
- Hugging Face: token in Integrations › Hugging Face, checked with `whoami-v2` and the Krea 2
  repositories' `auth-check` (`services/huggingface.py`). Accepting the license is the user's
  own click on huggingface.co (no API exists); the studio links there and checks again, and
  training checks it before renting a GPU.
- Image Library shows generated images and AI model photos. AI models hold up to 200 photos.
- Assistant photos: up to 4 per message, the user's own uploads only, sent downscaled and only
  with that message.

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
- Commit and push only after the owner approves it. When the work is done and all checks
  pass, show the result and ask for approval; never commit or push without it.
- Say plainly what was not tested against the real WaveSpeed / Atlas API.
- End every reply with the launch command:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/hladikmartin06-maker/ai-image-studio/main/scripts/install.sh)"
```
