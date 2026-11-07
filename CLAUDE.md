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

- Image generation, plus video with Wan 3.0 (the owner's request, 2026-11-02). Hosted models:
  Nano Banana Pro (SFW), Seedream (SFW), Wan 2.7 and Wan 2.7 Pro (SFW + NSFW), and the video
  models Wan 3.0 and Wan 3.0 Prime (SFW; WaveSpeed only; video from text, or from a photo as
  the first frame with an optional second photo as the last frame; 2–30 s, sound). Plus
  Krea 2 (RAW, Turbo) on the user's own ComfyUI (`backend/app/comfy/krea2.py` holds the
  verified facts), "Enhance photo" and Advanced (Krea 2 Turbo edit, below). No other models
  or modes. Videos are stored as delivered (MP4), shown with a player, and never used where a
  photo is needed (references, AI models, LoRA photos, Krea 2, Enhance photo, Advanced).
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
  are refused on the SFW-only models (`services/safety.py`) and on Wan 2.7, Wan 2.7 Pro and
  Krea 2 unless their "18+ content" switch is on (off by default; `adult` in the request,
  never sent to a provider). 18+ runs, and Krea 2 runs that start from an 18+ image, mark
  their images `nsfw` (blurred per Settings).
- Hosted runs cost money: Generate and Repeat first show the price and start only after the
  user confirms (`POST /api/generations/estimate`, `services/pricing.py`). Prices only from a
  provider API (WaveSpeed `model/pricing`, list `base_price`), always labelled estimates;
  Atlas Cloud has none, so "Price unknown" with a link to its pricing page.
- Reference images are sent in the order shown; clicking a photo makes it the first one.
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
  and loss only from the trainer's output; LoRAs only for fictional adults. Out of GPU memory
  = train again on the same GPU with more memory saving (`runner.memory_fallback`, at most
  twice, one change at a time: checkpointing, batch, quantization, low-VRAM), never a new
  GPU. GPUs with 80 GB or more always train in full precision (`full_precision_gb`; no
  quantization or low-VRAM). No baseline validation images; the interval is the user's choice (every 100/250/500/1000
  steps or only at the end); the newest set is shown, earlier steps on demand. AI captions
  run one image at a time and each shows as soon as it is written.
  Training defaults follow ai-toolkit's own Krea 2 choices (`timestep_type` linear,
  preservation off; sources in `docs/PROVIDERS.md`). Consistency checks with the face models:
  dataset "Check faces" (`/dataset/faces`), likeness per step of the validation images
  (`/versions/{v}/likeness`, best step → save its checkpoint as a version), and a warning when
  most captions name the hair colour. A finished version
  can be continued (`continue_from`: its file → `network.pretrained_lora_path`, same
  rank/alpha, all current photos, a new version). "Copy to a new LoRA"
  (`POST /api/loras/{id}/copy`): a new project with its own trigger word and base model, the
  same photos (shared files), marks, crops and optionally captions with the trigger replaced;
  no versions are copied (the base model of a trained project stays fixed).
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
  2 install the model with one click once ComfyUI answers (Turbo or RAW, files from
  `huggingface.co/Comfy-Org/Krea-2`), 3 AI model, LoRA, start from text or a photo (image to
  image: built-in nodes, `denoise` 0.1–0.95, size from the photo's aspect ratio), prompt and
  generate. Runs on one ComfyUI connection form a queue (`worker._run_comfy`): one at a time,
  in the order started; files (LoRA, photo) go to the GPU one upload per file at a time.
- RunPod through its REST API v2 (`api.runpod.io/v2`; v1 is retired 2026-11-15). GPU lists
  and CUDA versions come from its catalog (`/v2/catalog/gpus?include=AVAILABILITY&product=
  POD&cloud=…`): only GPUs with a free machine are offered (Community Cloud with its own price
  only when Secure has none); `allowedCudaVersions` match exactly, so only free versions are
  sent; no data center, vCPU/RAM minimums or network volume.
- Rented ComfyUI GPUs (the owner's design, `comfy/deploy.py`, sources in `docs/PROVIDERS.md`):
  ready in about 1–2 minutes, nothing installed at start. RunPod's image pinned to
  `runpod/comfyui:1.4.1-comfyuiv0.35.0-cuda12.8` (free 12.8/12.9 machines),
  `…-cuda13.0` (free 13.x) only for GPUs that offer CUDA 13 only. Start command `/bin/bash -c`: write the agent file (base64 env),
  start it in the background, `exec /start.sh`. Volume at /workspace. Only the agent's port
  8189/http is exposed (Bearer token per deployment); ComfyUI's 8188 stays private and listens
  on 127.0.0.1. Poll ~10 s; "Ready" only when ComfyUI answers through the agent; no answer in
  30 minutes (Modal 45) stops the GPU. Models: the agent downloads them on the GPU from
  huggingface.co only, as .part then renamed, into `/workspace/runpod-slim/ComfyUI/models/…`,
  only after ComfyUI answers. Modal: the same image with `.entrypoint([])`, agent,
  `cpu=4, memory=32768`, one Volume "ais-comfyui" at /workspace, `workdir="/"`, `timeout` =
  session + 10 min; created in the background and found again by its tag, never twice; "no
  free GPU" (Insufficient capacity) = wait and ask again (`modal_provider.CAPACITY_*`). Its
  start command (`COMFY_BOOT_MODAL`) runs ComfyUI from the image (not `/start.sh`) with only
  `models` on the Volume; an ended sandbox's exit code and last lines are shown.
- Prices before anything starts: RunPod live from the catalog; Modal live from the workspace's
  rates (`billing.rates()` in the SDK; names and units undocumented, read defensively in
  `gpu/modal_provider.parse_rates`), falling back per GPU to a fixed table of its list prices,
  labelled an estimate with a link to modal.com/pricing.
  The confirmation shows the hourly price, the session's maximum (hourly × session length,
  capped by the spending limit) and storage; the confirmed price is sent and a changed price
  refuses the start (`price_changed`). While running: "spent so far" and "time left"
  (estimates); the GPU stops at the session limit, the spending limit or after inactivity.
- Enhance photo (Generate › "Vylepšiť fotku", `comfy/enhance.py`): the workflow of the free
  RunPod template "PG - Free Upscaler" (PGUpscaler: Z-Image Turbo skin fix, optional SAM 3.1
  face mask, SeedVR2 7B upscale 1.5/2/3×; strength 0.15–0.4 like the template's picker;
  editable prompt prefilled with the template's text and face-only target, checked like
  prompts) rebuilt in API
  format from built-in ComfyUI nodes
  only — the template's setup.sh and custom nodes are not used. Runs on the same ComfyUI
  connection as Krea 2. Its 6 model files come only from `huggingface.co/AIPRIVATEGROUP/
  pgupscaler` through the agent (one click). The photo is the user's upload, an AI model photo
  or a generated image; it is uploaded to that ComfyUI only for the run. Image viewers have an
  "Enhance photo" action and show its results before/after (slider or side by side).
- Advanced (Generate › Advanced, the owner's request of 2026-11-07, `comfy/k2edit.py`,
  `comfy/advanced.py`): Krea 2 Turbo only, with the owner's Krea 2 LoRA; isolated from the
  Krea 2 tab (its graphs are not changed). Edit a photo from plain words with the in-context
  edit LoRA `krea2_general_edit_v2_r128` (conradlocke/krea2-identity-edit, pinned revision,
  SHA-256 checked on the GPU; V2 license = personal / non-commercial, say so in the UI) and
  the comfyui-krea2edit v2.0.0 nodes (pinned commit, SHA-256 checked on the Mac, sent with
  the GPU's start command, `comfy/custom_nodes.py`); or text to image. With a character LoRA
  the instruction asks for the character first (hair from the AI model when filled in) and
  a small face is redrawn close up (YuNet on the Mac, denoise 0.4); untouched parts keep the
  original's pixels (on by default); "Face only" edits the face crop and keeps the photo's
  hands (SAM 3.1); "Body" frees one part found by SAM 3.1 (two words per part, `ref_boost`
  0.05 there); "Strong" = lower `ref_boost` (new pose, hard expressions). The instruction can
  be rewritten in English by the user's AI provider, which also picks what to change (shown
  before use). Same safety rules and 18+ switch as Krea 2. SAM 3.1 is Enhance photo's file; no
  other models or nodes (no upscaler). Measured facts and limits in `docs/PROVIDERS.md` › Advanced.
  Modal mounts the ComfyUI Volume at `/studio` (a mount over `/workspace` kills the sandbox).
- Hugging Face: token in Integrations › Hugging Face, checked with `whoami-v2` and the Krea 2
  repositories' `auth-check` (`services/huggingface.py`). Accepting the license is the user's
  own click on huggingface.co (no API exists); the studio links there and checks again, and
  training checks it before renting a GPU.
- Image Library shows generated images and AI model photos. AI models hold up to 200 photos.
- Extras (the owner's request, 2026-11-02): "Animate" (viewer → Wan 3.0 first frame,
  `POST /api/reference-images/from-image`); LoRA dataset generator (`/api/loras/{id}/dataset/
  generate`, Krea 2 runs marked `dataset_lora_id`, results added with Add to LoRA); "Create a
  dataset from an AI model" (2026-11-04, `/api/loras/{id}/dataset/hosted` + `/estimate`: one
  click, 12/24/36 hosted runs of the studio's shot list with the AI model's photos as
  references on Nano Banana Pro, Seedream or Wan 2.7 Pro, the whole set's price confirmed
  first, runs marked `Generation.dataset_lora_id`; the user picks the best, face similarity
  measured on demand); spending
  overview (`/api/stats/spending`: confirmed `Generation.cost_estimate`, GPU hours ×
  `GPUWorkload.meta.cost_per_hour`, Modal's `billing.report`); export for social media
  (`/api/images/export`, JPEG without metadata); prompt snippets (`/api/snippets`); face match
  (`services/faces.py`, OpenCV YuNet + SFace on the Mac, files pinned by SHA-256 from
  huggingface.co/opencv, GitHub's OpenCV Model Zoo as the second source; never sent anywhere).
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
