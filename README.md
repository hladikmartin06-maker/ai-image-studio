# AI Image Studio

A private, premium AI image generation studio that runs on your Mac. Generate with
**Nano Banana Pro**, **Seedream**, **Wan 2.7** and **Wan 2.7 Pro** through your own
**WaveSpeed** or **Atlas Cloud** account, and keep every image in full quality in a fast,
searchable library.

## Install or update (one command)

Open Terminal and run:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/hladikmartin06-maker/ai-image-studio/main/scripts/install.sh)"
```

It installs what is missing (Homebrew, Node.js, uv, PostgreSQL 16), downloads the studio to
`~/ai-image-studio`, prepares the database, builds the app, starts it in the background and opens
it. It also adds **AI Image Studio** to Applications — open it from Launchpad or the Dock next
time. Running the same command again updates the studio.

First steps in the app:

1. The studio opens right away — there is no sign-in for now. It uses the first account on
   this Mac (with all its images) or creates one by itself.
2. **Integrations** → Connect WaveSpeed and/or Atlas Cloud with your API key.
3. **Generate** → choose a model, write a prompt, press Generate (⌘↵).

To bring the sign-in screen back, add `AUTO_LOGIN=false` to `~/ai-image-studio/backend/.env`
and run the update command again.

## What is inside

The menu is short on purpose: Dashboard, Generate, AI Models, LoRA, Image Library, Assistant,
Integrations and Settings.

- **Dashboard** — real statistics, recent images, live activity, provider usage and WaveSpeed
  balance.
- **Generate** — two choices at the top:
  - **Online models** (Nano Banana Pro, Seedream, Wan 2.7, Wan 2.7 Pro through WaveSpeed or
    Atlas Cloud): model cards, prompt editor with history, reference images (drag & drop), only
    the settings the selected model and provider support, live progress.
  - **Krea 2** (open weights, RAW and Turbo) through ComfyUI, in three steps: **1** choose
    where it runs — **Rent a GPU** (pick an NVIDIA GPU with 24 GB or more; the studio starts it
    on your RunPod or Modal account with RunPod's ready-made ComfyUI image, so it is ready in
    about 1–2 minutes; it is reached only through a token-protected agent, shows the progress
    and stops when idle) or **This Mac** (your own ComfyUI on port 8188);
    **2** when it is connected, install the model with one click (Krea 2 Turbo, ≈ 17 GB,
    downloaded from Hugging Face straight to the GPU, with progress);
    **3** pick an AI model and LoRA, write the prompt and generate. Seed, steps and CFG under
    "More settings"; model files, logs and custom workflows under "Manage".
- **AI Models** — your own fictional adult characters with up to 200 photos (shown across the
  whole page, primary photo first). The profile (description, age, hair, face, body, style, …)
  is optional and opens in a window. Select one on Generate: its description is added to the
  prompt and its photos are attached when the chosen model and provider accept references
  (primary first, within that endpoint's limit), for the same face and look in every image.
- **LoRA** — one page for everything: create a character LoRA for a fictional adult, add
  photos (upload, library or AI model import, duplicates, resolution check, crop on a copy),
  captions (templates, trigger word, AI captions), choose a training type and a GPU, train on
  your RunPod or Modal account with live progress, loss, test images, logs and cost. The LoRA
  is saved to this Mac and the GPU is always removed afterwards. Versions, preferred version,
  download, copy to a ComfyUI, base-vs-LoRA tests with a comparison slider, import of existing
  Krea 2 LoRA files and the training history with time and cost.
- **Image Library** — generated images and **AI model photos** in one place; search, filters,
  full-resolution viewer, bulk download (ZIP), add to an AI model and delete; **History** tab
  with every request, its exact settings and "regenerate".
- **Assistant** — chat that improves prompts, picks models and settings and explains failed
  generations; you can **attach photos** (button or paste) and ask it to describe them or write
  a prompt for a similar image. Works with Claude, GPT, Grok, Gemini or any OpenAI-compatible
  API, in Low cost, Balanced or High quality mode.
- **Integrations** — image providers, AI assistant, GPU compute (RunPod, Modal) and
  **Hugging Face** (token plus a check of the Krea 2 license, needed for LoRA training); keys
  encrypted. **Settings** — theme, language EN/SK, defaults, appearance, assistant, storage,
  account.

## Everyday commands

| | |
|---|---|
| Stop | `~/ai-image-studio/scripts/stop.sh` |
| Start in the background | `~/ai-image-studio/scripts/start.sh` |
| Logs | `~/Library/Logs/AI Image Studio` |
| Images on disk | `~/ai-image-studio/backend/storage` |

Both servers listen only on this computer (127.0.0.1: web 3000, API 8000). API keys are
stored encrypted in the local backend and are never sent back to the browser.

## Development

```bash
./scripts/setup.sh
```

```bash
./scripts/dev.sh
```

Checks: see `CLAUDE.md`. Architecture: `docs/ARCHITECTURE.md`. Provider endpoints and what is
verified: `docs/PROVIDERS.md`. To try the whole app without a provider account, run the local
fake provider described in `backend/tests/fake_server.py`.
