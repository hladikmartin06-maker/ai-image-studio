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

1. Create your account (it stays on this Mac).
2. **Integrations** → Connect WaveSpeed and/or Atlas Cloud with your API key.
3. **Generate** → choose a model, write a prompt, press Generate (⌘↵).

## What is inside

- **Dashboard** — real statistics, recent images, live activity, quick generate, provider usage
  and WaveSpeed balance.
- **Generate** — model cards, large prompt editor with history, reference images (drag & drop),
  only the settings the selected model and provider support, live progress.
- **AI Models** — your own fictional adult characters: name, description, appearance (age,
  hair, face, body, style, …) and a collection of reference images with a primary image.
  Select one on Generate: its description is added to the prompt and its reference images are
  attached when the chosen model and provider accept references (primary first, within that
  endpoint's limit), for the same face and look in every image. Duplicate, edit, delete, add
  images from the library, and see everything generated with each AI model.
- **Image Library** — justified grid, search (prompt, model, provider, size, seed), filters,
  full-resolution viewer with details, bulk download (ZIP), favorite and delete;
  **History** tab with every request, its exact settings and "regenerate".
- **Assistant** — chat that improves prompts, picks models and settings and explains failed
  generations; it sees your current setup and can apply its suggestions to the studio. Works
  with Claude, GPT, Grok, Gemini or any OpenAI-compatible API, in Low cost, Balanced or High
  quality mode, with automatic routing to cheaper models for simple questions.
- **GPU Compute** — start, stop and terminate GPU workloads on your own RunPod (pods,
  serverless endpoints) or Modal (sandboxes) account, with GPU catalog and usage.
- **Favorites**, **Integrations** (image providers, AI assistant, GPU compute — each separate,
  keys encrypted), **Settings** (theme, language EN/SK, defaults, appearance, assistant,
  storage, account and sessions).

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
