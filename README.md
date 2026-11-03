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

- **Dashboard** — real statistics, recent images, live activity, provider usage, WaveSpeed
  balance and **Spending this month** (the prices you confirmed before each paid run, GPU hours
  × their hourly price, and Modal's own bill; estimates, the providers' bills count).
- **Generate** — three choices at the top (Online models, Krea 2, Enhance photo). Results
  appear on the right; one click on the bin deletes an image or a whole generation, with
  "Undo" for a few seconds.
  - **Online models** (Nano Banana Pro, Seedream, Wan 2.7, Wan 2.7 Pro through WaveSpeed or
    Atlas Cloud): model cards with the price per image, prompt editor with history, reference
    images (drag & drop; click a photo to make it the first one — they are sent in that order,
    so the prompt can say "image 1", "image 2"), only the settings the selected model and
    provider support, live progress. **Generate** first shows the price (an estimate from
    WaveSpeed's price API; Atlas Cloud has no price API, so it says "Price unknown" with a link
    to its pricing) and starts only after you confirm it; Repeat asks the same way. Wan 2.7 and
    Wan 2.7 Pro have an **"18+ content"** switch at the bottom (off by default): off refuses
    explicit prompts, on allows adult content for fictional adults and marks the images 18+.
    **Wan 3.0** and **Wan 3.0 Prime** (WaveSpeed) make **videos**: from a prompt, or from a
    photo (photo 1 = first frame, an optional photo 2 = last frame); length 2–30 s, 480p–1080p,
    sound on or off. The price depends on length and resolution and is shown before it starts.
    Videos play in the results, the viewer and the Image Library. **Animate** in the image
    viewer sends a picture to Wan 3.0 as the first frame of a video.
    **Snippets** (bookmark button in the prompt editor): save pieces of a prompt — lighting,
    outfits, places — for every AI model or only one, and add them with one click (also on
    Krea 2).
  - **Krea 2** (open weights, RAW and Turbo) through ComfyUI, in three steps: **1** choose
    where it runs — **Rent a GPU** (pick an NVIDIA GPU with 24 GB or more; the studio starts it
    on your RunPod or Modal account with RunPod's ready-made ComfyUI image, so it is ready in
    about 1–2 minutes; it is reached only through a token-protected agent, shows the progress
    and stops when idle) or **This Mac** (your own ComfyUI on port 8188);
    **2** when it is connected, install the model with one click (Krea 2 Turbo or RAW, each
    ≈ 17 GB, downloaded from Hugging Face straight to the GPU, with progress);
    **3** pick an AI model and LoRA, start from text or **from a photo** (image to image: your
    upload or a library image, and how much changes, 0.1–0.95), write the prompt and generate.
    Several runs started at once wait in a queue and go one after another on the GPU.
    Krea 2 is SFW + NSFW: the **"18+ content"** switch (off by default) allows adult content
    for fictional adult characters and marks those images 18+ (blurred if Settings say so).
    Seed, steps and CFG under "More settings"; model files, logs and custom workflows under
    "Manage". The image viewer compares a photo-based result with its photo (slider or side
    by side).
  - **Enhance photo** — fixes plastic-looking AI skin, then upscales and sharpens a photo
    (PGUpscaler, the workflow of the free RunPod template "PG - Free Upscaler"). It runs on the
    same GPU or Mac as Krea 2: install its models once with one click (≈ 21 GB from Hugging
    Face, straight to the GPU), choose a photo (upload or "From library"), the strength
    (0.15–0.2 subtle · 0.3 default · 0.35–0.4 strong), "Face only" and the upscale (1.5×, 2×,
    3× — 3× needs about 48 GB of GPU memory). The prompt (what to change, in English) is
    prefilled with the template's text; edit it to steer the fix, e.g. add "freckles" or say
    what the photo shows. With "Face only" you can name what to fix (e.g. "face, neck, hand").
    The image viewer compares before and after (slider or side by side); every image viewer
    has an "Enhance photo" button.
- **AI Models** — your own fictional adult characters with up to 200 photos (shown across the
  whole page, primary photo first). The profile (description, age, hair, face, body, style, …)
  is optional and opens in a window. Select one on Generate: its description is added to the
  prompt and its photos are attached when the chosen model and provider accept references
  (primary first, within that endpoint's limit), for the same face and look in every image.
- **LoRA** — one page for everything: create a character LoRA for a fictional adult, add
  photos (upload, library or AI model import, duplicates, resolution check, crop on a copy),
  captions (templates, trigger word, AI captions), choose a training type and a GPU, train on
  your RunPod or Modal account with live progress, test images, logs and cost (Advanced:
  test images only at the end, base-model protection and memory saving can be switched off
  for faster runs; if the GPU runs out of memory, training starts again on the same GPU with
  memory saving on). **Continue training**: add photos and train a finished version further —
  the training starts from that LoRA instead of from zero and makes a new version. **Add to
  LoRA** (on every result in Generate, in the image viewer and in the library selection)
  copies your best generated images into a LoRA's photos. **Generate new photos with Krea 2**
  (LoRA › Photos): a list of prompts (or the examples) runs on your Krea 2 GPU with the LoRA,
  one run per prompt; tick the best results and add them to the LoRA. "Test & compare" has a **Body**
  test: full body from the front, the side and the back with the same seeds (optionally
  18+), to check that a version keeps the same proportions. The LoRA
  is saved to this Mac and the GPU is always removed afterwards. Versions, preferred version,
  download, copy to a ComfyUI, base-vs-LoRA tests with a comparison slider, import of existing
  Krea 2 LoRA files and the training history with time and cost.
- **Image Library** — generated images, videos and **AI model photos** in one place; search,
  filters, full-resolution viewer, bulk download (ZIP), **export for social media** (9:16, 4:5,
  1:1 or original as JPEG without metadata, optional watermark), add to an AI model and delete;
  **History** tab with every request, its exact settings and "regenerate".
- **Face match** (image viewer, and LoRA › Test & compare): how close the face is to the AI
  model's photos or the LoRA's photos — runs on this Mac with OpenCV's face models (YuNet +
  SFace, about 39 MB, downloaded the first time); 36 % or more = the same person.
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
