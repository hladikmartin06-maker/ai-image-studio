"use client";

import {
  ChevronLeft,
  ChevronRight,
  Copy,
  Download,
  Film,
  LibraryBig,
  RefreshCw,
  SlidersHorizontal,
  Sparkles,
  Trash2,
  Wand2,
  X,
} from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";
import { createPortal } from "react-dom";

import { useImageActions } from "@/components/images/actions";
import { ExportSocial } from "@/components/images/export-social";
import { FaceMatch } from "@/components/images/face-match";
import { AddToLora } from "@/components/lora/add-to-lora";
import { AddToPersona } from "@/components/personas/shared";
import { Badge, Button, IconButton, Notice, Segmented } from "@/components/ui";
import { aspectLabel, formatBytes, formatDate } from "@/lib/format";
import { useI18n } from "@/lib/i18n";
import { StatusBadge } from "@/components/images/status";
import { isVideo, MODE_LABELS, type StudioImage } from "@/lib/types";

export function ImageViewer({
  images,
  index,
  onClose,
  onIndexChange,
}: {
  images: StudioImage[];
  index: number | null;
  onClose: () => void;
  onIndexChange: (index: number) => void;
}) {
  const { t, locale } = useI18n();
  const actions = useImageActions();
  const image = index !== null ? images[index] : undefined;
  const [loadedId, setLoadedId] = useState<string | null>(null);
  // Enhance photo: the result, before/after with a slider, or both side by side.
  const [view, setView] = useState<"result" | "slider" | "side">("result");
  const [missingOriginal, setMissingOriginal] = useState<string | null>(null);

  useEffect(() => {
    if (index === null) return;
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.closest?.(".modal")) return;
      // The before/after slider uses the arrow keys itself.
      if (e.key !== "Escape" && (e.target as HTMLElement)?.closest?.(".viewer-compare")) return;
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowLeft" && index > 0) onIndexChange(index - 1);
      if (e.key === "ArrowRight" && index < images.length - 1) onIndexChange(index + 1);
    };
    window.addEventListener("keydown", onKey);
    const overflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      window.removeEventListener("keydown", onKey);
      document.body.style.overflow = overflow;
    };
  }, [index, images.length, onClose, onIndexChange]);

  useEffect(() => {
    if (index !== null && !image) onClose();
  }, [image, index, onClose]);

  if (!image || index === null) return null;
  const s = image.settings;
  // Enhance photo and Krea 2 image to image keep the photo they started from.
  const input = image.comfy?.input ?? null;
  const original = input && missingOriginal !== input.url ? input.url : null;
  const shown = original ? view : "result";

  const settingsRows: [string, string][] = [];
  if (s.resolution) settingsRows.push([t("gen.resolution"), isVideo(image) ? s.resolution : s.resolution.toUpperCase()]);
  if (s.aspect_ratio)
    settingsRows.push([
      t("gen.aspectRatio"),
      s.aspect_ratio === "match" ? t("gen.matchInput") : s.aspect_ratio,
    ]);
  if (s.output_format) settingsRows.push([t("gen.outputFormat"), s.output_format.toUpperCase()]);
  if (s.thinking_mode !== undefined)
    settingsRows.push([t("gen.thinkingMode"), s.thinking_mode ? t("common.on") : t("common.off")]);
  // Enhance photo
  if (s.strength !== undefined) settingsRows.push([t("en.strength"), String(s.strength)]);
  if (s.face_only !== undefined) settingsRows.push([t("en.faceOnly"), s.face_only ? t("common.on") : t("common.off")]);
  if (s.upscale !== undefined) settingsRows.push([t("en.upscale"), `${s.upscale}×`]);
  if (s.denoise !== undefined) settingsRows.push([t("k2.denoiseShort"), s.denoise.toFixed(2)]);
  if (s.adult) settingsRows.push([t("k2.adultShort"), t("common.on")]);
  // Video (Wan 3.0)
  if (s.duration !== undefined) settingsRows.push([t("gen.duration"), t("gen.durationValue", { n: s.duration })]);
  if (s.generate_audio !== undefined)
    settingsRows.push([t("gen.audio"), s.generate_audio ? t("common.on") : t("common.off")]);
  if (s.enable_prompt_expansion) settingsRows.push([t("gen.expansion"), t("common.on")]);

  const onDelete = async () => {
    const ok = await actions.deleteImage(image);
    if (ok && images.length <= 1) onClose();
    else if (ok && index >= images.length - 1) onIndexChange(Math.max(0, index - 1));
  };

  return createPortal(
    <div className="viewer" role="dialog" aria-modal="true" aria-label={t("viewer.title")}>
      <div className="viewer-stage" onClick={(e) => e.target === e.currentTarget && onClose()}>
        <div className="viewer-top">
          <IconButton label={t("common.close")} onMedia onClick={onClose}>
            <X />
          </IconButton>
          <span className="tiny" style={{ color: "#bbb" }}>
            {index + 1} / {images.length}
          </span>
        </div>
        {original && shown !== "result" ? (
          <BeforeAfter
            key={`${image.id}:${shown}`}
            mode={shown}
            before={original}
            after={image.url}
            labels={[t("en.before"), t("en.after")]}
            onBeforeError={() => setMissingOriginal(original)}
          />
        ) : (
          <>
            {isVideo(image) ? (
              <video key={image.id} src={image.url} controls autoPlay playsInline aria-label={image.prompt} />
            ) : (
              <>
                {loadedId !== image.id ? (
                  <img className="thumb-layer" src={image.thumb_url} alt="" aria-hidden />
                ) : null}
                <img
                  key={image.id}
                  src={image.url}
                  alt={image.prompt}
                  onLoad={() => setLoadedId(image.id)}
                  style={{ opacity: loadedId === image.id ? 1 : 0, transition: "opacity 200ms" }}
                />
              </>
            )}
          </>
        )}
        {index > 0 ? (
          <span className="viewer-nav prev">
            <IconButton label={t("viewer.previous")} onMedia onClick={() => onIndexChange(index - 1)}>
              <ChevronLeft />
            </IconButton>
          </span>
        ) : null}
        {index < images.length - 1 ? (
          <span className="viewer-nav next">
            <IconButton label={t("viewer.next")} onMedia onClick={() => onIndexChange(index + 1)}>
              <ChevronRight />
            </IconButton>
          </span>
        ) : null}
      </div>
      <aside className="viewer-side">
        <div className="viewer-side-section">
          <div className="row" style={{ marginBottom: 14 }}>
            <Badge tone="accent">{image.model_name}</Badge>
            <Badge>{image.provider_name}</Badge>
            {!image.in_library ? <Badge tone="warning">{t("images.notSaved")}</Badge> : null}
            <span className="spacer" />
            <StatusBadge status={image.status} />
          </div>
          <div className="row wrap" style={{ gap: 6 }}>
            <Button size="sm" icon={<Download />} onClick={() => actions.download(image)}>
              {t("common.download")}
            </Button>
            <Button
              size="sm"
              icon={<RefreshCw />}
              loading={actions.regenerating}
              onClick={() => actions.regenerateImage(image)}
            >
              {t("images.regenerate")}
            </Button>
            {isVideo(image) ? null : (
              <>
                <Button size="sm" icon={<Film />} onClick={() => void actions.animateImage(image)}>
                  {t("images.animate")}
                </Button>
                <Button size="sm" icon={<Sparkles />} onClick={() => actions.enhanceImage(image)}>
                  {t("images.enhance")}
                </Button>
                <Button size="sm" icon={<Wand2 />} onClick={() => actions.editImage(image)}>
                  {t("images.editAdvanced")}
                </Button>
              </>
            )}
            {!image.in_library ? (
              <Button size="sm" icon={<LibraryBig />} onClick={() => actions.saveToLibrary(image)}>
                {t("images.saveToLibrary")}
              </Button>
            ) : null}
            {isVideo(image) ? null : (
              <>
                <ExportSocial imageIds={[image.id]} />
                <AddToPersona imageIds={[image.id]} />
                <AddToLora imageIds={[image.id]} />
              </>
            )}
          </div>
        </div>
        {!isVideo(image) && (image.persona_id || image.comfy?.lora?.lora_id) ? (
          <div className="viewer-side-section">
            <div className="eyebrow" style={{ marginBottom: 10 }}>
              {t("fm.title")}
            </div>
            <FaceMatch
              key={image.id}
              imageId={image.id}
              target={image.persona_id ? { persona_id: image.persona_id } : { lora_id: image.comfy?.lora?.lora_id ?? "" }}
              name={image.persona_id ? (image.persona_name ?? "") : (image.comfy?.lora?.name ?? "LoRA")}
            />
          </div>
        ) : null}
        {input ? (
          <div className="viewer-side-section">
            <div className="eyebrow" style={{ marginBottom: 10 }}>
              {t("en.view")}
            </div>
            {original ? (
              <Segmented
                label={t("en.view")}
                value={shown}
                block
                onChange={setView}
                options={[
                  { value: "result", label: t("en.viewResult") },
                  { value: "slider", label: t("en.viewSlider") },
                  { value: "side", label: t("en.viewSide") },
                ]}
              />
            ) : (
              <Notice>{t("en.originalGone")}</Notice>
            )}
          </div>
        ) : null}
        <div className="viewer-side-section">
          <div className="row" style={{ marginBottom: 8 }}>
            <div className="eyebrow grow">{t("viewer.prompt")}</div>
            <IconButton label={t("common.copyPrompt")} onClick={() => actions.copyPrompt(image)}>
              <Copy />
            </IconButton>
          </div>
          <div className="prompt-box">{image.prompt}</div>
        </div>
        <div className="viewer-side-section">
          <div className="eyebrow" style={{ marginBottom: 12 }}>
            {t("viewer.details")}
          </div>
          <dl className="kv">
            <dt>{t("viewer.model")}</dt>
            <dd>{image.model_name}</dd>
            <dt>{t("viewer.provider")}</dt>
            <dd>{image.provider_name}</dd>
            <dt>{t("viewer.mode")}</dt>
            <dd>{t(MODE_LABELS[image.mode as keyof typeof MODE_LABELS] ?? "gen.modeText")}</dd>
            {image.comfy ? (
              <>
                <dt>{t("k2.workflow")}</dt>
                <dd>{image.comfy.workflow?.name ?? "—"}</dd>
                {image.comfy.model_file ? (
                  <>
                    <dt>{t("k2.modelFile")}</dt>
                    <dd className="mono truncate" title={image.comfy.model_file}>{image.comfy.model_file}</dd>
                  </>
                ) : null}
                {image.comfy.lora ? (
                  <>
                    <dt>LoRA</dt>
                    <dd>
                      <Link href={`/loras/${image.comfy.lora.lora_id}`} className="link" onClick={onClose}>
                        {image.comfy.lora.name} v{image.comfy.lora.version}
                      </Link>{" "}
                      · {image.comfy.lora.strength}
                    </dd>
                  </>
                ) : null}
                {image.settings.steps ? (
                  <>
                    <dt>{t("k2.steps")} · CFG</dt>
                    <dd>
                      {image.settings.steps} · {image.settings.cfg} · {image.settings.sampler}
                    </dd>
                  </>
                ) : null}
              </>
            ) : null}
            {image.persona_name ? (
              <>
                <dt>{t("pm.section")}</dt>
                <dd>
                  {image.persona_id ? (
                    <Link href={`/ai-models/${image.persona_id}`} className="link" onClick={onClose}>
                      {image.persona_name}
                    </Link>
                  ) : (
                    `${image.persona_name} · ${t("pm.deletedTag")}`
                  )}
                </dd>
              </>
            ) : null}
            <dt>{t("viewer.created")}</dt>
            <dd>{formatDate(image.created_at, locale)}</dd>
            <dt>{t("viewer.resolution")}</dt>
            <dd>
              {image.width} × {image.height}
            </dd>
            <dt>{t("gen.aspectRatio")}</dt>
            <dd>{aspectLabel(image.width, image.height)}</dd>
            <dt>{t("gen.seed")}</dt>
            <dd>{image.seed ?? t("gen.seedRandom")}</dd>
            <dt>{t("viewer.file")}</dt>
            <dd>
              {image.mime_type.replace("image/", "").toUpperCase()} · {formatBytes(image.size_bytes, locale)}
            </dd>
          </dl>
        </div>
        {settingsRows.length ? (
          <div className="viewer-side-section">
            <div className="eyebrow" style={{ marginBottom: 12 }}>
              {t("viewer.settings")}
            </div>
            <dl className="kv">
              {settingsRows.map(([k, v]) => (
                <div key={k} style={{ display: "contents" }}>
                  <dt>{k}</dt>
                  <dd>{v}</dd>
                </div>
              ))}
            </dl>
          </div>
        ) : null}
        <div className="viewer-side-section row wrap" style={{ borderBottom: 0, marginTop: "auto" }}>
          <Button size="sm" variant="ghost" icon={<SlidersHorizontal />} onClick={() => actions.openInStudio(image)}>
            {t("images.openInStudio")}
          </Button>
          <span className="spacer" />
          <Button size="sm" variant="danger" icon={<Trash2 />} onClick={onDelete}>
            {t("common.delete")}
          </Button>
        </div>
      </aside>
    </div>,
    document.body,
  );
}

/** Enhance photo: the original and the result, with a slider (the original on the left of
 * the line) or side by side. The result sets the size; the original is scaled to it. */
function BeforeAfter({
  mode,
  before,
  after,
  labels,
  onBeforeError,
}: {
  mode: "slider" | "side";
  before: string;
  after: string;
  labels: [string, string];
  onBeforeError: () => void;
}) {
  const [split, setSplit] = useState(50);
  if (mode === "side") {
    return (
      <div className="viewer-side-by-side">
        <figure>
          <img src={before} alt={labels[0]} onError={onBeforeError} />
          <figcaption>{labels[0]}</figcaption>
        </figure>
        <figure>
          <img src={after} alt={labels[1]} />
          <figcaption>{labels[1]}</figcaption>
        </figure>
      </div>
    );
  }
  return (
    <div className="viewer-compare">
      <img src={after} alt={labels[1]} />
      <img
        className="before"
        src={before}
        alt={labels[0]}
        onError={onBeforeError}
        style={{ clipPath: `inset(0 ${100 - split}% 0 0)` }}
      />
      <div className="compare-line" style={{ left: `${split}%` }} />
      <span className="compare-label left">{labels[0]}</span>
      <span className="compare-label right">{labels[1]}</span>
      <input
        type="range"
        min={0}
        max={100}
        value={split}
        onChange={(e) => setSplit(Number(e.target.value))}
        aria-label={`${labels[0]} / ${labels[1]}`}
      />
    </div>
  );
}
