"use client";

import {
  ChevronLeft,
  ChevronRight,
  Copy,
  Download,
  Heart,
  LibraryBig,
  RefreshCw,
  SlidersHorizontal,
  Trash2,
  X,
} from "lucide-react";
import { useEffect, useState } from "react";
import { createPortal } from "react-dom";

import { useImageActions } from "@/components/images/actions";
import { Badge, Button, IconButton } from "@/components/ui";
import { aspectLabel, formatBytes, formatDate } from "@/lib/format";
import { useI18n } from "@/lib/i18n";
import { StatusBadge } from "@/components/images/status";
import type { StudioImage } from "@/lib/types";

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

  useEffect(() => {
    if (index === null) return;
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.closest?.(".modal")) return;
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

  const settingsRows: [string, string][] = [];
  if (s.resolution) settingsRows.push([t("gen.resolution"), s.resolution.toUpperCase()]);
  if (s.aspect_ratio)
    settingsRows.push([
      t("gen.aspectRatio"),
      s.aspect_ratio === "match" ? t("gen.matchInput") : s.aspect_ratio,
    ]);
  if (s.output_format) settingsRows.push([t("gen.outputFormat"), s.output_format.toUpperCase()]);
  if (s.thinking_mode !== undefined)
    settingsRows.push([t("gen.thinkingMode"), s.thinking_mode ? t("common.on") : t("common.off")]);

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
            <Button
              size="sm"
              variant={image.is_favorite ? "primary" : "default"}
              icon={<Heart fill={image.is_favorite ? "currentColor" : "none"} />}
              onClick={() => actions.toggleFavorite(image)}
            >
              {image.is_favorite ? t("images.favorited") : t("images.favorite")}
            </Button>
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
            {!image.in_library ? (
              <Button size="sm" icon={<LibraryBig />} onClick={() => actions.saveToLibrary(image)}>
                {t("images.saveToLibrary")}
              </Button>
            ) : null}
          </div>
        </div>
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
            <dd>{image.mode === "image-edit" ? t("gen.modeEdit") : t("gen.modeText")}</dd>
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
