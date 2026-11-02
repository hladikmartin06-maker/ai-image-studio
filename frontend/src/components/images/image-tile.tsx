"use client";

import { Check, Download, EyeOff, Play, Trash2 } from "lucide-react";
import type { CSSProperties, KeyboardEvent, MouseEvent, ReactNode } from "react";

import { IconButton, cx } from "@/components/ui";
import { formatRelative } from "@/lib/format";
import { useI18n } from "@/lib/i18n";
import { isVideo, type StudioImage } from "@/lib/types";

export function ImageTile({
  image,
  onOpen,
  onDownload,
  onDelete,
  extra,
  selecting,
  selected,
  onSelect,
  showPrompt = true,
  blurNsfw = true,
  style,
  className,
}: {
  image: StudioImage;
  onOpen: () => void;
  onDownload: () => void;
  /** One-click delete (Generate results). */
  onDelete?: () => void;
  /** More buttons next to download (e.g. "Add to LoRA"). */
  extra?: ReactNode;
  selecting?: boolean;
  selected?: boolean;
  onSelect?: (shift: boolean) => void;
  showPrompt?: boolean;
  blurNsfw?: boolean;
  style?: CSSProperties;
  className?: string;
}) {
  const { t, locale } = useI18n();
  const blurred = blurNsfw && image.nsfw === true;

  const activate = (e: MouseEvent | KeyboardEvent) => {
    if (selecting && onSelect) onSelect("shiftKey" in e && e.shiftKey);
    else onOpen();
  };
  const stop = (e: MouseEvent) => e.stopPropagation();

  return (
    <div
      className={cx("tile", blurred && "blurred", className)}
      style={style}
      data-selected={selected || undefined}
      data-selecting={selecting || undefined}
      role="button"
      tabIndex={0}
      aria-label={image.prompt}
      onClick={activate}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          activate(e);
        }
      }}
    >
      {isVideo(image) ? (
        <>
          {/* A video has no picture of its own: its first frame, played while hovered. */}
          <video
            src={`${image.url}#t=0.1`}
            muted
            loop
            playsInline
            preload="metadata"
            onMouseEnter={(e) => void e.currentTarget.play().catch(() => undefined)}
            onMouseLeave={(e) => e.currentTarget.pause()}
          />
          <span className="tile-video" aria-label={t("gen.video")}>
            <Play size={12} /> {t("gen.video")}
          </span>
        </>
      ) : (
        <img src={image.thumb_url} alt="" loading="lazy" decoding="async" draggable={false} />
      )}
      {blurred ? (
        <div className="tile-nsfw">
          <span className="row gap-4">
            <EyeOff size={14} /> {t("images.mature")}
          </span>
        </div>
      ) : null}
      <div className="tile-overlay">
        <div className="row">
          {selecting || onSelect ? (
            <button
              type="button"
              className="tile-check"
              aria-label={selected ? t("common.deselect") : t("common.select")}
              aria-pressed={selected}
              onClick={(e) => {
                e.stopPropagation();
                onSelect?.(e.shiftKey);
              }}
            >
              {selected ? <Check /> : null}
            </button>
          ) : null}
          <span className="spacer" />
          <div className="tile-actions" onClick={stop} onKeyDown={(e) => e.stopPropagation()}>
            {extra}
            <IconButton label={t("common.download")} onMedia onClick={onDownload}>
              <Download />
            </IconButton>
            {onDelete ? (
              <IconButton label={t("common.delete")} onMedia onClick={onDelete}>
                <Trash2 />
              </IconButton>
            ) : null}
          </div>
        </div>
        <div className="tile-meta">
          {showPrompt ? <div className="clamp-2">{image.prompt}</div> : null}
          <div className="tiny">
            {image.model_name} · {formatRelative(image.created_at, locale)}
          </div>
        </div>
      </div>
    </div>
  );
}
