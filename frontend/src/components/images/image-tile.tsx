"use client";

import { Check, Download, EyeOff, Trash2 } from "lucide-react";
import type { CSSProperties, KeyboardEvent, MouseEvent } from "react";

import { IconButton, cx } from "@/components/ui";
import { formatRelative } from "@/lib/format";
import { useI18n } from "@/lib/i18n";
import type { StudioImage } from "@/lib/types";

export function ImageTile({
  image,
  onOpen,
  onDownload,
  onDelete,
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
      <img src={image.thumb_url} alt="" loading="lazy" decoding="async" draggable={false} />
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
          <div className="tile-actions" onClick={stop}>
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
