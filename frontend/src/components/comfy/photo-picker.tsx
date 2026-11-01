"use client";

import { ImageIcon, Images, X } from "lucide-react";
import { useState } from "react";

import { ReferenceImages } from "@/components/generate/controls";
import { Button, EmptyState, IconButton, Modal, Spinner } from "@/components/ui";
import { useI18n } from "@/lib/i18n";
import { useImages } from "@/lib/queries";
import type { EnhanceRequest } from "@/lib/studio-bridge";
import type { ReferenceImage } from "@/lib/types";

/** A photo of the user's: an upload (or an AI model's photo) or a generated image. */
export type PickedPhoto = EnhanceRequest;

/** One photo for Enhance photo or Krea 2 image to image: upload it, or pick a generated
 * image from the library. */
export function PhotoPicker({
  title,
  value,
  onChange,
}: {
  title: React.ReactNode;
  value: PickedPhoto | null;
  onChange: (photo: PickedPhoto | null) => void;
}) {
  const { t } = useI18n();
  const [picking, setPicking] = useState(false);
  return (
    <div>
      <div className="section-title">
        <span className="eyebrow grow">{title}</span>
        <Button size="sm" variant="ghost" icon={<Images />} onClick={() => setPicking(true)}>
          {t("en.fromLibrary")}
        </Button>
      </div>
      {value ? (
        <div className="refs">
          <div className="ref">
            <img src={value.thumb_url} alt={t("en.photo")} />
            <div className="ref-actions">
              <IconButton label={t("refs.remove")} onMedia onClick={() => onChange(null)}>
                <X />
              </IconButton>
            </div>
          </div>
        </div>
      ) : (
        <ReferenceImages
          refs={[]}
          max={1}
          onChange={(refs: ReferenceImage[]) =>
            onChange(refs[0] ? { kind: "reference", id: refs[0].id, thumb_url: refs[0].thumb_url } : null)
          }
        />
      )}
      {picking ? (
        <LibraryPicker
          onClose={() => setPicking(false)}
          onPick={(p) => {
            onChange(p);
            setPicking(false);
          }}
        />
      ) : null}
    </div>
  );
}

/** Pick one of the user's generated images. */
function LibraryPicker({ onClose, onPick }: { onClose: () => void; onPick: (p: PickedPhoto) => void }) {
  const { t } = useI18n();
  const images = useImages({});
  const items = (images.data?.pages ?? []).flatMap((p) => p.items);
  return (
    <Modal open wide onClose={onClose} title={t("en.fromLibrary")} description={t("en.fromLibraryHint")}>
      {images.isLoading ? (
        <Spinner />
      ) : !items.length ? (
        <EmptyState icon={<ImageIcon />} title={t("en.libraryEmpty")} description={t("en.libraryEmptyBody")} />
      ) : (
        <div className="pick-grid">
          {items.map((img) => (
            <button
              key={img.id}
              type="button"
              className="pick-tile"
              aria-label={img.model_name}
              onClick={() => onPick({ kind: "image", id: img.id, thumb_url: img.thumb_url })}
            >
              <img src={img.thumb_url} alt="" loading="lazy" />
            </button>
          ))}
          {images.hasNextPage ? (
            <Button size="sm" loading={images.isFetchingNextPage} onClick={() => void images.fetchNextPage()}>
              {t("common.loadMore")}
            </Button>
          ) : null}
        </div>
      )}
    </Modal>
  );
}
