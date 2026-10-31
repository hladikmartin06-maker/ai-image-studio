"use client";

import { useEffect, useRef } from "react";

import { ImageTile } from "@/components/images/image-tile";
import { Skeleton, Spinner } from "@/components/ui";
import type { StudioImage } from "@/lib/types";

/** Row height per grid density; tiles keep their real aspect ratio (justified rows). */
export function useRowHeight() {
  if (typeof document === "undefined") return 220;
  const value = getComputedStyle(document.documentElement).getPropertyValue("--row-h");
  return parseInt(value, 10) || 220;
}

function ratio(img: StudioImage) {
  return Math.min(3, Math.max(0.4, img.width / Math.max(1, img.height)));
}

export function ImageGrid({
  images,
  onOpen,
  onDownload,
  selection,
  selecting,
  onSelect,
  hasMore,
  loadMore,
  loadingMore,
  showPrompts,
  blurNsfw,
}: {
  images: StudioImage[];
  onOpen: (index: number) => void;
  onDownload: (img: StudioImage) => void;
  selection?: Set<string> | null;
  selecting?: boolean;
  onSelect?: (img: StudioImage, index: number, shift: boolean) => void;
  hasMore?: boolean;
  loadMore?: () => void;
  loadingMore?: boolean;
  showPrompts?: boolean;
  blurNsfw?: boolean;
}) {
  const sentinel = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!hasMore || !loadMore || !sentinel.current) return;
    const el = sentinel.current;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((e) => e.isIntersecting)) loadMore();
      },
      { rootMargin: "800px 0px" },
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [hasMore, loadMore, images.length]);

  return (
    <>
      <div className="jgrid">
        {images.map((img, index) => {
          const ar = ratio(img);
          return (
            <ImageTile
              key={img.id}
              image={img}
              style={{
                flexGrow: ar,
                flexBasis: `calc(${ar} * var(--row-h))`,
                aspectRatio: String(ar),
                maxHeight: "calc(var(--row-h) * 1.6)",
              }}
              onOpen={() => onOpen(index)}
              onDownload={() => onDownload(img)}
              selecting={Boolean(selecting || (selection && selection.size > 0))}
              selected={selection?.has(img.id)}
              onSelect={onSelect ? (shift) => onSelect(img, index, shift) : undefined}
              showPrompt={showPrompts}
              blurNsfw={blurNsfw}
            />
          );
        })}
      </div>
      {hasMore ? (
        <div ref={sentinel} className="row" style={{ justifyContent: "center", padding: 28 }}>
          {loadingMore ? <Spinner /> : null}
        </div>
      ) : null}
    </>
  );
}

export function GridSkeleton({ count = 12 }: { count?: number }) {
  const ratios = [1, 1.5, 0.75, 1.33, 0.66, 1.77, 1, 0.8, 1.25, 1.5, 0.66, 1];
  return (
    <div className="jgrid" aria-busy>
      {Array.from({ length: count }, (_, i) => {
        const ar = ratios[i % ratios.length];
        return (
          <Skeleton
            key={i}
            style={{
              flexGrow: ar,
              flexBasis: `calc(${ar} * var(--row-h))`,
              aspectRatio: String(ar),
              borderRadius: 12,
            }}
          />
        );
      })}
    </div>
  );
}
