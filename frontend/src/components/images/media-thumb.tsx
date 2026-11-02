"use client";

import { isVideo, type StudioImage } from "@/lib/types";

/** A small preview: the image's thumbnail, or a video's first frame. */
export function MediaThumb({
  image,
  className,
}: {
  image: Pick<StudioImage, "url" | "thumb_url" | "mime_type" | "media">;
  className?: string;
}) {
  if (isVideo(image)) {
    return <video className={className} src={`${image.url}#t=0.1`} muted playsInline preload="metadata" />;
  }
  return <img className={className} src={image.thumb_url} alt="" loading="lazy" />;
}
