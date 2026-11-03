"use client";

import { Share2 } from "lucide-react";
import { useState } from "react";

import { Button, Field, Popover, Segmented } from "@/components/ui";
import { downloadUrl } from "@/lib/api";
import { useT } from "@/lib/i18n";

type Format = "story" | "portrait" | "square" | "original";
const WATERMARK_KEY = "studio:export-watermark";

/** "Export for social media": a ZIP of JPEGs cropped to 9:16, 4:5 or 1:1, without metadata
 * (no EXIF or generation data), with an optional watermark. Videos are left out. */
export function ExportSocial({ imageIds }: { imageIds: string[] }) {
  const t = useT();
  const [format, setFormat] = useState<Format>("portrait");
  const [watermark, setWatermark] = useState(() => {
    try {
      return localStorage.getItem(WATERMARK_KEY) ?? "";
    } catch {
      return "";
    }
  });

  const run = (close: () => void) => {
    try {
      localStorage.setItem(WATERMARK_KEY, watermark);
    } catch {
      /* storage blocked */
    }
    const params = new URLSearchParams({ ids: imageIds.join(","), format, watermark: watermark.trim() });
    downloadUrl(`/api/images/export?${params.toString()}`);
    close();
  };

  return (
    <Popover
      width={320}
      trigger={({ toggle, ref }) => (
        <button ref={ref} type="button" className="btn btn-sm" onClick={toggle} disabled={!imageIds.length}>
          <Share2 /> {t("ex.title")}
        </button>
      )}
    >
      {(close) => (
        <div className="col" style={{ gap: 12, padding: 12 }}>
          <div>
            <div className="h3">{t("ex.title")}</div>
            <div className="tiny subtle">{t("ex.hint", { n: imageIds.length })}</div>
          </div>
          <Field label={t("ex.format")}>
            <Segmented
              label={t("ex.format")}
              block
              value={format}
              onChange={setFormat}
              options={[
                { value: "story", label: "9:16" },
                { value: "portrait", label: "4:5" },
                { value: "square", label: "1:1" },
                { value: "original", label: t("ex.original") },
              ]}
            />
          </Field>
          <div className="tiny subtle">{t(`ex.format.${format}`)}</div>
          <Field label={t("ex.watermark")} hint={t("ex.watermarkHint")}>
            <input
              className="input"
              value={watermark}
              maxLength={40}
              placeholder="@moja_modelka"
              onChange={(e) => setWatermark(e.target.value)}
            />
          </Field>
          <Button variant="primary" icon={<Share2 />} onClick={() => run(close)}>
            {t("ex.download")}
          </Button>
        </div>
      )}
    </Popover>
  );
}
