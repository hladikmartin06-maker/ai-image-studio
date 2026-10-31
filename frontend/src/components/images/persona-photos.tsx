"use client";

import { Download, ExternalLink, SquareUserRound } from "lucide-react";
import Link from "next/link";
import { type ReactNode, useMemo, useState } from "react";

import { GridSkeleton } from "@/components/images/image-grid";
import { Button, EmptyState, IconButton, Modal, Notice, Select } from "@/components/ui";
import { downloadUrl } from "@/lib/api";
import { useT } from "@/lib/i18n";
import { usePersonas } from "@/lib/queries";
import type { Persona, PersonaImage } from "@/lib/types";

interface Photo {
  image: PersonaImage;
  persona: Persona;
}

/** The photos saved in AI models, shown in the library next to the generated images. */
export function PersonaPhotos({ header, toolbarStart }: { header: (total: number | undefined) => ReactNode; toolbarStart?: ReactNode }) {
  const t = useT();
  const personas = usePersonas();
  const [persona, setPersona] = useState("any");
  const [open, setOpen] = useState<Photo | null>(null);

  const groups = useMemo(
    () =>
      (personas.data?.items ?? [])
        .filter((p) => p.images.length && (persona === "any" || p.id === persona))
        .map((p) => ({ persona: p, photos: p.images.map((image) => ({ image, persona: p })) })),
    [personas.data, persona],
  );
  const total = groups.reduce((n, g) => n + g.photos.length, 0);

  return (
    <>
      {header(personas.data ? total : undefined)}
      <div className="toolbar">
        {toolbarStart}
        {personas.data?.items.length ? (
          <Select
            label={t("pm.section")}
            value={persona}
            onChange={setPersona}
            highlight={persona !== "any"}
            options={[
              { value: "any", label: t("pm.filterAny") },
              ...personas.data.items.map((p) => ({ value: p.id, label: p.name })),
            ]}
          />
        ) : null}
      </div>
      {personas.isLoading ? (
        <GridSkeleton />
      ) : personas.isError ? (
        <Notice tone="danger">{(personas.error as Error).message}</Notice>
      ) : !groups.length ? (
        <EmptyState
          icon={<SquareUserRound />}
          title={t("lib.noModelPhotos")}
          description={t("lib.noModelPhotosBody")}
          action={
            <Link href="/ai-models" className="btn btn-primary btn-sm">
              {t("nav.aiModels")}
            </Link>
          }
        />
      ) : (
        <div className="col" style={{ gap: 28 }}>
          {groups.map((g) => (
            <section key={g.persona.id}>
              <div className="section-title" style={{ marginBottom: 12 }}>
                <h2 className="h3 grow">{g.persona.name}</h2>
                <span className="tiny subtle">{t("pm.imageCount", { n: g.photos.length, max: g.persona.max_images })}</span>
                <Link href={`/ai-models/${g.persona.id}`} className="btn btn-ghost btn-sm">
                  {t("lib.openModel")} <ExternalLink />
                </Link>
              </div>
              <div className="persona-images">
                {g.photos.map((ph) => (
                  <div className="persona-image" key={ph.image.id} data-primary={ph.image.is_primary || undefined}>
                    <button type="button" className="persona-image-open" onClick={() => setOpen(ph)} aria-label={ph.image.original_name || g.persona.name}>
                      <img src={ph.image.thumb_url} alt="" loading="lazy" decoding="async" />
                    </button>
                    <div className="persona-image-actions">
                      <IconButton label={t("common.download")} onMedia onClick={() => downloadUrl(`${ph.image.url}?download=1`)}>
                        <Download />
                      </IconButton>
                    </div>
                  </div>
                ))}
              </div>
            </section>
          ))}
        </div>
      )}
      {open ? (
        <Modal
          open
          wide
          onClose={() => setOpen(null)}
          title={open.persona.name}
          description={open.image.width && open.image.height ? `${open.image.width}×${open.image.height}` : undefined}
          footer={
            <>
              <Link href={`/ai-models/${open.persona.id}`} className="btn btn-ghost">
                {t("lib.openModel")}
              </Link>
              <Button icon={<Download />} onClick={() => downloadUrl(`${open.image.url}?download=1`)}>
                {t("common.download")}
              </Button>
            </>
          }
        >
          <img src={open.image.url} alt="" style={{ display: "block", maxWidth: "100%", maxHeight: "70vh", margin: "0 auto", borderRadius: 12 }} />
        </Modal>
      ) : null}
    </>
  );
}
