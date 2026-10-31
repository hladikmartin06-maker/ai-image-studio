"use client";

import { ArrowLeft, CopyPlus, Images, PencilLine, Sparkles, Trash2, UserRound } from "lucide-react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useMemo, useState } from "react";

import { useImageActions } from "@/components/images/actions";
import { GridSkeleton, ImageGrid } from "@/components/images/image-grid";
import { ImageViewer } from "@/components/images/image-viewer";
import { PageHeader } from "@/components/layout/page-header";
import { usePersonaActions } from "@/components/personas/actions";
import { PersonaImages, PersonaProfile } from "@/components/personas/persona-editor";
import { PersonaThumb } from "@/components/personas/shared";
import { Button, EmptyState, Modal, Notice, Skeleton } from "@/components/ui";
import { ApiError } from "@/lib/api";
import { formatDate } from "@/lib/format";
import { useI18n } from "@/lib/i18n";
import { useImages, usePersona, useSettings } from "@/lib/queries";

export default function AIModelPage() {
  const { t, locale } = useI18n();
  const router = useRouter();
  const params = useParams<{ id: string }>();
  const id = params.id;
  const persona = usePersona(id);
  const actions = usePersonaActions();
  const settings = useSettings();
  const imageActions = useImageActions();
  const images = useImages({ persona: id, limit: 30, scope: "all" }, Boolean(persona.data));
  const items = useMemo(() => images.data?.pages.flatMap((p) => p.items) ?? [], [images.data]);
  const [viewer, setViewer] = useState<number | null>(null);
  const [profile, setProfile] = useState(false);

  const back = (
    <Link href="/ai-models" className="btn btn-ghost btn-sm" style={{ marginBottom: 12, marginLeft: -8 }}>
      <ArrowLeft /> {t("pm.title")}
    </Link>
  );

  if (persona.isLoading) {
    return (
      <div className="page">
        {back}
        <Skeleton style={{ height: 56, width: 320, marginBottom: 24 }} />
        <Skeleton style={{ height: 520 }} />
      </div>
    );
  }
  if (persona.isError || !persona.data) {
    const missing = persona.error instanceof ApiError && persona.error.status === 404;
    return (
      <div className="page">
        {back}
        {missing ? (
          <EmptyState
            icon={<UserRound />}
            title={t("pm.notFound")}
            description={t("pm.notFoundBody")}
            action={
              <Link href="/ai-models" className="btn btn-sm">
                {t("pm.title")}
              </Link>
            }
          />
        ) : (
          <Notice tone="danger">{(persona.error as Error | null)?.message ?? t("errors.generic")}</Notice>
        )}
      </div>
    );
  }

  const p = persona.data;
  return (
    <div className="page">
      {back}
      <PageHeader
        eyebrow={t("pm.eyebrow")}
        title={
          <span className="row" style={{ gap: 14 }}>
            <PersonaThumb persona={p} size={44} />
            <span className="truncate">{p.name}</span>
          </span>
        }
        subtitle={t("pm.detailSubtitle", {
          images: p.image_count,
          generations: p.generation_count,
          date: formatDate(p.created_at, locale, false),
        })}
        actions={
          <>
            <Button icon={<CopyPlus />} loading={actions.duplicating} onClick={() => actions.copy(p)}>
              {t("pm.duplicate")}
            </Button>
            <Button
              variant="ghost"
              icon={<Trash2 />}
              onClick={() => actions.destroy(p, () => router.replace("/ai-models"))}
            >
              {t("common.delete")}
            </Button>
            <Button icon={<PencilLine />} onClick={() => setProfile(true)}>
              {t("pm.editProfile")}
            </Button>
            <Button variant="primary" icon={<Sparkles />} onClick={() => actions.openInGenerate(p)}>
              {t("pm.useInGenerate")}
            </Button>
          </>
        }
      />

      <section className="card">
        <div className="card-head">
          <div className="grow">
            <div className="h3">{t("pm.referenceImages")}</div>
            <div className="tiny subtle">{t("pm.referenceImagesHint")}</div>
          </div>
        </div>
        <div className="card-body">
          <PersonaImages persona={p} />
        </div>
      </section>
      {profile ? (
        <Modal open wide onClose={() => setProfile(false)} title={t("pm.profile")} description={t("pm.profileHint")}>
          <PersonaProfile key={p.id} persona={p} onSaved={() => setProfile(false)} />
        </Modal>
      ) : null}

      <section style={{ marginTop: 32 }}>
        <div className="section-title" style={{ marginBottom: 14 }}>
          <h2 className="h2 grow">{t("pm.generated")}</h2>
          {images.data ? (
            <span className="small subtle">{t("pm.generatedCount", { n: images.data.pages[0]?.total ?? 0 })}</span>
          ) : null}
        </div>
        {images.isLoading ? (
          <GridSkeleton count={4} />
        ) : images.isError ? (
          <Notice tone="danger">{(images.error as Error).message}</Notice>
        ) : items.length === 0 ? (
          <EmptyState
            icon={<Images />}
            title={t("pm.generatedEmpty")}
            description={t("pm.generatedEmptyBody")}
            action={
              <Button size="sm" variant="primary" icon={<Sparkles />} onClick={() => actions.openInGenerate(p)}>
                {t("pm.useInGenerate")}
              </Button>
            }
          />
        ) : (
          <ImageGrid
            images={items}
            onOpen={setViewer}
            onDownload={imageActions.download}
            hasMore={images.hasNextPage}
            loadMore={() => !images.isFetchingNextPage && images.fetchNextPage()}
            loadingMore={images.isFetchingNextPage}
            showPrompts={settings.data?.show_prompts_on_cards}
            blurNsfw={settings.data?.blur_nsfw ?? true}
          />
        )}
      </section>
      <ImageViewer images={items} index={viewer} onClose={() => setViewer(null)} onIndexChange={setViewer} />
    </div>
  );
}
