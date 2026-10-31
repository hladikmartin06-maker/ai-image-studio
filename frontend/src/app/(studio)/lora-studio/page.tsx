"use client";

import { FileUp, Plus, Sparkles, Wand } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { PageHeader } from "@/components/layout/page-header";
import { CreateLoraModal, ImportLoraModal } from "@/components/lora/create";
import { HfTokenCard, Progress, StatusPill } from "@/components/studio/common";
import { Badge, Button, EmptyState, Notice, Skeleton } from "@/components/ui";
import { formatRelative } from "@/lib/format";
import { useI18n } from "@/lib/i18n";
import { useLoras } from "@/lib/queries";

export default function LoraStudioPage() {
  const { t, locale } = useI18n();
  const router = useRouter();
  const loras = useLoras();
  const [modal, setModal] = useState<"create" | "import" | null>(null);
  const projects = (loras.data?.items ?? []).filter((l) => l.source === "trained");

  return (
    <div className="page">
      <PageHeader
        title={t("nav.loraStudio")}
        subtitle={t("ls.subtitle")}
        actions={
          <>
            <Button icon={<FileUp />} onClick={() => setModal("import")}>
              {t("ls.importLora")}
            </Button>
            <Button variant="primary" icon={<Plus />} onClick={() => setModal("create")}>
              {t("ls.createNew")}
            </Button>
          </>
        }
      />
      <div className="comfy-layout">
        <div className="col gap-16">
          <ol className="steps-strip">
            {(["s1", "s2", "s3", "s4", "s5", "s6"] as const).map((s, i) => (
              <li key={s}>
                <span className="step-num">{i + 1}</span>
                {t(`ls.flow.${s}`)}
              </li>
            ))}
          </ol>
          {loras.isLoading ? (
            <Skeleton style={{ height: 200 }} />
          ) : loras.isError ? (
            <Notice tone="danger">{(loras.error as Error).message}</Notice>
          ) : !projects.length ? (
            <EmptyState
              icon={<Wand />}
              title={t("ls.emptyTitle")}
              description={t("ls.emptyBody")}
              action={
                <Button variant="primary" size="sm" icon={<Plus />} onClick={() => setModal("create")}>
                  {t("ls.createNew")}
                </Button>
              }
            />
          ) : (
            <div className="lora-grid">
              {projects.map((l) => {
                const v = l.latest_version;
                return (
                  <Link key={l.id} href={`/lora-studio/${l.id}`} className="card lora-card">
                    <div className="row" style={{ gap: 8 }}>
                      <span className="h3 grow truncate">{l.name}</span>
                      {v ? <StatusPill status={v.status} /> : <Badge>{t("ls.draft")}</Badge>}
                    </div>
                    <div className="tiny subtle">
                      <span className="mono">{l.trigger_word}</span> · Krea 2 {l.base_variant.toUpperCase()}
                      {l.persona_name ? ` · ${l.persona_name}` : ""}
                    </div>
                    <div className="tiny">
                      {t("ls.datasetCount", { n: l.dataset_count ?? 0 })} · {t("ls.versionCount", { n: l.version_count })}
                    </div>
                    {v && l.active ? (
                      <Progress value={v.step && v.total_steps ? (100 * v.step) / v.total_steps : null} />
                    ) : null}
                    <div className="tiny subtle">{formatRelative(l.updated_at, locale)}</div>
                  </Link>
                );
              })}
            </div>
          )}
        </div>
        <aside className="col gap-16">
          <HfTokenCard />
          <section className="card card-pad col gap-12">
            <div className="row" style={{ gap: 8 }}>
              <Sparkles size={16} />
              <span className="h3">{t("ls.howTitle")}</span>
            </div>
            <div className="tiny subtle">{t("ls.howBody")}</div>
            <Link href="/loras" className="link small">{t("nav.myLoras")} →</Link>
          </section>
        </aside>
      </div>
      {modal === "create" ? (
        <CreateLoraModal onClose={() => setModal(null)} onCreated={(l) => router.push(`/lora-studio/${l.id}`)} />
      ) : null}
      {modal === "import" ? (
        <ImportLoraModal onClose={() => setModal(null)} onCreated={(l) => router.push(`/loras/${l.id}`)} />
      ) : null}
    </div>
  );
}
