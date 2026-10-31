"use client";

import { ArrowLeft, Library, Sparkles, Trash2, Wand } from "lucide-react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useState } from "react";
import { toast } from "sonner";

import { PageHeader } from "@/components/layout/page-header";
import { TestPanel } from "@/components/lora/compare";
import { VersionCard } from "@/components/lora/runs";
import { Badge, Button, EmptyState, Field, Notice, Segmented, Select, Skeleton, useConfirm } from "@/components/ui";
import { ApiError } from "@/lib/api";
import { formatDate } from "@/lib/format";
import { useI18n } from "@/lib/i18n";
import { useDeleteLora, useLora, usePersonas, useUpdateLora } from "@/lib/queries";
import type { LoraDetail } from "@/lib/types";

type Tab = "versions" | "test" | "details";

function Details({ lora }: { lora: LoraDetail }) {
  const { t, locale } = useI18n();
  const update = useUpdateLora(lora.id);
  const personas = usePersonas();
  const [tags, setTags] = useState(lora.tags.join(", "));
  const [notes, setNotes] = useState(lora.notes);
  const save = (body: Record<string, unknown>) =>
    update.mutate(body, {
      onSuccess: () => toast.success(t("ds.saved")),
      onError: (e) => toast.error((e as Error).message),
    });
  return (
    <div className="comfy-layout">
      <section className="card card-pad col gap-16">
        <Field label={t("ml.tags")} hint={t("ml.tagsHint")}>
          <input className="input" value={tags} onChange={(e) => setTags(e.target.value)} onBlur={() => save({ tags: tags.split(",").map((s) => s.trim()).filter(Boolean) })} />
        </Field>
        <Field label={t("ls.notes")}>
          <textarea className="textarea" rows={4} value={notes} onChange={(e) => setNotes(e.target.value)} onBlur={() => notes !== lora.notes && save({ notes })} />
        </Field>
        <Field label={t("pm.section")} hint={t("ml.personaHint")}>
          <Select
            label={t("pm.section")}
            value={lora.persona_id ?? ""}
            onChange={(v) => save({ persona_id: v || null })}
            options={[{ value: "", label: t("pm.useNone") }, ...(personas.data?.items ?? []).map((p) => ({ value: p.id, label: p.name }))]}
          />
        </Field>
      </section>
      <section className="card card-pad">
        <dl className="kv">
          <dt>{t("ls.trigger")}</dt>
          <dd className="mono">{lora.trigger_word}</dd>
          <dt>{t("ls.classWord")}</dt>
          <dd>{lora.class_word || "—"}</dd>
          <dt>{t("k2.architecture")}</dt>
          <dd>Krea 2 (krea2)</dd>
          <dt>{t("ls.baseModel")}</dt>
          <dd>{lora.base_variant === "raw" ? "krea/Krea-2-Raw" : "krea/Krea-2-Turbo"}</dd>
          <dt>{t("ml.usableOn")}</dt>
          <dd>{lora.base_variant === "raw" ? "Krea 2 RAW, Krea 2 Turbo" : "Krea 2 Turbo"}</dd>
          <dt>{t("ml.source")}</dt>
          <dd>{lora.source === "imported" ? t("ls.importedBadge") : t("ml.trained")}</dd>
          <dt>{t("ml.dataset")}</dt>
          <dd>{t("ls.datasetCount", { n: lora.dataset.length })}</dd>
          <dt>{t("viewer.created")}</dt>
          <dd>{formatDate(lora.created_at, locale)}</dd>
        </dl>
      </section>
    </div>
  );
}

export default function LoraLibraryPage() {
  const { t } = useI18n();
  const router = useRouter();
  const confirm = useConfirm();
  const { id } = useParams<{ id: string }>();
  const lora = useLora(id);
  const remove = useDeleteLora();
  const [tab, setTab] = useState<Tab>("versions");
  const back = (
    <Link href="/loras" className="btn btn-ghost btn-sm" style={{ marginBottom: 12, marginLeft: -8 }}>
      <ArrowLeft /> {t("nav.myLoras")}
    </Link>
  );
  if (lora.isLoading) {
    return (
      <div className="page">
        {back}
        <Skeleton style={{ height: 400 }} />
      </div>
    );
  }
  if (lora.isError || !lora.data) {
    const missing = lora.error instanceof ApiError && lora.error.status === 404;
    return (
      <div className="page">
        {back}
        {missing ? <EmptyState icon={<Library />} title={t("ls.notFound")} /> : <Notice tone="danger">{(lora.error as Error)?.message}</Notice>}
      </div>
    );
  }
  const l = lora.data;
  const onDelete = async () => {
    const ok = await confirm({
      title: t("ml.deleteTitle", { name: l.name }),
      description: t("ml.deleteBody", { n: l.versions.length }),
      confirmLabel: t("common.delete"),
      danger: true,
    });
    if (!ok) return;
    try {
      await remove.mutateAsync(l.id);
      toast.success(t("pm.deleted", { name: l.name }));
      router.replace("/loras");
    } catch (e) {
      toast.error((e as Error).message);
    }
  };
  return (
    <div className="page">
      {back}
      <PageHeader
        eyebrow={t("nav.myLoras")}
        title={l.name}
        subtitle={
          <span className="row wrap" style={{ gap: 6 }}>
            <Badge><span className="mono">{l.trigger_word}</span></Badge>
            <Badge>Krea 2 {l.base_variant.toUpperCase()}</Badge>
            {l.preferred_version_id ? <Badge tone="success">{t("ml.readyForGenerate")}</Badge> : <Badge tone="warning">{t("ml.noUsable")}</Badge>}
          </span>
        }
        actions={
          <>
            <Button variant="ghost" icon={<Trash2 />} onClick={onDelete}>{t("common.delete")}</Button>
            {l.source === "trained" ? (
              <Link href={`/lora-studio/${l.id}`} className="btn">
                <Wand /> {t("ml.openStudio")}
              </Link>
            ) : null}
            <Link href={`/generate?lora=${l.preferred_version_id ?? ""}`} className="btn btn-primary" aria-disabled={!l.preferred_version_id}>
              <Sparkles /> {t("pm.useInGenerate")}
            </Link>
          </>
        }
      />
      <div style={{ marginBottom: 18 }}>
        <Segmented
          label={t("nav.myLoras")}
          value={tab}
          onChange={setTab}
          options={[
            { value: "versions", label: `${t("ml.versions")} (${l.versions.length})` },
            { value: "test", label: t("ml.test") },
            { value: "details", label: t("ml.details") },
          ]}
        />
      </div>
      {tab === "versions" ? (
        l.versions.length ? (
          <div className="col gap-16">
            {l.versions.map((v) => (
              <VersionCard key={v.id} lora={l} version={v} />
            ))}
          </div>
        ) : (
          <EmptyState icon={<Wand />} title={t("ls.noRuns")} description={t("ls.noRunsBody")} />
        )
      ) : null}
      {tab === "test" ? <TestPanel lora={l} /> : null}
      {tab === "details" ? <Details lora={l} /> : null}
    </div>
  );
}
