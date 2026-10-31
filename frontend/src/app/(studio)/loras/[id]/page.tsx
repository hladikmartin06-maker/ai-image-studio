"use client";

import { ArrowLeft, Library, Sparkles, Trash2, Wand } from "lucide-react";
import Link from "next/link";
import { useParams, useRouter } from "next/navigation";
import { useState } from "react";
import { toast } from "sonner";

import { PageHeader } from "@/components/layout/page-header";
import { CaptionsTab } from "@/components/lora/captions";
import { TestPanel } from "@/components/lora/compare";
import { DatasetTab } from "@/components/lora/dataset";
import { RunsTab, VersionCard } from "@/components/lora/runs";
import { TrainingTab } from "@/components/lora/training";
import { Badge, Button, EmptyState, Field, Notice, Segmented, Select, Skeleton, useConfirm } from "@/components/ui";
import { ApiError } from "@/lib/api";
import { formatDate } from "@/lib/format";
import { useI18n } from "@/lib/i18n";
import { useDeleteLora, useLora, usePersonas, useUpdateLora } from "@/lib/queries";
import type { LoraDetail, TrainingConfig } from "@/lib/types";

type Tab = "dataset" | "captions" | "training" | "runs" | "versions" | "test" | "details";

function Details({ lora }: { lora: LoraDetail }) {
  const { t, locale } = useI18n();
  const update = useUpdateLora(lora.id);
  const personas = usePersonas();
  const [tags, setTags] = useState(lora.tags.join(", "));
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

export default function LoraPage() {
  const { t } = useI18n();
  const router = useRouter();
  const confirm = useConfirm();
  const { id } = useParams<{ id: string }>();
  const lora = useLora(id);
  const remove = useDeleteLora();
  const [tab, setTab] = useState<Tab | null>(null);
  const [initial, setInitial] = useState<Partial<TrainingConfig> | null>(null);
  const back = (
    <Link href="/loras" className="btn btn-ghost btn-sm" style={{ marginBottom: 12, marginLeft: -8 }}>
      <ArrowLeft /> LoRA
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
  const trained = l.source === "trained";
  // The first tab follows the project's progress; after that it only changes on a click.
  if (tab === null) {
    setTab(!trained ? "versions" : l.versions.length ? "runs" : l.dataset.length < 10 ? "dataset" : "captions");
  }
  const current: Tab = tab ?? (trained ? "dataset" : "versions");
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
        eyebrow="LoRA"
        title={l.name}
        subtitle={
          <span className="row wrap" style={{ gap: 6 }}>
            <Badge><span className="mono">{l.trigger_word}</span></Badge>
            <Badge>Krea 2 {l.base_variant.toUpperCase()}</Badge>
            {l.persona_name ? <Badge tone="accent">{l.persona_name}</Badge> : null}
            {l.preferred_version_id ? <Badge tone="success">{t("ml.readyForGenerate")}</Badge> : null}
          </span>
        }
        actions={
          <>
            <Button variant="ghost" icon={<Trash2 />} onClick={onDelete}>{t("common.delete")}</Button>
            {l.preferred_version_id ? (
              <Link href={`/generate?lora=${l.preferred_version_id}`} className="btn btn-primary">
                <Sparkles /> {t("pm.useInGenerate")}
              </Link>
            ) : null}
          </>
        }
      />
      <div style={{ marginBottom: 18 }}>
        <Segmented
          label="LoRA"
          value={current}
          onChange={setTab}
          options={[
            ...(trained
              ? [
                  { value: "dataset" as Tab, label: `1 · ${t("ls.tab.dataset")} (${l.analysis.train})` },
                  { value: "captions" as Tab, label: `2 · ${t("ls.tab.captions")}` },
                  { value: "training" as Tab, label: `3 · ${t("ls.tab.training")}` },
                  { value: "runs" as Tab, label: `4 · ${t("ls.tab.runs")} (${l.versions.length})` },
                ]
              : [{ value: "versions" as Tab, label: `${t("ml.versions")} (${l.versions.length})` }]),
            { value: "test", label: t("ml.test") },
            { value: "details", label: t("ml.details") },
          ]}
        />
      </div>
      {current === "dataset" ? <DatasetTab lora={l} /> : null}
      {current === "captions" ? <CaptionsTab lora={l} /> : null}
      {current === "training" ? (
        <TrainingTab key={JSON.stringify(initial)} lora={l} initial={initial} onStarted={() => setTab("runs")} />
      ) : null}
      {current === "runs" ? (
        <RunsTab
          lora={l}
          onNewExperiment={(config) => {
            setInitial(config);
            setTab("training");
          }}
        />
      ) : null}
      {current === "versions" ? (
        l.versions.length ? (
          <div className="col gap-16">
            {l.versions.map((v) => (
              <VersionCard key={v.id} lora={l} version={v} />
            ))}
          </div>
        ) : (
          <EmptyState icon={<Wand />} title={t("ls.noRuns")} />
        )
      ) : null}
      {current === "test" ? <TestPanel lora={l} /> : null}
      {current === "details" ? <Details lora={l} /> : null}
    </div>
  );
}
