"use client";

import { ArrowLeft, Library } from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";
import { useState } from "react";

import { PageHeader } from "@/components/layout/page-header";
import { CaptionsTab } from "@/components/lora/captions";
import { DatasetTab } from "@/components/lora/dataset";
import { RunsTab } from "@/components/lora/runs";
import { TrainingTab } from "@/components/lora/training";
import { Badge, EmptyState, Notice, Segmented, Skeleton } from "@/components/ui";
import { ApiError } from "@/lib/api";
import { useI18n } from "@/lib/i18n";
import { useLora } from "@/lib/queries";
import type { TrainingConfig } from "@/lib/types";

type Tab = "dataset" | "captions" | "training" | "runs";

export default function LoraWorkspacePage() {
  const { t } = useI18n();
  const { id } = useParams<{ id: string }>();
  const lora = useLora(id);
  const [tab, setTab] = useState<Tab | null>(null);
  const [initial, setInitial] = useState<Partial<TrainingConfig> | null>(null);

  const back = (
    <Link href="/lora-studio" className="btn btn-ghost btn-sm" style={{ marginBottom: 12, marginLeft: -8 }}>
      <ArrowLeft /> {t("nav.loraStudio")}
    </Link>
  );
  if (lora.isLoading) {
    return (
      <div className="page">
        {back}
        <Skeleton style={{ height: 480 }} />
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
  // The first tab follows the project's progress; after that it only changes on a click.
  if (tab === null) setTab(l.versions.length ? "runs" : l.dataset.length < 10 ? "dataset" : "captions");
  const current: Tab = tab ?? "dataset";

  return (
    <div className="page">
      {back}
      <PageHeader
        eyebrow={t("nav.loraStudio")}
        title={l.name}
        subtitle={
          <span className="row wrap" style={{ gap: 6 }}>
            <Badge>
              <span className="mono">{l.trigger_word}</span>
            </Badge>
            <Badge>Krea 2 {l.base_variant.toUpperCase()}</Badge>
            {l.class_word ? <Badge>{l.class_word}</Badge> : null}
            {l.persona_name ? <Badge tone="accent">{l.persona_name}</Badge> : null}
          </span>
        }
        actions={
          <Link href={`/loras/${l.id}`} className="btn">
            <Library /> {t("ls.openInLibrary")}
          </Link>
        }
      />
      <div style={{ marginBottom: 18 }}>
        <Segmented
          label={t("nav.loraStudio")}
          value={current}
          onChange={setTab}
          options={[
            { value: "dataset", label: `1 · ${t("ls.tab.dataset")} (${l.analysis.train})` },
            { value: "captions", label: `2 · ${t("ls.tab.captions")} (${l.analysis.captioned}/${l.analysis.train})` },
            { value: "training", label: `3 · ${t("ls.tab.training")}` },
            { value: "runs", label: `4 · ${t("ls.tab.runs")} (${l.versions.length})` },
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
    </div>
  );
}
