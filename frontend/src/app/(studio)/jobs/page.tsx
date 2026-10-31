"use client";

import { ListChecks } from "lucide-react";
import Link from "next/link";
import { useState } from "react";

import { PageHeader } from "@/components/layout/page-header";
import { formatElapsed, formatMoney, Progress, StatusPill } from "@/components/studio/common";
import { EmptyState, Notice, Select, Skeleton } from "@/components/ui";
import { formatRelative } from "@/lib/format";
import { useI18n } from "@/lib/i18n";
import type { MessageKey } from "@/lib/i18n";
import { useJobs } from "@/lib/queries";
import type { JobItem } from "@/lib/types";

const STATUSES = ["queued", "preparing", "running", "completed", "failed", "cancelled", "interrupted"];

function Row({ job }: { job: JobItem }) {
  const { t, locale } = useI18n();
  const running = job.status === "running" || job.status === "preparing";
  return (
    <Link href={job.link} className="job-row">
      <div className="col" style={{ gap: 4, minWidth: 0 }}>
        <div className="row" style={{ gap: 8 }}>
          <span className="badge">{t(`jb.type.${job.type}`)}</span>
          <span className="small truncate" style={{ fontWeight: 600 }}>{job.name}</span>
        </div>
        <div className="tiny subtle truncate">
          {[job.model, job.lora ? `LoRA ${job.lora}` : null, job.provider_name, job.gpu].filter(Boolean).join(" · ")}
        </div>
        {job.error && (job.status === "failed" || job.status === "interrupted") ? <div className="tiny error-text truncate">{job.error}</div> : null}
        {job.stage_detail && running ? <div className="tiny subtle truncate">{job.stage_detail}</div> : null}
      </div>
      <div className="col" style={{ gap: 6 }}>
        <StatusPill status={job.status} />
        {running ? <Progress value={job.progress} /> : null}
        {job.queue_position ? <span className="tiny subtle">{t("jb.queue", { n: job.queue_position })}</span> : null}
        {job.step && job.total_steps ? <span className="tiny subtle">{job.step}/{job.total_steps}</span> : null}
      </div>
      <div className="tiny subtle">
        <div>{formatRelative(job.created_at, locale)}</div>
        <div>{formatElapsed(job.elapsed_seconds)}</div>
      </div>
      <div className="tiny">
        {job.cost !== null ? (
          <>
            {formatMoney(job.cost, locale)}
            <div className="subtle">{job.cost_is_final ? t("jb.costFinal") : t("jb.costEstimate")}</div>
          </>
        ) : (
          <span className="subtle">—</span>
        )}
      </div>
    </Link>
  );
}

export default function JobsPage() {
  const { t } = useI18n();
  const [type, setType] = useState("training");
  const [status, setStatus] = useState("");
  const [provider, setProvider] = useState("");
  const jobs = useJobs({ type, status: status || undefined, provider: provider || undefined });
  const items = jobs.data?.items ?? [];
  return (
    <div className="page">
      <PageHeader title={t("nav.jobs")} subtitle={t("jb.subtitle")} />
      <div className="toolbar">
        <Select
          label={t("jb.type")}
          value={type}
          onChange={setType}
          options={[
            { value: "all", label: t("jb.type.all") },
            { value: "training", label: t("jb.type.training") },
            { value: "generation", label: t("jb.type.generation") },
            { value: "deployment", label: t("jb.type.deployment") },
          ]}
        />
        <Select
          label={t("lib.status")}
          value={status}
          onChange={setStatus}
          highlight={Boolean(status)}
          options={[{ value: "", label: t("lib.statusAny") }, ...STATUSES.map((s) => ({ value: s, label: t(`job.status.${s}` as MessageKey) }))]}
        />
        <Select
          label={t("lib.provider")}
          value={provider}
          onChange={setProvider}
          highlight={Boolean(provider)}
          options={[
            { value: "", label: t("common.all") },
            { value: "runpod", label: "RunPod" },
            { value: "modal", label: "Modal" },
            { value: "comfyui", label: "ComfyUI" },
            { value: "wavespeed", label: "WaveSpeed" },
            { value: "atlas", label: "Atlas Cloud" },
          ]}
        />
        <span className="spacer" />
        <span className="tiny subtle">
          {Object.entries(jobs.data?.counts ?? {})
            .map(([k, v]) => `${t(`job.status.${k}` as MessageKey)} ${v}`)
            .join(" · ")}
        </span>
      </div>
      {jobs.isLoading ? (
        <Skeleton style={{ height: 300 }} />
      ) : jobs.isError ? (
        <Notice tone="danger">{(jobs.error as Error).message}</Notice>
      ) : !items.length ? (
        <EmptyState icon={<ListChecks />} title={t("jb.empty")} description={t("jb.emptyBody")} />
      ) : (
        <div className="card">
          <div className="job-row job-head tiny subtle">
            <span>{t("jb.job")}</span>
            <span>{t("lib.status")}</span>
            <span>{t("jb.time")}</span>
            <span>{t("jb.cost")}</span>
          </div>
          {items.map((j) => (
            <Row key={`${j.type}-${j.id}`} job={j} />
          ))}
        </div>
      )}
    </div>
  );
}
