"use client";

import { FileUp, Library, ListChecks, Plus, Search, X } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { PageHeader } from "@/components/layout/page-header";
import { CreateLoraModal, ImportLoraModal } from "@/components/lora/create";
import { formatElapsed, formatMoney, Progress, StatusPill } from "@/components/studio/common";
import { Badge, Button, EmptyState, IconButton, Notice, Segmented, Skeleton } from "@/components/ui";
import { formatRelative } from "@/lib/format";
import { useDebounced } from "@/lib/hooks";
import { useI18n } from "@/lib/i18n";
import { useJobs, useLoras } from "@/lib/queries";
import type { JobItem, LoraSummary } from "@/lib/types";

type Tab = "loras" | "history";

function LoraCard({ l }: { l: LoraSummary }) {
  const { t, locale } = useI18n();
  const p = l.preferred_version;
  const v = l.latest_version;
  const sample = p?.samples.at(-1) ?? v?.samples.at(-1);
  return (
    <Link href={`/loras/${l.id}`} className="card lora-card">
      {sample ? <img className="lora-cover" src={sample.url} alt="" loading="lazy" /> : null}
      <div className="row" style={{ gap: 8 }}>
        <span className="h3 grow truncate">{l.name}</span>
        {l.active && v ? (
          <StatusPill status={v.status} />
        ) : p ? (
          <Badge tone="success">{t("lo.ready")}</Badge>
        ) : v ? (
          <StatusPill status={v.status} />
        ) : (
          <Badge>{t("ls.draft")}</Badge>
        )}
      </div>
      <div className="tiny subtle">
        <span className="mono">{l.trigger_word}</span>
        {l.persona_name ? ` · ${l.persona_name}` : ""}
        {l.source === "imported" ? ` · ${t("ls.importedBadge")}` : ""}
      </div>
      {l.active && v ? <Progress value={v.step && v.total_steps ? (100 * v.step) / v.total_steps : null} /> : null}
      {!v && l.source === "trained" ? (
        <div className="tiny">{t("lo.photosCount", { n: l.dataset_count ?? 0 })}</div>
      ) : null}
      <div className="tiny subtle">{formatRelative(l.updated_at, locale)}</div>
    </Link>
  );
}

function HistoryRow({ job }: { job: JobItem }) {
  const { t, locale } = useI18n();
  const running = job.status === "running" || job.status === "preparing";
  return (
    <Link href={job.link} className="job-row">
      <div className="col" style={{ gap: 4, minWidth: 0 }}>
        <span className="small truncate" style={{ fontWeight: 600 }}>{job.name}</span>
        <div className="tiny subtle truncate">{[job.provider_name, job.gpu].filter(Boolean).join(" · ")}</div>
        {job.error && (job.status === "failed" || job.status === "interrupted") ? (
          <div className="tiny error-text truncate">{job.error}</div>
        ) : null}
        {job.stage_detail && running ? <div className="tiny subtle truncate">{job.stage_detail}</div> : null}
      </div>
      <div className="col" style={{ gap: 6 }}>
        <StatusPill status={job.status} />
        {running ? <Progress value={job.progress} /> : null}
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

function History() {
  const { t } = useI18n();
  const jobs = useJobs({ type: "training" });
  const items = jobs.data?.items ?? [];
  if (jobs.isLoading) return <Skeleton style={{ height: 300 }} />;
  if (jobs.isError) return <Notice tone="danger">{(jobs.error as Error).message}</Notice>;
  if (!items.length) return <EmptyState icon={<ListChecks />} title={t("lo.historyEmpty")} description={t("lo.historyEmptyBody")} />;
  return (
    <div className="card">
      <div className="job-row job-head tiny subtle">
        <span>{t("jb.job")}</span>
        <span>{t("lib.status")}</span>
        <span>{t("jb.time")}</span>
        <span>{t("jb.cost")}</span>
      </div>
      {items.map((j) => (
        <HistoryRow key={j.id} job={j} />
      ))}
    </div>
  );
}

export default function LorasPage() {
  const { t } = useI18n();
  const router = useRouter();
  const [tab, setTab] = useState<Tab>("loras");
  const [search, setSearch] = useState("");
  const q = useDebounced(search.trim(), 250);
  const loras = useLoras(q);
  const [modal, setModal] = useState<"create" | "import" | null>(null);
  const items = loras.data?.items ?? [];

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- deep link (?tab=history)
    if (new URLSearchParams(window.location.search).get("tab") === "history") setTab("history");
  }, []);

  return (
    <div className="page">
      <PageHeader
        title="LoRA"
        subtitle={t("lo.subtitle")}
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
      <ol className="steps-strip" style={{ marginBottom: 18 }}>
        {(["s1", "s2", "s3", "s4"] as const).map((s, i) => (
          <li key={s}>
            <span className="step-num">{i + 1}</span>
            {t(`lo.flow.${s}`)}
          </li>
        ))}
      </ol>
      <div className="toolbar">
        <Segmented
          label="LoRA"
          value={tab}
          onChange={setTab}
          options={[
            { value: "loras", label: t("lo.tabLoras") },
            { value: "history", label: t("lo.tabHistory") },
          ]}
        />
        {tab === "loras" ? (
          <div className="input-wrap search">
            <Search />
            <input className="input" value={search} placeholder={t("ml.search")} onChange={(e) => setSearch(e.target.value)} aria-label={t("common.search")} />
            {search ? (
              <span className="input-action">
                <IconButton label={t("common.clear")} onClick={() => setSearch("")}>
                  <X />
                </IconButton>
              </span>
            ) : null}
          </div>
        ) : null}
      </div>
      {tab === "history" ? (
        <History />
      ) : loras.isLoading ? (
        <Skeleton style={{ height: 240 }} />
      ) : loras.isError ? (
        <Notice tone="danger">{(loras.error as Error).message}</Notice>
      ) : !items.length ? (
        <EmptyState
          icon={<Library />}
          title={q ? t("ml.noMatches") : t("lo.emptyTitle")}
          description={q ? undefined : t("lo.emptyBody")}
          action={
            q ? undefined : (
              <Button variant="primary" size="sm" icon={<Plus />} onClick={() => setModal("create")}>
                {t("ls.createNew")}
              </Button>
            )
          }
        />
      ) : (
        <div className="lora-grid">
          {items.map((l) => (
            <LoraCard key={l.id} l={l} />
          ))}
        </div>
      )}
      {modal === "create" ? (
        <CreateLoraModal onClose={() => setModal(null)} onCreated={(l) => router.push(`/loras/${l.id}`)} />
      ) : null}
      {modal === "import" ? (
        <ImportLoraModal onClose={() => setModal(null)} onCreated={(l) => router.push(`/loras/${l.id}`)} />
      ) : null}
    </div>
  );
}
