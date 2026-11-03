"use client";

import { CopyPlus, Download, FastForward, ScrollText, Send, Square, Star, Trash2 } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

import {
  formatElapsed,
  formatMoney,
  GpuMeter,
  isActive,
  Progress,
  StatusPill,
} from "@/components/studio/common";
import { Badge, Button, EmptyState, IconButton, Modal, Notice, Select, useConfirm } from "@/components/ui";
import { ApiError, downloadUrl } from "@/lib/api";
import { formatBytes, formatDate } from "@/lib/format";
import { useI18n } from "@/lib/i18n";
import type { MessageKey } from "@/lib/i18n";
import { useComfyConnections, useLoraLogs, useUpdateLora, useVersionAction, useVersionLikeness } from "@/lib/queries";
import type { LoraDetail, LoraVersion, TrainingConfig } from "@/lib/types";

function elapsed(v: LoraVersion) {
  if (!v.started_at) return null;
  const end = v.completed_at ? new Date(v.completed_at) : new Date();
  return (end.getTime() - new Date(v.started_at).getTime()) / 1000;
}

// Continuing a LoRA: fewer steps and the gentle learning speed by default (the studio's
// starting point; it refines the earlier LoRA instead of overwriting it).
export const CONTINUE_STEPS = 1000;
export const CONTINUE_LR = 0.00006;

/** A version's training settings without what belongs to that run only (trainer image,
 * cloud, the version it continued), so they can be used for a new run. */
export function trainingSettings<T extends Partial<TrainingConfig>>(
  config: T & { image?: string | null; cloud_type?: string; continue_from_number?: number },
): Omit<T, "image" | "cloud_type" | "continue_from_number"> {
  const { image: _image, cloud_type: _cloud, continue_from_number: _from, ...settings } = config;
  void _image;
  void _cloud;
  void _from;
  return settings;
}

/** One image per prompt at a step (the newest file wins, e.g. after a restart). */
function samplesAt(samples: LoraVersion["samples"], step: number) {
  const byIndex = new Map<number, LoraVersion["samples"][number]>();
  for (const s of samples) if (s.step === step) byIndex.set(s.index ?? 0, s);
  return [...byIndex.entries()].sort((a, b) => a[0] - b[0]).map(([, s]) => s);
}

/** The newest validation images: one per prompt from the latest step (older steps and
 *  images from before a restart are kept on the Mac but not shown). */
export function latestSamples(samples: LoraVersion["samples"]) {
  const stepped = samples.filter((s) => s.step !== null);
  if (!stepped.length) return { step: null, items: samples.slice(-3) };
  const step = Math.max(...stepped.map((s) => s.step as number));
  return { step, items: samplesAt(stepped, step) };
}

/** Validation images: the newest step large, older steps on demand (history), each with
 *  the face likeness of that step to the LoRA's photos; the best step's checkpoint can become
 *  its own version. */
function SamplesAndLikeness({
  lora,
  version,
  onSaveStep,
}: {
  lora: LoraDetail;
  version: LoraVersion;
  onSaveStep: (step: number) => void;
}) {
  const { t } = useI18n();
  const likeness = useVersionLikeness(lora.id, version, true);
  const [history, setHistory] = useState(false);
  const latest = latestSamples(version.samples);
  if (!latest.items.length) return null;
  const data = likeness.data;
  const threshold = data?.threshold ?? 0.363;
  const pct = (v: number) => Math.round(v * 100);
  const scoreOf = new Map((data?.steps ?? []).map((r) => [r.step, r.score]));
  // Shown against her own photos (100 % = as alike as they are to each other); the raw
  // similarity only when that mark is unknown.
  const shownOf = new Map((data?.steps ?? []).map((r) => [r.step, r.relative ?? r.score]));
  const baseline = data?.baseline ?? null;
  const best = data?.best_step ?? null;
  const bestScore = best !== null ? (scoreOf.get(best) ?? null) : null;
  const bestShown = best !== null ? (shownOf.get(best) ?? null) : null;
  const steps = [...new Set(version.samples.filter((s) => s.step !== null).map((s) => s.step as number))].sort((a, b) => b - a);
  const older = steps.filter((step) => step !== latest.step);
  const finished = !isActive(version.status);
  const hasCheckpoint = best !== null && version.checkpoints.some((c) => c.step === best);
  const label = (step: number) => {
    const shown = shownOf.get(step);
    return `${t("ls.likenessStep", { step })}${shown != null ? ` · ${pct(shown)} %` : ""}`;
  };
  const strip = (items: LoraVersion["samples"]) => (
    <div className="sample-strip">
      {items.map((s) => (
        <a key={s.file} href={s.url} target="_blank" rel="noreferrer" className="sample">
          <img src={s.url} alt="" loading="lazy" />
        </a>
      ))}
    </div>
  );
  return (
    <div className="col" style={{ gap: 8 }}>
      <div className="eyebrow row" style={{ gap: 6 }}>
        {latest.step !== null ? (
          <>
            {latest.step === best ? <Star size={11} /> : null}
            {t("ls.samples")} · {label(latest.step)}
          </>
        ) : (
          t("ls.samples")
        )}
      </div>
      {strip(latest.items)}
      {bestScore !== null ? (
        bestScore < threshold ? (
          <Notice tone="warning">{t("ls.likenessWeak", { pct: pct(threshold) })}</Notice>
        ) : (
          <div className="row wrap small" style={{ gap: 8, alignItems: "center" }}>
            <span>{t("ls.likenessBest", { step: best!, pct: pct(bestShown ?? bestScore) })}</span>
            {finished && best !== latest.step ? (
              hasCheckpoint ? (
                <Button size="sm" icon={<Star />} onClick={() => onSaveStep(best!)}>
                  {t("ls.likenessSave", { step: best! })}
                </Button>
              ) : (
                <span className="tiny subtle">{t("ls.likenessNoCheckpoint", { step: best! })}</span>
              )
            ) : null}
          </div>
        )
      ) : null}
      {data && data.steps.some((r) => r.score !== null) ? (
        <div className="tiny subtle">
          {baseline !== null
            ? t("ls.likenessHintRelative", { base: pct(baseline), pct: pct(threshold) })
            : t("ls.likenessHint", { pct: pct(threshold) })}
        </div>
      ) : likeness.isFetching ? (
        <div className="tiny subtle">{t("ls.likenessMeasuring")}</div>
      ) : likeness.error ? (
        <div className="tiny subtle">{(likeness.error as Error).message}</div>
      ) : null}
      {older.length ? (
        <>
          <button type="button" className="link-button tiny" style={{ alignSelf: "flex-start" }} onClick={() => setHistory((v) => !v)} aria-expanded={history}>
            {history ? t("ls.historyHide") : t("ls.historyShow", { n: older.length })}
          </button>
          {history ? (
            <div className="sample-history">
              {older.map((step) => (
                <div key={step} className="sample-row">
                  <div className="tiny subtle row" style={{ gap: 4 }}>
                    {step === best ? <Star size={11} /> : null}
                    {label(step)}
                  </div>
                  {strip(samplesAt(version.samples, step))}
                </div>
              ))}
            </div>
          ) : null}
        </>
      ) : null}
    </div>
  );
}

export function VersionCard({
  lora,
  version,
  onNewExperiment,
}: {
  lora: LoraDetail;
  version: LoraVersion;
  onNewExperiment?: (config: Partial<TrainingConfig>) => void;
}) {
  const { t, locale } = useI18n();
  const confirm = useConfirm();
  const action = useVersionAction(lora.id);
  const updateLora = useUpdateLora(lora.id);
  const conns = useComfyConnections();
  const [logs, setLogs] = useState(false);
  const [deployTo, setDeployTo] = useState("");
  const active = isActive(version.status);
  const preferred = lora.preferred_version_id === version.id;
  const percent =
    version.step !== null && version.total_steps ? Math.round((100 * version.step) / version.total_steps) : null;
  const uploadable = (conns.data?.items ?? []).filter((c) => c.status === "ready" && c.file_capability !== "none");

  const run = async (fn: () => Promise<unknown>, done?: string) => {
    try {
      await fn();
      if (done) toast.success(done);
    } catch (e) {
      toast.error(e instanceof ApiError ? e.message : t("errors.generic"));
    }
  };

  return (
    <article className="card version-card" data-preferred={preferred || undefined}>
      <div className="card-head">
        <span className="h3">v{version.number}</span>
        <StatusPill status={version.status} />
        {preferred ? (
          <Badge tone="accent">
            <Star size={11} /> {t("ls.preferred")}
          </Badge>
        ) : null}
        {version.source === "imported" ? <Badge>{t("ls.importedBadge")}</Badge> : null}
        {version.config.continue_from_number ? (
          <Badge>{t("ls.continuedFrom", { n: version.config.continue_from_number })}</Badge>
        ) : null}
        <span className="grow tiny subtle truncate">
          {version.preset ? t(`ls.preset.${version.preset}` as MessageKey) : ""}
          {version.gpu_type ? ` · ${version.gpu_count}× ${version.gpu_type} · ${version.gpu_provider === "modal" ? "Modal" : "RunPod"}` : ""}
        </span>
        <span className="tiny subtle">{formatDate(version.created_at, locale)}</span>
      </div>
      <div className="card-body col gap-12">
        {active ? (
          <>
            <div className="row tiny" style={{ gap: 8 }}>
              <span className="grow">{version.stage_detail || t(`job.status.${version.status}`)}</span>
              {version.step !== null ? (
                <span>
                  {t("ls.stepOf", { step: version.step, total: version.total_steps ?? "?" })}
                </span>
              ) : null}
            </div>
            <Progress value={version.status === "running" ? percent : null} />
          </>
        ) : null}
        {version.memory_retries?.length ? (
          <Notice tone="warning">
            {version.memory_retries.map((r, i) => (
              <div key={i}>
                {t("ls.memoryRetry", {
                  changes: r.changes.map((c) => t(`ls.memoryChange.${c}` as MessageKey)).join(", "),
                })}
              </div>
            ))}
          </Notice>
        ) : null}
        {version.error_message && !active ? (
          <Notice tone={version.status === "cancelled" ? undefined : "danger"}>
            <pre className="error-pre">{version.error_message}</pre>
          </Notice>
        ) : null}
        <div className="comfy-grid">
          <div className="comfy-stat">
            <span className="tiny subtle">{t("jb.elapsed")}</span>
            <span className="small">{formatElapsed(elapsed(version))}</span>
          </div>
          <div className="comfy-stat">
            <span className="tiny subtle">{version.cost_final !== null ? t("ls.costFinal") : t("ls.costEstimate")}</span>
            <span className="small">
              {version.cost_estimate !== null ? formatMoney(version.cost_estimate, locale) : version.gpu_provider === "modal" ? t("jb.noPriceApi") : "—"}
            </span>
          </div>
          <div className="comfy-stat">
            <span className="tiny subtle">{t("ls.datasetSize")}</span>
            <span className="small">{version.dataset_size || "—"}</span>
          </div>
          {version.file_size ? (
            <div className="comfy-stat">
              <span className="tiny subtle">{t("ls.file")}</span>
              <span className="small">{formatBytes(version.file_size, locale)}</span>
            </div>
          ) : null}
          {version.config.rank ? (
            <div className="comfy-stat">
              <span className="tiny subtle">{t("ls.settings")}</span>
              <span className="small">
                r{version.config.rank} · α{version.config.alpha} · {version.config.learning_rate} · {version.config.steps}
              </span>
            </div>
          ) : null}
        </div>
        {active ? <GpuMeter gpu={version.gpu} /> : null}
        {version.samples.length ? (
          <SamplesAndLikeness
            lora={lora}
            version={version}
            onSaveStep={(step) =>
              run(() => action.mutateAsync({ versionId: version.id, action: `checkpoints/${step}/as-version` }), t("ls.checkpointSaved"))
            }
          />
        ) : null}
        {version.checkpoints.length ? (
          <div className="col" style={{ gap: 4 }}>
            <div className="eyebrow">{t("ls.checkpoints")}</div>
            {version.checkpoints.map((c) => (
              <div key={c.step} className="row tiny" style={{ gap: 8 }}>
                <span className="grow">{t("ls.checkpointAt", { step: c.step })}</span>
                {c.size ? <span className="subtle">{formatBytes(c.size, locale)}</span> : null}
                <Button size="sm" variant="ghost" icon={<Download />} onClick={() => downloadUrl(c.url)}>
                  {t("common.download")}
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={() => run(() => action.mutateAsync({ versionId: version.id, action: `checkpoints/${c.step}/as-version` }), t("ls.checkpointSaved"))}
                >
                  {t("ls.asVersion")}
                </Button>
              </div>
            ))}
          </div>
        ) : null}
        <div className="row wrap" style={{ gap: 6 }}>
          {active ? (
            <Button
              size="sm"
              variant="danger"
              icon={<Square />}
              onClick={async () => {
                if (await confirm({ title: t("ls.cancelTitle"), description: t("ls.cancelBody"), confirmLabel: t("ls.cancelRun"), danger: true }))
                  await run(() => action.mutateAsync({ versionId: version.id, action: "cancel", body: { stop_gpu: true } }));
              }}
            >
              {t("ls.cancelRun")}
            </Button>
          ) : null}
          {version.gpu_provider ? (
            <Button size="sm" icon={<ScrollText />} onClick={() => setLogs(true)}>
              {t("cf.logs")}
            </Button>
          ) : null}
          {version.status === "completed" && version.file_url ? (
            <>
              <Button size="sm" icon={<Download />} onClick={() => downloadUrl(version.file_url!)}>
                {t("common.download")}
              </Button>
              {!preferred ? (
                <Button size="sm" icon={<Star />} onClick={() => run(() => updateLora.mutateAsync({ preferred_version_id: version.id }), t("ls.preferredSet"))}>
                  {t("ls.makePreferred")}
                </Button>
              ) : null}
              {uploadable.length ? (
                <span className="row" style={{ gap: 6 }}>
                  <Select
                    label={t("ls.copyTo")}
                    value={deployTo || null}
                    placeholder={t("ls.copyTo")}
                    onChange={setDeployTo}
                    options={uploadable.map((c) => ({ value: c.id, label: c.name }))}
                  />
                  <Button
                    size="sm"
                    icon={<Send />}
                    disabled={!deployTo}
                    onClick={() => run(() => action.mutateAsync({ versionId: version.id, action: "deploy", body: { connection_id: deployTo } }), t("ls.copied"))}
                  >
                    {t("ls.copy")}
                  </Button>
                </span>
              ) : null}
            </>
          ) : null}
          <span className="spacer" />
          {onNewExperiment && version.config.rank && version.status === "completed" && version.file_url ? (
            <Button
              size="sm"
              icon={<FastForward />}
              onClick={() =>
                onNewExperiment({ ...trainingSettings(version.config), continue_from: version.id, steps: CONTINUE_STEPS, learning_rate: CONTINUE_LR })
              }
            >
              {t("ls.continueTraining")}
            </Button>
          ) : null}
          {onNewExperiment && version.config.rank ? (
            <Button size="sm" variant="ghost" icon={<CopyPlus />} onClick={() => onNewExperiment({ ...trainingSettings(version.config), continue_from: null })}>
              {t("ls.newExperiment")}
            </Button>
          ) : null}
          {!active ? (
            <IconButton
              label={t("common.delete")}
              onClick={async () => {
                if (await confirm({ title: t("ls.deleteVersionTitle", { n: version.number }), description: t("ls.deleteVersionBody"), confirmLabel: t("common.delete"), danger: true }))
                  await run(() => action.mutateAsync({ versionId: version.id, action: "delete" }));
              }}
            >
              <Trash2 />
            </IconButton>
          ) : null}
        </div>
      </div>
      {logs ? <LogsModal lora={lora} version={version} onClose={() => setLogs(false)} /> : null}
    </article>
  );
}

function LogsModal({ lora, version, onClose }: { lora: LoraDetail; version: LoraVersion; onClose: () => void }) {
  const { t } = useI18n();
  const logs = useLoraLogs(lora.id, version.id, isActive(version.status));
  return (
    <Modal open wide onClose={onClose} title={t("cf.logs")} description={`${lora.name} v${version.number}`}>
      <pre className="log-box">{logs.data?.text || t("cf.noLogs")}</pre>
    </Modal>
  );
}

export function RunsTab({ lora, onNewExperiment }: { lora: LoraDetail; onNewExperiment: (c: Partial<TrainingConfig>) => void }) {
  const { t } = useI18n();
  if (!lora.versions.length) {
    return <EmptyState icon={<ScrollText />} title={t("ls.noRuns")} description={t("ls.noRunsBody")} />;
  }
  return (
    <div className="col gap-16">
      {lora.versions.map((v) => (
        <VersionCard key={v.id} lora={lora} version={v} onNewExperiment={onNewExperiment} />
      ))}
    </div>
  );
}
