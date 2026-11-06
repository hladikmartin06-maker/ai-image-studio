"use client";

import {
  Check,
  ChevronDown,
  CircleSlash,
  Copy,
  Download,
  ListChecks,
  Loader2,
  Maximize2,
  RefreshCw,
  SlidersHorizontal,
  Sparkles,
  Square,
  Trash2,
  X,
} from "lucide-react";
import { useMemo, useRef, useState } from "react";
import { toast } from "sonner";

import { ComfyConnectionBar } from "@/components/comfy/connection-bar";
import { loraFits } from "@/components/comfy/generate-panel";
import { type PickedPhoto, PhotoPicker } from "@/components/comfy/photo-picker";
import { CompareSlider } from "@/components/lora/compare";
import { copyText, useImageActions, useQuickDelete } from "@/components/images/actions";
import { ImageTile } from "@/components/images/image-tile";
import { ImageViewer } from "@/components/images/image-viewer";
import { AddToLora } from "@/components/lora/add-to-lora";
import { newRequestId } from "@/components/studio/common";
import { Badge, Button, EmptyState, Field, IconButton, Notice, Segmented, Select, Skeleton, Switch } from "@/components/ui";
import { ApiError } from "@/lib/api";
import { formatBytes } from "@/lib/format";
import { useI18n } from "@/lib/i18n";
import type { MessageKey } from "@/lib/i18n";
import { isHidden, useHiddenVersion } from "@/lib/pending-deletes";
import {
  useAdvancedCapabilities,
  useAdvancedJobs,
  useCancelAdvanced,
  useComfyConnections,
  useComfyModelDownload,
  useLoras,
  usePlanAdvanced,
  useSettings,
  useStartAdvanced,
  useTestComfy,
} from "@/lib/queries";
import type { AdvancedJob, AdvancedReport, AdvancedSpec, AdvancedStage, AdvancedTarget } from "@/lib/types";

type Mode = "text-to-image" | "edit";
const TARGETS: AdvancedTarget[] = ["hair", "mouth", "eyes", "face", "expression", "clothing", "background", "body", "whole"];
const RATIOS = ["1:1", "4:5", "3:4", "2:3", "9:16", "5:4", "4:3", "3:2", "16:9"];

function pct(v: number | null | undefined) {
  return v === null || v === undefined ? "—" : `${Math.round(v * 100)} %`;
}

/** The engine's stage keys as people read them. */
function stageLabel(stage: AdvancedStage, t: (k: MessageKey, v?: Record<string, string | number>) => string) {
  const target = (name: string) => (TARGETS.includes(name as AdvancedTarget) ? t(`adv.target.${name}` as MessageKey) : name);
  if (stage.key === "plan") return t("adv.stage.plan");
  if (stage.key === "prepare") return t(stage.status === "skipped" ? "adv.stage.prepareMissing" : "adv.stage.prepare");
  if (stage.key === "generate") return t("adv.stage.generate");
  if (stage.key === "enhance") return t("adv.stage.enhance");
  if (stage.key.startsWith("check")) return t("adv.stage.check");
  if (stage.key.startsWith("edit")) return t("adv.stage.edit", { target: target(stage.label.split(" ")[0]) });
  if (stage.key.startsWith("fix")) {
    const name = stage.key.split("-")[1];
    return name ? t("adv.stage.fixRegion", { target: target(name) }) : t("adv.stage.fixAll");
  }
  return stage.label;
}

function seconds(stage: AdvancedStage) {
  if (!stage.finished_at) return null;
  const s = (new Date(stage.finished_at).getTime() - new Date(stage.started_at).getTime()) / 1000;
  return `${s < 10 ? s.toFixed(1) : Math.round(s)} s`;
}

function StageIcon({ status }: { status: AdvancedStage["status"] }) {
  if (status === "running") return <Loader2 size={14} className="spin" />;
  if (status === "completed") return <Check size={14} color="var(--success)" />;
  if (status === "failed") return <X size={14} color="var(--danger)" />;
  return <CircleSlash size={14} color="var(--text-3)" />;
}

function Report({ report, job }: { report: AdvancedReport; job: AdvancedJob }) {
  const { t } = useI18n();
  return (
    <div className="col" style={{ gap: 8 }}>
      {report.checks.length ? (
        <ul className="adv-checks">
          {report.checks.map((c) => (
            <li key={c.id}>
              <span className={`adv-mark ${c.passed === true ? "ok" : c.passed === false ? "bad" : "unknown"}`}>
                {c.passed === true ? <Check size={12} /> : c.passed === false ? <X size={12} /> : "?"}
              </span>
              <span className="grow">
                <span className="small">{c.measured ? t("adv.faceMatch") : c.question}</span>
                {c.note ? <span className="tiny subtle"> — {c.note}</span> : null}
              </span>
              <Badge>{t(c.kind === "modify" ? "adv.kindModify" : "adv.kindPreserve")}</Badge>
            </li>
          ))}
        </ul>
      ) : null}
      <div className="row wrap tiny" style={{ gap: 12 }}>
        {report.identity !== null ? (
          <span>
            {t(report.identity_reference === "lora" ? "adv.identityLora" : "adv.identityOriginal")}:{" "}
            <strong>{pct(report.identity)}</strong>
          </span>
        ) : null}
        {report.changed_area !== null ? (
          <span>
            {t("adv.changedArea")}: <strong>{report.changed_area.toLocaleString()} %</strong>
          </span>
        ) : null}
        {report.ai ? (
          <span className="subtle">
            {t("adv.aiUsed", { model: report.ai.model, tokens: report.ai.input_tokens + report.ai.output_tokens })}
          </span>
        ) : null}
      </div>
      {report.artifacts ? <div className="tiny">{t("adv.artifacts", { text: report.artifacts })}</div> : null}
      {report.error ? <div className="tiny subtle">{report.error}</div> : null}
      <div className="tiny subtle">
        {t("adv.reportHint", { pct: pct(report.threshold) })}
        {job.mode === "edit" ? ` ${t("adv.changedHint")}` : ""}
      </div>
    </div>
  );
}

function JobView({
  job,
  blurNsfw,
  onLoad,
}: {
  job: AdvancedJob;
  blurNsfw: boolean;
  onLoad: (job: AdvancedJob) => void;
}) {
  const { t } = useI18n();
  const cancel = useCancelAdvanced();
  const actions = useImageActions();
  const quick = useQuickDelete();
  useHiddenVersion();
  const [viewer, setViewer] = useState<number | null>(null);
  const active = job.status === "queued" || job.status === "running";
  const last = job.reports.at(-1);
  // Deleted images disappear at once (they go after the "Undo" moment).
  const images = job.images.filter((img) => !isHidden(img.id));
  const result = job.result && !isHidden(job.result.id) ? job.result : null;
  const open = (id: string) => {
    const index = images.findIndex((img) => img.id === id);
    if (index >= 0) setViewer(index);
  };
  const ar = result ? result.width / result.height : 3 / 4;
  return (
    <div className="card card-pad col" style={{ gap: 14 }}>
      <div className="row wrap" style={{ gap: 8 }}>
        <Badge tone={job.status === "completed" ? "success" : job.status === "failed" ? "danger" : active ? "accent" : undefined}>
          {t(`adv.status.${job.status}` as MessageKey)}
        </Badge>
        <span className="small grow truncate" title={job.prompt}>{job.prompt}</span>
        {job.seconds !== null ? <span className="tiny subtle">{t("adv.took", { s: Math.round(job.seconds) })}</span> : null}
        <div className="row" style={{ gap: 2 }}>
          <IconButton
            label={t("common.copyPrompt")}
            onClick={async () => {
              if (await copyText(job.prompt)) toast.success(t("common.promptCopied"));
            }}
          >
            <Copy />
          </IconButton>
          <IconButton label={t("gen.loadSettings")} onClick={() => onLoad(job)}>
            <SlidersHorizontal />
          </IconButton>
          {active ? null : (
            <IconButton
              label={job.steps_running ? t("adv.deleteBusy") : t("adv.deleteJob")}
              disabled={job.steps_running > 0}
              onClick={() => quick.deleteAdvancedQuick(job)}
            >
              <Trash2 />
            </IconButton>
          )}
        </div>
        {active ? (
          <Button size="sm" variant="ghost" icon={<Square />} loading={cancel.isPending} onClick={() => cancel.mutate(job.id)}>
            {t("adv.cancel")}
          </Button>
        ) : null}
      </div>
      {job.stages.length ? (
        <ol className="adv-stages">
          {job.stages.map((s) => (
            <li key={s.key} data-status={s.status}>
              <StageIcon status={s.status} />
              <span className="grow small">{stageLabel(s, t)}</span>
              <span className="tiny subtle">{seconds(s) ?? ""}</span>
            </li>
          ))}
        </ol>
      ) : null}
      {job.error_message ? <Notice tone={job.status === "cancelled" ? undefined : "danger"}>{job.error_message}</Notice> : null}
      {result ? (
        <div className="col" style={{ gap: 8 }}>
          {job.source ? (
            <CompareSlider a={job.source.url} b={result.url} labelA={t("adv.original")} labelB={t("adv.result")} />
          ) : (
            <div className="job-grid adv-result" style={{ ["--ar" as string]: String(ar) }}>
              <ImageTile
                image={result}
                onOpen={() => open(result.id)}
                onDownload={() => actions.download(result)}
                onDelete={() => quick.deleteImageQuick(result)}
                showPrompt={false}
                blurNsfw={blurNsfw}
              />
            </div>
          )}
          <div className="row wrap" style={{ gap: 6 }}>
            <Button size="sm" icon={<Maximize2 />} onClick={() => open(result.id)}>
              {t("adv.open")}
            </Button>
            <Button size="sm" icon={<Download />} onClick={() => actions.download(result)}>
              {t("common.download")}
            </Button>
            <Button size="sm" icon={<Sparkles />} onClick={() => actions.enhanceImage(result)}>
              {t("images.enhance")}
            </Button>
            <AddToLora imageIds={[result.id]} />
            <span className="spacer" />
            <Button size="sm" variant="ghost" icon={<Trash2 />} onClick={() => quick.deleteImageQuick(result)}>
              {t("adv.deletePhoto")}
            </Button>
          </div>
        </div>
      ) : null}
      {last ? (
        <div className="col" style={{ gap: 6 }}>
          <div className="eyebrow">
            {t("adv.report")}
            {job.reports.length > 1 ? ` · ${t("adv.attempts", { n: job.reports.length })}` : ""}
          </div>
          <Report report={job.reports.find((r) => r.image_id === job.result?.image_id) ?? last} job={job} />
        </div>
      ) : null}
      {images.length > 1 || (images.length > 0 && !result) ? (
        <details>
          <summary className="tiny subtle">{t("adv.allImages", { n: images.length })}</summary>
          <div className="tiny subtle" style={{ margin: "6px 0 8px" }}>
            {t("adv.allImagesHint")}
          </div>
          <div className="job-grid adv-all" style={{ ["--ar" as string]: String(ar) }}>
            {images.map((img) => (
              <ImageTile
                key={img.id}
                image={img}
                onOpen={() => open(img.id)}
                onDownload={() => actions.download(img)}
                onDelete={() => quick.deleteImageQuick(img)}
                extra={<AddToLora imageIds={[img.id]} compact />}
                showPrompt={false}
                blurNsfw={blurNsfw}
              />
            ))}
          </div>
        </details>
      ) : null}
      {job.spec && (job.spec.preserve.length || job.spec.modify.length) ? (
        <details>
          <summary className="tiny subtle">{t("adv.planUsed")}</summary>
          <PlanSummary spec={job.spec} />
        </details>
      ) : null}
      <ImageViewer images={images} index={viewer} onClose={() => setViewer(null)} onIndexChange={setViewer} />
    </div>
  );
}

function PlanSummary({ spec }: { spec: AdvancedSpec }) {
  const { t } = useI18n();
  return (
    <div className="col" style={{ gap: 6, marginTop: 6 }}>
      {spec.summary ? <div className="small">{spec.summary}</div> : null}
      {spec.modify.length ? (
        <div className="tiny">
          <strong>{t("adv.modify")}:</strong> {spec.modify.join(" · ")}
        </div>
      ) : null}
      {spec.preserve.length ? (
        <div className="tiny">
          <strong>{t("adv.preserve")}:</strong> {spec.preserve.join(" · ")}
        </div>
      ) : null}
    </div>
  );
}

/** Generate › Advanced: one request runs everything — plan, Krea 2, masked edits, checks and
 * corrections — on the user's ComfyUI. */
export function AdvancedPanel({ header }: { header: React.ReactNode }) {
  const { t, locale } = useI18n();
  const conns = useComfyConnections();
  const jobs = useAdvancedJobs();
  const loras = useLoras("", true);
  const settings = useSettings();
  const start = useStartAdvanced();
  const plan = usePlanAdvanced();
  const download = useComfyModelDownload();
  const testConn = useTestComfy();
  const submitting = useRef(false);
  const [connectionId, setConnectionId] = useState("");
  const [mode, setMode] = useState<Mode>("text-to-image");
  const [variant, setVariant] = useState<"turbo" | "raw">("turbo");
  const [photo, setPhoto] = useState<PickedPhoto | null>(null);
  const [loraVersion, setLoraVersion] = useState("");
  // null = the default for the mode: text to image draws the LoRA's face and figure closer
  // at 1.3 (the owner's hard prompts, 2026-11-06: 42–53 % → 44–69 % alike, no artifacts).
  const [chosenStrength, setStrength] = useState<number | null>(null);
  const strength = chosenStrength ?? (mode === "text-to-image" ? 1.3 : 1);
  const [prompt, setPrompt] = useState("");
  const [analysis, setAnalysis] = useState(true);
  const [quality, setQuality] = useState(true);
  const [corrections, setCorrections] = useState(1);
  const [improveBackground, setImproveBackground] = useState(false);
  const [targets, setTargets] = useState<AdvancedTarget[]>([]);
  const [identity, setIdentity] = useState(true);
  const [more, setMore] = useState(false);
  const [aspect, setAspect] = useState("3:4");
  const [resolution, setResolution] = useState("1K");
  const [seed, setSeed] = useState("");
  const [denoise, setDenoise] = useState<number | null>(null);
  const [adult, setAdult] = useState(false);
  const [draft, setDraft] = useState<AdvancedSpec | null>(null);
  const [selected, setSelected] = useState<string | null>(null);

  const all = conns.data?.items ?? [];
  const conn = all.find((c) => c.id === connectionId) ?? all.find((c) => c.status === "ready") ?? all[0];
  const caps = useAdvancedCapabilities(conn?.status === "ready" ? conn.id : null);
  const c = caps.data;
  const loraOptions = useMemo(() => {
    const out: { value: string; label: string; disabled?: boolean; trigger: string }[] = [];
    for (const l of loras.data?.items ?? []) {
      for (const v of l.usable_versions) {
        out.push({
          value: v.id,
          label: `${l.name} · v${v.number}${loraFits(l, variant) ? "" : ` (${t("adv.loraOnlyTurbo")})`}`,
          disabled: !loraFits(l, variant),
          trigger: l.trigger_word,
        });
      }
    }
    return out;
  }, [loras.data, variant, t]);
  const chosenLora = loraOptions.find((o) => o.value === loraVersion);
  useHiddenVersion();
  // A request just deleted disappears at once (it goes after the "Undo" moment).
  const items = (jobs.data?.items ?? []).filter((j) => !isHidden(j.id));
  const job = items.find((j) => j.id === selected) ?? items[0];

  // "Load these settings": the form as that request ran, with the plan it used (shown
  // and editable) and the first attempt's seed, so Run repeats it.
  const loadJob = (j: AdvancedJob) => {
    const o = j.options;
    setMode(j.mode);
    setVariant(j.variant);
    setPrompt(j.prompt);
    setPhoto(j.mode === "edit" && j.source ? { kind: j.source.kind as "reference" | "image", id: j.source.id, thumb_url: j.source.thumb_url } : null);
    const loraId = o.lora?.version_id ?? "";
    const loraExists = Boolean(loraId) && (loras.data?.items ?? []).some((l) => l.usable_versions.some((v) => v.id === loraId));
    setLoraVersion(loraExists ? loraId : "");
    setStrength(o.lora?.strength ?? null);
    setAspect(o.aspect_ratio ?? "3:4");
    setResolution(o.resolution ?? "1K");
    const seedUsed = o.seed ?? o.first_seed;
    setSeed(seedUsed !== undefined && seedUsed !== null ? String(seedUsed) : "");
    // A request from before "turn the person into the LoRA's character" (no `identity` in its
    // options) is loaded with today's defaults: the switch on, no "whole photo" and no old
    // plan — its plan would skip the face and hair steps.
    const older = o.identity === undefined;
    setTargets(older ? (o.targets ?? []).filter((x) => x !== "whole") : (o.targets ?? []));
    setIdentity(older ? true : Boolean(o.identity));
    setDenoise(o.denoise ?? null);
    setAnalysis(o.analysis ?? true);
    setQuality(o.quality ?? true);
    setCorrections(o.corrections ?? 1);
    setImproveBackground(Boolean(o.improve_background));
    setAdult(Boolean(o.adult));
    setDraft(!older && j.spec && (j.spec.prompt || j.spec.edits.length) ? { ...j.spec } : null);
    setMore(true);
    if (loraId && !loraExists) toast.warning(t("adv.loraGone"));
    else toast.success(t("adv.loaded"));
  };

  // A photo and a LoRA: the person becomes the LoRA's character (unless switched off).
  const identityOn = mode === "edit" && Boolean(loraVersion) && identity;
  const problems: string[] = [];
  if (!conn) problems.push(t("k2.noConnection"));
  else if (conn.status !== "ready") problems.push(conn.status === "deploying" ? t("k2.starting") : t("k2.notReady"));
  else if (c && !c.variants[variant]) problems.push(t("adv.modelMissing"));
  else if (c && (mode === "edit" ? c.missing_nodes_edit : c.missing_nodes_text).length)
    problems.push(t("adv.nodesMissing", { nodes: (mode === "edit" ? c.missing_nodes_edit : c.missing_nodes_text).join(", ") }));
  if (mode === "edit" && !photo) problems.push(t("adv.photoFirst"));
  if (mode === "edit" && c && !c.sam3 && (analysis || identityOn || improveBackground || targets.some((x) => x !== "whole"))) problems.push(t("adv.sam3Missing"));
  if (mode === "edit" && !analysis && !identityOn && !improveBackground && !targets.length && !draft) problems.push(t("adv.chooseRegion"));
  if (analysis && c && !c.ai) problems.push(t("adv.aiMissing"));
  if (!prompt.trim()) problems.push(t("adv.promptFirst"));
  if (chosenLora?.disabled) problems.push(t("adv.loraIncompatible"));
  if (seed && !/^\d+$/.test(seed)) problems.push(t("gen.problemSeed"));

  const body = (withPlan: boolean) => ({
    connection_id: conn?.id,
    mode,
    variant,
    prompt: prompt.trim(),
    image: mode === "edit" && photo ? { kind: photo.kind, id: photo.id } : null,
    lora: loraVersion ? { version_id: loraVersion, strength } : null,
    aspect_ratio: aspect,
    resolution,
    seed: seed ? Number(seed) : null,
    targets,
    identity: identityOn,
    improve_background: mode === "edit" && improveBackground,
    denoise,
    analysis,
    quality,
    corrections: quality ? corrections : 0,
    adult,
    spec: withPlan && draft ? draft : null,
    client_request_id: newRequestId(),
  });

  const run = async () => {
    if (submitting.current) return;
    if (problems.length) {
      toast.error(problems[0]);
      return;
    }
    submitting.current = true;
    try {
      const created = await start.mutateAsync(body(true));
      setSelected(created.id);
      setDraft(null);
      toast.success(t("adv.started"));
    } catch (e) {
      toast.error(e instanceof ApiError ? e.message : t("errors.generic"));
    } finally {
      submitting.current = false;
    }
  };

  const showPlan = async () => {
    try {
      setDraft(
        await plan.mutateAsync({
          mode,
          prompt: prompt.trim(),
          image: mode === "edit" && photo ? { kind: photo.kind, id: photo.id } : null,
          lora: loraVersion ? { version_id: loraVersion, strength } : null,
          variant,
          analysis,
          targets,
          identity: identityOn,
          improve_background: mode === "edit" && improveBackground,
          adult,
        }),
      );
    } catch (e) {
      toast.error(e instanceof ApiError ? e.message : t("errors.generic"));
    }
  };

  const installSam = async () => {
    if (!conn || !c) return;
    try {
      await download.mutateAsync({ id: conn.id, ...c.sam3_file });
      toast.success(t("adv.sam3Downloading"));
    } catch (e) {
      toast.error(e instanceof ApiError ? e.message : t("errors.generic"));
    }
  };

  const recheck = async () => {
    if (!conn) return;
    try {
      await testConn.mutateAsync(conn.id);
      await caps.refetch();
    } catch (e) {
      toast.error(e instanceof ApiError ? e.message : t("errors.generic"));
    }
  };

  const toggleTarget = (target: AdvancedTarget) =>
    setTargets((list) => (list.includes(target) ? list.filter((x) => x !== target) : [...list, target].slice(-4)));

  if (conns.isLoading) {
    return (
      <section className="studio-controls" style={{ padding: 22 }}>
        {header}
        <Skeleton style={{ height: 400, marginTop: 16 }} />
      </section>
    );
  }

  return (
    <>
      <section className="studio-controls" aria-label={t("gen.controls")}>
        <div className="studio-scroll">
          {header}
          <div className="tiny subtle">{t("adv.lede")}</div>
          <div>
            <div className="section-title">
              <span className="eyebrow grow">{t("adv.stepGpu")}</span>
            </div>
            <ComfyConnectionBar selected={conn} onSelect={setConnectionId} />
          </div>
          {conn?.status !== "ready" ? null : (
            <>
              <Field label={t("adv.mode")}>
                <Segmented
                  label={t("adv.mode")}
                  value={mode}
                  block
                  onChange={(m) => {
                    setMode(m);
                    setDraft(null);
                  }}
                  options={[
                    { value: "text-to-image", label: t("adv.modeText") },
                    { value: "edit", label: t("adv.modeEdit") },
                  ]}
                />
              </Field>
              <Field label={t("adv.model")}>
                <Segmented
                  label={t("adv.model")}
                  value={variant}
                  block
                  onChange={(v) => {
                    setVariant(v);
                    if (v === "raw") setResolution("1K");
                  }}
                  options={[
                    { value: "turbo", label: "Krea 2 Turbo", disabled: c ? !c.variants.turbo : false },
                    { value: "raw", label: "Krea 2 RAW", disabled: c ? !c.variants.raw : false },
                  ]}
                />
              </Field>
              {mode === "edit" ? (
                <>
                  <PhotoPicker title={t("adv.photo")} value={photo} onChange={(p) => { setPhoto(p); setDraft(null); }} />
                  {c && !c.sam3 ? (
                    <Notice tone="warning">
                      <div>{t("adv.sam3Need", { size: formatBytes(c.sam3_file.size, locale) })}</div>
                      <div className="row wrap" style={{ gap: 8, marginTop: 8 }}>
                        {c.file_capability === "agent" ? (
                          <Button size="sm" icon={<Download />} loading={download.isPending} onClick={installSam}>
                            {t("adv.sam3Install")}
                          </Button>
                        ) : (
                          <span className="tiny mono">models/{c.sam3_file.folder}/{c.sam3_file.filename}</span>
                        )}
                        <Button size="sm" variant="ghost" icon={<RefreshCw />} loading={testConn.isPending} onClick={recheck}>
                          {t("adv.recheck")}
                        </Button>
                      </div>
                    </Notice>
                  ) : null}
                </>
              ) : null}
              <Field label={t("adv.lora")} hint={chosenLora ? t("adv.loraHint", { trigger: chosenLora.trigger }) : t("adv.loraNone")}>
                <Select
                  label={t("adv.lora")}
                  value={loraVersion}
                  onChange={setLoraVersion}
                  options={[{ value: "", label: t("adv.noLora") }, ...loraOptions.map((o) => ({ value: o.value, label: o.label, disabled: o.disabled }))]}
                />
              </Field>
              {loraVersion ? (
                <Field label={t("k2.strength", { n: strength.toFixed(2) })}>
                  <input type="range" min={0} max={1.5} step={0.05} value={strength} onChange={(e) => setStrength(Number(e.target.value))} />
                </Field>
              ) : null}
              {loraVersion && mode === "edit" ? (
                <label className="row" style={{ gap: 10, alignItems: "flex-start" }}>
                  <Switch checked={identity} onChange={(v) => { setIdentity(v); setDraft(null); }} label={t("adv.identity")} />
                  <div>
                    <div className="small">{t("adv.identity")}</div>
                    <div className="hint">{t("adv.identityHint")}</div>
                  </div>
                </label>
              ) : null}
              <Field label={t("adv.prompt")} hint={t(mode === "edit" ? "adv.promptEditHint" : "adv.promptTextHint")}>
                <textarea
                  className="textarea"
                  rows={4}
                  maxLength={2000}
                  value={prompt}
                  onChange={(e) => {
                    setPrompt(e.target.value);
                    setDraft(null);
                  }}
                  placeholder={t(mode === "edit" ? "adv.promptEditPlaceholder" : "adv.promptTextPlaceholder")}
                />
              </Field>
              <div className="col" style={{ gap: 10 }}>
                <div className="eyebrow">{t("adv.processing")}</div>
                <label className="row" style={{ gap: 10, alignItems: "flex-start" }}>
                  <Switch checked={analysis} onChange={(v) => { setAnalysis(v); setDraft(null); }} label={t("adv.analysis")} />
                  <div>
                    <div className="small">{t("adv.analysis")}</div>
                    <div className="hint">{t(c && !c.ai ? "adv.analysisNoAi" : "adv.analysisHint")}</div>
                  </div>
                </label>
                {mode === "edit" && !analysis ? (
                  <Field label={t("adv.regions")} hint={t("adv.regionsHint")}>
                    <div className="chips">
                      {TARGETS.map((target) => (
                        <button key={target} type="button" className="chip" aria-pressed={targets.includes(target)} onClick={() => toggleTarget(target)}>
                          {t(`adv.target.${target}` as MessageKey)}
                        </button>
                      ))}
                    </div>
                  </Field>
                ) : null}
                <label className="row" style={{ gap: 10, alignItems: "flex-start" }}>
                  <Switch checked={quality} onChange={setQuality} label={t("adv.quality")} />
                  <div>
                    <div className="small">{t("adv.quality")}</div>
                    <div className="hint">{t(analysis ? "adv.qualityHint" : "adv.qualityMeasuredHint")}</div>
                  </div>
                </label>
                {quality && (analysis || identityOn) ? (
                  <Field label={t("adv.corrections")} hint={t("adv.correctionsHint")}>
                    <Segmented label={t("adv.corrections")} value={corrections} onChange={setCorrections} options={[0, 1, 2, 3].map((n) => ({ value: n, label: n ? String(n) : t("adv.off") }))} />
                  </Field>
                ) : null}
                {mode === "edit" ? (
                  <label className="row" style={{ gap: 10, alignItems: "flex-start" }}>
                    <Switch checked={improveBackground} onChange={(v) => { setImproveBackground(v); setDraft(null); }} label={t("adv.improveBackground")} />
                    <div>
                      <div className="small">{t("adv.improveBackground")}</div>
                      <div className="hint">{t("adv.improveBackgroundHint")}</div>
                    </div>
                  </label>
                ) : null}
              </div>
              <section className="card">
                <button type="button" className="card-head adv-toggle" onClick={() => setMore((v) => !v)} aria-expanded={more}>
                  <span className="h3 grow" style={{ textAlign: "left" }}>{t("ls.advanced")}</span>
                  <ChevronDown size={16} style={{ transform: more ? "rotate(180deg)" : undefined }} />
                </button>
                {more ? (
                  <div className="card-body col gap-16">
                    {mode === "text-to-image" ? (
                      <Field label={t("adv.aspect")}>
                        <Select label={t("adv.aspect")} value={aspect} onChange={setAspect} options={RATIOS.map((r) => ({ value: r, label: r }))} />
                      </Field>
                    ) : (
                      <Field label={t("adv.strength", { n: denoise === null ? t("adv.auto") : denoise.toFixed(2) })} hint={t("adv.strengthHint")}>
                        <div className="row" style={{ gap: 8 }}>
                          <input type="range" min={0.2} max={0.95} step={0.05} value={denoise ?? 0.7} onChange={(e) => setDenoise(Number(e.target.value))} />
                          <Button size="sm" variant="ghost" onClick={() => setDenoise(null)} disabled={denoise === null}>
                            {t("adv.auto")}
                          </Button>
                        </div>
                      </Field>
                    )}
                    <Field label={t("adv.resolution")}>
                      <Segmented
                        label={t("adv.resolution")}
                        value={resolution}
                        onChange={setResolution}
                        options={[
                          { value: "1K", label: "1K" },
                          { value: "2K", label: "2K", disabled: variant === "raw" },
                        ]}
                      />
                    </Field>
                    <Field label={t("adv.seed")} hint={t("adv.seedHint")}>
                      <input className="input" inputMode="numeric" value={seed} onChange={(e) => setSeed(e.target.value.trim())} placeholder={t("adv.random")} />
                    </Field>
                    <label className="row" style={{ gap: 10, alignItems: "flex-start" }}>
                      <Switch checked={adult} onChange={setAdult} label={t("k2.adult")} />
                      <div>
                        <div className="small">{t("k2.adult")}</div>
                        <div className="hint">{t(adult ? "k2.adultOnHint" : "k2.adultOffHint")}</div>
                      </div>
                    </label>
                  </div>
                ) : null}
              </section>
              {draft ? (
                <div className="card card-pad col" style={{ gap: 10 }}>
                  <div className="row">
                    <span className="eyebrow grow">{t(draft.source === "ai" ? "adv.planAi" : "adv.planRules")}</span>
                    <Button size="sm" variant="ghost" onClick={() => setDraft(null)}>
                      {t("adv.planDiscard")}
                    </Button>
                  </div>
                  <PlanSummary spec={draft} />
                  {draft.mode === "text-to-image" ? (
                    <Field label={t("adv.kreaPrompt")}>
                      <textarea className="textarea" rows={4} value={draft.prompt} onChange={(e) => setDraft({ ...draft, prompt: e.target.value })} />
                    </Field>
                  ) : (
                    draft.edits.map((edit, i) => {
                      const change = (patch: Partial<typeof edit>) =>
                        setDraft({ ...draft, edits: draft.edits.map((x, j) => (j === i ? { ...x, ...patch } : x)) });
                      return (
                        <Field key={i} label={t("adv.editFor", { target: t(`adv.target.${edit.target}` as MessageKey), region: edit.region || t("adv.target.whole") })}>
                          <div className="col" style={{ gap: 6 }}>
                            <textarea className="textarea" rows={3} value={edit.prompt} onChange={(e) => change({ prompt: e.target.value })} />
                            {edit.target === "hair" || edit.target === "clothing" ? (
                              <div className="row" style={{ gap: 8 }}>
                                {edit.color ? (
                                  <input
                                    type="color"
                                    className="adv-color"
                                    value={edit.color}
                                    aria-label={t("adv.color")}
                                    onChange={(e) => change({ color: e.target.value.toUpperCase() })}
                                  />
                                ) : null}
                                <span className="tiny subtle grow">{t(edit.color ? "adv.colorOn" : "adv.colorOff")}</span>
                                <Button size="sm" variant="ghost" onClick={() => change({ color: edit.color ? "" : "#5A3E2B" })}>
                                  {t(edit.color ? "adv.colorRemove" : "adv.colorAdd")}
                                </Button>
                              </div>
                            ) : null}
                          </div>
                        </Field>
                      );
                    })
                  )}
                  {draft.checks.length ? (
                    <div className="tiny subtle">
                      {t("adv.checksPlanned")}: {draft.checks.map((x) => x.question).join(" · ")}
                    </div>
                  ) : null}
                </div>
              ) : null}
              {problems.length ? <div className="tiny subtle">{problems[0]}</div> : null}
              <div className="row wrap" style={{ gap: 8 }}>
                <Button icon={<ListChecks />} loading={plan.isPending} disabled={!prompt.trim() || (mode === "edit" && !photo)} onClick={showPlan}>
                  {t("adv.showPlan")}
                </Button>
                <Button variant="primary" className="grow" icon={<Sparkles />} loading={start.isPending} disabled={problems.length > 0} onClick={run}>
                  {t(draft ? "adv.runPlan" : "adv.run")}
                </Button>
              </div>
            </>
          )}
        </div>
      </section>
      <section className="studio-results" aria-label={t("adv.results")}>
        <div className="row" style={{ marginBottom: 18 }}>
          <div className="grow">
            <h2 className="h2">{t("adv.results")}</h2>
            <div className="small subtle">{t("adv.resultsHint")}</div>
          </div>
        </div>
        {jobs.isLoading ? (
          <Skeleton style={{ height: 320 }} />
        ) : !job ? (
          <EmptyState icon={<Sparkles />} title={t("adv.emptyTitle")} description={t("adv.emptyBody")} />
        ) : (
          <div className="col gap-16">
            <JobView key={job.id} job={job} blurNsfw={settings.data?.blur_nsfw ?? true} onLoad={loadJob} />
            {items.length > 1 ? (
              <div className="col" style={{ gap: 6 }}>
                <div className="eyebrow">{t("adv.history")}</div>
                <div className="adv-history">
                  {items.map((j) => (
                    <button key={j.id} type="button" className="adv-history-item" aria-pressed={j.id === job.id} onClick={() => setSelected(j.id)}>
                      {j.result && !isHidden(j.result.id) ? (
                        <img src={j.result.thumb_url} alt="" className={settings.data?.blur_nsfw !== false && j.result.nsfw ? "adv-blur" : undefined} />
                      ) : (
                        <span className="adv-history-empty">{t(`adv.status.${j.status}` as MessageKey)}</span>
                      )}
                      <span className="tiny truncate">{j.prompt}</span>
                    </button>
                  ))}
                </div>
              </div>
            ) : null}
          </div>
        )}
      </section>
    </>
  );
}
