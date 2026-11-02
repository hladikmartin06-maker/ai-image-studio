"use client";

import {
  CircleCheck,
  CircleDashed,
  CircleX,
  CloudCog,
  FolderOpen,
  HardDriveDownload,
  Laptop,
  Play,
  Plug,
  RefreshCw,
  ScrollText,
  Server,
  Square,
  Trash2,
  Upload,
} from "lucide-react";
import { useQueryClient } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { toast } from "sonner";

import { formatMoney, GpuMeter, type GpuChoice, GpuPicker, hourlyPrice, MIN_GPU_GB, StatusPill } from "@/components/studio/common";
import { Badge, Button, Field, IconButton, Modal, Notice, Select, Switch, useConfirm } from "@/components/ui";
import { ApiError, uploadInChunks } from "@/lib/api";
import { formatBytes, formatRelative } from "@/lib/format";
import { useI18n, useT } from "@/lib/i18n";
import type { MessageKey } from "@/lib/i18n";
import {
  useAddComfy,
  useComfyCatalog,
  useComfyModelDownload,
  useComfyModelUpload,
  useComfyPower,
  useDeployComfy,
  useRemoveComfy,
  useTestComfy,
  useUpdateComfy,
} from "@/lib/queries";
import type { ComfyConnection } from "@/lib/types";

const KIND_ICON = { local: Laptop, remote: Server, runpod: CloudCog, modal: CloudCog };

function Check({ ok, label }: { ok: boolean; label: string }) {
  return (
    <div className="row tiny" style={{ gap: 6, color: ok ? "var(--text)" : "var(--text-3)" }}>
      {ok ? <CircleCheck size={14} color="var(--success)" /> : <CircleDashed size={14} />}
      <span className="truncate">{label}</span>
    </div>
  );
}

export function ConnectionCard({ conn }: { conn: ComfyConnection }) {
  const { t, locale } = useI18n();
  const confirm = useConfirm();
  const test = useTestComfy();
  const power = useComfyPower();
  const remove = useRemoveComfy();
  const update = useUpdateComfy();
  const [models, setModels] = useState(false);
  const [logs, setLogs] = useState(false);
  const Icon = KIND_ICON[conn.kind];
  const info = conn.info ?? {};
  const k2 = info.krea2;
  const deployed = conn.auth_type === "agent";
  const device = info.devices?.[0];

  const run = async (fn: () => Promise<unknown>, done?: string) => {
    try {
      await fn();
      if (done) toast.success(done);
    } catch (e) {
      toast.error(e instanceof ApiError ? e.message : t("errors.generic"));
    }
  };

  const onRemove = async () => {
    const running = deployed && conn.status !== "stopped";
    const ok = await confirm({
      title: t("cf.removeTitle", { name: conn.name }),
      description: deployed ? t("cf.removeDeployedBody") : t("cf.removeBody"),
      confirmLabel: deployed ? t("cf.removeAndTerminate") : t("common.delete"),
      danger: true,
    });
    if (ok) await run(() => remove.mutateAsync({ id: conn.id, terminate: deployed || running }), t("cf.removed"));
  };

  const onStart = async () => {
    const ok = await confirm({
      title: t("gpu.billingTitle"),
      description: t("gpu.billingBody", { provider: conn.gpu_provider === "modal" ? "Modal" : "RunPod" }),
      confirmLabel: t("cf.start"),
    });
    if (ok) await run(() => power.mutateAsync({ id: conn.id, action: "start" }));
  };

  return (
    <article className="card comfy-card">
      <div className="card-head">
        <span className="logo">
          <Icon />
        </span>
        <div className="grow" style={{ minWidth: 0 }}>
          <div className="row" style={{ gap: 8 }}>
            <span className="h3 truncate">{conn.name}</span>
            <StatusPill status={conn.status} />
          </div>
          <div className="tiny subtle truncate">
            {t(`cf.kind.${conn.kind}`)}
            {conn.base_url ? ` · ${conn.base_url}` : ""}
            {info.version ? ` · ComfyUI ${info.version}` : ""}
          </div>
        </div>
        <IconButton label={t("cf.test")} onClick={() => run(() => test.mutateAsync(conn.id), t("cf.tested"))} disabled={test.isPending}>
          <RefreshCw />
        </IconButton>
        <IconButton label={t("common.delete")} onClick={onRemove}>
          <Trash2 />
        </IconButton>
      </div>
      <div className="card-body col gap-12">
        {conn.status === "deploying" ? (
          <Notice tone="accent">
            <strong>{t(`cf.stage.${conn.deploy.stage ?? "starting"}`)}</strong>
            <div className="tiny">{t("cf.deployingHint")}</div>
          </Notice>
        ) : null}
        {conn.last_error && conn.status !== "ready" ? <Notice tone="danger">{conn.last_error}</Notice> : null}

        {device || info.agent ? (
          <div className="comfy-grid">
            {device ? (
              <div className="comfy-stat">
                <span className="tiny subtle">{t("cf.device")}</span>
                <span className="small truncate" title={device.name}>
                  {device.name.replace(/^cuda:\d+\s*/, "").split(":")[0]}
                </span>
                {device.vram_total ? (
                  <span className="tiny subtle">
                    {t("cf.vram", {
                      free: formatBytes(device.vram_free ?? 0, locale),
                      total: formatBytes(device.vram_total, locale),
                    })}
                  </span>
                ) : null}
              </div>
            ) : null}
            {info.pytorch ? (
              <div className="comfy-stat">
                <span className="tiny subtle">PyTorch · Python</span>
                <span className="small">
                  {info.pytorch} · {info.python}
                </span>
              </div>
            ) : null}
            {info.node_count ? (
              <div className="comfy-stat">
                <span className="tiny subtle">{t("cf.nodes")}</span>
                <span className="small">
                  {t("cf.nodeCount", { n: info.node_count, packs: info.custom_node_packs?.length ?? 0 })}
                </span>
              </div>
            ) : null}
            {deployed && conn.deploy.cost_per_hour ? (
              <div className="comfy-stat">
                <span className="tiny subtle">{t("cf.rate")}</span>
                <span className="small">{formatMoney(conn.deploy.cost_per_hour, locale)}/h</span>
              </div>
            ) : null}
          </div>
        ) : null}
        {info.agent?.gpus?.length ? <GpuMeter gpu={info.agent.gpus[0]} /> : null}

        {k2 ? (
          <div className="col" style={{ gap: 6 }}>
            <div className="row">
              <span className="eyebrow grow">Krea 2</span>
              {k2.ready ? <Badge tone="success">{t("cf.k2ready")}</Badge> : <Badge tone="warning">{t("cf.k2missing")}</Badge>}
            </div>
            <Check ok={k2.supported} label={t("cf.k2support")} />
            <Check ok={k2.variants.turbo.installed} label={`Turbo ${k2.variants.turbo.files.map((f) => f.precision).join(", ")}`} />
            <Check ok={k2.variants.raw.installed} label={`RAW ${k2.variants.raw.files.map((f) => f.precision).join(", ")}`} />
            <Check ok={k2.text_encoders.length > 0} label={`${t("cf.textEncoder")} ${k2.text_encoders[0] ?? ""}`} />
            <Check ok={k2.vaes.length > 0} label={`VAE ${k2.vaes[0] ?? ""}`} />
            <Check ok={k2.loras.length > 0} label={t("cf.loraCount", { n: k2.loras.length })} />
          </div>
        ) : null}

        <div className="row wrap" style={{ gap: 6 }}>
          {conn.file_capability !== "none" ? (
            <Button size="sm" icon={<HardDriveDownload />} onClick={() => setModels(true)} disabled={conn.status !== "ready"}>
              {t("cf.models")}
            </Button>
          ) : (
            <span className="tiny subtle">{t("cf.noUploads")}</span>
          )}
          {deployed ? (
            <>
              <Button size="sm" icon={<ScrollText />} onClick={() => setLogs(true)}>
                {t("cf.logs")}
              </Button>
              {conn.status === "stopped" || conn.status === "error" ? (
                <Button size="sm" icon={<Play />} onClick={onStart} loading={power.isPending}>
                  {t("cf.start")}
                </Button>
              ) : (
                <Button
                  size="sm"
                  icon={<Square />}
                  onClick={() => run(() => power.mutateAsync({ id: conn.id, action: "stop" }), t("cf.stopped"))}
                  loading={power.isPending}
                >
                  {conn.gpu_provider === "modal" ? t("cf.terminate") : t("cf.stop")}
                </Button>
              )}
              <span className="spacer" />
              <label className="row tiny subtle" style={{ gap: 6 }}>
                {t("cf.autoStop")}
                <Select
                  label={t("cf.autoStop")}
                  value={conn.deploy.auto_stop_minutes ?? 0}
                  onChange={(m) => run(() => update.mutateAsync({ id: conn.id, auto_stop_minutes: Number(m) }))}
                  options={[0, 15, 30, 60, 120].map((m) => ({ value: m, label: m ? `${m} min` : t("common.off") }))}
                />
              </label>
            </>
          ) : null}
        </div>
        {conn.last_checked_at ? (
          <div className="tiny subtle">{t("cf.checked", { when: formatRelative(conn.last_checked_at, locale) })}</div>
        ) : null}
      </div>
      {models ? <ModelsModal conn={conn} onClose={() => setModels(false)} /> : null}
      {logs ? (
        <Modal open wide onClose={() => setLogs(false)} title={t("cf.logs")} description={conn.name}>
          <pre className="log-box">{conn.deploy.log_tail || t("cf.noLogs")}</pre>
        </Modal>
      ) : null}
    </article>
  );
}

// --- add / deploy ------------------------------------------------------------------------

export type AddMode = "local" | "deploy";

export function AddConnectionModal({ open, onClose, mode }: { open: boolean; onClose: () => void; mode?: AddMode }) {
  const t = useT();
  if (!open) return null;
  return (
    <Modal open wide onClose={onClose} icon={<Plug />} title={t("cf.addTitle")} description={t("cf.addBody")}>
      <ConnectForm initialMode={mode ?? "local"} onDone={onClose} onCancel={onClose} />
    </Modal>
  );
}

// The rented GPU's disks (backend defaults) and RunPod's storage prices (docs.runpod.io,
// accounts-billing/billing: volume 0.10 USD/GB/month running, 0.20 stopped; container disk
// 0.10 running).
const VOLUME_GB = 80;
const CONTAINER_GB = 30;

export function formatSession(minutes: number, t: (k: MessageKey, v?: Record<string, string | number>) => string) {
  return minutes < 60 ? t("cf.minutes", { n: minutes }) : t("cf.hours", { n: Math.round((minutes / 60) * 10) / 10 });
}

/** Where Krea 2 runs: ComfyUI on this Mac, or a GPU the studio rents and sets up. */
export function ConnectForm({
  initialMode,
  onDone,
  onCancel,
}: {
  initialMode: AddMode;
  onDone?: () => void;
  onCancel?: () => void;
}) {
  const { t, locale } = useI18n();
  const confirm = useConfirm();
  const add = useAddComfy();
  const deploy = useDeployComfy();
  const [mode, setMode] = useState<AddMode>(initialMode);
  const [url, setUrl] = useState("http://127.0.0.1:8188");
  const [folder, setFolder] = useState("");
  const [gpu, setGpu] = useState<GpuChoice>({ provider: "", gpu_type: "", gpu_count: 1, cloud_type: "SECURE" });
  const [autoStop, setAutoStop] = useState(30);
  const [session, setSession] = useState(120);
  const [limit, setLimit] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const qc = useQueryClient();

  const submit = async () => {
    setError(null);
    try {
      if (mode === "deploy") {
        if (!gpu.provider || !gpu.gpu_type) {
          setError(t("gpu.chooseGpu"));
          return;
        }
        const providerName = gpu.provider === "modal" ? "Modal" : "RunPod";
        const hourly = hourlyPrice(gpu);
        const max = hourly != null ? Math.min((hourly * session) / 60, limit || Infinity) : null;
        const ok = await confirm({
          title: t("gpu.billingTitle"),
          description: (
            <div className="col" style={{ gap: 8 }}>
              <div>
                {hourly != null
                  ? t("cf.deployBillingPrice", {
                      provider: providerName,
                      price: `${gpu.price_estimate ? "≈ " : ""}${formatMoney(hourly, locale)}`,
                      cloud: gpu.provider === "modal" ? "Modal" : t(`gpu.cloud.${gpu.cloud_type}`),
                    })
                  : t("cf.deployBilling", { provider: providerName })}
              </div>
              {max != null ? (
                <div>
                  {t("cf.maxSession", {
                    max: formatMoney(max, locale),
                    hours: formatSession(session, t),
                    limit: limit ? formatMoney(limit, locale) : t("cf.noLimit").toLowerCase(),
                  })}
                </div>
              ) : null}
              <div className="small subtle">
                {gpu.provider === "modal"
                  ? t("cf.storageModal")
                  : t("cf.storageRunpod", {
                      gb: VOLUME_GB,
                      running: formatMoney(VOLUME_GB * 0.1, locale),
                      stopped: formatMoney(VOLUME_GB * 0.2, locale),
                      disk: CONTAINER_GB,
                      diskRunning: formatMoney(CONTAINER_GB * 0.1, locale),
                    })}
              </div>
              {gpu.price_estimate ? <div className="small subtle">{t("cf.estimateNote")}</div> : null}
            </div>
          ),
          confirmLabel: t("cf.deploy"),
        });
        if (!ok) return;
        try {
          await deploy.mutateAsync({
            name: `${providerName} · ${gpu.gpu_type}`.slice(0, 80),
            provider: gpu.provider,
            gpu_type: gpu.gpu_type,
            gpu_count: 1,
            cloud_type: gpu.cloud_type,
            auto_stop_minutes: autoStop,
            session_minutes: session,
            spend_limit: limit || null,
            // Nothing starts at another price than the one just confirmed.
            // RunPod's catalog and Modal's live rates are checked again at the start.
            confirmed_price:
              gpu.provider === "runpod" || gpu.price_source === "api" ? (gpu.price_per_hour ?? null) : null,
            confirm_billing: true,
          });
        } catch (e) {
          if (e instanceof ApiError && (e.code === "price_changed" || e.code === "unavailable")) {
            // Fresh prices and availability; the owner confirms again.
            void qc.invalidateQueries({ queryKey: ["gpu"] });
            setGpu({ ...gpu, gpu_type: "", price_per_hour: null });
          }
          throw e;
        }
        toast.success(t("cf.deployStarted"));
      } else {
        const created = await add.mutateAsync({
          name: t("cf.localName"),
          kind: "local",
          base_url: url.trim(),
          auth_type: "none",
          secret: null,
          models_dir: folder.trim() || null,
        });
        const c = created as ComfyConnection;
        if (c.status === "ready") toast.success(t("cf.connected"));
        else toast.error(c.last_error ?? t("cf.notReachable"));
      }
      onDone?.();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : t("errors.generic"));
    }
  };

  return (
    <div className="col gap-16">
      <div className="connect-choices" role="radiogroup" aria-label={t("cf.addTitle")}>
        {(["deploy", "local"] as const).map((m) => (
          <button
            key={m}
            type="button"
            role="radio"
            aria-checked={mode === m}
            className="model-card"
            onClick={() => {
              setMode(m);
              setError(null);
            }}
          >
            <span className="row" style={{ gap: 8 }}>
              {m === "local" ? <Laptop size={18} /> : <CloudCog size={18} />}
              <span className="name">{t(`cf.mode.${m}`)}</span>
            </span>
            <span className="desc">{m === "local" ? t("cf.choiceLocal") : t("cf.choiceDeploy")}</span>
          </button>
        ))}
      </div>

      {mode === "local" ? (
        <>
          <Field label={t("cf.url")} hint={t("cf.urlLocalHint")}>
            <input className="input mono" value={url} onChange={(e) => setUrl(e.target.value)} />
          </Field>
          <Field label={t("cf.folder")} hint={t("cf.folderHint")}>
            <input className="input mono" value={folder} placeholder="~/ComfyUI" onChange={(e) => setFolder(e.target.value)} />
          </Field>
        </>
      ) : (
        <>
          <GpuPicker value={gpu} onChange={setGpu} minMemoryGb={MIN_GPU_GB} purpose="comfyui" />
          <div className="hint">{t("cf.gpuMemoryHint")}</div>
          <div className="form-grid">
            <Field label={t("cf.session")} hint={t("cf.sessionHint")}>
              <Select
                label={t("cf.session")}
                value={session}
                onChange={(m) => setSession(Number(m))}
                options={[30, 60, 120, 240, 480, 720].map((m) => ({ value: m, label: formatSession(m, t) }))}
              />
            </Field>
            <Field label={t("cf.spendLimit")} hint={t("cf.spendLimitHint")}>
              <Select
                label={t("cf.spendLimit")}
                value={limit}
                onChange={(m) => setLimit(Number(m))}
                options={[0, 2, 5, 10, 20, 50].map((m) => ({ value: m, label: m ? formatMoney(m, locale) : t("cf.noLimit") }))}
              />
            </Field>
          </div>
          <Field label={t("cf.autoStop")} hint={t("cf.autoStopHint")}>
            <Select
              label={t("cf.autoStop")}
              value={autoStop}
              onChange={(m) => setAutoStop(Number(m))}
              options={[15, 30, 60, 120, 0].map((m) => ({ value: m, label: m ? `${m} min` : t("common.off") }))}
            />
          </Field>
          <div className="hint">{t("cf.deployNote")}</div>
        </>
      )}
      {error ? <div className="error-text">{error}</div> : null}
      <div className="row" style={{ justifyContent: "flex-end", gap: 8 }}>
        {onCancel ? (
          <Button variant="ghost" onClick={onCancel}>
            {t("common.cancel")}
          </Button>
        ) : null}
        <Button variant="primary" onClick={submit} loading={add.isPending || deploy.isPending}>
          {mode === "deploy" ? t("cf.deploy") : t("cf.connect")}
        </Button>
      </div>
    </div>
  );
}

// --- model files ------------------------------------------------------------------------

function ModelsModal({ conn, onClose }: { conn: ComfyConnection; onClose: () => void }) {
  const { t, locale } = useI18n();
  const catalog = useComfyCatalog();
  const download = useComfyModelDownload();
  const upload = useComfyModelUpload();
  const [folder, setFolder] = useState("loras");
  const [url, setUrl] = useState("");
  const [filename, setFilename] = useState("");
  const [progress, setProgress] = useState<number | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const files = conn.info.files ?? {};
  const known = [
    ...(catalog.data?.krea2.variants.flatMap((v) => v.known_files) ?? []),
    ...(catalog.data ? [{ ...catalog.data.krea2.text_encoder, folder: "text_encoders" }, { ...catalog.data.krea2.vae, folder: "vae" }] : []),
  ];

  const startDownload = async (u: string, f: string, name: string) => {
    try {
      const res = await download.mutateAsync({ id: conn.id, url: u, folder: f, filename: name });
      toast.success(t("cf.downloadStarted", { name }), { description: res.job.id });
    } catch (e) {
      toast.error((e as Error).message);
    }
  };

  const onFile = async (file: File) => {
    setProgress(0);
    try {
      const uploadId = await uploadInChunks(file, setProgress);
      await upload.mutateAsync({ id: conn.id, upload_id: uploadId, folder });
      toast.success(t("cf.uploaded", { name: file.name }));
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setProgress(null);
    }
  };

  return (
    <Modal open wide onClose={onClose} title={t("cf.models")} description={t("cf.modelsBody", { name: conn.name })}>
      <div className="col gap-16">
        <div className="form-grid">
          {(catalog.data?.model_folders ?? []).map((f) => (
            <div key={f} className="comfy-stat">
              <span className="tiny subtle">{f}</span>
              <span className="tiny">{(files[f] ?? []).slice(0, 6).join(", ") || "—"}</span>
            </div>
          ))}
        </div>
        <Field label={t("cf.folderTarget")}>
          <Select
            label={t("cf.folderTarget")}
            value={folder}
            onChange={setFolder}
            options={(catalog.data?.model_folders ?? ["loras"]).map((f) => ({ value: f, label: f }))}
          />
        </Field>
        <div className="row wrap" style={{ gap: 8 }}>
          <input ref={input} type="file" hidden accept=".safetensors,.sft,.ckpt,.pt,.gguf" onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])} />
          <Button icon={<Upload />} onClick={() => input.current?.click()} loading={progress !== null}>
            {progress !== null ? `${Math.round(progress * 100)}%` : t("cf.uploadFromMac")}
          </Button>
          <span className="tiny subtle">{t("cf.uploadHint")}</span>
        </div>
        {conn.file_capability === "agent" ? (
          <>
            <div className="divider" />
            <div className="eyebrow">{t("cf.downloadOnGpu")}</div>
            <div className="col" style={{ gap: 6 }}>
              {known.map((f) => {
                const have = (files[f.folder] ?? []).includes(f.filename);
                return (
                  <div key={f.filename} className="row" style={{ gap: 8 }}>
                    {have ? <CircleCheck size={14} color="var(--success)" /> : <CircleX size={14} />}
                    <span className="small grow truncate mono">
                      {f.folder}/{f.filename}
                    </span>
                    {f.size_gb ? <span className="tiny subtle">{formatBytes(f.size_gb * 1e9, locale)}</span> : null}
                    <Button size="sm" variant="ghost" disabled={have} onClick={() => startDownload(f.url, f.folder, f.filename)}>
                      {t("cf.download")}
                    </Button>
                  </div>
                );
              })}
            </div>
            <div className="form-grid">
              <Field label={t("cf.downloadUrl")} hint={t("cf.downloadUrlHint")}>
                <input className="input mono" value={url} placeholder="https://huggingface.co/…/resolve/main/…" onChange={(e) => setUrl(e.target.value)} />
              </Field>
              <Field label={t("cf.filename")}>
                <input className="input mono" value={filename} placeholder="model.safetensors" onChange={(e) => setFilename(e.target.value)} />
              </Field>
            </div>
            <div>
              <Button size="sm" icon={<FolderOpen />} disabled={!url || !filename} loading={download.isPending} onClick={() => startDownload(url, folder, filename)}>
                {t("cf.downloadTo", { folder })}
              </Button>
            </div>
          </>
        ) : null}
        <Notice>{t("cf.refreshAfter")}</Notice>
      </div>
    </Modal>
  );
}

export function useConnectionsReady(items: ComfyConnection[] | undefined) {
  return (items ?? []).filter((c) => c.status === "ready");
}

export function DeployToggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label: string }) {
  return (
    <label className="row small" style={{ gap: 10 }}>
      <Switch checked={checked} onChange={onChange} label={label} />
      {label}
    </label>
  );
}
