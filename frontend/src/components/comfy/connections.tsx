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
import { useRef, useState } from "react";
import { toast } from "sonner";

import { formatMoney, GpuMeter, type GpuChoice, GpuPicker, StatusPill } from "@/components/studio/common";
import { Badge, Button, Field, IconButton, Modal, Notice, Segmented, Select, Switch, useConfirm } from "@/components/ui";
import { ApiError, api, uploadInChunks } from "@/lib/api";
import { formatBytes, formatRelative } from "@/lib/format";
import { useI18n, useT } from "@/lib/i18n";
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
import type { ComfyConnection, GpuWorkload } from "@/lib/types";

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

type AddMode = "local" | "remote" | "existing" | "deploy";

export function AddConnectionModal({ open, onClose }: { open: boolean; onClose: () => void }) {
  return open ? <AddConnection onClose={onClose} /> : null;
}

function AddConnection({ onClose }: { onClose: () => void }) {
  const t = useT();
  const confirm = useConfirm();
  const add = useAddComfy();
  const deploy = useDeployComfy();
  const [mode, setMode] = useState<AddMode>("local");
  const [name, setName] = useState("");
  const [url, setUrl] = useState("http://127.0.0.1:8188");
  const [auth, setAuth] = useState<"none" | "bearer" | "basic">("none");
  const [secret, setSecret] = useState("");
  const [folder, setFolder] = useState("");
  const [provider, setProvider] = useState<"runpod" | "modal">("runpod");
  const [workloads, setWorkloads] = useState<GpuWorkload[] | null>(null);
  const [gpu, setGpu] = useState<GpuChoice>({ provider: "", gpu_type: "", gpu_count: 1, cloud_type: "SECURE" });
  const [variants, setVariants] = useState<string[]>(["turbo"]);
  const [autoStop, setAutoStop] = useState(30);
  const [image, setImage] = useState("");
  const [error, setError] = useState<string | null>(null);

  const loadWorkloads = async (p: "runpod" | "modal") => {
    setProvider(p);
    setWorkloads(null);
    try {
      const res = await api<{ items: GpuWorkload[] }>(`/api/comfy/workloads?provider=${p}`);
      setWorkloads(res.items);
    } catch (e) {
      setError((e as Error).message);
      setWorkloads([]);
    }
  };

  const submit = async () => {
    setError(null);
    try {
      if (mode === "deploy") {
        if (!gpu.provider || !gpu.gpu_type) {
          setError(t("gpu.chooseGpu"));
          return;
        }
        const ok = await confirm({
          title: t("gpu.billingTitle"),
          description: t("cf.deployBilling", { provider: gpu.provider === "modal" ? "Modal" : "RunPod" }),
          confirmLabel: t("cf.deploy"),
        });
        if (!ok) return;
        await deploy.mutateAsync({
          name: name.trim() || `ComfyUI ${gpu.provider === "modal" ? "Modal" : "RunPod"}`,
          provider: gpu.provider,
          gpu_type: gpu.gpu_type,
          gpu_count: gpu.gpu_count,
          cloud_type: gpu.cloud_type,
          image: image.trim() || null,
          variants,
          auto_stop_minutes: autoStop,
          confirm_billing: true,
        });
        toast.success(t("cf.deployStarted"));
      } else {
        const kind = mode === "existing" ? provider : mode;
        const created = await add.mutateAsync({
          name: name.trim() || (mode === "local" ? "ComfyUI on this Mac" : "ComfyUI"),
          kind,
          base_url: url.trim(),
          auth_type: auth,
          secret: auth === "none" ? null : secret,
          models_dir: mode === "local" ? folder.trim() || null : null,
        });
        const c = created as ComfyConnection;
        if (c.status === "ready") toast.success(t("cf.connected"));
        else toast.error(c.last_error ?? t("cf.notReachable"));
      }
      onClose();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : t("errors.generic"));
    }
  };

  return (
    <Modal
      open
      wide
      onClose={onClose}
      icon={<Plug />}
      title={t("cf.addTitle")}
      description={t("cf.addBody")}
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>
            {t("common.cancel")}
          </Button>
          <Button variant="primary" onClick={submit} loading={add.isPending || deploy.isPending}>
            {mode === "deploy" ? t("cf.deploy") : t("cf.connect")}
          </Button>
        </>
      }
    >
      <div className="col gap-16">
        <Segmented
          label={t("cf.addTitle")}
          value={mode}
          block
          onChange={(m) => {
            setMode(m);
            setError(null);
            if (m === "local") setUrl("http://127.0.0.1:8188");
            if (m === "remote") setUrl("https://");
            if (m === "existing") void loadWorkloads(provider);
          }}
          options={[
            { value: "local", label: t("cf.mode.local") },
            { value: "remote", label: t("cf.mode.remote") },
            { value: "existing", label: t("cf.mode.existing") },
            { value: "deploy", label: t("cf.mode.deploy") },
          ]}
        />
        <div className="hint">{t(`cf.modeHint.${mode}`)}</div>
        <Field label={t("cf.name")}>
          <input className="input" value={name} maxLength={80} onChange={(e) => setName(e.target.value)} placeholder={t("cf.namePlaceholder")} />
        </Field>

        {mode === "existing" ? (
          <>
            <Segmented
              label={t("gpu.provider")}
              value={provider}
              onChange={(p) => void loadWorkloads(p)}
              options={[
                { value: "runpod", label: "RunPod" },
                { value: "modal", label: "Modal" },
              ]}
            />
            {workloads === null ? (
              <div className="hint">{t("common.loading")}</div>
            ) : workloads.length === 0 ? (
              <div className="hint">{t("cf.noWorkloads")}</div>
            ) : (
              <div className="col" style={{ gap: 6 }}>
                {workloads.flatMap((w) =>
                  w.urls.map((u) => (
                    <button
                      key={u}
                      type="button"
                      className="chip"
                      aria-pressed={url === u}
                      onClick={() => {
                        setUrl(u);
                        if (!name) setName(w.name);
                      }}
                    >
                      {w.name} · {u}
                    </button>
                  )),
                )}
              </div>
            )}
          </>
        ) : null}

        {mode !== "deploy" ? (
          <>
            <Field label={t("cf.url")} hint={mode === "local" ? t("cf.urlLocalHint") : t("cf.urlRemoteHint")}>
              <input className="input mono" value={url} onChange={(e) => setUrl(e.target.value)} />
            </Field>
            <div className="form-grid">
              <Field label={t("cf.auth")}>
                <Select
                  label={t("cf.auth")}
                  value={auth}
                  onChange={setAuth}
                  options={[
                    { value: "none", label: t("cf.authNone") },
                    { value: "bearer", label: t("cf.authBearer") },
                    { value: "basic", label: t("cf.authBasic") },
                  ]}
                />
              </Field>
              {auth !== "none" ? (
                <Field label={auth === "basic" ? t("cf.userPass") : t("cf.token")}>
                  <input
                    className="input"
                    type="password"
                    autoComplete="off"
                    value={secret}
                    placeholder={auth === "basic" ? "user:password" : ""}
                    onChange={(e) => setSecret(e.target.value)}
                  />
                </Field>
              ) : null}
            </div>
            {mode === "local" ? (
              <Field label={t("cf.folder")} hint={t("cf.folderHint")}>
                <input className="input mono" value={folder} placeholder="~/ComfyUI" onChange={(e) => setFolder(e.target.value)} />
              </Field>
            ) : null}
          </>
        ) : (
          <>
            <GpuPicker value={gpu} onChange={setGpu} minMemoryGb={24} />
            <Field label={t("cf.variantsToInstall")} hint={t("cf.variantsHint")}>
              <div className="chips">
                {(["turbo", "raw"] as const).map((v) => (
                  <button
                    key={v}
                    type="button"
                    className="chip"
                    aria-pressed={variants.includes(v)}
                    disabled={v === "raw"}
                    title={v === "raw" ? t("cf.rawNoUrl") : undefined}
                    onClick={() => setVariants((cur) => (cur.includes(v) ? cur.filter((x) => x !== v) : [...cur, v]))}
                  >
                    Krea 2 {v === "turbo" ? "Turbo" : "RAW"}
                  </button>
                ))}
              </div>
            </Field>
            <div className="form-grid">
              <Field label={t("cf.autoStop")} hint={t("cf.autoStopHint")}>
                <Select
                  label={t("cf.autoStop")}
                  value={autoStop}
                  onChange={(m) => setAutoStop(Number(m))}
                  options={[0, 15, 30, 60, 120].map((m) => ({ value: m, label: m ? `${m} min` : t("common.off") }))}
                />
              </Field>
              <Field label={t("cf.image")} hint={t("cf.imageHint")}>
                <input className="input mono" value={image} placeholder={t("cf.imageDefault")} onChange={(e) => setImage(e.target.value)} />
              </Field>
            </div>
            <Notice>{t("cf.deployNote")}</Notice>
          </>
        )}
        {error ? <div className="error-text">{error}</div> : null}
      </div>
    </Modal>
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
