"use client";

import { useQueryClient } from "@tanstack/react-query";
import {
  Cpu,
  ExternalLink,
  Play,
  Plug,
  Plus,
  RefreshCw,
  Server,
  Square,
  Trash2,
  Wallet,
} from "lucide-react";
import Link from "next/link";
import { type FormEvent, useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

import { PageHeader } from "@/components/layout/page-header";
import {
  Badge,
  Button,
  EmptyState,
  Field,
  IconButton,
  Modal,
  Notice,
  Segmented,
  Select,
  Skeleton,
  useConfirm,
} from "@/components/ui";
import { api, ApiError } from "@/lib/api";
import { formatRelative } from "@/lib/format";
import { useI18n } from "@/lib/i18n";
import {
  gpuKeys,
  useGPUProviders,
  useGpuEndpoints,
  useGpuTypes,
  useGpuUsage,
  useGpuWorkloads,
} from "@/lib/queries";
import type { GPUProvider, GPUProviderKey, GpuEndpoint, GpuWorkload } from "@/lib/types";

function money(value: number | null | undefined, locale: string) {
  if (value === null || value === undefined) return "—";
  return value.toLocaleString(locale, { style: "currency", currency: "USD", maximumFractionDigits: 3 });
}

const STATUSES = ["running", "starting", "stopped", "exited", "terminated"] as const;

function statusKey(status: string): (typeof STATUSES)[number] | "unknown" {
  return (STATUSES as readonly string[]).includes(status)
    ? (status as (typeof STATUSES)[number])
    : "unknown";
}

function statusTone(status: string) {
  if (status === "running") return "success" as const;
  if (status === "starting" || status === "created") return "accent" as const;
  if (status === "error" || status === "failed") return "danger" as const;
  return undefined;
}

function NewWorkload({
  provider,
  open,
  onClose,
}: {
  provider: GPUProvider;
  open: boolean;
  onClose: () => void;
}) {
  const { t, locale } = useI18n();
  const qc = useQueryClient();
  const types = useGpuTypes(provider.key, open);
  const [name, setName] = useState("studio-gpu");
  const [gpu, setGpu] = useState<string>("");
  const [count, setCount] = useState(1);
  const [image, setImage] = useState(provider.default_image);
  const [ports, setPorts] = useState("");
  const [env, setEnv] = useState("");
  const [cloud, setCloud] = useState<"SECURE" | "COMMUNITY">("SECURE");
  const [disk, setDisk] = useState(20);
  const [command, setCommand] = useState("");
  const [timeout, setTimeoutMin] = useState(60);
  const [agree, setAgree] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const isRunPod = provider.key === "runpod";
  const selected = types.data?.items.find((g) => g.id === gpu);
  const price = selected ? (cloud === "SECURE" ? selected.price_secure : selected.price_community) : null;

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    const portList = ports
      .split(/[,\s]+/)
      .filter(Boolean)
      .map(Number);
    if (portList.some((p) => !Number.isInteger(p) || p < 1 || p > 65535)) {
      setError(t("gpu.badPorts"));
      return;
    }
    const envMap: Record<string, string> = {};
    for (const line of env.split("\n").map((l) => l.trim()).filter(Boolean)) {
      const idx = line.indexOf("=");
      if (idx < 1) {
        setError(t("gpu.badEnv"));
        return;
      }
      envMap[line.slice(0, idx).trim()] = line.slice(idx + 1).trim();
    }
    setBusy(true);
    try {
      await api(`/api/gpu/${provider.key}/workloads`, {
        body: {
          name,
          image,
          gpu_type: gpu,
          gpu_count: count,
          ports: portList,
          env: envMap,
          cloud_type: cloud,
          container_disk_gb: disk,
          command: command.trim() ? command.trim().split(/\s+/) : [],
          timeout_minutes: timeout,
          confirm_billing: agree,
        },
      });
      toast.success(t("gpu.started", { name }));
      qc.invalidateQueries({ queryKey: gpuKeys.workloads(provider.key) });
      qc.invalidateQueries({ queryKey: gpuKeys.usage(provider.key) });
      onClose();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : t("errors.generic"));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal open={open} onClose={onClose} wide title={t("gpu.newTitle", { name: provider.name })} description={t("gpu.newLede")}>
      <form onSubmit={submit} className="col" style={{ gap: 14, paddingBottom: 16 }}>
        <div className="form-grid">
          <Field label={t("gpu.name")} htmlFor="wl-name">
            <input id="wl-name" className="input" value={name} maxLength={60} onChange={(e) => setName(e.target.value)} />
          </Field>
          <Field label={t("gpu.gpuType")}>
            {types.isLoading ? (
              <Skeleton style={{ height: 36 }} />
            ) : (
              <Select
                label={t("gpu.gpuType")}
                value={gpu || undefined}
                placeholder={t("gpu.chooseGpu")}
                onChange={setGpu}
                options={(types.data?.items ?? []).map((g) => ({
                  value: g.id,
                  label: g.name,
                  hint: [
                    g.memory_gb ? `${g.memory_gb} GB` : null,
                    g.price_secure !== null ? `${money(g.price_secure, locale)}/h` : null,
                  ]
                    .filter(Boolean)
                    .join(" · "),
                }))}
              />
            )}
          </Field>
          <Field label={t("gpu.count")}>
            <Segmented
              label={t("gpu.count")}
              value={count}
              onChange={setCount}
              options={[1, 2, 4, 8].map((n) => ({ value: n, label: String(n) }))}
            />
          </Field>
          {isRunPod ? (
            <Field label={t("gpu.cloud")}>
              <Segmented
                label={t("gpu.cloud")}
                value={cloud}
                onChange={setCloud}
                options={[
                  { value: "SECURE", label: t("gpu.secure") },
                  { value: "COMMUNITY", label: t("gpu.community") },
                ]}
              />
            </Field>
          ) : (
            <Field label={t("gpu.timeout")} hint={t("gpu.timeoutHint")}>
              <input
                className="input"
                type="number"
                min={1}
                max={1440}
                value={timeout}
                onChange={(e) => setTimeoutMin(Number(e.target.value))}
              />
            </Field>
          )}
        </div>
        <Field label={t("gpu.image")} hint={t("gpu.imageHint")} htmlFor="wl-image">
          <input id="wl-image" className="input mono" value={image} onChange={(e) => setImage(e.target.value)} />
        </Field>
        <div className="form-grid">
          <Field label={t("gpu.ports")} hint={isRunPod ? t("gpu.portsHintRunpod") : t("gpu.portsHintModal")} htmlFor="wl-ports">
            <input id="wl-ports" className="input" placeholder="8188, 8888" value={ports} onChange={(e) => setPorts(e.target.value)} />
          </Field>
          {isRunPod ? (
            <Field label={t("gpu.disk")} htmlFor="wl-disk">
              <input id="wl-disk" className="input" type="number" min={5} max={500} value={disk} onChange={(e) => setDisk(Number(e.target.value))} />
            </Field>
          ) : (
            <Field label={t("gpu.command")} hint={t("gpu.commandHint")} htmlFor="wl-cmd">
              <input id="wl-cmd" className="input mono" placeholder="sleep infinity" value={command} onChange={(e) => setCommand(e.target.value)} />
            </Field>
          )}
        </div>
        <Field label={t("gpu.env")} hint={t("gpu.envHint")} htmlFor="wl-env">
          <textarea id="wl-env" className="textarea mono" rows={3} value={env} onChange={(e) => setEnv(e.target.value)} placeholder="HF_HOME=/workspace/hf" />
        </Field>
        <label className="notice warning" style={{ cursor: "pointer" }}>
          <input type="checkbox" checked={agree} onChange={(e) => setAgree(e.target.checked)} style={{ marginTop: 3 }} />
          <div className="grow">
            <strong>{t("gpu.billingTitle")}</strong>
            <div>
              {t("gpu.billingBody", { provider: provider.name })}
              {price !== null && price !== undefined ? ` ${t("gpu.priceNow", { price: money(price * count, locale) })}` : ""}
            </div>
          </div>
        </label>
        {error ? <Notice tone="danger">{error}</Notice> : null}
        <div className="row" style={{ justifyContent: "flex-end" }}>
          <Button variant="ghost" onClick={onClose}>
            {t("common.cancel")}
          </Button>
          <Button type="submit" variant="primary" loading={busy} disabled={!gpu || !image || !name || !agree} icon={<Play />}>
            {t("gpu.start")}
          </Button>
        </div>
      </form>
    </Modal>
  );
}

function EndpointRow({ provider, endpoint }: { provider: GPUProviderKey; endpoint: GpuEndpoint }) {
  const { t } = useI18n();
  const qc = useQueryClient();
  const confirm = useConfirm();
  const [min, setMin] = useState(endpoint.workers_min);
  const [max, setMax] = useState(endpoint.workers_max);
  const changed = min !== endpoint.workers_min || max !== endpoint.workers_max;

  const save = async () => {
    if (min > 0) {
      const ok = await confirm({
        title: t("gpu.alwaysOnTitle"),
        description: t("gpu.alwaysOnBody", { n: min }),
        confirmLabel: t("common.save"),
      });
      if (!ok) return;
    }
    try {
      await api(`/api/gpu/${provider}/endpoints/${endpoint.id}`, {
        method: "PATCH",
        body: { workers_min: min, workers_max: max, confirm_billing: min > 0 },
      });
      toast.success(t("set.saved"));
      qc.invalidateQueries({ queryKey: gpuKeys.endpoints(provider) });
    } catch (e) {
      toast.error((e as Error).message);
    }
  };

  const remove = async () => {
    const ok = await confirm({
      title: t("gpu.deleteEndpointTitle", { name: endpoint.name }),
      description: t("gpu.deleteEndpointBody"),
      confirmLabel: t("common.delete"),
      danger: true,
    });
    if (!ok) return;
    try {
      await api(`/api/gpu/${provider}/endpoints/${endpoint.id}`, { method: "DELETE" });
      qc.invalidateQueries({ queryKey: gpuKeys.endpoints(provider) });
    } catch (e) {
      toast.error((e as Error).message);
    }
  };

  const workers = endpoint.health?.workers;
  return (
    <tr>
      <td>
        <div style={{ fontWeight: 600 }}>{endpoint.name}</div>
        <div className="tiny subtle mono">{endpoint.url}</div>
      </td>
      <td className="small">{endpoint.gpu_types.join(", ") || "—"}</td>
      <td className="small">
        {workers
          ? Object.entries(workers)
              .map(([k, v]) => `${k} ${v}`)
              .join(" · ")
          : "—"}
      </td>
      <td>
        <div className="row" style={{ gap: 6 }}>
          <input className="input" style={{ width: 64 }} type="number" min={0} max={100} value={min} onChange={(e) => setMin(Number(e.target.value))} aria-label={t("gpu.workersMin")} />
          <span className="subtle">–</span>
          <input className="input" style={{ width: 64 }} type="number" min={0} max={100} value={max} onChange={(e) => setMax(Number(e.target.value))} aria-label={t("gpu.workersMax")} />
          {changed ? (
            <Button size="sm" variant="primary" onClick={save}>
              {t("common.save")}
            </Button>
          ) : null}
        </div>
      </td>
      <td style={{ textAlign: "right" }}>
        <IconButton label={t("common.delete")} onClick={remove}>
          <Trash2 />
        </IconButton>
      </td>
    </tr>
  );
}

function ProviderView({ provider }: { provider: GPUProvider }) {
  const { t, locale } = useI18n();
  const qc = useQueryClient();
  const confirm = useConfirm();
  const workloads = useGpuWorkloads(provider.key, true);
  const usage = useGpuUsage(provider.key, true);
  const endpoints = useGpuEndpoints(provider.key, provider.supports_endpoints);
  const types = useGpuTypes(provider.key, true);
  const [creating, setCreating] = useState(false);
  const [busyId, setBusyId] = useState<string | null>(null);

  const refresh = () => {
    qc.invalidateQueries({ queryKey: ["gpu", provider.key] });
  };

  const act = async (w: GpuWorkload, action: "start" | "stop" | "terminate") => {
    if (action !== "stop") {
      const ok = await confirm({
        title: action === "start" ? t("gpu.startTitle", { name: w.name }) : t("gpu.terminateTitle", { name: w.name }),
        description: action === "start" ? t("gpu.billingBody", { provider: provider.name }) : t("gpu.terminateBody"),
        confirmLabel: action === "start" ? t("gpu.start") : t("gpu.terminate"),
        danger: action === "terminate",
      });
      if (!ok) return;
    }
    setBusyId(w.id);
    try {
      if (action === "terminate") await api(`/api/gpu/${provider.key}/workloads/${w.id}`, { method: "DELETE" });
      else
        await api(`/api/gpu/${provider.key}/workloads/${w.id}/${action}`, {
          body: action === "start" ? { confirm_billing: true } : {},
        });
      toast.success(t(`gpu.done.${action}`, { name: w.name }));
      refresh();
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setBusyId(null);
    }
  };

  const items = workloads.data?.items ?? [];
  const running = items.filter((w) => w.status === "running").length;
  const u = usage.data;

  return (
    <>
      <div className="stats" style={{ gridTemplateColumns: "repeat(4, minmax(0, 1fr))" }}>
        <div className="card stat">
          <div className="stat-top">
            <Wallet /> {t("gpu.balance")}
          </div>
          <div className="stat-value">{usage.isLoading ? <Skeleton style={{ height: 28, width: 80 }} /> : money(u?.balance, locale)}</div>
          <div className="stat-foot">{u?.balance === null ? t("gpu.noBalanceApi", { name: provider.name }) : ""}</div>
        </div>
        <div className="card stat">
          <div className="stat-top">
            <Cpu /> {t("gpu.spend")}
          </div>
          <div className="stat-value">{usage.isLoading ? <Skeleton style={{ height: 28, width: 80 }} /> : u?.spend_per_hour !== null && u?.spend_per_hour !== undefined ? `${money(u.spend_per_hour, locale)}/h` : "—"}</div>
          <div className="stat-foot">{u?.source === "pods" ? t("gpu.spendFromPods") : u?.spend_per_hour === null ? t("gpu.noPrices") : ""}</div>
        </div>
        <div className="card stat">
          <div className="stat-top">
            <Server /> {t("gpu.running")}
          </div>
          <div className="stat-value">{workloads.isLoading ? "—" : running}</div>
          <div className="stat-foot">{t("gpu.total", { n: items.length })}</div>
        </div>
        <div className="card stat">
          <div className="stat-top">
            <Cpu /> {t("gpu.gpuHours")}
          </div>
          <div className="stat-value">{u ? u.studio_gpu_hours.toLocaleString(locale) : "—"}</div>
          <div className="stat-foot">{t("gpu.gpuHoursFoot", { n: u?.studio_workloads ?? 0 })}</div>
        </div>
      </div>

      <div className="card" style={{ marginBottom: 20 }}>
        <div className="card-head">
          <Server size={16} color="var(--text-3)" />
          <div className="h3 grow">{provider.supports_stop_start ? t("gpu.pods") : t("gpu.sandboxes")}</div>
          <IconButton label={t("gpu.refresh")} onClick={refresh}>
            <RefreshCw />
          </IconButton>
          <a className="btn btn-ghost btn-sm" href={provider.dashboard_url} target="_blank" rel="noreferrer">
            {t("gpu.dashboard")} <ExternalLink />
          </a>
          <Button size="sm" variant="primary" icon={<Plus />} onClick={() => setCreating(true)}>
            {t("gpu.new")}
          </Button>
        </div>
        {workloads.isLoading ? (
          <div className="card-body">
            <Skeleton style={{ height: 80 }} />
          </div>
        ) : workloads.isError ? (
          <div className="card-body">
            <Notice tone="danger">{(workloads.error as Error).message}</Notice>
          </div>
        ) : items.length === 0 ? (
          <div className="card-body small subtle">{t("gpu.noWorkloads")}</div>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table className="matrix">
              <thead>
                <tr>
                  <th>{t("gpu.name")}</th>
                  <th>{t("lib.status")}</th>
                  <th>{t("gpu.gpuType")}</th>
                  <th>{t("gpu.costHour")}</th>
                  <th>{t("gpu.endpoints")}</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {items.map((w) => (
                  <tr key={w.id}>
                    <td>
                      <div style={{ fontWeight: 600 }}>{w.name}</div>
                      <div className="tiny subtle mono truncate" style={{ maxWidth: 280 }} title={w.image}>
                        {w.image || w.id}
                      </div>
                      {w.from_studio ? <span className="tiny subtle">{t("gpu.fromStudio")}</span> : null}
                    </td>
                    <td>
                      <Badge tone={statusTone(w.status)}>
                        <span className={w.status === "starting" ? "dot pulse" : "dot"} />{" "}
                        {t(`gpu.status.${statusKey(w.status)}`)}
                      </Badge>
                    </td>
                    <td className="small">
                      {w.gpu_type || "—"}
                      {w.gpu_count > 1 ? ` × ${w.gpu_count}` : ""}
                    </td>
                    <td className="small">{w.cost_per_hour !== null ? `${money(w.cost_per_hour, locale)}/h` : "—"}</td>
                    <td className="small">
                      {w.urls.length
                        ? w.urls.map((u) => (
                            <a key={u} href={u} target="_blank" rel="noreferrer" className="row gap-4" style={{ color: "var(--accent-text)" }}>
                              {u.replace("https://", "").slice(0, 38)} <ExternalLink size={12} />
                            </a>
                          ))
                        : "—"}
                    </td>
                    <td style={{ textAlign: "right", whiteSpace: "nowrap" }}>
                      {w.can_start ? (
                        <Button size="sm" icon={<Play />} loading={busyId === w.id} onClick={() => act(w, "start")}>
                          {t("gpu.startShort")}
                        </Button>
                      ) : null}
                      {w.can_stop ? (
                        <Button size="sm" icon={<Square />} loading={busyId === w.id} onClick={() => act(w, "stop")}>
                          {t("gpu.stop")}
                        </Button>
                      ) : null}
                      {w.can_terminate && w.status !== "terminated" ? (
                        <IconButton label={t("gpu.terminate")} onClick={() => act(w, "terminate")}>
                          <Trash2 />
                        </IconButton>
                      ) : null}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {provider.supports_endpoints ? (
        <div className="card" style={{ marginBottom: 20 }}>
          <div className="card-head">
            <Plug size={16} color="var(--text-3)" />
            <div className="h3 grow">{t("gpu.serverless")}</div>
            <span className="tiny subtle">{t("gpu.serverlessHint")}</span>
          </div>
          {endpoints.isLoading ? (
            <div className="card-body">
              <Skeleton style={{ height: 60 }} />
            </div>
          ) : endpoints.isError ? (
            <div className="card-body">
              <Notice tone="danger">{(endpoints.error as Error).message}</Notice>
            </div>
          ) : (endpoints.data?.items.length ?? 0) === 0 ? (
            <div className="card-body small subtle">{t("gpu.noEndpoints")}</div>
          ) : (
            <div style={{ overflowX: "auto" }}>
              <table className="matrix">
                <thead>
                  <tr>
                    <th>{t("gpu.endpoint")}</th>
                    <th>{t("gpu.gpuType")}</th>
                    <th>{t("gpu.health")}</th>
                    <th>{t("gpu.workers")}</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {endpoints.data!.items.map((e) => (
                    <EndpointRow key={e.id} provider={provider.key} endpoint={e} />
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      ) : null}

      <div className="card">
        <div className="card-head">
          <Cpu size={16} color="var(--text-3)" />
          <div className="h3 grow">{t("gpu.catalog")}</div>
          {provider.key === "modal" ? <span className="tiny subtle">{t("gpu.modalPricesNote")}</span> : null}
        </div>
        {types.isLoading ? (
          <div className="card-body">
            <Skeleton style={{ height: 80 }} />
          </div>
        ) : types.isError ? (
          <div className="card-body">
            <Notice tone="danger">{(types.error as Error).message}</Notice>
          </div>
        ) : (
          <div style={{ overflowX: "auto", maxHeight: 380 }}>
            <table className="matrix">
              <thead>
                <tr>
                  <th>{t("gpu.gpuType")}</th>
                  <th>{t("gpu.memory")}</th>
                  {provider.key === "runpod" ? (
                    <>
                      <th>{t("gpu.secure")}</th>
                      <th>{t("gpu.community")}</th>
                    </>
                  ) : null}
                </tr>
              </thead>
              <tbody>
                {(types.data?.items ?? []).map((g) => (
                  <tr key={g.id}>
                    <td>
                      <div style={{ fontWeight: 500 }}>{g.name}</div>
                      <div className="tiny subtle mono">{g.id}</div>
                    </td>
                    <td className="small">{g.memory_gb ? `${g.memory_gb} GB` : "—"}</td>
                    {provider.key === "runpod" ? (
                      <>
                        <td className="small">{g.price_secure !== null ? `${money(g.price_secure, locale)}/h` : "—"}</td>
                        <td className="small">{g.price_community !== null ? `${money(g.price_community, locale)}/h` : "—"}</td>
                      </>
                    ) : null}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {u?.recent.length ? (
        <div className="card" style={{ marginTop: 20 }}>
          <div className="card-head">
            <div className="h3 grow">{t("gpu.history")}</div>
          </div>
          <div className="card-body col" style={{ gap: 6 }}>
            {u.recent.map((r) => (
              <div key={r.id} className="row small">
                <span className="grow truncate">{r.name}</span>
                <span className="subtle">
                  {r.gpu_type}
                  {r.gpu_count > 1 ? ` × ${r.gpu_count}` : ""}
                </span>
                <span className="subtle">{formatRelative(r.created_at, locale)}</span>
                <Badge tone={statusTone(r.status)}>{t(`gpu.status.${statusKey(r.status)}`)}</Badge>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      <NewWorkload key={String(creating)} provider={provider} open={creating} onClose={() => setCreating(false)} />
    </>
  );
}

export default function ComputePage() {
  const { t } = useI18n();
  const providers = useGPUProviders();
  const connected = useMemo(
    () => (providers.data?.providers ?? []).filter((p) => p.status !== "not_connected"),
    [providers.data],
  );
  const [selected, setSelected] = useState<GPUProviderKey | null>(null);

  useEffect(() => {
    if (!selected && connected.length) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- default to the first connected
      setSelected(connected[0].key);
    }
  }, [connected, selected]);

  const provider = connected.find((p) => p.key === selected);

  return (
    <div className="page">
      <PageHeader
        title={t("gpu.title")}
        subtitle={t("gpu.subtitle")}
        actions={
          connected.length > 1 ? (
            <Segmented
              label={t("gpu.provider")}
              value={selected}
              onChange={setSelected}
              options={connected.map((p) => ({ value: p.key, label: p.name }))}
            />
          ) : (
            <Link href="/integrations?tab=gpu" className="btn btn-ghost">
              <Plug /> {t("gpu.manage")}
            </Link>
          )
        }
      />
      {providers.isLoading ? (
        <Skeleton style={{ height: 240 }} />
      ) : !connected.length ? (
        <EmptyState
          icon={<Cpu />}
          title={t("gpu.emptyTitle")}
          description={t("gpu.emptyBody")}
          action={
            <Link href="/integrations?tab=gpu" className="btn btn-primary btn-sm">
              <Plug /> {t("gpu.connect")}
            </Link>
          }
        />
      ) : provider ? (
        <ProviderView key={provider.key} provider={provider} />
      ) : null}
    </div>
  );
}
