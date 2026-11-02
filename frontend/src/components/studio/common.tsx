"use client";

import { ArrowRight, Cpu, RotateCcw } from "lucide-react";
import Link from "next/link";

import { Badge, Button, Field, Notice, Select, Spinner } from "@/components/ui";
import { ApiError } from "@/lib/api";
import { formatBytes } from "@/lib/format";
import { useI18n, useT } from "@/lib/i18n";
import type { MessageKey } from "@/lib/i18n";
import { useGPUProviders, useGpuTypes } from "@/lib/queries";
import type { AgentGpu, GpuType, JobStatus, TrainingStatus } from "@/lib/types";

// --- status -----------------------------------------------------------------------------

const TONE: Record<string, "accent" | "success" | "warning" | "danger" | undefined> = {
  queued: undefined,
  preparing: "accent",
  uploading: "accent",
  running: "accent",
  downloading: "accent",
  completed: "success",
  failed: "danger",
  cancelled: undefined,
  interrupted: "warning",
  ready: "success",
  deploying: "accent",
  error: "danger",
  stopped: undefined,
  new: undefined,
};

const ACTIVE = new Set(["queued", "preparing", "uploading", "running", "downloading", "deploying"]);

export function StatusPill({ status }: { status: JobStatus | TrainingStatus | string }) {
  const t = useT();
  return (
    <Badge tone={TONE[status]}>
      <span className={ACTIVE.has(status) ? "dot pulse" : "dot"} />
      {t(`job.status.${status}` as MessageKey)}
    </Badge>
  );
}

export function isActive(status: string) {
  return ACTIVE.has(status);
}

export function Progress({ value }: { value: number | null | undefined }) {
  if (value === null || value === undefined) return <div className="progress" aria-hidden />;
  return (
    <div className="meter" role="progressbar" aria-valuenow={value} aria-valuemin={0} aria-valuemax={100}>
      <span style={{ width: `${Math.max(2, Math.min(100, value))}%` }} />
    </div>
  );
}

export function formatElapsed(seconds: number | null | undefined) {
  if (seconds === null || seconds === undefined) return "—";
  const s = Math.round(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  return h ? `${h}h ${m}m` : m ? `${m}m ${sec}s` : `${sec}s`;
}

export function formatMoney(value: number | null | undefined, locale: string) {
  if (value === null || value === undefined) return "—";
  const format = (v: number, digits: number) =>
    new Intl.NumberFormat(locale, { style: "currency", currency: "USD", maximumFractionDigits: digits }).format(v);
  if (value > 0 && value < 0.01) return `< ${format(0.01, 2)}`;
  return format(value, value < 1 ? 3 : 2);
}

// --- GPU usage --------------------------------------------------------------------------

export function GpuMeter({ gpu }: { gpu: AgentGpu | null | undefined }) {
  const { t, locale } = useI18n();
  if (!gpu) return null;
  const memory =
    gpu.memory_total_mb && gpu.memory_used_mb !== null
      ? Math.round((100 * gpu.memory_used_mb) / gpu.memory_total_mb)
      : null;
  return (
    <div className="gpu-meter">
      <div className="row tiny" style={{ gap: 6 }}>
        <Cpu size={13} />
        <span className="grow truncate" style={{ fontWeight: 600 }}>
          {gpu.name}
        </span>
        {gpu.temperature !== null ? <span className="subtle">{gpu.temperature}°C</span> : null}
      </div>
      <div className="gpu-meter-row">
        <span className="tiny subtle">{t("gpu.utilization")}</span>
        <Progress value={gpu.utilization ?? 0} />
        <span className="tiny">{gpu.utilization ?? "—"}%</span>
      </div>
      <div className="gpu-meter-row">
        <span className="tiny subtle">VRAM</span>
        <Progress value={memory ?? 0} />
        <span className="tiny">
          {gpu.memory_used_mb !== null && gpu.memory_total_mb
            ? `${formatBytes(gpu.memory_used_mb * 2 ** 20, locale)} / ${formatBytes(gpu.memory_total_mb * 2 ** 20, locale)}`
            : "—"}
        </span>
      </div>
    </div>
  );
}

// --- GPU selection ----------------------------------------------------------------------

export interface GpuChoice {
  provider: "runpod" | "modal" | "";
  gpu_type: string;
  gpu_count: number;
  cloud_type: "SECURE" | "COMMUNITY";
  // The provider's hourly list price of the chosen GPU (RunPod: live; Modal: estimate from
  // its published list prices), and Modal's CPU/memory on top.
  price_per_hour?: number | null;
  price_extra?: number | null;
  price_estimate?: boolean;
  // Modal: "api" = its live rates (then the confirmed price is checked again at the start).
  price_source?: "api" | "list" | null;
  // GPU memory of the chosen type, to refuse a GPU that is too small for the task.
  memory_gb?: number | null;
}

/** Hourly price of a GPU choice (GPU + Modal's CPU/memory), or null when unknown. */
export function hourlyPrice(c: GpuChoice): number | null {
  return c.price_per_hour != null ? c.price_per_hour + (c.price_extra ?? 0) : null;
}

/** GPUs with less memory are never offered (the owner's rule). */
export const MIN_GPU_GB = 24;

type StockLevel = "NONE" | "LOW" | "MEDIUM" | "HIGH";
const STOCK_RANK: Record<string, number> = { HIGH: 3, MEDIUM: 2, LOW: 1, NONE: 0 };

export function GpuPicker({
  value,
  onChange,
  minMemoryGb,
  purpose,
}: {
  value: GpuChoice;
  onChange: (v: GpuChoice) => void;
  minMemoryGb?: number;
  // What the GPU is for: RunPod reports which GPUs have a free machine whose driver fits it.
  purpose?: "comfyui" | "training";
}) {
  const { t, locale } = useI18n();
  const providers = useGPUProviders();
  const connected = (providers.data?.providers ?? []).filter((p) => p.status !== "not_connected");
  const types = useGpuTypes(value.provider || "runpod", Boolean(value.provider), purpose);
  // Only GPUs that can do the job: 24 GB or more (and at least what this task needs), NVIDIA
  // only (the studio's images are CUDA images), smallest and cheapest first. When the provider
  // reports availability (RunPod's catalog), only GPUs with a free machine right now: Secure
  // Cloud first, Community Cloud (its own price) only when Secure has none.
  const floor = Math.max(MIN_GPU_GB, minMemoryGb ?? 0);
  const all = types.data?.items ?? [];
  const reportsStock = all.some((g) => g.stock_secure != null || g.stock_community != null);
  type Offer = {
    key: string;
    gpu: GpuType;
    cloud: "SECURE" | "COMMUNITY";
    price: number | null;
    stock: StockLevel | null;
    busy?: boolean;
  };
  const fitting = all
    .filter((g) => !g.memory_gb || g.memory_gb >= floor)
    .filter((g) => (g.manufacturer ?? "").toLowerCase() !== "amd" && !/^amd\b/i.test(g.id) && !/^amd\b/i.test(g.name));
  const offers: Offer[] = [];
  for (const g of fitting) {
    if (!reportsStock) {
      const p = value.cloud_type === "COMMUNITY" ? g.price_community : g.price_secure;
      const priced = all.some((x) => x.price_secure != null || x.price_community != null);
      if (!priced || (p ?? 0) > 0) offers.push({ key: g.id, gpu: g, cloud: value.cloud_type, price: p, stock: null });
      continue;
    }
    const secure = (g.stock_secure ?? "NONE") as StockLevel;
    const community = (g.stock_community ?? "NONE") as StockLevel;
    if (secure !== "NONE" && (g.price_secure ?? 0) > 0) {
      offers.push({ key: `${g.id}|SECURE`, gpu: g, cloud: "SECURE", price: g.price_secure, stock: secure });
    } else if (community !== "NONE" && (g.price_community ?? 0) > 0) {
      offers.push({ key: `${g.id}|COMMUNITY`, gpu: g, cloud: "COMMUNITY", price: g.price_community, stock: community });
    }
  }
  offers.sort(
    (a, b) =>
      (a.gpu.memory_gb ?? 999) - (b.gpu.memory_gb ?? 999) ||
      (a.price ?? 999) - (b.price ?? 999) ||
      STOCK_RANK[b.stock ?? "NONE"] - STOCK_RANK[a.stock ?? "NONE"],
  );
  // GPUs without a free machine that fits are listed too, but cannot be picked.
  const busy: Offer[] = reportsStock
    ? fitting
        .filter((g) => !offers.some((o) => o.gpu.id === g.id))
        .map((g) => ({ key: `${g.id}|busy`, gpu: g, cloud: "SECURE", price: g.price_secure, stock: "NONE", busy: true }))
    : [];
  const estimates = all.some((g) => g.price_estimate);
  const livePrices = all.some((g) => g.price_source === "api");
  const currentKey = reportsStock ? (value.gpu_type ? `${value.gpu_type}|${value.cloud_type}` : "") : value.gpu_type;
  const selected = offers.find((o) => o.key === currentKey);
  const providerName = connected.find((p) => p.key === value.provider)?.name ?? value.provider;
  const stockLabel = (s: StockLevel | null) =>
    s === "HIGH" ? t("gpu.stock.high") : s === "MEDIUM" ? t("gpu.stock.medium") : s === "LOW" ? t("gpu.stock.low") : "";

  if (providers.isLoading) return <Spinner />;
  if (!connected.length) {
    return (
      <Notice tone="accent">
        <div>{t("gpu.noProviderConnected")}</div>
        <Link href="/integrations?tab=gpu" className="btn btn-sm btn-primary" style={{ marginTop: 10 }}>
          {t("gpu.connectProvider")} <ArrowRight />
        </Link>
      </Notice>
    );
  }
  return (
    <div className="col gap-12">
      {/* One field per row: the GPU label (name · memory · price per hour) stays readable. */}
      <div className="col gap-12">
        <Field label={t("gpu.provider")}>
          <Select
            label={t("gpu.provider")}
            value={value.provider || null}
            placeholder={t("gpu.chooseProvider")}
            onChange={(p) => onChange({ ...value, provider: p as GpuChoice["provider"], gpu_type: "", cloud_type: "SECURE" })}
            options={connected.map((p) => ({ value: p.key, label: p.name }))}
          />
        </Field>
        <Field label={t("gpu.gpuType")}>
          {types.isLoading && value.provider ? (
            <Spinner />
          ) : (
            <Select
              label={t("gpu.gpuType")}
              value={selected ? selected.key : null}
              placeholder={t("gpu.chooseGpu")}
              onChange={(key) => {
                const offer = offers.find((o) => o.key === String(key));
                if (!offer) return;
                onChange({
                  ...value,
                  gpu_type: offer.gpu.id,
                  cloud_type: offer.cloud,
                  price_per_hour: offer.price,
                  price_extra: offer.gpu.price_extra ?? null,
                  price_estimate: Boolean(offer.gpu.price_estimate),
                  price_source: offer.gpu.price_source ?? null,
                  memory_gb: offer.gpu.memory_gb ?? null,
                });
              }}
              options={[...offers, ...busy].map((o) => ({
                value: o.key,
                disabled: o.busy,
                label: [
                  o.gpu.name,
                  o.gpu.memory_gb ? `${o.gpu.memory_gb} GB` : null,
                  o.price != null
                    ? `${o.gpu.price_estimate ? "≈ " : ""}${formatMoney(o.price, locale)}/h${o.gpu.price_extra ? ` + ${t("gpu.cpuRam")}` : ""}`
                    : null,
                  o.cloud === "COMMUNITY" && reportsStock && !o.busy ? t("gpu.community") : null,
                  o.busy ? t("gpu.booked") : stockLabel(o.stock) || null,
                ]
                  .filter(Boolean)
                  .join(" · "),
              }))}
            />
          )}
        </Field>
      </div>
      {value.provider && types.isError ? (
        <Notice tone="danger">
          <div>{t("gpu.typesFailed", { provider: providerName })}</div>
          {types.error instanceof ApiError ? <div className="tiny" style={{ marginTop: 4 }}>{types.error.message}</div> : null}
          <Button size="sm" icon={<RotateCcw />} loading={types.isFetching} onClick={() => void types.refetch()} style={{ marginTop: 10 }}>
            {t("cf.tryAgain")}
          </Button>
        </Notice>
      ) : value.provider && types.data && !offers.length ? (
        <Notice tone="warning">
          {reportsStock ? t("gpu.noneFree", { provider: providerName, n: floor }) : t("gpu.noGpus", { provider: providerName, n: floor })}
          <div style={{ marginTop: 10 }}>
            <Button size="sm" icon={<RotateCcw />} loading={types.isFetching} onClick={() => void types.refetch()}>
              {t("gpu.refresh")}
            </Button>
          </div>
        </Notice>
      ) : null}
      {(estimates || livePrices) && value.provider === "modal" ? (
        <div className="hint">
          {livePrices ? t(estimates ? "gpu.modalLivePartly" : "gpu.modalLive") : t("gpu.modalEstimate")}
          {estimates ? (
            <>
              {" "}
              <a href="https://modal.com/pricing" target="_blank" rel="noreferrer" style={{ color: "var(--accent-text)" }}>
                modal.com/pricing
              </a>
            </>
          ) : null}
        </div>
      ) : null}
      {selected?.cloud === "COMMUNITY" && reportsStock ? <div className="hint">{t("gpu.communityHint")}</div> : null}
      {selected && minMemoryGb && selected.gpu.memory_gb && selected.gpu.memory_gb < minMemoryGb ? (
        <div className="error-text">{t("gpu.tooSmall", { n: minMemoryGb })}</div>
      ) : null}
      {selected && !selected.price ? <div className="hint">{t("gpu.noPrice")}</div> : null}
    </div>
  );
}

export function newRequestId() {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}
