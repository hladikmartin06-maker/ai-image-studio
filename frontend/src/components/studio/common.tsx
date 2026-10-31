"use client";

import { ArrowRight, Cpu } from "lucide-react";
import Link from "next/link";

import { Badge, Field, Notice, Select, Spinner } from "@/components/ui";
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

// --- loss chart -------------------------------------------------------------------------

/** Training loss as reported by the trainer (not a measure of identity or image quality). */
export function LossChart({ points, height = 120 }: { points: [number, number][]; height?: number }) {
  const t = useT();
  if (points.length < 2) {
    return <div className="hint">{t("ls.noLossYet")}</div>;
  }
  const width = 520;
  const xs = points.map((p) => p[0]);
  const ys = points.map((p) => p[1]);
  const [x0, x1] = [Math.min(...xs), Math.max(...xs)];
  const [y0, y1] = [Math.min(...ys), Math.max(...ys)];
  const px = (x: number) => ((x - x0) / Math.max(1, x1 - x0)) * (width - 8) + 4;
  const py = (y: number) => height - 6 - ((y - y0) / Math.max(1e-9, y1 - y0)) * (height - 12);
  // A light moving average makes the trend readable without hiding the raw values.
  const smooth = points.map((_, i) => {
    const slice = points.slice(Math.max(0, i - 4), i + 1);
    return [points[i][0], slice.reduce((a, p) => a + p[1], 0) / slice.length] as [number, number];
  });
  const line = (pts: [number, number][]) => pts.map((p, i) => `${i ? "L" : "M"}${px(p[0]).toFixed(1)},${py(p[1]).toFixed(1)}`).join(" ");
  return (
    <figure className="loss-chart">
      <svg viewBox={`0 0 ${width} ${height}`} role="img" aria-label={t("ls.lossChart")}>
        <path d={line(points)} className="loss-raw" />
        <path d={line(smooth)} className="loss-smooth" />
      </svg>
      <figcaption className="row tiny subtle">
        <span className="grow">
          {t("ls.lossRange", { from: y1.toPrecision(3), to: ys[ys.length - 1].toPrecision(3) })}
        </span>
        <span>{t("ls.lossNote")}</span>
      </figcaption>
    </figure>
  );
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
}

export function GpuPicker({
  value,
  onChange,
  minMemoryGb,
}: {
  value: GpuChoice;
  onChange: (v: GpuChoice) => void;
  minMemoryGb?: number;
}) {
  const { t, locale } = useI18n();
  const providers = useGPUProviders();
  const connected = (providers.data?.providers ?? []).filter((p) => p.status !== "not_connected");
  const types = useGpuTypes(value.provider || "runpod", Boolean(value.provider));
  const selected = types.data?.items.find((g) => g.id === value.gpu_type);
  const price = (g: GpuType) => (value.cloud_type === "COMMUNITY" ? g.price_community : g.price_secure);

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
      <div className="form-grid">
        <Field label={t("gpu.provider")}>
          <Select
            label={t("gpu.provider")}
            value={value.provider || null}
            placeholder={t("gpu.chooseProvider")}
            onChange={(p) => onChange({ ...value, provider: p as GpuChoice["provider"], gpu_type: "" })}
            options={connected.map((p) => ({ value: p.key, label: p.name }))}
          />
        </Field>
        <Field label={t("gpu.gpuType")}>
          {types.isLoading && value.provider ? (
            <Spinner />
          ) : (
            <Select
              label={t("gpu.gpuType")}
              value={value.gpu_type || null}
              placeholder={t("gpu.chooseGpu")}
              onChange={(g) => onChange({ ...value, gpu_type: String(g) })}
              options={(types.data?.items ?? []).map((g) => ({
                value: g.id,
                label: `${g.name}${g.memory_gb ? ` · ${g.memory_gb} GB` : ""}${
                  price(g) !== null && price(g) !== undefined ? ` · ${formatMoney(price(g), locale)}/h` : ""
                }`,
                disabled: Boolean(minMemoryGb && g.memory_gb && g.memory_gb < minMemoryGb),
              }))}
            />
          )}
        </Field>
      </div>
      {selected && minMemoryGb && selected.memory_gb && selected.memory_gb < minMemoryGb ? (
        <div className="error-text">{t("gpu.tooSmall", { n: minMemoryGb })}</div>
      ) : null}
      {selected ? (
        <div className="hint">
          {price(selected) !== null && price(selected) !== undefined
            ? t("gpu.priceHint", { price: formatMoney(price(selected) ?? 0, locale) })
            : t("gpu.noPrice")}
        </div>
      ) : null}
    </div>
  );
}

export function newRequestId() {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}
