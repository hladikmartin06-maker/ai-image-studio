"use client";

import { ArrowRight, ChevronDown, Play } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { toast } from "sonner";

import { hfProblem } from "@/components/integrations/huggingface";
import { type GpuChoice, GpuPicker, newRequestId } from "@/components/studio/common";
import { Button, Field, Notice, Segmented, Skeleton, Switch, useConfirm } from "@/components/ui";
import { ApiError } from "@/lib/api";
import { useT } from "@/lib/i18n";
import type { MessageKey } from "@/lib/i18n";
import { useHfStatus, useLoraCatalog, useStartTraining } from "@/lib/queries";
import type { LoraDetail, LoraPreset, TrainingConfig } from "@/lib/types";

const SPEEDS = [
  { value: 0.00006, label: "ls.speed.gentle" },
  { value: 0.0001, label: "ls.speed.normal" },
  { value: 0.0002, label: "ls.speed.fast" },
] as const;

export function TrainingTab({ lora, onStarted, initial }: { lora: LoraDetail; onStarted: () => void; initial?: Partial<TrainingConfig> | null }) {
  const t = useT();
  const router = useRouter();
  const confirm = useConfirm();
  const catalog = useLoraCatalog();
  const hf = useHfStatus();
  const start = useStartTraining(lora.id);
  const [preset, setPreset] = useState<LoraPreset["key"]>("character_identity");
  const [config, setConfig] = useState<TrainingConfig | null>(null);
  const [advanced, setAdvanced] = useState(false);
  const [gpu, setGpu] = useState<GpuChoice>({ provider: "", gpu_type: "", gpu_count: 1, cloud_type: "SECURE" });
  const [prompts, setPrompts] = useState("");

  if (catalog.isLoading || !catalog.data) return <Skeleton style={{ height: 400 }} />;
  const arch = catalog.data.architectures.find((a) => a.key === lora.architecture);
  const presets = catalog.data.presets;
  const base = presets.find((p) => p.key === preset)!.config;
  const cfg: TrainingConfig = config ?? { ...base, ...(initial ?? {}), base_variant: lora.base_variant };
  const set = (patch: Partial<TrainingConfig>) => setConfig({ ...cfg, ...patch });
  const minVram = cfg.quantize ? arch?.vram_quantized_gb : arch?.vram_full_gb;
  const gated = Boolean(arch?.variants[lora.base_variant]?.gated);
  const hfIssue = gated ? (hf.data ? hfProblem(hf.data, lora.base_variant) : catalog.data.hf_token_set ? null : "hf.needToken") : null;
  const blocked = !lora.analysis.ready || !gpu.provider || !gpu.gpu_type || Boolean(hfIssue);

  const submit = async () => {
    const ok = await confirm({
      title: t("gpu.billingTitle"),
      description: t("ls.billing", {
        provider: gpu.provider === "modal" ? "Modal" : "RunPod",
        steps: cfg.steps,
        gpu: gpu.gpu_type,
      }),
      confirmLabel: t("ls.startTraining"),
    });
    if (!ok) return;
    try {
      await start.mutateAsync({
        preset,
        config: {
          ...cfg,
          validation_prompts: prompts
            .split("\n")
            .map((p) => p.trim())
            .filter(Boolean)
            .slice(0, 6),
        },
        provider: gpu.provider,
        gpu_type: gpu.gpu_type,
        gpu_count: 1,
        cloud_type: "SECURE",
        image: null,
        // The GPU is always removed after the run (or a failure), so billing stops.
        auto_cleanup: true,
        notes: "",
        client_request_id: newRequestId(),
        confirm_billing: true,
      });
      toast.success(t("ls.trainingStarted"));
      onStarted();
    } catch (e) {
      const err = e as ApiError;
      const hfCode = err instanceof ApiError && err.code?.startsWith("hf_");
      toast.error(err instanceof ApiError ? err.message : t("errors.generic"), {
        action: hfCode ? { label: t("hf.openSetup"), onClick: () => router.push("/integrations?tab=hf") } : undefined,
      });
      if (hfCode) void hf.refetch();
    }
  };

  return (
    <div className="col gap-16">
      {!lora.analysis.ready ? <Notice tone="warning">{t("ls.datasetNotReady")}</Notice> : null}
      {hfIssue ? (
        <Notice tone="warning">
          <div>{t(hfIssue)}</div>
          <Link href="/integrations?tab=hf" className="btn btn-sm btn-primary" style={{ marginTop: 10 }}>
            {t("hf.openSetup")} <ArrowRight />
          </Link>
        </Notice>
      ) : null}

      <section>
        <div className="section-title">
          <span className="eyebrow grow">{t("ls.presets")}</span>
        </div>
        <div className="preset-grid" role="radiogroup" aria-label={t("ls.presets")}>
          {presets.map((p) => (
            <button
              key={p.key}
              type="button"
              role="radio"
              aria-checked={preset === p.key}
              className="model-card"
              onClick={() => {
                setPreset(p.key);
                setConfig(null);
              }}
            >
              <span className="name">{t(`ls.preset.${p.key}` as MessageKey)}</span>
              <span className="desc">{t(`ls.presetSimple.${p.key}` as MessageKey)}</span>
              <span className="tiny subtle">{t("ls.presetSteps", { n: p.config.steps })}</span>
            </button>
          ))}
        </div>
      </section>

      <section className="card card-pad col gap-16">
        <div className="eyebrow">{t("ls.whereToTrain")}</div>
        <GpuPicker value={gpu} onChange={setGpu} minMemoryGb={minVram} />
        <div className="hint">{t("ls.vramHint", { n: minVram ?? 24 })}</div>
      </section>

      <section className="card">
        <button type="button" className="card-head adv-toggle" onClick={() => setAdvanced((v) => !v)} aria-expanded={advanced}>
          <span className="h3 grow" style={{ textAlign: "left" }}>{t("ls.advanced")}</span>
          <span className="tiny subtle">{t("ls.advancedHint")}</span>
          <ChevronDown size={16} style={{ transform: advanced ? "rotate(180deg)" : undefined }} />
        </button>
        {advanced && arch ? (
          <div className="card-body col gap-16">
            <div className="form-grid">
              <Field label={t("ls.steps")} hint={t("ls.stepsHint")}>
                <input
                  className="input"
                  type="number"
                  min={100}
                  max={20000}
                  step={50}
                  value={cfg.steps}
                  onChange={(e) => set({ steps: Number(e.target.value) })}
                />
              </Field>
              <Field label={t("ls.size")} hint={t("ls.sizeHint")}>
                <Segmented
                  label={t("ls.size")}
                  value={cfg.rank}
                  onChange={(r) => set({ rank: r, alpha: Math.min(cfg.alpha, r) })}
                  options={[16, 32, 64].map((r) => ({ value: r, label: String(r) }))}
                />
              </Field>
              <Field label={t("ls.speed")} hint={t("ls.speedHint")}>
                <Segmented
                  label={t("ls.speed")}
                  value={SPEEDS.find((s) => s.value === cfg.learning_rate)?.value ?? cfg.learning_rate}
                  onChange={(v) => set({ learning_rate: v })}
                  options={[
                    ...SPEEDS.map((s) => ({ value: s.value as number, label: t(s.label) })),
                    ...(SPEEDS.some((s) => s.value === cfg.learning_rate) ? [] : [{ value: cfg.learning_rate, label: String(cfg.learning_rate) }]),
                  ]}
                />
              </Field>
              <Field label={t("ls.resolutions")} hint={t("ls.resolutionsHint")}>
                <div className="chips">
                  {arch.resolutions.map((r) => (
                    <button
                      key={r}
                      type="button"
                      className="chip"
                      aria-pressed={cfg.resolutions.includes(r)}
                      onClick={() => {
                        const next = cfg.resolutions.includes(r) ? cfg.resolutions.filter((x) => x !== r) : [...cfg.resolutions, r];
                        if (next.length) set({ resolutions: next.sort((a, b) => a - b) });
                      }}
                    >
                      {r}px
                    </button>
                  ))}
                </div>
              </Field>
            </div>
            <div className="row" style={{ gap: 10, alignItems: "flex-start" }}>
              <Switch checked={cfg.quantize} onChange={(v) => set({ quantize: v })} label={t("ls.quantize")} />
              <div>
                <div className="small">{t("ls.quantize")}</div>
                <div className="hint">{t("ls.quantizeHint", { a: arch.vram_quantized_gb, b: arch.vram_full_gb })}</div>
              </div>
            </div>
            <Field label={t("ls.validationPrompts")} hint={t("ls.validationPromptsHint", { trigger: lora.trigger_word })}>
              <textarea
                className="textarea"
                rows={3}
                value={prompts}
                onChange={(e) => setPrompts(e.target.value)}
                placeholder={`portrait photo of ${lora.trigger_word} ${lora.class_word}, soft light`}
              />
            </Field>
          </div>
        ) : null}
      </section>

      <div className="row">
        <span className="tiny subtle grow">
          {t("ls.summarySimple", { steps: cfg.steps, n: lora.analysis.train })}
          {" · "}
          {t("ls.autoCleanupNote")}
        </span>
        <Button variant="primary" size="lg" icon={<Play />} disabled={blocked} loading={start.isPending} onClick={submit}>
          {t("ls.startTraining")}
        </Button>
      </div>
    </div>
  );
}
