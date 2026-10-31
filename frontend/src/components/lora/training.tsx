"use client";

import { ChevronDown, Play, TriangleAlert } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

import { type GpuChoice, GpuPicker, newRequestId } from "@/components/studio/common";
import { Badge, Button, Field, Notice, Select, Skeleton, Switch, useConfirm } from "@/components/ui";
import { ApiError } from "@/lib/api";
import { useT } from "@/lib/i18n";
import type { MessageKey } from "@/lib/i18n";
import { useLoraCatalog, useStartTraining } from "@/lib/queries";
import type { LoraDetail, LoraPreset, TrainingConfig } from "@/lib/types";

function Num({
  label,
  hint,
  value,
  onChange,
  min,
  max,
  step = 1,
}: {
  label: string;
  hint?: string;
  value: number;
  onChange: (v: number) => void;
  min: number;
  max: number;
  step?: number;
}) {
  return (
    <Field label={label} hint={hint}>
      <input
        className="input"
        type="number"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </Field>
  );
}

function Toggle({ label, hint, value, onChange }: { label: string; hint?: string; value: boolean; onChange: (v: boolean) => void }) {
  return (
    <div className="row" style={{ gap: 10, alignItems: "flex-start" }}>
      <Switch checked={value} onChange={onChange} label={label} />
      <div>
        <div className="small">{label}</div>
        {hint ? <div className="hint">{hint}</div> : null}
      </div>
    </div>
  );
}

export function TrainingTab({ lora, onStarted, initial }: { lora: LoraDetail; onStarted: () => void; initial?: Partial<TrainingConfig> | null }) {
  const t = useT();
  const confirm = useConfirm();
  const catalog = useLoraCatalog();
  const start = useStartTraining(lora.id);
  const [preset, setPreset] = useState<LoraPreset["key"]>("character_identity");
  const [config, setConfig] = useState<TrainingConfig | null>(null);
  const [advanced, setAdvanced] = useState(false);
  const [gpu, setGpu] = useState<GpuChoice>({ provider: "", gpu_type: "", gpu_count: 1, cloud_type: "SECURE" });
  const [autoCleanup, setAutoCleanup] = useState(true);
  const [image, setImage] = useState("");
  const [notes, setNotes] = useState("");
  const [prompts, setPrompts] = useState("");

  if (catalog.isLoading || !catalog.data) return <Skeleton style={{ height: 400 }} />;
  const arch = catalog.data.architectures.find((a) => a.key === lora.architecture);
  const presets = catalog.data.presets;
  const base = presets.find((p) => p.key === preset)!.config;
  const cfg: TrainingConfig = config ?? { ...base, ...(initial ?? {}), base_variant: lora.base_variant };
  const set = (patch: Partial<TrainingConfig>) => setConfig({ ...cfg, ...patch });
  const minVram = cfg.quantize ? arch?.vram_quantized_gb : arch?.vram_full_gb;
  const blocked = !lora.analysis.ready || !gpu.provider || !gpu.gpu_type;
  const gated = arch?.variants[lora.base_variant]?.gated && !catalog.data.hf_token_set;

  const submit = async () => {
    const ok = await confirm({
      title: t("gpu.billingTitle"),
      description: t("ls.billing", {
        provider: gpu.provider === "modal" ? "Modal" : "RunPod",
        steps: cfg.steps,
        gpu: `${gpu.gpu_count}× ${gpu.gpu_type}`,
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
        gpu_count: gpu.gpu_count,
        cloud_type: gpu.cloud_type,
        image: image.trim() || null,
        auto_cleanup: autoCleanup,
        notes,
        client_request_id: newRequestId(),
        confirm_billing: true,
      });
      toast.success(t("ls.trainingStarted"));
      onStarted();
    } catch (e) {
      toast.error(e instanceof ApiError ? e.message : t("errors.generic"));
    }
  };

  return (
    <div className="col gap-16">
      {!lora.analysis.ready ? <Notice tone="warning">{t("ls.datasetNotReady")}</Notice> : null}
      {gated ? <Notice tone="warning">{t("ls.needHf")}</Notice> : null}
      <section>
        <div className="section-title">
          <span className="eyebrow grow">{t("ls.presets")}</span>
          <Badge>{arch?.variants[lora.base_variant]?.name}</Badge>
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
              <span className="desc">{p.summary}</span>
              <span className="tiny subtle">{p.tradeoff}</span>
              <span className="tiny mono">
                {p.config.steps} steps · rank {p.config.rank} · lr {p.config.learning_rate}
              </span>
            </button>
          ))}
        </div>
        <div className="hint" style={{ marginTop: 8 }}>{t("ls.presetNote")}</div>
      </section>

      <section className="card">
        <button type="button" className="card-head adv-toggle" onClick={() => setAdvanced((v) => !v)} aria-expanded={advanced}>
          <span className="h3 grow" style={{ textAlign: "left" }}>{t("ls.advanced")}</span>
          <span className="tiny subtle">{t("ls.advancedHint")}</span>
          <ChevronDown size={16} style={{ transform: advanced ? "rotate(180deg)" : undefined }} />
        </button>
        {advanced && arch ? (
          <div className="card-body col gap-16">
            <div className="eyebrow">{t("ls.grp.network")}</div>
            <div className="form-grid">
              <Num label={t("ls.rank")} hint={t("ls.rankHint")} value={cfg.rank} min={4} max={128} onChange={(v) => set({ rank: v })} />
              <Num label={t("ls.alpha")} hint={t("ls.alphaHint")} value={cfg.alpha} min={1} max={128} onChange={(v) => set({ alpha: v })} />
            </div>
            <div className="eyebrow">{t("ls.grp.optim")}</div>
            <div className="form-grid">
              <Num label={t("ls.lr")} value={cfg.learning_rate} min={0.000001} max={0.001} step={0.000005} onChange={(v) => set({ learning_rate: v })} />
              <Field label={t("ls.optimizer")}>
                <Select label={t("ls.optimizer")} value={cfg.optimizer} onChange={(v) => set({ optimizer: v })} options={arch.optimizers.map((o) => ({ value: o, label: o }))} />
              </Field>
              <Field label={t("ls.scheduler")}>
                <Select label={t("ls.scheduler")} value={cfg.lr_scheduler} onChange={(v) => set({ lr_scheduler: v })} options={arch.schedulers.map((o) => ({ value: o, label: o }))} />
              </Field>
              <Field label={t("ls.timestep")}>
                <Select label={t("ls.timestep")} value={cfg.timestep_type} onChange={(v) => set({ timestep_type: v })} options={arch.timestep_types.map((o) => ({ value: o, label: o }))} />
              </Field>
            </div>
            <div className="eyebrow">{t("ls.grp.schedule")}</div>
            <div className="form-grid">
              <Num label={t("ls.steps")} hint={t("ls.stepsHint")} value={cfg.steps} min={100} max={20000} step={50} onChange={(v) => set({ steps: v })} />
              <Num label={t("ls.batch")} value={cfg.batch_size} min={1} max={8} onChange={(v) => set({ batch_size: v })} />
              <Num label={t("ls.accum")} value={cfg.gradient_accumulation} min={1} max={16} onChange={(v) => set({ gradient_accumulation: v })} />
              <Num label={t("ls.dropout")} hint={t("ls.dropoutHint")} value={cfg.caption_dropout_rate} min={0} max={0.5} step={0.01} onChange={(v) => set({ caption_dropout_rate: v })} />
            </div>
            <div className="eyebrow">{t("ls.grp.memory")}</div>
            <div className="form-grid">
              <Field label={t("ls.precision")}>
                <Select label={t("ls.precision")} value={cfg.mixed_precision} onChange={(v) => set({ mixed_precision: v })} options={[{ value: "bf16", label: "bf16" }, { value: "fp16", label: "fp16" }]} />
              </Field>
              <Field label={t("ls.saveDtype")}>
                <Select label={t("ls.saveDtype")} value={cfg.save_dtype} onChange={(v) => set({ save_dtype: v })} options={[{ value: "float16", label: "float16" }, { value: "bf16", label: "bf16" }]} />
              </Field>
              <Toggle label={t("ls.checkpointing")} hint={t("ls.checkpointingHint")} value={cfg.gradient_checkpointing} onChange={(v) => set({ gradient_checkpointing: v })} />
              <Toggle label={t("ls.quantize")} hint={t("ls.quantizeHint", { a: arch.vram_quantized_gb, b: arch.vram_full_gb })} value={cfg.quantize} onChange={(v) => set({ quantize: v })} />
              <Toggle label={t("ls.lowVram")} hint={t("ls.lowVramHint")} value={cfg.low_vram} onChange={(v) => set({ low_vram: v })} />
            </div>
            <div className="eyebrow">{t("ls.grp.data")}</div>
            <Field label={t("ls.resolutions")} hint={arch.bucketing}>
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
            <div className="eyebrow">{t("ls.grp.preservation")}</div>
            <div className="form-grid">
              <Toggle label={t("ls.dop")} hint={t("ls.dopHint", { cls: lora.class_word || "—" })} value={cfg.diff_output_preservation} onChange={(v) => set({ diff_output_preservation: v })} />
              <Num label={t("ls.dopStrength")} value={cfg.diff_output_preservation_multiplier} min={0.1} max={3} step={0.1} onChange={(v) => set({ diff_output_preservation_multiplier: v })} />
            </div>
            {!lora.class_word && cfg.diff_output_preservation ? <div className="error-text">{t("ls.dopNeedsClass")}</div> : null}
            <div className="eyebrow">{t("ls.grp.output")}</div>
            <div className="form-grid">
              <Num label={t("ls.saveEvery")} value={cfg.save_every} min={50} max={5000} step={50} onChange={(v) => set({ save_every: v })} />
              <Num label={t("ls.keep")} value={cfg.max_checkpoints} min={1} max={12} onChange={(v) => set({ max_checkpoints: v })} />
              <Num label={t("ls.sampleEvery")} hint={t("ls.sampleEveryHint")} value={cfg.sample_every} min={0} max={5000} step={50} onChange={(v) => set({ sample_every: v })} />
              <Num label={t("ls.seed")} hint={t("ls.seedHint")} value={cfg.seed} min={0} max={2147483647} onChange={(v) => set({ seed: v })} />
            </div>
            <Field label={t("ls.validationPrompts")} hint={t("ls.validationPromptsHint", { trigger: lora.trigger_word })}>
              <textarea className="textarea" rows={3} value={prompts} onChange={(e) => setPrompts(e.target.value)} placeholder={`portrait photo of ${lora.trigger_word} ${lora.class_word}, soft light`} />
            </Field>
            <div className="eyebrow">{t("ls.unavailable")}</div>
            <ul className="unavailable">
              {Object.entries(arch.unavailable).map(([k, v]) => (
                <li key={k} className="tiny subtle">
                  <TriangleAlert size={12} /> <strong>{t(`ls.na.${k}` as MessageKey)}</strong> — {v}
                </li>
              ))}
            </ul>
          </div>
        ) : null}
      </section>

      <section className="card card-pad col gap-16">
        <div className="eyebrow">{t("ls.whereToTrain")}</div>
        <GpuPicker value={gpu} onChange={setGpu} minMemoryGb={minVram} />
        <div className="hint">{t("ls.vramHint", { n: minVram ?? 24 })}</div>
        <div className="form-grid">
          <Toggle label={t("ls.autoCleanup")} hint={t("ls.autoCleanupHint")} value={autoCleanup} onChange={setAutoCleanup} />
          <Field label={t("ls.image")} hint={t("ls.imageHint")}>
            <input className="input mono" value={image} placeholder={arch?.trainer_image} onChange={(e) => setImage(e.target.value)} />
          </Field>
        </div>
        <Field label={t("ls.notes")}>
          <input className="input" value={notes} maxLength={2000} onChange={(e) => setNotes(e.target.value)} placeholder={t("ls.notesPlaceholder")} />
        </Field>
        <div className="row">
          <span className="tiny subtle grow">{t("ls.summary", { steps: cfg.steps, rank: cfg.rank, res: cfg.resolutions.join("/"), n: lora.analysis.train })}</span>
          <Button variant="primary" icon={<Play />} disabled={blocked || Boolean(gated)} loading={start.isPending} onClick={submit}>
            {t("ls.startTraining")}
          </Button>
        </div>
      </section>
    </div>
  );
}
