"use client";

import { ArrowRight, Dices, Plug, Sparkles, TriangleAlert } from "lucide-react";
import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";

import { PromptEditor } from "@/components/generate/controls";
import { newPersonaDraft, type PersonaDraft, PersonaSection } from "@/components/personas/persona-picker";
import { newRequestId } from "@/components/studio/common";
import { Badge, Button, Field, IconButton, Notice, Segmented, Select, Skeleton, Switch } from "@/components/ui";
import { ApiError } from "@/lib/api";
import { useT } from "@/lib/i18n";
import {
  useComfyCatalog,
  useComfyConnections,
  useComfyGenerate,
  useComfyWorkflows,
  useLoras,
  usePersonas,
} from "@/lib/queries";
import type { Generation, LoraSummary } from "@/lib/types";

export interface ComfyDraft {
  connectionId: string;
  variant: "turbo" | "raw";
  modelFile: string;
  workflowId: string;
  prompt: string;
  negative: string;
  aspect: string;
  resolution: string;
  seed: string;
  steps: string;
  cfg: string;
  count: number;
  loraVersionId: string;
  strength: number;
  insertTrigger: boolean;
  persona: PersonaDraft | null;
}

const EMPTY: ComfyDraft = {
  connectionId: "",
  variant: "turbo",
  modelFile: "",
  workflowId: "",
  prompt: "",
  negative: "",
  aspect: "3:4",
  resolution: "1K",
  seed: "",
  steps: "",
  cfg: "",
  count: 1,
  loraVersionId: "",
  strength: 1,
  insertTrigger: true,
  persona: null,
};

/** Krea 2 compatibility, as the backend enforces it: RAW LoRAs work on RAW and Turbo,
 * Turbo LoRAs (trained with the adapter) only on Turbo. */
export function loraFits(lora: LoraSummary, variant: "turbo" | "raw") {
  if (lora.architecture !== "krea2") return false;
  return lora.base_variant === variant || (lora.base_variant === "raw" && variant === "turbo");
}

function fromGeneration(gen: Generation): Partial<ComfyDraft> {
  const c = gen.comfy;
  return {
    connectionId: c?.connection_id ?? "",
    variant: c?.variant ?? "turbo",
    modelFile: c?.model_file ?? "",
    workflowId: c?.workflow?.id ?? "",
    prompt: gen.prompt,
    negative: gen.settings.negative_prompt ?? "",
    aspect: gen.settings.aspect_ratio ?? "3:4",
    resolution: gen.settings.resolution ?? "1K",
    seed: gen.settings.seed !== undefined ? String(gen.settings.seed) : "",
    count: gen.image_count,
    loraVersionId: c?.lora?.version_id ?? "",
    strength: c?.lora?.strength ?? 1,
    persona: gen.persona?.id
      ? { ...newPersonaDraft(gen.persona.id), useRefs: false, useDescription: gen.persona.description_used }
      : null,
  };
}

export function ComfyPanel({
  storageKey,
  initial,
  loraFromUrl,
  header,
}: {
  storageKey: string | null;
  initial: Generation | null;
  loraFromUrl: string | null;
  header: React.ReactNode;
}) {
  const t = useT();
  const catalog = useComfyCatalog();
  const conns = useComfyConnections();
  const workflows = useComfyWorkflows();
  const loras = useLoras("", true);
  const personas = usePersonas();
  const generate = useComfyGenerate();
  const submitting = useRef(false);
  const [draft, setDraft] = useState<ComfyDraft>(() => {
    let saved: Partial<ComfyDraft> = {};
    if (storageKey && typeof window !== "undefined") {
      try {
        saved = JSON.parse(localStorage.getItem(storageKey) ?? "{}");
      } catch {
        saved = {};
      }
    }
    return {
      ...EMPTY,
      ...saved,
      ...(initial ? fromGeneration(initial) : {}),
      ...(loraFromUrl ? { loraVersionId: loraFromUrl } : {}),
    };
  });

  useEffect(() => {
    if (!storageKey) return;
    try {
      localStorage.setItem(storageKey, JSON.stringify(draft));
    } catch {
      /* storage blocked */
    }
  }, [draft, storageKey]);

  const set = (patch: Partial<ComfyDraft>) => setDraft((d) => ({ ...d, ...patch }));
  const ready = (conns.data?.items ?? []).filter((c) => c.status === "ready");
  const conn = ready.find((c) => c.id === draft.connectionId) ?? ready[0];
  const detection = conn?.info.krea2;
  const variantSpec = catalog.data?.krea2.variants.find((v) => v.key === draft.variant);
  const files = detection?.variants[draft.variant]?.files ?? [];
  const modelFile = files.find((f) => f.filename === draft.modelFile)?.filename ?? files[0]?.filename ?? "";
  const tiers = variantSpec?.tiers ?? ["1K"];
  const resolution = tiers.includes(draft.resolution) ? draft.resolution : tiers[0];
  const size = catalog.data?.krea2.sizes[resolution]?.[draft.aspect];
  const runnable = (workflows.data?.items ?? []).filter((w) => w.runnable && (!w.family || w.family === "krea2"));

  const loraOptions = useMemo(() => {
    const out: { value: string; label: string; disabled?: boolean; lora: LoraSummary; number: number }[] = [];
    for (const l of loras.data?.items ?? []) {
      for (const v of l.usable_versions) {
        out.push({
          value: v.id,
          label: `${l.name} · v${v.number}${v.preferred ? " ★" : ""} · ${l.trigger_word}`,
          disabled: !loraFits(l, draft.variant),
          lora: l,
          number: v.number,
        });
      }
    }
    return out;
  }, [loras.data, draft.variant]);
  const chosenLora = loraOptions.find((o) => o.value === draft.loraVersionId);
  const loraProblem = chosenLora && chosenLora.disabled ? t("k2.loraIncompatible", { name: chosenLora.lora.name }) : null;
  const persona = draft.persona ? personas.data?.items.find((p) => p.id === draft.persona?.id) : undefined;

  const pickPersona = (value: PersonaDraft | null) => {
    const patch: Partial<ComfyDraft> = { persona: value ? { ...value, useRefs: false } : null };
    // The AI model's own LoRA (preferred version) is chosen automatically.
    if (value && !draft.loraVersionId) {
      const own = (loras.data?.items ?? []).find(
        (l) => l.persona_id === value.id && l.preferred_version_id && loraFits(l, draft.variant),
      );
      if (own?.preferred_version_id) patch.loraVersionId = own.preferred_version_id;
    }
    set(patch);
  };

  const problems: string[] = [];
  if (!conn) problems.push(t("k2.noConnection"));
  else if (!detection?.supported) problems.push(t("k2.notSupported"));
  else if (!files.length) problems.push(t("k2.variantMissing", { name: variantSpec?.name ?? "" }));
  else if (!detection.text_encoders.length || !detection.vaes.length) problems.push(detection.missing.join(" · "));
  if (loraProblem) problems.push(loraProblem);
  if (draft.seed && !/^\d+$/.test(draft.seed)) problems.push(t("gen.problemSeed"));

  const submit = async () => {
    if (submitting.current) return;
    if (!draft.prompt.trim()) {
      toast.error(t("gen.problemPrompt"));
      return;
    }
    if (problems.length || !conn) {
      toast.error(problems[0]);
      return;
    }
    submitting.current = true;
    try {
      await generate.mutateAsync({
        connection_id: conn.id,
        variant: draft.variant,
        model_file: modelFile || null,
        workflow_id: draft.workflowId || null,
        prompt: draft.prompt.trim(),
        negative_prompt: variantSpec?.negative_prompt ? draft.negative : "",
        aspect_ratio: draft.aspect,
        resolution,
        seed: draft.seed ? Number(draft.seed) : null,
        steps: draft.steps ? Number(draft.steps) : null,
        cfg: draft.cfg ? Number(draft.cfg) : null,
        count: draft.count,
        lora: draft.loraVersionId
          ? { version_id: draft.loraVersionId, strength: draft.strength, insert_trigger: draft.insertTrigger }
          : null,
        persona: persona && draft.persona ? { id: persona.id, use_description: draft.persona.useDescription } : null,
        client_request_id: newRequestId(),
      });
    } catch (e) {
      toast.error(e instanceof ApiError ? e.message : t("errors.generic"));
    } finally {
      submitting.current = false;
    }
  };

  if (conns.isLoading || catalog.isLoading) {
    return (
      <section className="studio-controls" style={{ padding: 22 }}>
        {header}
        <Skeleton style={{ height: 400, marginTop: 16 }} />
      </section>
    );
  }

  return (
    <section className="studio-controls" aria-label={t("gen.controls")}>
      <div className="studio-scroll">
        {header}
        {!ready.length ? (
          <Notice tone="accent" icon={<Plug />}>
            <strong>{t("k2.connectTitle")}</strong>
            <div style={{ margin: "4px 0 10px" }}>{t("k2.connectBody")}</div>
            <Link href="/comfyui" className="btn btn-sm btn-primary">
              ComfyUI <ArrowRight />
            </Link>
          </Notice>
        ) : (
          <>
            <div className="form-grid">
              <Field label="ComfyUI">
                <Select
                  label="ComfyUI"
                  value={conn?.id ?? null}
                  onChange={(id) => set({ connectionId: id })}
                  options={ready.map((c) => ({ value: c.id, label: c.name }))}
                />
              </Field>
              <Field label={t("k2.workflow")}>
                <Select
                  label={t("k2.workflow")}
                  value={draft.workflowId}
                  onChange={(id) => set({ workflowId: id })}
                  options={[{ value: "", label: t("wf.builtin") }, ...runnable.map((w) => ({ value: w.id, label: w.name }))]}
                />
              </Field>
            </div>
            <div>
              <div className="section-title">
                <span className="eyebrow grow">{t("k2.variant")}</span>
                {variantSpec ? <span className="tiny subtle">{variantSpec.settings_source}</span> : null}
              </div>
              <div className="models" role="radiogroup" aria-label={t("k2.variant")}>
                {(catalog.data?.krea2.variants ?? []).map((v) => {
                  const state = detection?.variants[v.key];
                  return (
                    <button
                      key={v.key}
                      type="button"
                      role="radio"
                      aria-checked={draft.variant === v.key}
                      className="model-card"
                      onClick={() => set({ variant: v.key, modelFile: "" })}
                    >
                      <div className="row" style={{ gap: 6 }}>
                        <span className="name grow">{v.name}</span>
                        {state?.installed ? <Badge tone="success">{t("k2.installed")}</Badge> : <Badge tone="warning">{t("k2.notInstalled")}</Badge>}
                      </div>
                      <div className="desc clamp-2">{v.description}</div>
                      <div className="avail tiny subtle">
                        {v.steps} steps · CFG {v.cfg} · {state?.files.map((f) => f.precision).join(", ") || "—"}
                      </div>
                    </button>
                  );
                })}
              </div>
              {files.length > 1 ? (
                <Field label={t("k2.precision")}>
                  <Select
                    label={t("k2.precision")}
                    value={modelFile}
                    onChange={(f) => set({ modelFile: f })}
                    options={files.map((f) => ({ value: f.filename, label: `${f.precision} · ${f.filename}` }))}
                  />
                </Field>
              ) : null}
            </div>

            <div>
              <div className="section-title">
                <span className="eyebrow grow">{t("pm.section")}</span>
              </div>
              <PersonaSection
                value={persona ? draft.persona : draft.persona}
                persona={persona}
                loading={personas.isLoading}
                hasPersonas={(personas.data?.total ?? 0) > 0}
                onChange={pickPersona}
                attached={[]}
                maxReferences={null}
                modelName="Krea 2"
                providerName="ComfyUI"
                manualCount={0}
              />
            </div>

            <div>
              <div className="section-title">
                <span className="eyebrow grow">LoRA</span>
                <Link href="/loras" className="tiny subtle">{t("nav.myLoras")}</Link>
              </div>
              <Select
                label="LoRA"
                value={draft.loraVersionId}
                onChange={(id) => set({ loraVersionId: id })}
                options={[{ value: "", label: t("k2.noLora") }, ...loraOptions.map(({ value, label, disabled }) => ({ value, label, disabled }))]}
              />
              {chosenLora && !chosenLora.disabled ? (
                <div className="col gap-12" style={{ marginTop: 10 }}>
                  <label className="small">
                    {t("k2.strength", { n: draft.strength.toFixed(2) })}
                    <input type="range" min={0} max={1.5} step={0.05} value={draft.strength} onChange={(e) => set({ strength: Number(e.target.value) })} />
                  </label>
                  <label className="row small" style={{ gap: 10 }}>
                    <Switch checked={draft.insertTrigger} onChange={(v) => set({ insertTrigger: v })} label={t("k2.insertTrigger")} />
                    {t("k2.insertTrigger")} <span className="mono">{chosenLora.lora.trigger_word}</span>
                  </label>
                  {chosenLora.lora.base_variant === "raw" && draft.variant === "turbo" ? (
                    <div className="hint">{t("k2.rawOnTurbo")}</div>
                  ) : null}
                </div>
              ) : null}
              {loraProblem ? (
                <div className="error-text row" style={{ gap: 6, marginTop: 6 }}>
                  <TriangleAlert size={14} /> {loraProblem}
                </div>
              ) : null}
            </div>

            <div>
              <div className="section-title">
                <span className="eyebrow grow">{t("gen.prompt")}</span>
              </div>
              <PromptEditor value={draft.prompt} onChange={(p) => set({ prompt: p })} max={4000} onSubmit={submit} />
            </div>
            {variantSpec?.negative_prompt ? (
              <Field label={t("k2.negative")}>
                <textarea className="textarea" rows={2} value={draft.negative} maxLength={2000} onChange={(e) => set({ negative: e.target.value })} />
              </Field>
            ) : (
              <div className="hint">{t("k2.noNegative")}</div>
            )}

            <div>
              <div className="section-title">
                <span className="eyebrow grow">{t("gen.settings")}</span>
                {size ? <span className="tiny subtle">{size[0]}×{size[1]}</span> : null}
              </div>
              <div className="col gap-12">
                <div className="chips">
                  {(catalog.data?.krea2.aspect_ratios ?? []).map((a) => (
                    <button key={a} type="button" className="chip" aria-pressed={draft.aspect === a} onClick={() => set({ aspect: a })}>
                      {a}
                    </button>
                  ))}
                </div>
                <div className="row wrap" style={{ gap: 10 }}>
                  <Segmented label={t("gen.resolution")} value={resolution} onChange={(r) => set({ resolution: r })} options={tiers.map((r) => ({ value: r, label: r }))} />
                  <Segmented label={t("gen.imageCount")} value={draft.count} onChange={(n) => set({ count: n })} options={[1, 2, 3, 4].map((n) => ({ value: n, label: String(n) }))} />
                </div>
                <div className="form-grid">
                  <Field label={t("gen.seed")} hint={t("gen.seedHint")}>
                    <div className="input-wrap">
                      <input className="input mono" value={draft.seed} placeholder={t("gen.seedRandom")} onChange={(e) => set({ seed: e.target.value.trim() })} />
                      <span className="input-action">
                        <IconButton label={t("gen.randomSeed")} onClick={() => set({ seed: String(Math.floor(Math.random() * 2 ** 32)) })}>
                          <Dices />
                        </IconButton>
                      </span>
                    </div>
                  </Field>
                  <Field label={t("k2.steps")} hint={t("k2.default", { n: variantSpec?.steps ?? "" })}>
                    <input className="input" type="number" min={1} max={150} value={draft.steps} placeholder={String(variantSpec?.steps ?? "")} onChange={(e) => set({ steps: e.target.value })} />
                  </Field>
                  <Field label="CFG" hint={t("k2.default", { n: variantSpec?.cfg ?? "" })}>
                    <input className="input" type="number" min={0} max={20} step={0.1} value={draft.cfg} placeholder={String(variantSpec?.cfg ?? "")} onChange={(e) => set({ cfg: e.target.value })} />
                  </Field>
                </div>
              </div>
            </div>
            {problems.length ? (
              <Notice tone="warning">
                {problems.map((p) => (
                  <div key={p}>{p}</div>
                ))}
              </Notice>
            ) : null}
          </>
        )}
      </div>
      <div className="studio-foot">
        <Button
          variant="primary"
          size="lg"
          block
          loading={generate.isPending}
          disabled={!ready.length || problems.length > 0 || generate.isPending}
          onClick={submit}
          icon={<Sparkles />}
        >
          {draft.count > 1 ? t("gen.generateN", { n: draft.count }) : t("gen.generate")}
          <span className="kbd" aria-hidden>⌘↵</span>
        </Button>
      </div>
    </section>
  );
}
