"use client";

import { FlaskConical } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

import { StatusPill } from "@/components/studio/common";
import { Button, EmptyState, Field, Notice, Select, Skeleton, Switch, useConfirm } from "@/components/ui";
import { ApiError } from "@/lib/api";
import { formatRelative } from "@/lib/format";
import { useI18n } from "@/lib/i18n";
import type { MessageKey } from "@/lib/i18n";
import { useComfyConnections, useLoraTests, useRunLoraTest } from "@/lib/queries";
import type { Generation, LoraDetail } from "@/lib/types";

/** Two images on top of each other; the slider reveals the second one. */
export function CompareSlider({ a, b, labelA, labelB }: { a: string; b: string; labelA: string; labelB: string }) {
  const [split, setSplit] = useState(50);
  return (
    <div className="compare">
      <img src={a} alt={labelA} />
      <img src={b} alt={labelB} style={{ clipPath: `inset(0 0 0 ${split}%)` }} />
      <div className="compare-line" style={{ left: `${split}%` }} />
      <span className="compare-label left">{labelA}</span>
      <span className="compare-label right">{labelB}</span>
      <input
        type="range"
        min={0}
        max={100}
        value={split}
        onChange={(e) => setSplit(Number(e.target.value))}
        aria-label={`${labelA} / ${labelB}`}
      />
    </div>
  );
}

function label(run: Generation, lora: LoraDetail, t: (k: MessageKey) => string) {
  if (run.comfy?.test_label === "base") return t("cmp.base");
  const v = lora.versions.find((x) => x.id === run.comfy?.test_label);
  return v ? `v${v.number}` : t("cmp.lora");
}

export function TestPanel({ lora }: { lora: LoraDetail }) {
  const { t, locale } = useI18n();
  const confirm = useConfirm();
  const conns = useComfyConnections();
  const tests = useLoraTests(lora.id);
  const run = useRunLoraTest(lora.id);
  const ready = (conns.data?.items ?? []).filter((c) => c.status === "ready" && c.info.krea2?.ready);
  const usable = lora.versions.filter((v) => v.status === "completed" && v.file_url);
  const [connection, setConnection] = useState("");
  const [variant, setVariant] = useState<"turbo" | "raw">(lora.base_variant === "turbo" ? "turbo" : "turbo");
  const [prompts, setPrompts] = useState(`portrait photo of ${lora.trigger_word} ${lora.class_word}, soft window light`);
  const [seeds, setSeeds] = useState("7, 42");
  const [versions, setVersions] = useState<string[]>(lora.preferred_version_id ? [lora.preferred_version_id] : usable.slice(0, 1).map((v) => v.id));
  const [includeBase, setIncludeBase] = useState(true);
  const [strength, setStrength] = useState(1);
  const conn = connection || ready[0]?.id || "";
  const promptList = prompts.split("\n").map((p) => p.trim()).filter(Boolean).slice(0, 4);
  const seedList = seeds.split(/[,\s]+/).map(Number).filter((n) => Number.isInteger(n) && n >= 0).slice(0, 4);
  const count = promptList.length * seedList.length * (versions.length + (includeBase ? 1 : 0));

  const submit = async () => {
    if (!(await confirm({ title: t("cmp.confirmTitle", { n: count }), description: t("cmp.confirmBody"), confirmLabel: t("cmp.run") })))
      return;
    try {
      await run.mutateAsync({
        connection_id: conn,
        variant,
        prompts: promptList,
        seeds: seedList,
        version_ids: versions,
        include_base: includeBase,
        strength,
        confirm_count: count,
      });
      toast.success(t("cmp.started", { n: count }));
    } catch (e) {
      toast.error(e instanceof ApiError ? e.message : t("errors.generic"));
    }
  };

  return (
    <div className="col gap-16">
      <section className="card card-pad col gap-12">
        {!ready.length ? <Notice tone="warning">{t("cmp.noComfy")}</Notice> : null}
        <div className="form-grid">
          <Field label="ComfyUI">
            <Select label="ComfyUI" value={conn || null} placeholder={t("cmp.chooseConn")} onChange={setConnection} options={ready.map((c) => ({ value: c.id, label: c.name }))} />
          </Field>
          <Field label={t("k2.variant")} hint={lora.base_variant === "turbo" ? t("cmp.turboOnly") : t("cmp.rawBoth")}>
            <Select
              label={t("k2.variant")}
              value={variant}
              onChange={setVariant}
              options={[
                { value: "turbo", label: "Krea 2 Turbo" },
                { value: "raw", label: "Krea 2 RAW", disabled: lora.base_variant === "turbo" },
              ]}
            />
          </Field>
        </div>
        <Field label={t("cmp.prompts")} hint={t("cmp.promptsHint")}>
          <textarea className="textarea" rows={3} value={prompts} onChange={(e) => setPrompts(e.target.value)} />
        </Field>
        <div className="form-grid">
          <Field label={t("cmp.seeds")} hint={t("cmp.seedsHint")}>
            <input className="input mono" value={seeds} onChange={(e) => setSeeds(e.target.value)} />
          </Field>
          <Field label={t("k2.strength", { n: strength.toFixed(2) })}>
            <input type="range" min={0} max={1.5} step={0.05} value={strength} onChange={(e) => setStrength(Number(e.target.value))} />
          </Field>
        </div>
        <Field label={t("cmp.versions")}>
          <div className="chips">
            {usable.map((v) => (
              <button
                key={v.id}
                type="button"
                className="chip"
                aria-pressed={versions.includes(v.id)}
                onClick={() => setVersions((cur) => (cur.includes(v.id) ? cur.filter((x) => x !== v.id) : [...cur, v.id].slice(-3)))}
              >
                v{v.number}
                {lora.preferred_version_id === v.id ? " ★" : ""}
              </button>
            ))}
          </div>
        </Field>
        <div className="row">
          <label className="row small grow" style={{ gap: 8 }}>
            <Switch checked={includeBase} onChange={setIncludeBase} label={t("cmp.includeBase")} />
            {t("cmp.includeBase")}
          </label>
          <Button variant="primary" icon={<FlaskConical />} disabled={!conn || !count || !versions.length} loading={run.isPending} onClick={submit}>
            {t("cmp.runN", { n: count })}
          </Button>
        </div>
      </section>

      {tests.isLoading ? (
        <Skeleton style={{ height: 200 }} />
      ) : !tests.data?.items.length ? (
        <EmptyState icon={<FlaskConical />} title={t("cmp.empty")} description={t("cmp.emptyBody")} />
      ) : (
        tests.data.items.map((group) => <TestGroup key={group.group} lora={lora} runs={group.runs} when={formatRelative(group.created_at, locale)} />)
      )}
    </div>
  );
}

function TestGroup({ lora, runs, when }: { lora: LoraDetail; runs: Generation[]; when: string }) {
  const { t } = useI18n();
  const tt = (k: MessageKey) => t(k);
  // Rows: prompt + seed; columns: base and versions.
  const rows = new Map<string, Generation[]>();
  for (const r of runs) {
    const key = `${r.comfy?.test_prompt ?? r.prompt}|${r.settings.seed}`;
    rows.set(key, [...(rows.get(key) ?? []), r]);
  }
  return (
    <section className="card">
      <div className="card-head">
        <span className="h3 grow">{t("cmp.group", { n: runs.length })}</span>
        <span className="tiny subtle">{when}</span>
      </div>
      <div className="card-body col gap-16">
        {[...rows.entries()].map(([key, cells]) => {
          const done = cells.filter((c) => c.images[0]);
          return (
            <div key={key} className="col gap-12">
              <div className="tiny subtle">
                “{cells[0].comfy?.test_prompt ?? cells[0].prompt}” · seed {cells[0].settings.seed}
              </div>
              <div className="compare-row">
                {cells.map((c) => (
                  <figure key={c.id} className="compare-cell">
                    {c.images[0] ? <img src={c.images[0].thumb_url} alt="" /> : <div className="compare-wait"><StatusPill status={c.status === "processing" ? "running" : c.status} /></div>}
                    <figcaption className="tiny">{label(c, lora, tt)}</figcaption>
                  </figure>
                ))}
              </div>
              {done.length >= 2 ? (
                <CompareSlider
                  a={done[0].images[0].url}
                  b={done[done.length - 1].images[0].url}
                  labelA={label(done[0], lora, tt)}
                  labelB={label(done[done.length - 1], lora, tt)}
                />
              ) : null}
            </div>
          );
        })}
      </div>
    </section>
  );
}
