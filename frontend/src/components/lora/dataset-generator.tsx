"use client";

import { ArrowDownWideNarrow, CheckCheck, ChevronDown, Images, Plus, ScanFace, Sparkles, Wand2 } from "lucide-react";
import Link from "next/link";
import { useQueryClient } from "@tanstack/react-query";
import { useMemo, useState } from "react";
import { toast } from "sonner";

import { usePriceConfirm } from "@/components/generate/price-confirm";
import { compareFaces, FaceBadge, type FaceResult } from "@/components/images/face-match";
import { ImageTile } from "@/components/images/image-tile";
import { Button, Field, Notice, Segmented, Select, Switch, useConfirm } from "@/components/ui";
import { api, ApiError, downloadUrl } from "@/lib/api";
import { useT } from "@/lib/i18n";
import { useAddToLoraDataset, useComfyConnections, useLoraGenerated, usePersonas, useSettings } from "@/lib/queries";
import type { LoraDetail } from "@/lib/types";

// Varied situations, so the LoRA sees the character in many backgrounds, outfits, angles and
// lights (the studio's suggestions; edit them freely).
const EXAMPLES = [
  "close-up portrait, soft window light, neutral background",
  "half body, walking in a city street at golden hour, casual jacket",
  "full body, standing on a beach at sunset, white summer dress",
  "sitting in a cozy cafe, side view, warm indoor light",
  "portrait from a low angle, overcast daylight, park in the background",
  "three-quarter view, studio photo, grey seamless background, black turtleneck",
  "laughing, candid photo at a dinner party, evening light",
  "full body, hiking trail in the mountains, sporty outfit",
  "close-up, looking over the shoulder, night city lights bokeh",
  "half body, reading a book by a window, morning light, knit sweater",
];

/** LoRA › Photos: makes new dataset photos with Krea 2 and this LoRA, one run per prompt (they
 * wait in the GPU's queue); the best ones go into the LoRA with "Add to LoRA". */
export function DatasetGenerator({ lora }: { lora: LoraDetail }) {
  const t = useT();
  const confirm = useConfirm();
  const [open, setOpen] = useState(false);
  const conns = useComfyConnections(open);
  const qc = useQueryClient();
  const ready = (conns.data?.items ?? []).filter((c) => c.status === "ready");
  const versions = lora.versions.filter((v) => v.status === "completed");
  const [connectionId, setConnectionId] = useState("");
  const [versionId, setVersionId] = useState(lora.preferred_version_id ?? "");
  const [variant, setVariant] = useState<"turbo" | "raw">("turbo");
  const [prompts, setPrompts] = useState("");
  const [count, setCount] = useState(1);
  const [adult, setAdult] = useState(false);
  const [busy, setBusy] = useState(false);

  const conn = ready.find((c) => c.id === connectionId) ?? ready[0];
  const version = versions.find((v) => v.id === versionId) ?? versions[0];
  const lines = prompts.split("\n").map((p) => p.trim()).filter(Boolean).slice(0, 30);
  const total = lines.length * count;
  // Krea 2 compatibility (as the backend enforces it): RAW LoRAs on RAW and Turbo, Turbo only on Turbo.
  const fits = lora.architecture === "krea2" && (lora.base_variant === variant || (lora.base_variant === "raw" && variant === "turbo"));
  const start = async () => {
    if (!conn || !version || !lines.length) return;
    const ok = await confirm({
      title: t("dg.confirmTitle"),
      description: t("dg.confirmBody", { n: total, runs: lines.length, gpu: conn.name }),
      confirmLabel: t("dg.start"),
    });
    if (!ok) return;
    setBusy(true);
    try {
      await api(`/api/loras/${lora.id}/dataset/generate`, {
        body: {
          connection_id: conn.id,
          version_id: version.id,
          variant,
          prompts: lines,
          count,
          adult,
          confirm_count: total,
        },
      });
      toast.success(t("dg.started", { n: total }));
      void qc.invalidateQueries({ queryKey: ["loras", "generated", lora.id] });
    } catch (e) {
      toast.error(e instanceof ApiError ? e.message : t("errors.generic"));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card">
      <button type="button" className="card-head adv-toggle" style={{ textAlign: "left" }} onClick={() => setOpen((v) => !v)} aria-expanded={open}>
        <Wand2 size={16} color="var(--text-3)" />
        <div className="grow">
          <div className="h3">{t("dg.title")}</div>
          <div className="tiny subtle">{t("dg.subtitle")}</div>
        </div>
        <ChevronDown size={16} style={{ transform: open ? "rotate(180deg)" : undefined }} />
      </button>
      {open ? (
        <div className="card-body col" style={{ gap: 14 }}>
          {!versions.length ? (
            <Notice tone="accent">{t("dg.needVersion")}</Notice>
          ) : !ready.length ? (
            <Notice tone="accent">
              {t("dg.needGpu")}{" "}
              <Link href="/generate?engine=comfy" className="btn btn-sm" style={{ marginTop: 8 }}>
                {t("dg.openKrea")}
              </Link>
            </Notice>
          ) : (
            <>
              <div className="grid-2" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 12 }}>
                <Field label={t("dg.gpu")}>
                  <Select label={t("dg.gpu")} value={conn?.id ?? null} onChange={(v) => setConnectionId(String(v))} options={ready.map((c) => ({ value: c.id, label: c.name }))} />
                </Field>
                <Field label={t("dg.version")}>
                  <Select
                    label={t("dg.version")}
                    value={version?.id ?? null}
                    onChange={(v) => setVersionId(String(v))}
                    options={versions.map((v) => ({ value: v.id, label: `v${v.number}` }))}
                  />
                </Field>
              </div>
              <Field label={t("dg.model")}>
                <Segmented
                  label={t("dg.model")}
                  value={variant}
                  onChange={setVariant}
                  options={[
                    { value: "turbo", label: "Krea 2 Turbo" },
                    { value: "raw", label: "Krea 2 RAW", disabled: lora.base_variant === "turbo" },
                  ]}
                />
              </Field>
              {!fits ? <div className="error-text small">{t("dg.incompatible")}</div> : null}
              <Field label={t("dg.prompts")} hint={t("dg.promptsHint", { trigger: `${lora.trigger_word} ${lora.class_word}`.trim() })}>
                <textarea
                  className="textarea"
                  rows={6}
                  value={prompts}
                  onChange={(e) => setPrompts(e.target.value)}
                  placeholder={EXAMPLES.slice(0, 3).join("\n")}
                />
              </Field>
              <div className="row wrap" style={{ gap: 8 }}>
                <Button size="sm" icon={<Plus />} onClick={() => setPrompts((p) => [p.trim(), ...EXAMPLES].filter(Boolean).join("\n"))}>
                  {t("dg.examples")}
                </Button>
                <span className="tiny subtle">{t("dg.lines", { n: lines.length })}</span>
              </div>
              <Field label={t("dg.count")}>
                <Segmented label={t("dg.count")} value={count} onChange={setCount} options={[1, 2, 3, 4].map((n) => ({ value: n, label: String(n) }))} />
              </Field>
              <div className="row" style={{ gap: 10, alignItems: "flex-start" }}>
                <Switch checked={adult} onChange={setAdult} label={t("k2.adult")} />
                <div>
                  <div className="small">{t("k2.adult")}</div>
                  <div className="hint">{t(adult ? "k2.adultOnHint" : "k2.adultOffHint")}</div>
                </div>
              </div>
              <Button variant="primary" icon={<Sparkles />} loading={busy} disabled={!lines.length || !fits} onClick={start}>
                {t("dg.generate", { n: total })}
              </Button>
            </>
          )}

        </div>
      ) : null}
    </div>
  );
}

const DATASET_MODELS = [
  { value: "nano-banana-pro", label: "Nano Banana Pro" },
  { value: "seedream", label: "Seedream" },
  { value: "wan-2.7-pro", label: "Wan 2.7 Pro" },
] as const;
type DatasetModel = (typeof DATASET_MODELS)[number]["value"];

/** LoRA › Photos: one click makes a set of photos of the AI model (its photos as references,
 * a hosted model, one shot each: close-ups, half and full body); the price of the whole set is
 * confirmed first. The best ones go into the LoRA from "Generated photos". */
export function HostedDatasetCard({ lora }: { lora: LoraDetail }) {
  const t = useT();
  const qc = useQueryClient();
  const priceConfirm = usePriceConfirm();
  const personas = usePersonas();
  const [open, setOpen] = useState(lora.dataset.length < 10);
  const [personaId, setPersonaId] = useState(lora.persona_id ?? "");
  const [model, setModel] = useState<DatasetModel>("nano-banana-pro");
  const [count, setCount] = useState(24);
  const [adult, setAdult] = useState(false);
  const [busy, setBusy] = useState(false);
  const list = personas.data?.items ?? [];
  const persona = list.find((p) => p.id === personaId) ?? (lora.persona_id ? undefined : list[0]);
  const mature = model === "wan-2.7-pro";

  const start = async () => {
    if (!persona) return;
    const body = { persona_id: persona.id, model, count, adult: mature && adult };
    setBusy(true);
    try {
      const confirmed = await priceConfirm.forDataset(lora.id, body);
      if (!confirmed) return;
      await api(`/api/loras/${lora.id}/dataset/hosted`, {
        body: { ...body, confirmed: true, price_each: confirmed.price !== null ? confirmed.price / count : null },
      });
      toast.success(t("hd.started", { n: count }));
      void qc.invalidateQueries({ queryKey: ["loras", "generated", lora.id] });
    } catch (e) {
      toast.error(e instanceof ApiError ? e.message : t("errors.generic"));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card">
      <button type="button" className="card-head adv-toggle" style={{ textAlign: "left" }} onClick={() => setOpen((v) => !v)} aria-expanded={open}>
        <Images size={16} color="var(--text-3)" />
        <div className="grow">
          <div className="h3">{t("hd.title")}</div>
          <div className="tiny subtle">{t("hd.subtitle")}</div>
        </div>
        <ChevronDown size={16} style={{ transform: open ? "rotate(180deg)" : undefined }} />
      </button>
      {open ? (
        <div className="card-body col" style={{ gap: 14 }}>
          {!list.length && !personas.isLoading ? (
            <Notice tone="accent">
              {t("hd.needPersona")}{" "}
              <Link href="/ai-models" className="btn btn-sm" style={{ marginTop: 8 }}>
                {t("nav.aiModels")}
              </Link>
            </Notice>
          ) : (
            <>
              <div className="grid-2" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 12 }}>
                <Field label={t("pm.section")} hint={t("hd.personaHint")}>
                  <Select
                    label={t("pm.section")}
                    value={persona?.id ?? null}
                    onChange={(v) => setPersonaId(String(v))}
                    options={list.map((p) => ({ value: p.id, label: p.name }))}
                  />
                </Field>
                <Field label={t("hd.count")}>
                  <Segmented label={t("hd.count")} value={count} onChange={setCount} options={[12, 24, 36].map((n) => ({ value: n, label: String(n) }))} />
                </Field>
              </div>
              <Field label={t("dg.model")} hint={t(mature ? "hd.modelHintMature" : "hd.modelHint")}>
                <Segmented label={t("dg.model")} value={model} onChange={setModel} options={DATASET_MODELS.map((m) => ({ value: m.value, label: m.label }))} />
              </Field>
              {mature ? (
                <div className="row" style={{ gap: 10, alignItems: "flex-start" }}>
                  <Switch checked={adult} onChange={setAdult} label={t("k2.adult")} />
                  <div>
                    <div className="small">{t("k2.adult")}</div>
                    <div className="hint">{t(adult ? "hd.adultOnHint" : "k2.adultOffHint")}</div>
                  </div>
                </div>
              ) : null}
              <div className="hint">{t("hd.shotsHint")}</div>
              <Button variant="primary" icon={<Sparkles />} loading={busy} disabled={!persona} onClick={start}>
                {t("hd.generate", { n: count })}
              </Button>
            </>
          )}
        </div>
      ) : null}
    </div>
  );
}

/** The photos made for this LoRA (online models or Krea 2): pick the best — with the face
 * similarity to the AI model's (or the LoRA's) photos when measured — and add them. */
export function GeneratedPhotos({ lora }: { lora: LoraDetail }) {
  const t = useT();
  const settings = useSettings();
  const generated = useLoraGenerated(lora.id, true);
  const add = useAddToLoraDataset();
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [faces, setFaces] = useState<Map<string, FaceResult> | null>(null);
  const [threshold, setThreshold] = useState(0.363);
  const [measuring, setMeasuring] = useState(false);
  const [sorted, setSorted] = useState(false);
  const images = useMemo(() => (generated.data?.items ?? []).flatMap((g) => g.images).filter((i) => !i.mime_type?.startsWith("video/")), [generated.data]);
  const running = (generated.data?.items ?? []).filter((g) => g.status === "queued" || g.status === "processing").length;
  const target = lora.persona_id ? { persona_id: lora.persona_id } : lora.dataset.length ? { lora_id: lora.id } : null;
  const shown = sorted && faces ? [...images].sort((a, b) => (faces.get(b.id)?.score ?? -1) - (faces.get(a.id)?.score ?? -1)) : images;
  if (!images.length && !running) return null;

  const measure = async () => {
    if (!target) return;
    setMeasuring(true);
    try {
      const out = new Map<string, FaceResult>();
      for (let i = 0; i < images.length; i += 60) {
        const res = await compareFaces(images.slice(i, i + 60).map((img) => img.id), target);
        res.items.forEach((r) => out.set(r.image_id, r));
        setThreshold(res.threshold);
      }
      setFaces(out);
      setSorted(true);
    } catch (e) {
      toast.error(e instanceof ApiError ? e.message : t("errors.generic"));
    } finally {
      setMeasuring(false);
    }
  };

  const addSelected = async () => {
    try {
      const res = await add.mutateAsync({ loraId: lora.id, imageIds: [...selected] });
      toast.success(t("lo.addedTo", { n: res.added, name: lora.name }));
      setSelected(new Set());
    } catch (e) {
      toast.error(e instanceof ApiError ? e.message : t("errors.generic"));
    }
  };

  return (
    <div className="card card-pad col" style={{ gap: 10 }}>
      <div className="row wrap" style={{ gap: 8 }}>
        <div className="eyebrow grow">
          {t("dg.results", { n: images.length })}
          {running ? ` · ${t("dg.running", { n: running })}` : ""}
        </div>
        {target && images.length ? (
          <Button size="sm" icon={<ScanFace />} loading={measuring} onClick={measure}>
            {t("hd.measure")}
          </Button>
        ) : null}
        {faces ? (
          <>
            <Button size="sm" variant="ghost" icon={<ArrowDownWideNarrow />} onClick={() => setSorted((v) => !v)}>
              {sorted ? t("hd.unsort") : t("hd.sort")}
            </Button>
            <Button
              size="sm"
              variant="ghost"
              icon={<CheckCheck />}
              onClick={() => setSelected(new Set(images.filter((i) => (faces.get(i.id)?.score ?? -1) >= threshold).map((i) => i.id)))}
            >
              {t("hd.selectMatches", { pct: Math.round(threshold * 100) })}
            </Button>
          </>
        ) : null}
        <Button size="sm" variant="primary" icon={<Wand2 />} disabled={!selected.size} loading={add.isPending} onClick={addSelected}>
          {t("dg.addSelected", { n: selected.size })}
        </Button>
      </div>
      <div className="grid" style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(120px, 1fr))", gap: 8 }}>
        {shown.map((img) => (
          <div key={img.id} className="col" style={{ gap: 4 }}>
            <ImageTile
              image={img}
              style={{ aspectRatio: `${img.width} / ${img.height}` }}
              selecting
              selected={selected.has(img.id)}
              onSelect={() =>
                setSelected((s) => {
                  const next = new Set(s);
                  if (next.has(img.id)) next.delete(img.id);
                  else next.add(img.id);
                  return next;
                })
              }
              onOpen={() => undefined}
              onDownload={() => downloadUrl(img.download_url)}
              showPrompt={false}
              blurNsfw={settings.data?.blur_nsfw ?? true}
            />
            {faces ? <FaceBadge result={faces.get(img.id)} /> : null}
          </div>
        ))}
      </div>
    </div>
  );
}
