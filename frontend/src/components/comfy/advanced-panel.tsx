"use client";

import { ChevronDown, Dices, Download, Sparkles, TriangleAlert, Wand2 } from "lucide-react";
import Link from "next/link";
import { useMemo, useRef, useState } from "react";
import { toast } from "sonner";

import { ComfyConnectionBar } from "@/components/comfy/connection-bar";
import { loraFits } from "@/components/comfy/generate-panel";
import { type PickedPhoto, PhotoPicker } from "@/components/comfy/photo-picker";
import { Progress, newRequestId } from "@/components/studio/common";
import { Badge, Button, Field, IconButton, Notice, Segmented, Select, Skeleton, Switch } from "@/components/ui";
import { ApiError } from "@/lib/api";
import { formatBytes } from "@/lib/format";
import { useI18n } from "@/lib/i18n";
import {
  useAdvancedEdit,
  useAdvancedPrompt,
  useComfyCatalog,
  useComfyConnections,
  useInstallKrea2Model,
  useLoras,
} from "@/lib/queries";
import type { Generation } from "@/lib/types";

type Mode = "edit" | "text";
type Target = "photo" | "face" | "head" | "hair" | "clothes" | "body";
const TARGETS: Target[] = ["photo", "face", "head", "hair", "clothes", "body"];
const BODY_PARTS = ["figure", "bust", "butt", "waist", "belly", "legs", "arms"] as const;
type BodyPart = (typeof BODY_PARTS)[number];

/** Slovak or Czech words: the edit model mainly reads English (a Slovak "smile" changed
 * nothing in the studio's tests), so the panel suggests "Improve instruction". */
const NOT_ENGLISH = /[áäčďéíĺľňóôŕšťúýžěřů]|\b(nech|zmeň|zmen|daj|aby|jej|ju|sa|na|je|alebo|ktor)\b/i;

/** Generate › Advanced: Krea 2 Turbo with the owner's LoRA — edit a photo from plain words
 * (an in-context edit LoRA keeps everything not asked for), or a new photo from text. */
export function AdvancedPanel({
  header,
  initialPhoto,
  initial,
}: {
  header: React.ReactNode;
  initialPhoto: PickedPhoto | null;
  /** "Load these settings" from an earlier Advanced run. */
  initial: Generation | null;
}) {
  const { t, locale } = useI18n();
  const catalog = useComfyCatalog();
  const conns = useComfyConnections();
  const install = useInstallKrea2Model();
  const edit = useAdvancedEdit();
  const improve = useAdvancedPrompt();
  const [suggestion, setSuggestion] = useState<{
    text: string;
    model: string;
    tokens: number | null;
    target: Target | null;
    part: BodyPart | null;
  } | null>(null);
  const loras = useLoras("", true);
  const submitting = useRef(false);
  const before = initial?.settings ?? {};
  const [mode, setMode] = useState<Mode>(initial && !initial.comfy?.input ? "text" : "edit");
  const [connectionId, setConnectionId] = useState(initial?.comfy?.connection_id ?? "");
  const [photo, setPhoto] = useState<PickedPhoto | null>(() => {
    if (initialPhoto) return initialPhoto;
    const input = initial?.comfy?.input;
    return input ? { kind: input.kind, id: input.id, thumb_url: input.thumb_url } : null;
  });
  const [prompt, setPrompt] = useState(initial?.prompt ?? "");
  const [loraVersionId, setLoraVersionId] = useState(initial?.comfy?.lora?.version_id ?? "");
  const [strength, setStrength] = useState(initial?.comfy?.lora?.strength ?? 1);
  const [keep, setKeep] = useState(before.keep_unchanged !== false);
  const [target, setTarget] = useState<Target>(
    before.face_only ? "face" : before.body_part ? "body" : TARGETS.includes(before.area as Target) ? (before.area as Target) : "photo",
  );
  const [part, setPart] = useState<BodyPart>((before.body_part as BodyPart) ?? "bust");
  const [change, setChange] = useState<"normal" | "strong">(before.change === "strong" ? "strong" : "normal");
  const [faceRefine, setFaceRefine] = useState(initial ? before.face_refine === true : true);
  const [allFaces, setAllFaces] = useState(true);
  const [adult, setAdult] = useState(before.adult === true);
  const [aspect, setAspect] = useState(before.aspect_ratio && before.aspect_ratio !== "match" ? before.aspect_ratio : "3:4");
  const [count, setCount] = useState(1);
  const [resolution, setResolution] = useState<"auto" | "1K" | "2K">("auto");
  const [seed, setSeed] = useState(typeof before.seed === "number" ? String(before.seed) : "");
  const [more, setMore] = useState(Boolean(seed));

  const all = conns.data?.items ?? [];
  const conn = all.find((c) => c.id === connectionId) ?? all.find((c) => c.status === "ready") ?? all[0];
  const connReady = conn?.status === "ready";
  const spec = catalog.data?.advanced;
  const detection = conn?.info.k2edit;
  const modelsReady = connReady && Boolean(detection?.ready);
  const job = conn?.deploy?.model_job ?? null;
  const downloading = job?.status === "running" && job.variant === "k2edit";

  const loraOptions = useMemo(() => {
    const out: { value: string; label: string; disabled?: boolean; trigger: string }[] = [];
    for (const l of loras.data?.items ?? []) {
      for (const v of l.usable_versions) {
        out.push({
          value: v.id,
          label: `${l.name} · v${v.number}${v.preferred ? " ★" : ""} · ${l.trigger_word}`,
          disabled: !loraFits(l, "turbo"),
          trigger: l.trigger_word,
        });
      }
    }
    return out;
  }, [loras.data]);
  const chosen = loraOptions.find((o) => o.value === loraVersionId);

  const problems: string[] = [];
  if (!conn) problems.push(t("k2.noConnection"));
  else if (!connReady) problems.push(conn.status === "deploying" ? t("k2.starting") : t("k2.notReady"));
  else if (detection?.missing_nodes.length) problems.push(t("ad.missingNodes", { nodes: detection.missing_nodes.join(", ") }));
  else if (!modelsReady) problems.push(t("en.installFirst"));
  else if (mode === "edit" && !photo) problems.push(t("ad.photoFirst"));
  if (chosen?.disabled) problems.push(t("ad.loraTurbo"));
  if (seed && !/^\d+$/.test(seed)) problems.push(t("gen.problemSeed"));

  const improvePrompt = async () => {
    if (!prompt.trim()) {
      toast.error(mode === "edit" ? t("ad.promptFirst") : t("gen.problemPrompt"));
      return;
    }
    try {
      const r = await improve.mutateAsync({ text: prompt.trim(), mode });
      const tokens = r.input_tokens !== null && r.output_tokens !== null ? r.input_tokens + r.output_tokens : null;
      setSuggestion({
        text: r.prompt,
        model: r.model,
        tokens,
        target: r.target,
        part: r.body_part && (BODY_PARTS as readonly string[]).includes(r.body_part) ? (r.body_part as BodyPart) : null,
      });
    } catch (e) {
      toast.error(e instanceof ApiError ? e.message : t("errors.generic"));
    }
  };

  const installModels = async () => {
    if (!conn) return;
    try {
      await install.mutateAsync({ id: conn.id, variant: "k2edit" });
      toast.success(t("k2.downloadStarted"));
    } catch (e) {
      toast.error(e instanceof ApiError ? e.message : t("errors.generic"));
    }
  };

  const submit = async () => {
    if (submitting.current) return;
    if (!prompt.trim()) {
      toast.error(mode === "edit" ? t("ad.promptFirst") : t("gen.problemPrompt"));
      return;
    }
    if (problems.length || !conn) {
      toast.error(problems[0]);
      return;
    }
    submitting.current = true;
    const lora = loraVersionId ? { version_id: loraVersionId, strength } : null;
    try {
      await edit.mutateAsync({
        connection_id: conn.id,
        image: mode === "edit" && photo ? { kind: photo.kind, id: photo.id } : null,
        prompt: prompt.trim(),
        aspect_ratio: aspect,
        count: mode === "edit" ? 1 : count,
        resolution,
        face_refine: Boolean(lora) && faceRefine,
        all_faces: allFaces,
        face_only: false,
        area: mode === "edit" ? target : "photo",
        body_part: mode === "edit" && target === "body" ? part : null,
        strength: change,
        lora,
        keep_unchanged: keep,
        adult,
        seed: seed ? Number(seed) : null,
        client_request_id: newRequestId(),
      });
      toast.success(t("ad.started"));
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

  const busy = edit.isPending;
  return (
    <section className="studio-controls" aria-label={t("gen.controls")}>
      <div className="studio-scroll">
        {header}
        <div className="tiny subtle">{t("ad.lede")}</div>
        <Segmented
          label={t("ad.mode")}
          value={mode}
          block
          onChange={setMode}
          options={[
            { value: "edit", label: t("ad.modeEdit") },
            { value: "text", label: t("ad.modeText") },
          ]}
        />
        <div>
          <div className="section-title">
            <span className="eyebrow grow">{t("en.stepGpu")}</span>
          </div>
          <ComfyConnectionBar selected={conn} onSelect={setConnectionId} />
        </div>

        {!connReady ? null : (
          <>
            <div>
              <div className="section-title">
                <span className="eyebrow grow">{t("en.stepModels")}</span>
                {modelsReady ? <Badge tone="success">{t("k2.installed")}</Badge> : null}
              </div>
              {modelsReady ? (
                <div className="tiny subtle">{t("ad.modelsReady")}</div>
              ) : detection?.missing_nodes.length ? (
                <Notice tone="warning">{t("ad.missingNodes", { nodes: detection.missing_nodes.join(", ") })}</Notice>
              ) : downloading ? (
                <div className="model-download">
                  <div className="row tiny" style={{ gap: 8 }}>
                    <span className="grow truncate">
                      {job.current ? t("k2.downloadingFile", { name: job.current }) : t("k2.downloadingHint")}
                    </span>
                    <span>{job.progress}%</span>
                  </div>
                  <Progress value={job.progress} />
                </div>
              ) : conn?.auth_type === "agent" ? (
                <div className="model-download">
                  {job?.status === "failed" && job.variant === "k2edit" ? <div className="error-text small">{job.error}</div> : null}
                  <div className="row wrap" style={{ gap: 10 }}>
                    <Button variant="primary" icon={<Download />} loading={install.isPending} disabled={job?.status === "running"} onClick={installModels}>
                      {t("en.install", { size: formatBytes(spec?.total_bytes ?? 0, locale) })}
                    </Button>
                    <span className="tiny subtle">{t("ad.installHint")}</span>
                  </div>
                </div>
              ) : (
                <div className="col" style={{ gap: 6 }}>
                  <div className="hint">{t("en.localFiles")}</div>
                  <ul className="tiny mono" style={{ margin: 0, paddingLeft: 18 }}>
                    {(spec?.files ?? []).map((f) => (
                      <li key={f.filename}>
                        models/{f.folder}/{f.filename}
                      </li>
                    ))}
                  </ul>
                  <div className="hint">{t("ad.localNodes")}</div>
                </div>
              )}
              {spec ? <div className="hint" style={{ marginTop: 6 }}>{t("ad.license", { license: spec.edit_license })}</div> : null}
            </div>

            {!modelsReady ? null : (
              <>
                {mode === "edit" ? <PhotoPicker title={t("ad.stepPhoto")} value={photo} onChange={setPhoto} /> : null}

                <div>
                  <div className="section-title">
                    <span className="eyebrow grow">{mode === "edit" ? t("ad.stepChange") : t("ad.stepPrompt")}</span>
                  </div>
                  <Field label={mode === "edit" ? t("ad.change") : t("gen.prompt")} hint={mode === "edit" ? t("ad.changeHint") : t("ad.promptHint")} htmlFor="advanced-prompt">
                    <textarea
                      id="advanced-prompt"
                      className="textarea"
                      rows={4}
                      value={prompt}
                      maxLength={spec?.max_prompt ?? 2000}
                      onChange={(e) => setPrompt(e.target.value)}
                    />
                  </Field>
                  {NOT_ENGLISH.test(prompt) && !suggestion ? <div className="hint" style={{ marginTop: 6 }}>{t("ad.notEnglish")}</div> : null}
                  <div className="row wrap" style={{ gap: 10, marginTop: 8 }}>
                    <Button size="sm" icon={<Wand2 />} loading={improve.isPending} onClick={improvePrompt}>
                      {t("ad.improve")}
                    </Button>
                    <span className="tiny subtle">{t("ad.improveHint")}</span>
                  </div>
                  {suggestion ? (
                    <div className="col" style={{ gap: 8, marginTop: 10 }}>
                      <div className="hint">{t("ad.suggestion")}</div>
                      <div className="small" style={{ whiteSpace: "pre-wrap" }}>{suggestion.text}</div>
                      {mode === "edit" && suggestion.target ? (
                        <div className="tiny subtle">
                          {t("ad.suggestionTarget", {
                            target: t(`ad.target.${suggestion.target}`),
                            part: suggestion.part ? ` · ${t(`ad.part.${suggestion.part}`)}` : "",
                          })}
                        </div>
                      ) : null}
                      <div className="tiny subtle">
                        {suggestion.tokens !== null ? t("ad.suggestionTokens", { model: suggestion.model, n: suggestion.tokens }) : suggestion.model}
                      </div>
                      <div className="row" style={{ gap: 8 }}>
                        <Button
                          size="sm"
                          variant="primary"
                          onClick={() => {
                            setPrompt(suggestion.text);
                            if (mode === "edit" && suggestion.target) setTarget(suggestion.target);
                            if (suggestion.part) setPart(suggestion.part);
                            setSuggestion(null);
                          }}
                        >
                          {t("ad.useSuggestion")}
                        </Button>
                        <Button size="sm" variant="ghost" onClick={() => setSuggestion(null)}>
                          {t("ad.dropSuggestion")}
                        </Button>
                      </div>
                    </div>
                  ) : null}
                </div>

                <div>
                  <div className="section-title">
                    <span className="eyebrow grow">{t("ad.stepLora")}</span>
                    <Link href="/loras" className="tiny subtle">{t("lo.tabLoras")}</Link>
                  </div>
                  <Select
                    label="LoRA"
                    value={loraVersionId}
                    onChange={setLoraVersionId}
                    options={[{ value: "", label: t("k2.noLora") }, ...loraOptions.map(({ value, label, disabled }) => ({ value, label, disabled }))]}
                  />
                  {chosen && !chosen.disabled ? (
                    <div className="col gap-12" style={{ marginTop: 10 }}>
                      <label className="small">
                        {t("k2.strength", { n: strength.toFixed(2) })}
                        <input type="range" min={0} max={1.5} step={0.05} value={strength} onChange={(e) => setStrength(Number(e.target.value))} />
                      </label>
                      <div className="hint">{t(mode === "edit" ? "ad.loraEditHint" : "ad.loraTextHint", { trigger: chosen.trigger })}</div>
                      <div className="row" style={{ gap: 10, alignItems: "flex-start" }}>
                        <Switch checked={faceRefine} onChange={setFaceRefine} label={t("ad.faceRefine")} />
                        <div>
                          <div className="small">{t("ad.faceRefine")}</div>
                          <div className="hint">{t("ad.faceRefineHint")}</div>
                        </div>
                      </div>
                      {mode === "edit" ? (
                        <div className="row" style={{ gap: 10, alignItems: "flex-start" }}>
                          <Switch checked={allFaces} onChange={setAllFaces} label={t("ad.allFaces")} />
                          <div>
                            <div className="small">{t("ad.allFaces")}</div>
                            <div className="hint">{t("ad.allFacesHint")}</div>
                          </div>
                        </div>
                      ) : null}
                    </div>
                  ) : null}
                  {chosen?.disabled ? (
                    <div className="error-text row" style={{ gap: 6, marginTop: 6 }}>
                      <TriangleAlert size={14} /> {t("ad.loraTurbo")}
                    </div>
                  ) : null}
                </div>

                <div className="col gap-16">
                  {mode === "edit" ? (
                    <>
                      <Field label={t("ad.targetLabel")} hint={t(`ad.target.${target}Hint`)}>
                        <Segmented
                          label={t("ad.targetLabel")}
                          value={target}
                          block
                          onChange={setTarget}
                          options={TARGETS.map((v) => ({ value: v, label: t(`ad.target.${v}`) }))}
                        />
                      </Field>
                      {target === "body" ? (
                        <div className="chips">
                          {BODY_PARTS.map((p) => (
                            <button key={p} type="button" className="chip" aria-pressed={part === p} onClick={() => setPart(p)}>
                              {t(`ad.part.${p}`)}
                            </button>
                          ))}
                        </div>
                      ) : null}
                      {target === "hair" || target === "clothes" ? null : (
                        <Field
                          label={t("ad.changeStrength")}
                          hint={t(
                            change === "strong"
                              ? target === "face"
                                ? "ad.changeStrongFaceHint"
                                : target === "body"
                                  ? "ad.bodyStrongHint"
                                  : "ad.changeStrongHint"
                              : target === "body"
                                ? "ad.bodyNormalHint"
                                : "ad.changeNormalHint",
                          )}
                        >
                          <Segmented
                            label={t("ad.changeStrength")}
                            value={change}
                            block
                            onChange={setChange}
                            options={[
                              { value: "normal", label: t("ad.changeNormal") },
                              { value: "strong", label: t("ad.changeStrong") },
                            ]}
                          />
                        </Field>
                      )}
                      {target === "face" ? null : (
                        <div className="row" style={{ gap: 10, alignItems: "flex-start" }}>
                          <Switch checked={keep} onChange={setKeep} label={t("ad.keep")} />
                          <div>
                            <div className="small">{t("ad.keep")}</div>
                            <div className="hint">{t("ad.keepHint")}</div>
                          </div>
                        </div>
                      )}
                    </>
                  ) : (
                    <div className="col gap-12">
                      <div className="chips">
                        {(catalog.data?.krea2.aspect_ratios ?? []).map((a) => (
                          <button key={a} type="button" className="chip" aria-pressed={aspect === a} onClick={() => setAspect(a)}>
                            {a}
                          </button>
                        ))}
                      </div>
                      <Segmented label={t("gen.imageCount")} value={count} onChange={setCount} options={[1, 2, 3, 4].map((n) => ({ value: n, label: String(n) }))} />

                    </div>
                  )}
                  <div className="row" style={{ gap: 10, alignItems: "flex-start" }}>
                    <Switch checked={adult} onChange={setAdult} label={t("k2.adult")} />
                    <div>
                      <div className="small">{t("k2.adult")}</div>
                      <div className="hint">{t(adult ? "k2.adultOnHint" : "k2.adultOffHint")}</div>
                    </div>
                  </div>
                </div>

                <div className="col gap-12">
                  <button type="button" className="link-button tiny" onClick={() => setMore((v) => !v)} aria-expanded={more}>
                    {t("gen.moreSettings")} <ChevronDown size={13} style={{ transform: more ? "rotate(180deg)" : undefined }} />
                  </button>
                  {more ? (
                    <Field label={t("gen.resolution")} hint={t("ad.resolutionHint")}>
                      <Segmented
                        label={t("gen.resolution")}
                        value={resolution}
                        onChange={setResolution}
                        options={[
                          { value: "auto", label: t("ad.resolutionAuto") },
                          { value: "2K", label: "2K" },
                          { value: "1K", label: "1K" },
                        ]}
                      />
                    </Field>
                  ) : null}
                  {more ? (
                    <Field label={t("gen.seed")} hint={t("en.seedHint")}>
                      <div className="input-wrap">
                        <input className="input mono" inputMode="numeric" value={seed} placeholder={t("gen.seedRandom")} onChange={(e) => setSeed(e.target.value.trim())} />
                        <span className="input-action">
                          <IconButton label={t("gen.randomSeed")} onClick={() => setSeed(String(Math.floor(Math.random() * 2 ** 32)))}>
                            <Dices />
                          </IconButton>
                        </span>
                      </div>
                    </Field>
                  ) : null}
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
          </>
        )}
      </div>
      <div className="studio-foot">
        <Button variant="primary" size="lg" block icon={<Sparkles />} loading={busy} disabled={problems.length > 0 || busy} onClick={submit}>
          {mode === "edit" ? t("ad.runEdit") : count > 1 ? t("gen.generateN", { n: count }) : t("gen.generate")}
        </Button>
      </div>
    </section>
  );
}
