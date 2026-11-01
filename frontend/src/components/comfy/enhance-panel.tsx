"use client";

import { ChevronDown, Dices, Download, ImageIcon, Images, Sparkles, X } from "lucide-react";
import { useRef, useState } from "react";
import { toast } from "sonner";

import { ComfyConnectionBar } from "@/components/comfy/connection-bar";
import { ReferenceImages } from "@/components/generate/controls";
import { Progress, newRequestId } from "@/components/studio/common";
import { Badge, Button, EmptyState, Field, IconButton, Modal, Notice, Segmented, Skeleton, Spinner, Switch } from "@/components/ui";
import { ApiError } from "@/lib/api";
import { formatBytes } from "@/lib/format";
import { useI18n } from "@/lib/i18n";
import { useComfyCatalog, useComfyConnections, useEnhancePhoto, useImages, useInstallKrea2Model } from "@/lib/queries";
import type { EnhanceRequest } from "@/lib/studio-bridge";
import type { Generation, ReferenceImage } from "@/lib/types";

/** The photo to enhance: an upload (or an AI model's photo) or a generated image. */
export type EnhancePhoto = EnhanceRequest;

/** The template's note: 0.15–0.2 subtle, 0.3 default, 0.35–0.4 strong. */
function strengthHint(s: number) {
  return s <= 0.2 ? "en.strength.subtleHint" : s >= 0.35 ? "en.strength.strongHint" : "en.strength.defaultHint";
}

/** Generate › Enhance photo: PGUpscaler (fixes plastic AI skin, then upscales and sharpens)
 * on the same ComfyUI as Krea 2 — this Mac or a rented GPU. */
export function EnhancePanel({
  header,
  initialPhoto,
  initial,
}: {
  header: React.ReactNode;
  initialPhoto: EnhancePhoto | null;
  /** "Load these settings" from an earlier Enhance photo run. */
  initial: Generation | null;
}) {
  const { t, locale } = useI18n();
  const catalog = useComfyCatalog();
  const conns = useComfyConnections();
  const install = useInstallKrea2Model();
  const enhance = useEnhancePhoto();
  const submitting = useRef(false);
  const before = initial?.settings ?? {};
  const [connectionId, setConnectionId] = useState(initial?.comfy?.connection_id ?? "");
  const [photo, setPhoto] = useState<EnhancePhoto | null>(() => {
    if (initialPhoto) return initialPhoto;
    const input = initial?.comfy?.input;
    return input ? { kind: input.kind, id: input.id, thumb_url: input.thumb_url } : null;
  });
  const [strength, setStrength] = useState(typeof before.strength === "number" ? before.strength : 0.3);
  const [faceOnly, setFaceOnly] = useState(before.face_only === true);
  const [upscale, setUpscale] = useState(typeof before.upscale === "number" ? before.upscale : 2);
  const [seed, setSeed] = useState(typeof before.seed === "number" ? String(before.seed) : "");
  const [skinExtra, setSkinExtra] = useState(before.skin_extra ?? "");
  const [faceTarget, setFaceTarget] = useState(before.face_target ?? "");
  const [more, setMore] = useState(Boolean(seed || skinExtra || faceTarget));
  const [picking, setPicking] = useState(false);

  const all = conns.data?.items ?? [];
  const conn = all.find((c) => c.id === connectionId) ?? all.find((c) => c.status === "ready") ?? all[0];
  const connReady = conn?.status === "ready";
  const spec = catalog.data?.enhance;
  const detection = conn?.info.enhance;
  const modelsReady = connReady && Boolean(detection?.ready);
  const job = conn?.deploy?.model_job ?? null;
  const downloading = job?.status === "running" && job.variant === "pgupscaler";
  const vramGb = (conn?.info.devices?.[0]?.vram_total ?? 0) / 1024 ** 3;
  const smallFor3x = upscale === 3 && vramGb > 0 && spec && vramGb < spec.min_vram_gb_3x - 1;

  const problems: string[] = [];
  if (!conn) problems.push(t("k2.noConnection"));
  else if (!connReady) problems.push(conn.status === "deploying" ? t("k2.starting") : t("k2.notReady"));
  else if (detection?.missing_nodes.length) problems.push(t("en.missingNodes", { nodes: detection.missing_nodes.join(", ") }));
  else if (!modelsReady) problems.push(t("en.installFirst"));
  else if (!photo) problems.push(t("en.photoFirst"));
  if (seed && !/^\d+$/.test(seed)) problems.push(t("gen.problemSeed"));

  const installModels = async () => {
    if (!conn) return;
    try {
      await install.mutateAsync({ id: conn.id, variant: "pgupscaler" });
      toast.success(t("k2.downloadStarted"));
    } catch (e) {
      toast.error(e instanceof ApiError ? e.message : t("errors.generic"));
    }
  };

  const submit = async () => {
    if (submitting.current) return;
    if (problems.length || !conn || !photo) {
      toast.error(problems[0] ?? t("en.photoFirst"));
      return;
    }
    submitting.current = true;
    try {
      await enhance.mutateAsync({
        connection_id: conn.id,
        image: { kind: photo.kind, id: photo.id },
        strength,
        face_only: faceOnly,
        upscale,
        seed: seed ? Number(seed) : null,
        skin_extra: skinExtra.trim(),
        face_target: faceOnly ? faceTarget.trim() || (spec?.face_target ?? "face") : (spec?.face_target ?? "face"),
        client_request_id: newRequestId(),
      });
      toast.success(t("en.started"));
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
        <div className="tiny subtle">{t("en.lede")}</div>
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
                <div className="tiny subtle">{t("en.modelsReady")}</div>
              ) : detection?.missing_nodes.length ? (
                <Notice tone="warning">{t("en.missingNodes", { nodes: detection.missing_nodes.join(", ") })}</Notice>
              ) : (
                <div className="model-download">
                  {downloading ? (
                    <>
                      <div className="row tiny" style={{ gap: 8 }}>
                        <span className="grow truncate">
                          {job.current ? t("k2.downloadingFile", { name: job.current }) : t("k2.downloadingHint")}
                        </span>
                        <span>{job.progress}%</span>
                      </div>
                      <Progress value={job.progress} />
                    </>
                  ) : conn?.auth_type === "agent" ? (
                    <>
                      {job?.status === "failed" && job.variant === "pgupscaler" ? (
                        <div className="error-text small">{job.error}</div>
                      ) : null}
                      <div className="row wrap" style={{ gap: 10 }}>
                        <Button
                          variant="primary"
                          icon={<Download />}
                          loading={install.isPending}
                          disabled={job?.status === "running"}
                          onClick={installModels}
                        >
                          {t("en.install", { size: formatBytes(spec?.total_bytes ?? 0, locale) })}
                        </Button>
                        <span className="tiny subtle">{t("en.installHint")}</span>
                      </div>
                    </>
                  ) : (
                    <div className="col" style={{ gap: 6 }}>
                      <div className="hint">{t("en.localFiles")}</div>
                      <ul className="tiny mono" style={{ margin: 0, paddingLeft: 18 }}>
                        {(detection?.files ?? []).filter((f) => !f.installed).map((f) => (
                          <li key={f.filename}>
                            models/{f.folder}/{f.filename}
                          </li>
                        ))}
                      </ul>
                      {spec ? (
                        <a className="tiny" href={spec.repo} target="_blank" rel="noreferrer" style={{ color: "var(--accent-text)" }}>
                          {spec.repo}
                        </a>
                      ) : null}
                    </div>
                  )}
                </div>
              )}
            </div>

            {!modelsReady ? null : (
              <>
                <div>
                  <div className="section-title">
                    <span className="eyebrow grow">{t("en.stepPhoto")}</span>
                    <Button size="sm" variant="ghost" icon={<Images />} onClick={() => setPicking(true)}>
                      {t("en.fromLibrary")}
                    </Button>
                  </div>
                  {photo ? (
                    <div className="refs">
                      <div className="ref">
                        <img src={photo.thumb_url} alt={t("en.photo")} />
                        <div className="ref-actions">
                          <IconButton label={t("refs.remove")} onMedia onClick={() => setPhoto(null)}>
                            <X />
                          </IconButton>
                        </div>
                      </div>
                    </div>
                  ) : (
                    <ReferenceImages
                      refs={[]}
                      max={1}
                      onChange={(refs: ReferenceImage[]) =>
                        refs[0] ? setPhoto({ kind: "reference", id: refs[0].id, thumb_url: refs[0].thumb_url }) : setPhoto(null)
                      }
                    />
                  )}
                </div>

                <div className="col gap-16">
                  <Field label={t("en.strength")} hint={t(strengthHint(strength))}>
                    <Segmented
                      label={t("en.strength")}
                      value={strength}
                      block
                      onChange={setStrength}
                      options={(spec?.strengths ?? [0.15, 0.2, 0.25, 0.3, 0.35, 0.4]).map((s) => ({
                        value: s,
                        label: String(s),
                        hint: s === spec?.default_strength ? t("en.strength.default") : undefined,
                      }))}
                    />
                  </Field>
                  <div className="row" style={{ gap: 10, alignItems: "flex-start" }}>
                    <Switch checked={faceOnly} onChange={setFaceOnly} label={t("en.faceOnly")} />
                    <div>
                      <div className="small">{t("en.faceOnly")}</div>
                      <div className="hint">{t("en.faceOnlyHint")}</div>
                    </div>
                  </div>
                  <Field label={t("en.upscale")} hint={t("en.upscaleHint")}>
                    <Segmented
                      label={t("en.upscale")}
                      value={upscale}
                      block
                      onChange={setUpscale}
                      options={(spec?.upscales ?? [1.5, 2, 3]).map((u) => ({ value: u, label: `${u}×` }))}
                    />
                  </Field>
                  {smallFor3x ? <Notice tone="warning">{t("en.vram3x", { n: spec?.min_vram_gb_3x ?? 48 })}</Notice> : null}
                </div>

                <div className="col gap-12">
                  <button type="button" className="link-button tiny" onClick={() => setMore((v) => !v)} aria-expanded={more}>
                    {t("gen.moreSettings")} <ChevronDown size={13} style={{ transform: more ? "rotate(180deg)" : undefined }} />
                  </button>
                  {more ? (
                    <Field label={t("en.skinExtra")} hint={t("en.skinExtraHint")}>
                      <textarea
                        className="textarea"
                        rows={2}
                        value={skinExtra}
                        maxLength={spec?.max_skin_extra ?? 500}
                        placeholder="freckles"
                        onChange={(e) => setSkinExtra(e.target.value)}
                      />
                    </Field>
                  ) : null}
                  {more && faceOnly ? (
                    <Field label={t("en.faceTarget")} hint={t("en.faceTargetHint")}>
                      <input
                        className="input"
                        value={faceTarget}
                        maxLength={spec?.max_face_target ?? 120}
                        placeholder={spec?.face_target ?? "face"}
                        onChange={(e) => setFaceTarget(e.target.value)}
                      />
                    </Field>
                  ) : null}
                  {more ? (
                    <Field label={t("gen.seed")} hint={t("en.seedHint")}>
                      <div className="input-wrap">
                        <input
                          className="input mono"
                          inputMode="numeric"
                          value={seed}
                          placeholder={t("gen.seedRandom")}
                          onChange={(e) => setSeed(e.target.value.trim())}
                        />
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
        <Button
          variant="primary"
          size="lg"
          block
          icon={<Sparkles />}
          loading={enhance.isPending}
          disabled={problems.length > 0 || enhance.isPending}
          onClick={submit}
        >
          {t("en.run")}
        </Button>
      </div>
      {picking ? (
        <LibraryPicker
          onClose={() => setPicking(false)}
          onPick={(p) => {
            setPhoto(p);
            setPicking(false);
          }}
        />
      ) : null}
    </section>
  );
}

/** Pick one of the user's generated images. */
function LibraryPicker({ onClose, onPick }: { onClose: () => void; onPick: (p: EnhancePhoto) => void }) {
  const { t } = useI18n();
  const images = useImages({});
  const items = (images.data?.pages ?? []).flatMap((p) => p.items);
  return (
    <Modal open wide onClose={onClose} title={t("en.fromLibrary")} description={t("en.fromLibraryHint")}>
      {images.isLoading ? (
        <Spinner />
      ) : !items.length ? (
        <EmptyState icon={<ImageIcon />} title={t("en.libraryEmpty")} description={t("en.libraryEmptyBody")} />
      ) : (
        <div className="pick-grid">
          {items.map((img) => (
            <button
              key={img.id}
              type="button"
              className="pick-tile"
              aria-label={img.model_name}
              onClick={() => onPick({ kind: "image", id: img.id, thumb_url: img.thumb_url })}
            >
              <img src={img.thumb_url} alt="" loading="lazy" />
            </button>
          ))}
          {images.hasNextPage ? (
            <Button size="sm" loading={images.isFetchingNextPage} onClick={() => void images.fetchNextPage()}>
              {t("common.loadMore")}
            </Button>
          ) : null}
        </div>
      )}
    </Modal>
  );
}
