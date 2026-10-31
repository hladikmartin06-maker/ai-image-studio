"use client";

import { ArrowLeft, ArrowRight, Check, ImagePlus, Images, Star, Trash2, Upload } from "lucide-react";
import { type DragEvent, useMemo, useRef, useState } from "react";
import { toast } from "sonner";

import { Badge, Button, EmptyState, Field, IconButton, Modal, Notice, Skeleton, Spinner, useConfirm } from "@/components/ui";
import { ApiError } from "@/lib/api";
import { formatBytes } from "@/lib/format";
import { useI18n, useT } from "@/lib/i18n";
import type { MessageKey } from "@/lib/i18n";
import {
  useAddPersonaImages,
  useCatalog,
  useImages,
  useRemovePersonaImage,
  useUpdatePersona,
  useUploadPersonaImage,
} from "@/lib/queries";
import type { Persona, PersonaAppearance, PersonaImage } from "@/lib/types";

const ACCEPT = ["image/png", "image/jpeg", "image/webp"];
const MAX_MB = 10;

// --- Profile ------------------------------------------------------------------------------

type TextKey = Exclude<keyof PersonaAppearance, "age">;

const FIELDS: { key: TextKey; max: number; long?: boolean }[] = [
  { key: "gender", max: 60 },
  { key: "ethnicity", max: 120 },
  { key: "skin", max: 160 },
  { key: "eyes", max: 160 },
  { key: "hair_color", max: 120 },
  { key: "height", max: 60 },
  { key: "hairstyle", max: 300, long: true },
  { key: "face", max: 400, long: true },
  { key: "body", max: 400, long: true },
  { key: "style", max: 300, long: true },
  { key: "distinctive", max: 300, long: true },
  { key: "extra", max: 400, long: true },
];

interface ProfileForm {
  name: string;
  description: string;
  notes: string;
  age: string;
  look: Record<TextKey, string>;
}

function toForm(p: Persona): ProfileForm {
  const look = {} as Record<TextKey, string>;
  for (const f of FIELDS) look[f.key] = p.appearance[f.key] ?? "";
  return {
    name: p.name,
    description: p.description,
    notes: p.notes,
    age: p.appearance.age ? String(p.appearance.age) : "",
    look,
  };
}

function sameForm(a: ProfileForm, b: ProfileForm) {
  const norm = (f: ProfileForm) =>
    JSON.stringify({
      ...f,
      name: f.name.trim(),
      description: f.description.trim(),
      notes: f.notes.trim(),
      look: Object.fromEntries(Object.entries(f.look).map(([k, v]) => [k, v.trim().replace(/\s+/g, " ")])),
    });
  return norm(a) === norm(b);
}

export function PersonaProfile({ persona }: { persona: Persona }) {
  const t = useT();
  const update = useUpdatePersona(persona.id);
  const baseline = toForm(persona);
  const [form, setForm] = useState<ProfileForm>(baseline);
  const dirty = !sameForm(form, baseline);
  const age = form.age.trim() ? Number(form.age) : null;
  const ageError = age !== null && (!Number.isInteger(age) || age < 18 || age > 120) ? t("pm.ageError") : null;
  const nameError = !form.name.trim() ? t("pm.nameRequired") : null;

  const setLook = (key: TextKey, value: string) => setForm((f) => ({ ...f, look: { ...f.look, [key]: value } }));

  const save = async () => {
    if (ageError || nameError) return;
    try {
      await update.mutateAsync({
        name: form.name.trim(),
        description: form.description,
        notes: form.notes,
        appearance: { ...form.look, age },
      });
      toast.success(t("pm.saved"));
    } catch (e) {
      toast.error(e instanceof ApiError ? e.message : t("errors.generic"));
    }
  };

  return (
    <form
      className="col gap-16"
      onSubmit={(e) => {
        e.preventDefault();
        void save();
      }}
    >
      <Field label={t("pm.name")} htmlFor="pf-name" error={nameError}>
        <input
          id="pf-name"
          className="input"
          maxLength={80}
          value={form.name}
          onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))}
        />
      </Field>
      <Field label={t("pm.description")} hint={t("pm.descriptionHint")} htmlFor="pf-desc">
        <textarea
          id="pf-desc"
          className="textarea"
          rows={3}
          maxLength={1500}
          placeholder={t("pm.descriptionPlaceholder")}
          value={form.description}
          onChange={(e) => setForm((f) => ({ ...f, description: e.target.value }))}
        />
      </Field>

      <div>
        <div className="eyebrow" style={{ marginBottom: 10 }}>
          {t("pm.appearance")}
        </div>
        <div className="form-grid">
          <Field label={t("pm.field.age")} htmlFor="pf-age" error={ageError} hint={t("pm.field.ageHint")}>
            <input
              id="pf-age"
              className="input"
              type="number"
              inputMode="numeric"
              min={18}
              max={120}
              value={form.age}
              aria-invalid={Boolean(ageError) || undefined}
              onChange={(e) => setForm((f) => ({ ...f, age: e.target.value }))}
            />
          </Field>
          {FIELDS.map((f) => (
            <Field key={f.key} label={t(`pm.field.${f.key}` as MessageKey)} htmlFor={`pf-${f.key}`}>
              {f.long ? (
                <textarea
                  id={`pf-${f.key}`}
                  className="textarea"
                  rows={2}
                  maxLength={f.max}
                  placeholder={t(`pm.ph.${f.key}` as MessageKey)}
                  value={form.look[f.key]}
                  onChange={(e) => setLook(f.key, e.target.value)}
                />
              ) : (
                <input
                  id={`pf-${f.key}`}
                  className="input"
                  maxLength={f.max}
                  placeholder={t(`pm.ph.${f.key}` as MessageKey)}
                  value={form.look[f.key]}
                  onChange={(e) => setLook(f.key, e.target.value)}
                />
              )}
            </Field>
          ))}
        </div>
      </div>

      <Field label={t("pm.notes")} hint={t("pm.notesHint")} htmlFor="pf-notes">
        <textarea
          id="pf-notes"
          className="textarea"
          rows={2}
          maxLength={4000}
          value={form.notes}
          onChange={(e) => setForm((f) => ({ ...f, notes: e.target.value }))}
        />
      </Field>

      <div className="persona-prompt">
        <span className="tiny subtle">{t("pm.promptPreview")}</span>
        <div>{persona.prompt_text || t("pm.promptPreviewEmpty")}</div>
        {dirty ? <span className="tiny subtle">{t("pm.previewAfterSave")}</span> : null}
      </div>

      <div className="row" style={{ justifyContent: "flex-end" }}>
        {dirty ? (
          <Button variant="ghost" onClick={() => setForm(baseline)}>
            {t("pm.discard")}
          </Button>
        ) : null}
        <Button
          type="submit"
          variant="primary"
          icon={<Check />}
          loading={update.isPending}
          disabled={!dirty || Boolean(ageError || nameError)}
        >
          {t("common.save")}
        </Button>
      </div>
    </form>
  );
}

// --- Reference image collection --------------------------------------------------------------

export function PersonaImages({ persona }: { persona: Persona }) {
  const { t, locale } = useI18n();
  const confirm = useConfirm();
  const upload = useUploadPersonaImage(persona.id);
  const remove = useRemovePersonaImage(persona.id);
  const update = useUpdatePersona(persona.id);
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const [pending, setPending] = useState(0);
  const [preview, setPreview] = useState<number | null>(null);
  const [library, setLibrary] = useState(false);
  const images = persona.images;
  const room = persona.max_images - images.length;

  const addFiles = async (files: File[]) => {
    if (room <= 0) {
      toast.error(t("pm.full", { n: persona.max_images }));
      return;
    }
    const chosen = files.slice(0, room);
    if (files.length > room) toast.warning(t("pm.full", { n: persona.max_images }));
    for (const file of chosen) {
      if (!ACCEPT.includes(file.type)) {
        toast.error(`${file.name}: ${t("refs.badType")}`);
        continue;
      }
      if (file.size > MAX_MB * 1024 * 1024) {
        toast.error(`${file.name}: ${t("refs.tooLarge", { mb: MAX_MB })}`);
        continue;
      }
      setPending((n) => n + 1);
      try {
        const res = await upload.mutateAsync(file);
        if (!res.added) toast.info(`${file.name}: ${t("pm.duplicateImage")}`);
      } catch (e) {
        toast.error(`${file.name}: ${e instanceof ApiError ? e.message : t("errors.generic")}`);
      } finally {
        setPending((n) => n - 1);
      }
    }
  };

  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setOver(false);
    void addFiles(Array.from(e.dataTransfer.files));
  };

  const run = (body: Parameters<typeof update.mutate>[0], done?: string) =>
    update.mutate(body, {
      onSuccess: () => (done ? toast.success(done) : undefined),
      onError: (e) => toast.error((e as Error).message),
    });

  const move = (index: number, delta: -1 | 1) => {
    const ids = images.map((i) => i.id);
    const target = index + delta;
    // The primary image always stays first.
    if (target < 1 || target >= ids.length) return;
    [ids[index], ids[target]] = [ids[target], ids[index]];
    run({ image_order: ids });
  };

  const makePrimary = (img: PersonaImage) => {
    const ids = [img.id, ...images.filter((i) => i.id !== img.id).map((i) => i.id)];
    run({ cover_reference_id: img.id, image_order: ids }, t("pm.primarySet"));
  };

  const removeImage = async (img: PersonaImage) => {
    const ok = await confirm({
      title: t("pm.removeImageTitle"),
      description: t("pm.removeImageBody"),
      confirmLabel: t("refs.remove"),
      danger: true,
    });
    if (!ok) return;
    remove.mutate(img.id, {
      onSuccess: () => toast.success(t("pm.imageRemoved")),
      onError: (e) => toast.error((e as Error).message),
    });
  };

  return (
    <div
      className="col gap-16"
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={onDrop}
    >
      <input
        ref={input}
        type="file"
        accept={ACCEPT.join(",")}
        multiple
        hidden
        onChange={(e) => {
          void addFiles(Array.from(e.target.files ?? []));
          e.target.value = "";
        }}
      />
      {images.length || pending ? (
        <div className="persona-images">
          {images.map((img, i) => (
            <div className="persona-image" key={img.id} data-primary={img.is_primary || undefined}>
              <button
                type="button"
                className="persona-image-open"
                onClick={() => setPreview(i)}
                aria-label={t("pm.preview")}
              >
                <img src={img.thumb_url} alt={img.original_name || t("refs.reference")} loading="lazy" />
              </button>
              {img.is_primary ? (
                <span className="persona-image-badge">
                  <Star /> {t("pm.primary")}
                </span>
              ) : null}
              <div className="persona-image-actions">
                {!img.is_primary ? (
                  <IconButton label={t("pm.makePrimary")} onMedia onClick={() => makePrimary(img)}>
                    <Star />
                  </IconButton>
                ) : null}
                {!img.is_primary && i > 1 ? (
                  <IconButton label={t("pm.moveEarlier")} onMedia onClick={() => move(i, -1)}>
                    <ArrowLeft />
                  </IconButton>
                ) : null}
                {!img.is_primary && i < images.length - 1 ? (
                  <IconButton label={t("pm.moveLater")} onMedia onClick={() => move(i, 1)}>
                    <ArrowRight />
                  </IconButton>
                ) : null}
                <IconButton label={t("refs.remove")} onMedia onClick={() => removeImage(img)}>
                  <Trash2 />
                </IconButton>
              </div>
            </div>
          ))}
          {Array.from({ length: pending }, (_, i) => (
            <div className="persona-image ref-uploading" key={`p${i}`}>
              <Spinner />
            </div>
          ))}
          {room > pending ? (
            <button
              type="button"
              className="persona-image dropzone"
              data-over={over || undefined}
              onClick={() => input.current?.click()}
              aria-label={t("pm.upload")}
            >
              <ImagePlus />
            </button>
          ) : null}
        </div>
      ) : (
        <button
          type="button"
          className="dropzone"
          data-over={over || undefined}
          onClick={() => input.current?.click()}
          style={{ padding: 22 }}
        >
          <Upload />
          <div className="grow">
            <div style={{ color: "var(--text)", fontWeight: 500 }}>{t("pm.dropTitle")}</div>
            <div className="tiny subtle">{t("pm.dropHint", { n: persona.max_images, mb: MAX_MB })}</div>
          </div>
        </button>
      )}
      <div className="row wrap" style={{ gap: 8 }}>
        <Button size="sm" icon={<Upload />} onClick={() => input.current?.click()} disabled={room <= 0}>
          {t("pm.upload")}
        </Button>
        <Button size="sm" icon={<Images />} onClick={() => setLibrary(true)} disabled={room <= 0}>
          {t("pm.fromLibrary")}
        </Button>
        <span className="spacer" />
        <span className="tiny subtle">{t("pm.imageCount", { n: images.length, max: persona.max_images })}</span>
      </div>
      <ReferenceLimits />

      {preview !== null && images[preview] ? (
        <Modal
          open
          wide
          onClose={() => setPreview(null)}
          title={images[preview].is_primary ? t("pm.primary") : t("pm.imageN", { n: preview + 1 })}
          description={`${images[preview].width} × ${images[preview].height} · ${formatBytes(images[preview].size_bytes ?? 0, locale)}`}
          footer={
            <>
              <Button size="sm" variant="ghost" disabled={preview === 0} onClick={() => setPreview(preview - 1)} icon={<ArrowLeft />}>
                {t("viewer.previous")}
              </Button>
              <Button
                size="sm"
                variant="ghost"
                disabled={preview >= images.length - 1}
                onClick={() => setPreview(preview + 1)}
                icon={<ArrowRight />}
              >
                {t("viewer.next")}
              </Button>
              <span className="spacer" />
              {!images[preview].is_primary ? (
                <Button size="sm" icon={<Star />} onClick={() => makePrimary(images[preview])}>
                  {t("pm.makePrimary")}
                </Button>
              ) : null}
            </>
          }
        >
          <div className="persona-preview">
            <img src={images[preview].url} alt="" />
          </div>
        </Modal>
      ) : null}
      <LibraryPicker
        open={library}
        persona={persona}
        room={room}
        onClose={() => setLibrary(false)}
      />
    </div>
  );
}

/** How many reference images each model takes per request (from the catalog). */
function ReferenceLimits() {
  const t = useT();
  const catalog = useCatalog();
  const rows = useMemo(
    () =>
      (catalog.data?.models ?? []).map((m) => ({
        name: m.name,
        limits: m.variants
          .filter((v) => v.mode === "image-edit")
          .map((v) => `${v.provider === "wavespeed" ? "WaveSpeed" : "Atlas"} ${v.max_reference_images}`),
      })),
    [catalog.data],
  );
  if (!rows.length) return null;
  return (
    <div className="persona-limits">
      <div className="tiny subtle" style={{ marginBottom: 6 }}>
        {t("pm.limitsTitle")}
      </div>
      <dl className="kv">
        {rows.map((r) => (
          <div key={r.name} style={{ display: "contents" }}>
            <dt>{r.name}</dt>
            <dd>{r.limits.length ? r.limits.join(" · ") : t("pm.noRefs")}</dd>
          </div>
        ))}
      </dl>
    </div>
  );
}

function LibraryPicker({
  open,
  persona,
  room,
  onClose,
}: {
  open: boolean;
  persona: Persona;
  room: number;
  onClose: () => void;
}) {
  return open ? <LibraryPickerBody persona={persona} room={room} onClose={onClose} /> : null;
}

function LibraryPickerBody({ persona, room, onClose }: { persona: Persona; room: number; onClose: () => void }) {
  const t = useT();
  const images = useImages({ limit: 60, scope: "library" });
  const add = useAddPersonaImages();
  const [selected, setSelected] = useState<string[]>([]);
  const items = images.data?.pages.flatMap((p) => p.items) ?? [];

  const toggle = (id: string) =>
    setSelected((s) => {
      if (s.includes(id)) return s.filter((x) => x !== id);
      if (s.length >= room) {
        toast.warning(t("pm.full", { n: persona.max_images }));
        return s;
      }
      return [...s, id];
    });

  const submit = async () => {
    try {
      const res = await add.mutateAsync({ id: persona.id, libraryImageIds: selected });
      toast.success(t("pm.addedTo", { n: res.added, name: persona.name }));
      onClose();
    } catch (e) {
      toast.error(e instanceof ApiError ? e.message : t("errors.generic"));
    }
  };

  return (
    <Modal
      open
      wide
      onClose={onClose}
      title={t("pm.fromLibraryTitle")}
      description={t("pm.fromLibraryBody")}
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>
            {t("common.cancel")}
          </Button>
          <Button variant="primary" disabled={!selected.length} loading={add.isPending} onClick={submit}>
            {t("pm.addN", { n: selected.length })}
          </Button>
        </>
      }
    >
      {images.isLoading ? (
        <div className="persona-library">
          {Array.from({ length: 12 }, (_, i) => (
            <Skeleton key={i} style={{ aspectRatio: "1" }} />
          ))}
        </div>
      ) : images.isError ? (
        <Notice tone="danger">{(images.error as Error).message}</Notice>
      ) : items.length === 0 ? (
        <EmptyState icon={<Images />} title={t("lib.emptyTitle")} description={t("pm.libraryEmpty")} />
      ) : (
        <>
          <div className="persona-library">
            {items.map((img) => {
              const on = selected.includes(img.id);
              return (
                <button
                  key={img.id}
                  type="button"
                  className="persona-library-item"
                  aria-pressed={on}
                  onClick={() => toggle(img.id)}
                  title={img.prompt}
                >
                  <img src={img.thumb_url} alt="" loading="lazy" />
                  {on ? (
                    <span className="persona-pick-check" aria-hidden>
                      <Check />
                    </span>
                  ) : null}
                </button>
              );
            })}
          </div>
          {images.hasNextPage ? (
            <div className="row" style={{ justifyContent: "center", marginTop: 12 }}>
              <Button size="sm" loading={images.isFetchingNextPage} onClick={() => images.fetchNextPage()}>
                {t("common.loadMore")}
              </Button>
            </div>
          ) : null}
        </>
      )}
      {persona.image_count ? (
        <div className="row" style={{ marginTop: 12 }}>
          <Badge>{t("pm.imageCount", { n: persona.image_count, max: persona.max_images })}</Badge>
        </div>
      ) : null}
    </Modal>
  );
}
