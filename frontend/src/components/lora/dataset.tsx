"use client";

import {
  CheckSquare,
  Crop,
  EyeOff,
  Images,
  ImagePlus,
  SquareUserRound,
  Trash2,
  TriangleAlert,
  Upload,
} from "lucide-react";
import { type DragEvent, useMemo, useRef, useState } from "react";
import { toast } from "sonner";

import { DatasetGenerator } from "@/components/lora/dataset-generator";
import { Badge, Button, EmptyState, Field, Modal, Notice, Segmented, Select, Skeleton, useConfirm } from "@/components/ui";
import { ApiError } from "@/lib/api";
import { formatBytes } from "@/lib/format";
import { useI18n, useT } from "@/lib/i18n";
import type { MessageKey } from "@/lib/i18n";
import {
  useAddDatasetImages,
  useBulkDataset,
  useImages,
  usePersonas,
  useUpdateDatasetItem,
  useUploadDatasetImage,
} from "@/lib/queries";
import type { LoraDetail, LoraItem } from "@/lib/types";

const ACCEPT = ["image/png", "image/jpeg", "image/webp"];
type Filter = "all" | "train" | "validation" | "excluded" | "warnings";

export function DatasetStats({ lora }: { lora: LoraDetail }) {
  const t = useT();
  const a = lora.analysis;
  return (
    <div className="dataset-stats">
      <div className="comfy-grid">
        <div className="comfy-stat">
          <span className="tiny subtle">{t("ds.train")}</span>
          <span className="h3">{a.train}</span>
        </div>
        <div className="comfy-stat">
          <span className="tiny subtle">{t("ds.validation")}</span>
          <span className="h3">{a.validation}</span>
        </div>
        <div className="comfy-stat">
          <span className="tiny subtle">{t("ds.excluded")}</span>
          <span className="h3">{a.excluded}</span>
        </div>
        <div className="comfy-stat">
          <span className="tiny subtle">{t("ds.captioned")}</span>
          <span className="h3">
            {a.captioned}/{a.train}
          </span>
        </div>
        <div className="comfy-stat">
          <span className="tiny subtle">{t("ds.minEdge")}</span>
          <span className="h3">{a.min_edge ? `${a.min_edge}px` : "—"}</span>
        </div>
        <div className="comfy-stat">
          <span className="tiny subtle">{t("ds.buckets")}</span>
          <span className="small">
            {Object.entries(a.aspect_buckets)
              .map(([k, v]) => `${k} ×${v}`)
              .join(" · ") || "—"}
          </span>
        </div>
      </div>
      {a.warnings.length ? (
        <div className="col" style={{ gap: 6, marginTop: 12 }}>
          {a.warnings.map((w) => (
            <div key={w.code} className={`ds-warning ${w.level}`}>
              <TriangleAlert size={14} />
              <span>
                {t(`ds.warn.${w.code}` as MessageKey, {
                  n: w.n ?? w.items?.length ?? w.pairs?.length ?? 0,
                  min: 10,
                })}
              </span>
            </div>
          ))}
        </div>
      ) : (
        <div className="ds-warning ok" style={{ marginTop: 12 }}>
          {t("ds.allGood")}
        </div>
      )}
    </div>
  );
}

export function DatasetTab({ lora }: { lora: LoraDetail }) {
  const { t, locale } = useI18n();
  const confirm = useConfirm();
  const upload = useUploadDatasetImage(lora.id);
  const bulk = useBulkDataset(lora.id);
  const input = useRef<HTMLInputElement>(null);
  const [pending, setPending] = useState(0);
  const [over, setOver] = useState(false);
  const [filter, setFilter] = useState<Filter>("all");
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [editing, setEditing] = useState<LoraItem | null>(null);
  const [picker, setPicker] = useState<"library" | "persona" | null>(null);

  const flagged = useMemo(() => {
    const ids = new Set<string>();
    for (const w of lora.analysis.warnings) {
      w.items?.forEach((i) => ids.add(i));
      w.pairs?.forEach((p) => p.forEach((i) => ids.add(i)));
    }
    return ids;
  }, [lora.analysis]);

  const items = lora.dataset.filter((i) =>
    filter === "all"
      ? true
      : filter === "excluded"
        ? i.excluded
        : filter === "warnings"
          ? flagged.has(i.id)
          : i.split === filter && !i.excluded,
  );

  const addFiles = async (files: File[]) => {
    for (const file of files) {
      if (!ACCEPT.includes(file.type)) {
        toast.error(`${file.name}: ${t("refs.badType")}`);
        continue;
      }
      setPending((n) => n + 1);
      try {
        const res = await upload.mutateAsync(file);
        if (!res.added) toast.info(`${file.name}: ${t("ds.duplicate")}`);
      } catch (e) {
        toast.error(`${file.name}: ${e instanceof ApiError ? e.message : t("errors.generic")}`);
      } finally {
        setPending((n) => n - 1);
      }
    }
  };

  const runBulk = async (action: string) => {
    if (action === "remove") {
      const ok = await confirm({
        title: t("ds.removeTitle", { n: selected.size }),
        description: t("ds.removeBody"),
        confirmLabel: t("refs.remove"),
        danger: true,
      });
      if (!ok) return;
    }
    try {
      await bulk.mutateAsync({ item_ids: [...selected], action });
      setSelected(new Set());
    } catch (e) {
      toast.error((e as Error).message);
    }
  };

  const toggle = (id: string) =>
    setSelected((s) => {
      const next = new Set(s);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });

  return (
    <div
      className="col gap-16"
      onDragOver={(e: DragEvent) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e: DragEvent) => {
        e.preventDefault();
        setOver(false);
        void addFiles(Array.from(e.dataTransfer.files));
      }}
    >
      <input
        ref={input}
        type="file"
        hidden
        multiple
        accept={ACCEPT.join(",")}
        onChange={(e) => {
          void addFiles(Array.from(e.target.files ?? []));
          e.target.value = "";
        }}
      />
      <DatasetStats lora={lora} />
      <DatasetGenerator lora={lora} />
      <div className="toolbar" style={{ marginBottom: 0 }}>
        <Button icon={<Upload />} onClick={() => input.current?.click()} loading={pending > 0}>
          {pending ? t("ds.uploading", { n: pending }) : t("ds.upload")}
        </Button>
        <Button icon={<Images />} onClick={() => setPicker("library")}>
          {t("pm.fromLibrary")}
        </Button>
        <Button icon={<SquareUserRound />} onClick={() => setPicker("persona")}>
          {t("ds.fromPersona")}
        </Button>
        <span className="spacer" />
        <Segmented
          label={t("ds.filter")}
          value={filter}
          onChange={setFilter}
          options={(["all", "train", "validation", "excluded", "warnings"] as Filter[]).map((f) => ({
            value: f,
            label: t(`ds.filter.${f}`),
          }))}
        />
      </div>
      {lora.dataset.length === 0 && !pending ? (
        <button type="button" className="dropzone" data-over={over || undefined} onClick={() => input.current?.click()} style={{ padding: 28 }}>
          <ImagePlus />
          <div className="grow">
            <div style={{ color: "var(--text)", fontWeight: 500 }}>{t("ds.dropTitle")}</div>
            <div className="tiny subtle">{t("ds.dropHint")}</div>
          </div>
        </button>
      ) : (
        <div className="ds-grid" data-over={over || undefined}>
          {items.map((item) => (
            <div key={item.id} className="ds-item" data-excluded={item.excluded || undefined} data-selected={selected.has(item.id) || undefined}>
              <button type="button" className="ds-img" onClick={() => setEditing(item)} aria-label={t("ds.edit")}>
                <img src={item.thumb_url} alt="" loading="lazy" />
              </button>
              <button
                type="button"
                className="ds-check"
                aria-pressed={selected.has(item.id)}
                aria-label={t("common.select")}
                onClick={() => toggle(item.id)}
              >
                <CheckSquare size={14} />
              </button>
              <div className="ds-badges">
                {item.split === "validation" ? <Badge tone="accent">{t("ds.val")}</Badge> : null}
                {item.excluded ? <Badge><EyeOff size={11} /></Badge> : null}
                {item.crop ? <Badge><Crop size={11} /></Badge> : null}
                {flagged.has(item.id) ? <Badge tone="warning"><TriangleAlert size={11} /></Badge> : null}
              </div>
              <div className="ds-meta">
                <span className="tiny subtle">
                  {item.width}×{item.height} · {formatBytes(item.size_bytes ?? 0, locale)}
                </span>
                <span className={item.caption ? "tiny clamp-2" : "tiny subtle"}>{item.caption || t("ds.noCaption")}</span>
              </div>
            </div>
          ))}
          {Array.from({ length: pending }, (_, i) => (
            <Skeleton key={`p${i}`} style={{ aspectRatio: "3 / 4" }} />
          ))}
        </div>
      )}
      {selected.size ? (
        <div className="bulkbar" role="toolbar" aria-label={t("lib.bulkActions")}>
          <span className="small" style={{ fontWeight: 600 }}>{t("common.nSelected", { n: selected.size })}</span>
          <Button size="sm" variant="ghost" onClick={() => setSelected(new Set(items.map((i) => i.id)))}>
            {t("lib.selectAll", { n: items.length })}
          </Button>
          <Button size="sm" onClick={() => runBulk("include")}>{t("ds.include")}</Button>
          <Button size="sm" onClick={() => runBulk("exclude")}>{t("ds.exclude")}</Button>
          <Button size="sm" onClick={() => runBulk("train")}>{t("ds.toTrain")}</Button>
          <Button size="sm" onClick={() => runBulk("validation")}>{t("ds.toValidation")}</Button>
          <Button size="sm" onClick={() => runBulk("add_trigger")}>{t("ds.addTrigger")}</Button>
          <Button size="sm" variant="danger" icon={<Trash2 />} onClick={() => runBulk("remove")}>{t("refs.remove")}</Button>
        </div>
      ) : null}
      {editing ? (
        <ItemEditor
          key={editing.id}
          lora={lora}
          item={lora.dataset.find((i) => i.id === editing.id) ?? editing}
          onClose={() => setEditing(null)}
          onNavigate={(dir) => {
            const index = items.findIndex((i) => i.id === editing.id);
            const next = items[index + dir];
            if (next) setEditing(next);
          }}
        />
      ) : null}
      {picker === "library" ? <LibraryImport lora={lora} onClose={() => setPicker(null)} /> : null}
      {picker === "persona" ? <PersonaImport lora={lora} onClose={() => setPicker(null)} /> : null}
    </div>
  );
}

// --- one image: crop, caption, split ----------------------------------------------------

const CROPS: { label: string; ratio: number | null }[] = [
  { label: "—", ratio: null },
  { label: "1:1", ratio: 1 },
  { label: "4:5", ratio: 0.8 },
  { label: "3:4", ratio: 0.75 },
  { label: "2:3", ratio: 2 / 3 },
  { label: "9:16", ratio: 9 / 16 },
  { label: "16:9", ratio: 16 / 9 },
];

function cropFor(width: number, height: number, ratio: number, zoom: number, cx: number, cy: number) {
  const imageRatio = width / height;
  let w = 1;
  let h = 1;
  if (ratio > imageRatio) h = imageRatio / ratio;
  else w = ratio / imageRatio;
  w *= zoom;
  h *= zoom;
  const x = Math.min(Math.max(cx - w / 2, 0), 1 - w);
  const y = Math.min(Math.max(cy - h / 2, 0), 1 - h);
  return { x: +x.toFixed(4), y: +y.toFixed(4), w: +w.toFixed(4), h: +h.toFixed(4) };
}

function ItemEditor({
  lora,
  item,
  onClose,
  onNavigate,
}: {
  lora: LoraDetail;
  item: LoraItem;
  onClose: () => void;
  onNavigate: (dir: -1 | 1) => void;
}) {
  const t = useT();
  const update = useUpdateDatasetItem(lora.id);
  const [caption, setCaption] = useState(item.caption);
  const [ratio, setRatio] = useState<string>(item.crop ? "custom" : "—");
  const [zoom, setZoom] = useState(item.crop ? Math.max(item.crop.w, item.crop.h) : 1);
  const [center, setCenter] = useState({
    x: item.crop ? item.crop.x + item.crop.w / 2 : 0.5,
    y: item.crop ? item.crop.y + item.crop.h / 2 : 0.5,
  });
  const preset = CROPS.find((c) => c.label === ratio);
  const crop =
    preset?.ratio && item.width && item.height
      ? cropFor(item.width, item.height, preset.ratio, zoom, center.x, center.y)
      : ratio === "custom"
        ? item.crop
        : null;

  const save = async (patch: Record<string, unknown>, done?: string) => {
    try {
      await update.mutateAsync({ itemId: item.id, ...patch });
      if (done) toast.success(done);
    } catch (e) {
      toast.error((e as Error).message);
    }
  };

  return (
    <Modal
      open
      wide
      onClose={onClose}
      title={item.original_name || t("ds.image")}
      description={`${item.width}×${item.height}`}
      footer={
        <>
          <Button size="sm" variant="ghost" onClick={() => onNavigate(-1)}>{t("viewer.previous")}</Button>
          <Button size="sm" variant="ghost" onClick={() => onNavigate(1)}>{t("viewer.next")}</Button>
          <span className="spacer" />
          <Button
            variant="primary"
            loading={update.isPending}
            onClick={() => save({ caption, crop: crop ?? null }, t("ds.saved"))}
          >
            {t("common.save")}
          </Button>
        </>
      }
    >
      <div className="ds-editor">
        <div className="ds-preview">
          <img src={item.url} alt="" />
          {crop ? (
            <div
              className="ds-crop"
              style={{
                left: `${crop.x * 100}%`,
                top: `${crop.y * 100}%`,
                width: `${crop.w * 100}%`,
                height: `${crop.h * 100}%`,
              }}
            />
          ) : null}
        </div>
        <div className="col gap-16">
          <Field label={t("ds.caption")} hint={t("ds.captionHint", { trigger: lora.trigger_word })}>
            <textarea className="textarea" rows={4} value={caption} maxLength={2000} onChange={(e) => setCaption(e.target.value)} />
          </Field>
          <Field label={t("ds.crop")} hint={t("ds.cropHint")}>
            <div className="chips">
              {CROPS.map((c) => (
                <button key={c.label} type="button" className="chip" aria-pressed={ratio === c.label} onClick={() => setRatio(c.label)}>
                  {c.label === "—" ? t("ds.noCrop") : c.label}
                </button>
              ))}
            </div>
          </Field>
          {preset?.ratio ? (
            <div className="col gap-12">
              <label className="small">
                {t("ds.zoom")}
                <input type="range" min={0.3} max={1} step={0.01} value={zoom} onChange={(e) => setZoom(Number(e.target.value))} />
              </label>
              <label className="small">
                {t("ds.horizontal")}
                <input type="range" min={0} max={1} step={0.01} value={center.x} onChange={(e) => setCenter((c) => ({ ...c, x: Number(e.target.value) }))} />
              </label>
              <label className="small">
                {t("ds.vertical")}
                <input type="range" min={0} max={1} step={0.01} value={center.y} onChange={(e) => setCenter((c) => ({ ...c, y: Number(e.target.value) }))} />
              </label>
              {crop && item.width && item.height ? (
                <span className="tiny subtle">
                  {t("ds.cropSize", { w: Math.round(crop.w * item.width), h: Math.round(crop.h * item.height) })}
                </span>
              ) : null}
            </div>
          ) : null}
          <div className="form-grid">
            <Field label={t("ds.split")}>
              <Select
                label={t("ds.split")}
                value={item.split}
                onChange={(s) => save({ split: s })}
                options={[
                  { value: "train", label: t("ds.filter.train") },
                  { value: "validation", label: t("ds.filter.validation") },
                ]}
              />
            </Field>
            <Field label={t("ds.use")}>
              <Select
                label={t("ds.use")}
                value={item.excluded ? "excluded" : "included"}
                onChange={(v) => save({ excluded: v === "excluded" })}
                options={[
                  { value: "included", label: t("ds.include") },
                  { value: "excluded", label: t("ds.exclude") },
                ]}
              />
            </Field>
          </div>
          <Notice>{t("ds.originalKept")}</Notice>
        </div>
      </div>
    </Modal>
  );
}

// --- import from the library / an AI model ---------------------------------------------

function LibraryImport({ lora, onClose }: { lora: LoraDetail; onClose: () => void }) {
  const t = useT();
  const images = useImages({ limit: 60, scope: "library", media: "image" });
  const add = useAddDatasetImages(lora.id);
  const [selected, setSelected] = useState<string[]>([]);
  const items = images.data?.pages.flatMap((p) => p.items) ?? [];
  return (
    <Modal
      open
      wide
      onClose={onClose}
      title={t("pm.fromLibraryTitle")}
      description={t("ds.libraryBody")}
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>{t("common.cancel")}</Button>
          <Button
            variant="primary"
            disabled={!selected.length}
            loading={add.isPending}
            onClick={async () => {
              try {
                const res = await add.mutateAsync({ library_image_ids: selected });
                toast.success(t("ds.added", { n: res.added }));
                onClose();
              } catch (e) {
                toast.error((e as Error).message);
              }
            }}
          >
            {t("pm.addN", { n: selected.length })}
          </Button>
        </>
      }
    >
      {images.isLoading ? (
        <Skeleton style={{ height: 200 }} />
      ) : items.length === 0 ? (
        <EmptyState icon={<Images />} title={t("lib.emptyTitle")} />
      ) : (
        <div className="persona-library">
          {items.map((img) => (
            <button
              key={img.id}
              type="button"
              className="persona-library-item"
              aria-pressed={selected.includes(img.id)}
              onClick={() => setSelected((s) => (s.includes(img.id) ? s.filter((x) => x !== img.id) : [...s, img.id]))}
              title={img.prompt}
            >
              <img src={img.thumb_url} alt="" loading="lazy" />
            </button>
          ))}
        </div>
      )}
    </Modal>
  );
}

function PersonaImport({ lora, onClose }: { lora: LoraDetail; onClose: () => void }) {
  const t = useT();
  const personas = usePersonas();
  const add = useAddDatasetImages(lora.id);
  const [chosen, setChosen] = useState(lora.persona_id ?? "");
  const persona = personas.data?.items.find((p) => p.id === chosen);
  return (
    <Modal
      open
      onClose={onClose}
      title={t("ds.fromPersona")}
      description={t("ds.personaBody")}
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>{t("common.cancel")}</Button>
          <Button
            variant="primary"
            disabled={!persona}
            loading={add.isPending}
            onClick={async () => {
              try {
                const res = await add.mutateAsync({ persona_id: chosen });
                toast.success(t("ds.added", { n: res.added }));
                onClose();
              } catch (e) {
                toast.error((e as Error).message);
              }
            }}
          >
            {t("pm.addN", { n: persona?.image_count ?? 0 })}
          </Button>
        </>
      }
    >
      <Field label={t("pm.section")}>
        <Select
          label={t("pm.section")}
          value={chosen}
          placeholder={t("pm.choose")}
          onChange={setChosen}
          options={(personas.data?.items ?? []).map((p) => ({ value: p.id, label: `${p.name} · ${p.image_count}` }))}
        />
      </Field>
      {persona ? (
        <div className="persona-strip" style={{ marginTop: 12 }}>
          {persona.images.map((img) => (
            <span key={img.id} className="persona-ref" data-state="sent">
              <img src={img.thumb_url} alt="" />
            </span>
          ))}
        </div>
      ) : null}
    </Modal>
  );
}
