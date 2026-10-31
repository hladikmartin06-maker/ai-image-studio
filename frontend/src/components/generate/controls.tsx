"use client";

import {
  Clipboard,
  Dices,
  Eraser,
  History,
  ImagePlus,
  Lightbulb,
  RefreshCcw,
  Trash2,
  Upload,
} from "lucide-react";
import { type DragEvent, useRef, useState } from "react";
import { toast } from "sonner";

import { copyText } from "@/components/images/actions";
import {
  Badge,
  IconButton,
  MenuItem,
  Popover,
  Segmented,
  Spinner,
  Switch,
  Tooltip,
} from "@/components/ui";
import { ApiError } from "@/lib/api";
import { formatRelative } from "@/lib/format";
import { useI18n, useT } from "@/lib/i18n";
import { usePromptHistory, useUploadReference } from "@/lib/queries";
import type { ModelInfo, ModelKey, ProviderKey, ReferenceImage, Variant } from "@/lib/types";

// --- Model picker ----------------------------------------------------------------------------

export function ModelPicker({
  models,
  value,
  onChange,
  connected,
}: {
  models: ModelInfo[];
  value: ModelKey;
  onChange: (key: ModelKey) => void;
  connected: ProviderKey[];
}) {
  const t = useT();
  return (
    <div className="models" role="radiogroup" aria-label={t("gen.model")}>
      {models.map((m) => (
        <button
          key={m.key}
          type="button"
          role="radio"
          aria-checked={m.key === value}
          className="model-card"
          onClick={() => onChange(m.key)}
        >
          <div className="row" style={{ gap: 6 }}>
            <span className="name grow truncate">{m.name}</span>
            <Badge tone={m.content === "sfw" ? undefined : "accent"} title={t(`content.${m.content}.hint`)}>
              {t(`content.${m.content}`)}
            </Badge>
          </div>
          <div className="desc clamp-2">{t(`model.${m.key}.short`)}</div>
          <div className="avail">
            {(["wavespeed", "atlas"] as ProviderKey[]).map((p) => (
              <span
                key={p}
                className="provider-dot"
                data-off={!m.providers.includes(p) || undefined}
                title={
                  m.providers.includes(p)
                    ? connected.includes(p)
                      ? t("gen.availableConnected")
                      : t("gen.availableNotConnected")
                    : t("gen.notAvailable")
                }
              >
                {p === "wavespeed" ? "WaveSpeed" : "Atlas"}
              </span>
            ))}
          </div>
        </button>
      ))}
    </div>
  );
}

// --- Prompt editor --------------------------------------------------------------------------

export function PromptEditor({
  value,
  onChange,
  max,
  onSubmit,
}: {
  value: string;
  onChange: (v: string) => void;
  max: number;
  onSubmit: () => void;
}) {
  const { t, locale } = useI18n();
  const [historyOpen, setHistoryOpen] = useState(false);
  const history = usePromptHistory(historyOpen);
  const over = value.length > max;

  return (
    <div className="prompt-editor">
      <label htmlFor="prompt" className="sr-only">
        {t("gen.prompt")}
      </label>
      <textarea
        id="prompt"
        value={value}
        placeholder={t("gen.promptPlaceholder")}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
            e.preventDefault();
            onSubmit();
          }
        }}
        spellCheck
        autoFocus
      />
      <div className="prompt-tools">
        <span className="tiny" style={{ color: over ? "var(--danger)" : "var(--text-3)" }}>
          {value.length.toLocaleString(locale)} / {max.toLocaleString(locale)}
        </span>
        <span className="spacer" />
        <Popover
          align="end"
          width={360}
          trigger={({ toggle, ref }) => (
            <span ref={ref}>
              <IconButton
                label={t("gen.promptHistory")}
                onClick={() => {
                  setHistoryOpen(true);
                  toggle();
                }}
              >
                <History />
              </IconButton>
            </span>
          )}
        >
          {(close) => (
            <>
              <div className="menu-heading eyebrow">{t("gen.promptHistory")}</div>
              {history.isLoading ? (
                <div className="row" style={{ padding: 14, justifyContent: "center" }}>
                  <Spinner />
                </div>
              ) : history.data?.items.length ? (
                history.data.items.map((item) => (
                  <MenuItem
                    key={item.prompt}
                    onClick={() => {
                      onChange(item.prompt);
                      close();
                    }}
                  >
                    <div className="clamp-2 small">{item.prompt}</div>
                    <div className="tiny subtle">{formatRelative(item.last_used, locale)}</div>
                  </MenuItem>
                ))
              ) : (
                <div className="small subtle" style={{ padding: "8px 10px 12px" }}>
                  {t("gen.noPromptHistory")}
                </div>
              )}
            </>
          )}
        </Popover>
        <IconButton
          label={t("common.copyPrompt")}
          disabled={!value}
          onClick={async () => {
            if (await copyText(value)) toast.success(t("common.promptCopied"));
          }}
        >
          <Clipboard />
        </IconButton>
        <IconButton label={t("gen.clearPrompt")} disabled={!value} onClick={() => onChange("")}>
          <Eraser />
        </IconButton>
      </div>
    </div>
  );
}

// --- Reference images ------------------------------------------------------------------------

const ACCEPT = ["image/png", "image/jpeg", "image/webp"];
const MAX_MB = 10;

export function ReferenceImages({
  refs,
  onChange,
  max,
}: {
  refs: ReferenceImage[];
  onChange: (refs: ReferenceImage[]) => void;
  max: number;
}) {
  const t = useT();
  const upload = useUploadReference();
  const input = useRef<HTMLInputElement>(null);
  const replaceIndex = useRef<number | null>(null);
  const [over, setOver] = useState(false);
  const [pending, setPending] = useState(0);

  const validate = (file: File) => {
    if (!ACCEPT.includes(file.type)) return t("refs.badType");
    if (file.size > MAX_MB * 1024 * 1024) return t("refs.tooLarge", { mb: MAX_MB });
    return null;
  };

  const addFiles = async (files: File[]) => {
    const slot = replaceIndex.current;
    replaceIndex.current = null;
    const room = slot !== null ? 1 : Math.max(0, max - refs.length);
    if (room === 0) {
      toast.error(t("refs.limit", { n: max }));
      return;
    }
    const chosen = files.slice(0, room);
    if (files.length > room) toast.warning(t("refs.limit", { n: max }));
    let next = [...refs];
    for (const file of chosen) {
      const problem = validate(file);
      if (problem) {
        toast.error(`${file.name}: ${problem}`);
        continue;
      }
      setPending((n) => n + 1);
      try {
        const ref = await upload.mutateAsync(file);
        if (slot !== null) next = next.map((r, i) => (i === slot ? ref : r));
        else next = [...next, ref];
        onChange(next);
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

  return (
    <div
      className="col"
      style={{ gap: 10 }}
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
      {refs.length || pending ? (
        <div className="refs">
          {refs.map((ref, i) => (
            <div className="ref" key={ref.id}>
              <img src={ref.thumb_url} alt={ref.original_name || t("refs.reference")} />
              <div className="ref-actions">
                <IconButton
                  label={t("refs.replace")}
                  onMedia
                  onClick={() => {
                    replaceIndex.current = i;
                    input.current?.click();
                  }}
                >
                  <RefreshCcw />
                </IconButton>
                <IconButton
                  label={t("refs.remove")}
                  onMedia
                  onClick={() => onChange(refs.filter((r) => r.id !== ref.id))}
                >
                  <Trash2 />
                </IconButton>
              </div>
            </div>
          ))}
          {Array.from({ length: pending }, (_, i) => (
            <div className="ref ref-uploading" key={`p${i}`}>
              <Spinner />
            </div>
          ))}
          {refs.length + pending < max ? (
            <Tooltip label={t("refs.add")}>
              <button
                type="button"
                className="ref dropzone"
                style={{ justifyContent: "center", padding: 0 }}
                data-over={over || undefined}
                aria-label={t("refs.add")}
                onClick={() => input.current?.click()}
              >
                <ImagePlus />
              </button>
            </Tooltip>
          ) : null}
        </div>
      ) : (
        <button
          type="button"
          className="dropzone"
          data-over={over || undefined}
          onClick={() => input.current?.click()}
        >
          <Upload />
          <div className="grow">
            <div style={{ color: "var(--text)", fontWeight: 500 }}>{t("refs.drop")}</div>
            <div className="tiny subtle">{t("refs.formats", { n: max, mb: MAX_MB })}</div>
          </div>
        </button>
      )}
    </div>
  );
}

// --- Settings controls -------------------------------------------------------------------------

export interface DraftSettings {
  aspect_ratio?: string;
  resolution?: string;
  output_format?: string;
  seed?: string;
  thinking_mode?: boolean;
}

function RatioGlyph({ ratio }: { ratio: string }) {
  if (ratio === "match") return null;
  const [w, h] = ratio.split(":").map(Number);
  const scale = 14 / Math.max(w, h);
  return <span className="ratio-glyph" style={{ width: w * scale, height: h * scale }} aria-hidden />;
}

export function SettingsControls({
  variant,
  value,
  onChange,
  count,
  onCount,
  maxImages,
}: {
  variant: Variant;
  value: DraftSettings;
  onChange: (patch: DraftSettings) => void;
  count: number;
  onCount: (n: number) => void;
  maxImages: number;
}) {
  const t = useT();
  const c = variant.controls;
  const ratios = variant.allow_match_input ? ["match", ...variant.aspect_ratios] : variant.aspect_ratios;

  let sizeLabel: string | null = null;
  if (variant.size_mode === "pixels" && value.resolution && value.aspect_ratio && value.aspect_ratio !== "match") {
    const px = variant.pixel_presets[value.resolution]?.[value.aspect_ratio];
    if (px) sizeLabel = `${px[0]} × ${px[1]} px`;
  } else if (variant.size_mode === "tier" && value.resolution) {
    sizeLabel = t("gen.tierSize", { tier: value.resolution });
  } else if (value.aspect_ratio === "match") {
    sizeLabel = t("gen.followsInput");
  }

  return (
    <div className="col" style={{ gap: 20 }}>
      {c.includes("aspect_ratio") ? (
        <div className="field">
          <div className="label">{t("gen.aspectRatio")}</div>
          <div className="chips" role="group" aria-label={t("gen.aspectRatio")}>
            {ratios.map((r) => (
              <button
                key={r}
                type="button"
                className="chip"
                aria-pressed={value.aspect_ratio === r}
                onClick={() => onChange({ aspect_ratio: r })}
              >
                <RatioGlyph ratio={r} />
                {r === "match" ? t("gen.matchInput") : r}
              </button>
            ))}
          </div>
        </div>
      ) : null}

      {c.includes("resolution") && !(variant.size_mode === "pixels" && value.aspect_ratio === "match") ? (
        <div className="field">
          <div className="row">
            <div className="label grow">{t("gen.resolution")}</div>
            {sizeLabel ? <span className="tiny subtle">{sizeLabel}</span> : null}
          </div>
          <Segmented
            block
            label={t("gen.resolution")}
            value={value.resolution}
            onChange={(v) => onChange({ resolution: v })}
            options={variant.resolutions.map((r) => ({ value: r, label: r.toUpperCase() }))}
          />
        </div>
      ) : sizeLabel ? (
        <div className="hint">{sizeLabel}</div>
      ) : null}

      <div className="field">
        <div className="row">
          <div className="label grow">{t("gen.imageCount")}</div>
          <span className="tiny subtle">
            {variant.batched ? t("gen.oneRequest") : count > 1 ? t("gen.separateRequests", { n: count }) : ""}
          </span>
        </div>
        <Segmented
          block
          label={t("gen.imageCount")}
          value={count}
          onChange={onCount}
          options={Array.from({ length: maxImages }, (_, i) => ({ value: i + 1, label: String(i + 1) }))}
        />
      </div>

      {c.includes("output_format") ? (
        <div className="field">
          <div className="label">{t("gen.outputFormat")}</div>
          <Segmented
            block
            label={t("gen.outputFormat")}
            value={value.output_format}
            onChange={(v) => onChange({ output_format: v })}
            options={variant.output_formats.map((f) => ({ value: f, label: f.toUpperCase() }))}
          />
        </div>
      ) : null}

      {c.includes("seed") ? (
        <div className="field">
          <label className="label" htmlFor="seed">
            {t("gen.seed")}
          </label>
          <div className="input-wrap">
            <input
              id="seed"
              className="input"
              inputMode="numeric"
              placeholder={t("gen.seedRandom")}
              value={value.seed ?? ""}
              onChange={(e) => onChange({ seed: e.target.value.replace(/[^\d-]/g, "").slice(0, 11) })}
              style={{ paddingRight: 72 }}
            />
            <span className="input-action row gap-4">
              <IconButton
                label={t("gen.randomSeed")}
                onClick={() => onChange({ seed: String(Math.floor(Math.random() * 2_147_483_647)) })}
              >
                <Dices />
              </IconButton>
              <IconButton label={t("gen.clearSeed")} disabled={!value.seed} onClick={() => onChange({ seed: "" })}>
                <Eraser />
              </IconButton>
            </span>
          </div>
          <div className="hint">{t("gen.seedHint")}</div>
        </div>
      ) : null}

      {c.includes("thinking_mode") ? (
        <div className="row" style={{ gap: 12 }}>
          <div className="grow">
            <div className="label" style={{ color: "var(--text)" }}>
              <Lightbulb size={14} /> {t("gen.thinkingMode")}
            </div>
            <div className="hint">{t("gen.thinkingHint")}</div>
          </div>
          <Switch
            label={t("gen.thinkingMode")}
            checked={value.thinking_mode ?? true}
            onChange={(v) => onChange({ thinking_mode: v })}
          />
        </div>
      ) : null}

      {variant.size_mode === "tier" ? <div className="hint">{t("gen.noteTier")}</div> : null}
      {variant.mode === "image-edit" && !c.includes("aspect_ratio") && !c.includes("resolution") ? (
        <div className="hint">{t("gen.followsInput")}</div>
      ) : null}
    </div>
  );
}
