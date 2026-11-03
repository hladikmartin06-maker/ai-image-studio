"use client";

import { Loader2, Sparkles, Square } from "lucide-react";
import { useRef, useState } from "react";
import { toast } from "sonner";

import { Badge, Button, Field, Notice, Select, Switch } from "@/components/ui";
import { ApiError } from "@/lib/api";
import { useT } from "@/lib/i18n";
import { useCaptionItem, useUpdateDatasetItem, useUpdateLora } from "@/lib/queries";
import type { LoraDetail, LoraItem } from "@/lib/types";

function CaptionRow({ lora, item, writing }: { lora: LoraDetail; item: LoraItem; writing: boolean }) {
  const t = useT();
  const update = useUpdateDatasetItem(lora.id);
  const [text, setText] = useState(item.caption);
  const [last, setLast] = useState(item.caption);
  // A caption written by the AI (or elsewhere) replaces the local text when it changes.
  if (item.caption !== last) {
    setLast(item.caption);
    setText(item.caption);
  }
  return (
    <div className="caption-row" data-excluded={item.excluded || undefined} data-writing={writing || undefined}>
      <img src={item.thumb_url} alt="" loading="lazy" />
      <div className="col grow" style={{ gap: 4 }}>
        <textarea
          className="textarea"
          rows={2}
          value={text}
          maxLength={2000}
          placeholder={t("ds.noCaption")}
          onChange={(e) => setText(e.target.value)}
          onBlur={() => {
            if (text !== item.caption)
              update.mutate({ itemId: item.id, caption: text }, { onError: (e) => toast.error((e as Error).message) });
          }}
        />
        <div className="row tiny subtle" style={{ gap: 6 }}>
          {writing ? (
            <Badge tone="accent">
              <Loader2 size={11} className="spin" /> {t("cap.writing")}
            </Badge>
          ) : (
            <Badge tone={item.caption_source === "ai" ? "accent" : undefined}>{t(`ds.source.${item.caption_source}`)}</Badge>
          )}
          {item.split === "validation" ? <Badge>{t("ds.val")}</Badge> : null}
          {item.excluded ? <Badge>{t("ds.excluded")}</Badge> : null}
        </div>
      </div>
    </div>
  );
}

export function CaptionsTab({ lora }: { lora: LoraDetail }) {
  const t = useT();
  const updateLora = useUpdateLora(lora.id);
  const caption = useCaptionItem(lora.id);
  const [template, setTemplate] = useState(lora.caption_template);
  const [tier, setTier] = useState<"low" | "balanced" | "high">("low");
  const [overwrite, setOverwrite] = useState(false);
  const [running, setRunning] = useState<{ done: number; total: number } | null>(null);
  const [writing, setWriting] = useState<string | null>(null);
  const stop = useRef(false);
  const targets = lora.dataset.filter((i) => !i.excluded && (overwrite || !i.caption.trim()));

  const runAll = async () => {
    stop.current = false;
    setRunning({ done: 0, total: targets.length });
    let failed = 0;
    for (const [index, item] of targets.entries()) {
      if (stop.current) break;
      setWriting(item.id);
      try {
        await caption.mutateAsync({ itemId: item.id, tier, overwrite });
      } catch (e) {
        failed += 1;
        const err = e as ApiError;
        if (err.code === "not_connected" || err.code === "invalid_credentials") {
          toast.error(err.message);
          break;
        }
      }
      setRunning({ done: index + 1, total: targets.length });
    }
    setWriting(null);
    setRunning(null);
    await updateLora.mutateAsync({});
    if (failed) toast.warning(t("cap.someFailed", { n: failed }));
    else toast.success(t("cap.done"));
  };

  return (
    <div className="col gap-16">
      <div className="card card-pad col gap-12">
        <Field label={t("cap.template")} hint={t("cap.templateHint")}>
          <div className="row" style={{ gap: 8 }}>
            <input className="input mono grow" value={template} maxLength={300} onChange={(e) => setTemplate(e.target.value)} />
            <Button
              size="sm"
              disabled={template === lora.caption_template}
              loading={updateLora.isPending}
              onClick={() => updateLora.mutate({ caption_template: template }, { onSuccess: () => toast.success(t("ds.saved")) })}
            >
              {t("common.save")}
            </Button>
          </div>
        </Field>
        <div className="tiny subtle">
          {t("cap.preview")}: <span className="mono">{template.replace("{trigger}", lora.trigger_word).replace("{class}", lora.class_word).replace("{caption}", t("cap.exampleCaption"))}</span>
        </div>
        <div className="row wrap" style={{ gap: 12 }}>
          <Field label={t("cap.model")}>
            <Select
              label={t("cap.model")}
              value={tier}
              onChange={setTier}
              options={[
                { value: "low", label: t("ai.mode.low") },
                { value: "balanced", label: t("ai.mode.balanced") },
                { value: "high", label: t("ai.mode.high") },
              ]}
            />
          </Field>
          <label className="row small" style={{ gap: 8, alignSelf: "flex-end", paddingBottom: 8 }}>
            <Switch checked={overwrite} onChange={setOverwrite} label={t("cap.overwrite")} />
            {t("cap.overwrite")}
          </label>
          <span className="spacer" />
          {running ? (
            <Button icon={<Square />} onClick={() => (stop.current = true)}>
              {t("cap.stop", { done: running.done, total: running.total })}
            </Button>
          ) : (
            <Button variant="primary" icon={<Sparkles />} disabled={!targets.length} onClick={runAll} style={{ alignSelf: "flex-end" }}>
              {t("cap.run", { n: targets.length })}
            </Button>
          )}
        </div>
        <Notice>{t("cap.costNote")}</Notice>
      </div>
      <div className="col" style={{ gap: 8 }}>
        {lora.dataset.map((item) => (
          <CaptionRow key={item.id} lora={lora} item={item} writing={writing === item.id} />
        ))}
      </div>
    </div>
  );
}
