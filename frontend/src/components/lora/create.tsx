"use client";

import { Copy, FileUp, Sparkles } from "lucide-react";
import { useRef, useState } from "react";
import { toast } from "sonner";

import { Button, Field, Modal, Notice, Segmented, Select, Switch } from "@/components/ui";
import { ApiError, uploadInChunks } from "@/lib/api";
import { useT } from "@/lib/i18n";
import { useCopyLora, useCreateLora, useImportLora, usePersonas } from "@/lib/queries";
import type { LoraDetail } from "@/lib/types";

const TRIGGER = /^[A-Za-z0-9][A-Za-z0-9_-]{1,39}$/;

function suggestTrigger(name: string) {
  const letters = name.toLowerCase().replace(/[^a-z]/g, "");
  return letters ? `${letters.slice(0, 3)}${letters.length}x` : "";
}

function Common({
  name,
  setName,
  trigger,
  setTrigger,
  classWord,
  setClassWord,
  personaId,
  setPersonaId,
  adult,
  setAdult,
}: {
  name: string;
  setName: (v: string) => void;
  trigger: string;
  setTrigger: (v: string) => void;
  classWord: string;
  setClassWord: (v: string) => void;
  personaId: string;
  setPersonaId: (v: string) => void;
  adult: boolean;
  setAdult: (v: boolean) => void;
}) {
  const t = useT();
  const personas = usePersonas();
  return (
    <>
      <div className="form-grid">
        <Field label={t("ls.name")}>
          <input className="input" autoFocus value={name} maxLength={80} onChange={(e) => setName(e.target.value)} placeholder={t("ls.namePlaceholder")} />
        </Field>
        <Field
          label={t("ls.trigger")}
          hint={t("ls.triggerHint")}
          error={trigger && !TRIGGER.test(trigger) ? t("ls.triggerInvalid") : null}
          aside={
            name && !trigger ? (
              <button type="button" className="link tiny" onClick={() => setTrigger(suggestTrigger(name))}>
                {t("ls.suggest")}
              </button>
            ) : null
          }
        >
          <input className="input mono" value={trigger} maxLength={40} onChange={(e) => setTrigger(e.target.value.trim())} placeholder="lnx" />
        </Field>
        <Field label={t("ls.classWord")} hint={t("ls.classWordHint")}>
          <input className="input" value={classWord} maxLength={40} onChange={(e) => setClassWord(e.target.value)} placeholder={t("ls.classWordPlaceholder")} />
        </Field>
        <Field label={t("pm.section")} hint={t("ls.personaHint")}>
          <Select
            label={t("pm.section")}
            value={personaId}
            onChange={setPersonaId}
            options={[{ value: "", label: t("pm.useNone") }, ...(personas.data?.items ?? []).map((p) => ({ value: p.id, label: p.name }))]}
          />
        </Field>
      </div>
      <label className="row" style={{ gap: 12, alignItems: "flex-start" }}>
        <Switch checked={adult} onChange={setAdult} label={t("pm.adultConfirm")} />
        <div>
          <div className="small">{t("pm.adultConfirm")}</div>
          <div className="hint">{t("ls.adultHint")}</div>
        </div>
      </label>
    </>
  );
}

export function CreateLoraModal({ onClose, onCreated }: { onClose: () => void; onCreated: (l: LoraDetail) => void }) {
  const t = useT();
  const create = useCreateLora();
  const [name, setName] = useState("");
  const [trigger, setTrigger] = useState("");
  const [classWord, setClassWord] = useState("woman");
  const [personaId, setPersonaId] = useState("");
  const [variant, setVariant] = useState<"raw" | "turbo">("raw");
  const [adult, setAdult] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    setError(null);
    try {
      const lora = await create.mutateAsync({
        name: name.trim(),
        trigger_word: trigger,
        class_word: classWord.trim(),
        architecture: "krea2",
        base_variant: variant,
        persona_id: personaId || null,
        adult_confirmed: adult,
      });
      toast.success(t("ls.created", { name: lora.name }));
      onCreated(lora);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : t("errors.generic"));
    }
  };

  return (
    <Modal
      open
      wide
      onClose={onClose}
      icon={<Sparkles />}
      title={t("ls.createTitle")}
      description={t("ls.createBody")}
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>{t("common.cancel")}</Button>
          <Button variant="primary" onClick={submit} loading={create.isPending} disabled={!name.trim() || !TRIGGER.test(trigger) || !adult}>
            {t("ls.create")}
          </Button>
        </>
      }
    >
      <div className="col gap-16">
        <Common {...{ name, setName, trigger, setTrigger, classWord, setClassWord, personaId, setPersonaId, adult, setAdult }} />
        <Field label={t("ls.baseModel")}>
          <Segmented
            label={t("ls.baseModel")}
            value={variant}
            block
            onChange={setVariant}
            options={[
              { value: "raw", label: t("ls.baseRaw") },
              { value: "turbo", label: t("ls.baseTurbo") },
            ]}
          />
        </Field>
        <Notice>{variant === "raw" ? t("ls.baseRawHint") : t("ls.baseTurboHint")}</Notice>
        {error ? <div className="error-text">{error}</div> : null}
      </div>
    </Modal>
  );
}

/** A new LoRA project with another one's photos (and captions), e.g. on RAW instead of Turbo. */
export function CopyLoraModal({ source, onClose, onCreated }: { source: LoraDetail; onClose: () => void; onCreated: (l: LoraDetail) => void }) {
  const t = useT();
  const copy = useCopyLora(source.id);
  // Name and trigger follow the chosen base model until the user edits them.
  const suggest = (v: "raw" | "turbo") => {
    const same = v === source.base_variant;
    return {
      name: `${source.name} ${same ? "2" : v === "raw" ? "RAW" : "Turbo"}`.slice(0, 80),
      trigger: `${source.trigger_word}${same ? "2" : v}`.slice(0, 40),
    };
  };
  const [variant, setVariant] = useState<"raw" | "turbo">("raw");
  const [name, setName] = useState(() => suggest("raw").name);
  const [trigger, setTrigger] = useState(() => suggest("raw").trigger);
  const [classWord, setClassWord] = useState(source.class_word);
  const [personaId, setPersonaId] = useState(source.persona_id ?? "");
  const changeVariant = (v: "raw" | "turbo") => {
    const before = suggest(variant);
    const after = suggest(v);
    if (name === before.name) setName(after.name);
    if (trigger === before.trigger) setTrigger(after.trigger);
    setVariant(v);
  };
  const [captions, setCaptions] = useState(true);
  const [adult, setAdult] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const n = source.dataset.length;

  const submit = async () => {
    setError(null);
    try {
      const lora = await copy.mutateAsync({
        name: name.trim(),
        trigger_word: trigger,
        class_word: classWord.trim(),
        base_variant: variant,
        persona_id: personaId || null,
        captions,
        adult_confirmed: adult,
      });
      toast.success(t("ls.copiedLora", { name: lora.name, n: lora.dataset.length }));
      onCreated(lora);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : t("errors.generic"));
    }
  };

  return (
    <Modal
      open
      wide
      onClose={onClose}
      icon={<Copy />}
      title={t("ls.copyTitle")}
      description={t("ls.copyBody", { n })}
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>{t("common.cancel")}</Button>
          <Button variant="primary" onClick={submit} loading={copy.isPending} disabled={!n || !name.trim() || !TRIGGER.test(trigger) || !adult}>
            {t("ls.copyStart", { n })}
          </Button>
        </>
      }
    >
      <div className="col gap-16">
        <Common {...{ name, setName, trigger, setTrigger, classWord, setClassWord, personaId, setPersonaId, adult, setAdult }} />
        <Field label={t("ls.baseModel")}>
          <Segmented
            label={t("ls.baseModel")}
            value={variant}
            block
            onChange={changeVariant}
            options={[
              { value: "raw", label: t("ls.baseRaw") },
              { value: "turbo", label: t("ls.baseTurbo") },
            ]}
          />
        </Field>
        <Notice>{variant === "raw" ? t("ls.baseRawHint") : t("ls.baseTurboHint")}</Notice>
        <label className="row" style={{ gap: 12, alignItems: "flex-start" }}>
          <Switch checked={captions} onChange={setCaptions} label={t("ls.copyCaptions")} />
          <div>
            <div className="small">{t("ls.copyCaptions")}</div>
            <div className="hint">{t("ls.copyCaptionsHint", { old: source.trigger_word })}</div>
          </div>
        </label>
        {error ? <div className="error-text">{error}</div> : null}
      </div>
    </Modal>
  );
}

export function ImportLoraModal({ onClose, onCreated }: { onClose: () => void; onCreated: (l: LoraDetail) => void }) {
  const t = useT();
  const importer = useImportLora();
  const input = useRef<HTMLInputElement>(null);
  const [file, setFile] = useState<File | null>(null);
  const [uploadId, setUploadId] = useState<string | null>(null);
  const [progress, setProgress] = useState<number | null>(null);
  const [name, setName] = useState("");
  const [trigger, setTrigger] = useState("");
  const [classWord, setClassWord] = useState("woman");
  const [personaId, setPersonaId] = useState("");
  const [adult, setAdult] = useState(false);
  const [declareArch, setDeclareArch] = useState(false);
  const [variant, setVariant] = useState<"" | "raw" | "turbo">("");
  const [error, setError] = useState<{ code: string; message: string } | null>(null);

  const submit = async () => {
    if (!file) return;
    setError(null);
    try {
      let id = uploadId;
      if (!id) {
        setProgress(0);
        id = await uploadInChunks(file, setProgress);
        setUploadId(id);
        setProgress(null);
      }
      const lora = await importer.mutateAsync({
        upload_id: id,
        name: name.trim() || file.name.replace(/\.safetensors$/i, ""),
        trigger_word: trigger,
        class_word: classWord.trim(),
        persona_id: personaId || null,
        declared_architecture: declareArch ? "krea2" : null,
        declared_variant: variant || null,
        adult_confirmed: adult,
      });
      toast.success(t("ls.imported", { name: lora.name }));
      onCreated(lora);
    } catch (e) {
      setProgress(null);
      const err = e as ApiError;
      setError({ code: err.code ?? "", message: err.message });
      if (err.code === "invalid_file" || err.code === "lora_incompatible" || err.code === "not_a_lora") setUploadId(null);
    }
  };

  return (
    <Modal
      open
      wide
      onClose={onClose}
      icon={<FileUp />}
      title={t("ls.importTitle")}
      description={t("ls.importBody")}
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>{t("common.cancel")}</Button>
          <Button variant="primary" onClick={submit} loading={importer.isPending || progress !== null} disabled={!file || !TRIGGER.test(trigger) || !adult}>
            {progress !== null ? `${Math.round(progress * 100)}%` : t("ls.import")}
          </Button>
        </>
      }
    >
      <div className="col gap-16">
        <input ref={input} type="file" hidden accept=".safetensors" onChange={(e) => {
          const f = e.target.files?.[0] ?? null;
          setFile(f);
          setUploadId(null);
          if (f && !name) setName(f.name.replace(/\.safetensors$/i, ""));
        }} />
        <button type="button" className="dropzone" onClick={() => input.current?.click()}>
          <FileUp />
          <div className="grow">
            <div style={{ color: "var(--text)", fontWeight: 500 }}>{file ? file.name : t("ls.pickFile")}</div>
            <div className="tiny subtle">{t("ls.pickFileHint")}</div>
          </div>
        </button>
        <Common {...{ name, setName, trigger, setTrigger, classWord, setClassWord, personaId, setPersonaId, adult, setAdult }} />
        <div className="form-grid">
          <Field label={t("ls.trainedOn")} hint={t("ls.trainedOnHint")}>
            <Select
              label={t("ls.trainedOn")}
              value={variant}
              onChange={setVariant}
              options={[
                { value: "", label: t("ls.fromMetadata") },
                { value: "raw", label: "Krea 2 RAW" },
                { value: "turbo", label: "Krea 2 Turbo" },
              ]}
            />
          </Field>
          {error?.code === "architecture_unknown" || declareArch ? (
            <label className="row small" style={{ gap: 10 }}>
              <Switch checked={declareArch} onChange={setDeclareArch} label={t("ls.declareKrea")} />
              {t("ls.declareKrea")}
            </label>
          ) : null}
        </div>
        {error ? <Notice tone={error.code === "architecture_unknown" || error.code === "variant_unknown" ? "warning" : "danger"}>{error.message}</Notice> : null}
      </div>
    </Modal>
  );
}
