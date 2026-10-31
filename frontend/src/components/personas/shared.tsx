"use client";

import { Plus, UserRound, UserRoundPlus } from "lucide-react";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { toast } from "sonner";

import { Button, Field, MenuItem, Modal, Popover, Spinner, Switch } from "@/components/ui";
import { ApiError } from "@/lib/api";
import { useT } from "@/lib/i18n";
import { useAddPersonaImages, useCreatePersona, usePersonas } from "@/lib/queries";
import type { Persona } from "@/lib/types";

export function personaInitials(name: string) {
  return name
    .split(/\s+/)
    .map((p) => p[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();
}

/** Square preview of an AI model: its primary reference image, else its initials. */
export function PersonaThumb({ persona, size = 36 }: { persona: Persona; size?: number }) {
  const image = persona.primary_image;
  return (
    <span className="persona-thumb" style={{ width: size, height: size }} aria-hidden>
      {image ? (
        <img src={image.thumb_url} alt="" loading="lazy" />
      ) : (
        <span style={{ fontSize: Math.max(10, Math.round(size / 3)) }}>
          {personaInitials(persona.name) || <UserRound />}
        </span>
      )}
    </span>
  );
}

// --- Create --------------------------------------------------------------------------------

export function CreatePersonaModal({
  open,
  onClose,
  onCreated,
}: {
  open: boolean;
  onClose: () => void;
  onCreated: (persona: Persona) => void;
}) {
  // A fresh form every time the dialog opens.
  return open ? <CreatePersonaForm onClose={onClose} onCreated={onCreated} /> : null;
}

function CreatePersonaForm({
  onClose,
  onCreated,
}: {
  onClose: () => void;
  onCreated: (persona: Persona) => void;
}) {
  const t = useT();
  const create = useCreatePersona();
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [adult, setAdult] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    if (!name.trim()) {
      setError(t("pm.nameRequired"));
      return;
    }
    if (!adult) {
      setError(t("pm.adultRequired"));
      return;
    }
    setError(null);
    try {
      const persona = await create.mutateAsync({
        name: name.trim(),
        description: description.trim(),
        adult_confirmed: true,
      });
      toast.success(t("pm.created", { name: persona.name }));
      onCreated(persona);
      onClose();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : t("errors.generic"));
    }
  };

  return (
    <Modal
      open
      onClose={onClose}
      icon={<UserRoundPlus />}
      title={t("pm.newTitle")}
      description={t("pm.newBody")}
      footer={
        <>
          <Button variant="ghost" onClick={onClose}>
            {t("common.cancel")}
          </Button>
          <Button variant="primary" loading={create.isPending} disabled={!name.trim() || !adult} onClick={submit}>
            {t("pm.create")}
          </Button>
        </>
      }
    >
      <form
        className="col gap-16"
        onSubmit={(e) => {
          e.preventDefault();
          void submit();
        }}
      >
        <Field label={t("pm.name")} htmlFor="persona-name">
          <input
            id="persona-name"
            className="input"
            value={name}
            maxLength={80}
            autoFocus
            placeholder={t("pm.namePlaceholder")}
            onChange={(e) => setName(e.target.value)}
          />
        </Field>
        <Field label={t("pm.description")} hint={t("pm.descriptionHint")} htmlFor="persona-desc">
          <textarea
            id="persona-desc"
            className="textarea"
            rows={3}
            maxLength={1500}
            value={description}
            placeholder={t("pm.descriptionPlaceholder")}
            onChange={(e) => setDescription(e.target.value)}
          />
        </Field>
        <div className="row" style={{ alignItems: "flex-start", gap: 12 }}>
          <Switch checked={adult} onChange={setAdult} label={t("pm.adultConfirm")} />
          <div className="grow">
            <div className="small" style={{ color: "var(--text)" }}>
              {t("pm.adultConfirm")}
            </div>
            <div className="hint">{t("pm.adultHint")}</div>
          </div>
        </div>
        {error ? <div className="error-text">{error}</div> : null}
        <button type="submit" hidden />
      </form>
    </Modal>
  );
}

// --- Add library images to an AI model ------------------------------------------------------

export function AddToPersona({
  imageIds,
  onDone,
  size = "sm",
}: {
  imageIds: string[];
  onDone?: () => void;
  size?: "sm" | "md";
}) {
  const t = useT();
  const router = useRouter();
  const [creating, setCreating] = useState(false);
  const personas = usePersonas("", true);
  const add = useAddPersonaImages();
  const max = personas.data?.max_images ?? 24;

  const run = async (persona: Persona) => {
    if (imageIds.length > max) {
      toast.error(t("pm.tooMany", { n: max }));
      return;
    }
    try {
      const res = await add.mutateAsync({ id: persona.id, libraryImageIds: imageIds });
      const open = { label: t("common.view"), onClick: () => router.push(`/ai-models/${persona.id}`) };
      if (res.added) toast.success(t("pm.addedTo", { n: res.added, name: persona.name }), { action: open });
      else toast.info(t("pm.alreadyIn", { name: persona.name }), { action: open });
      onDone?.();
    } catch (e) {
      toast.error(e instanceof ApiError ? e.message : t("errors.generic"));
    }
  };

  const items = personas.data?.items ?? [];
  return (
    <>
      <Popover
        width={280}
        trigger={({ toggle, ref }) => (
          <button
            ref={ref}
            type="button"
            className={size === "sm" ? "btn btn-sm" : "btn"}
            onClick={toggle}
            disabled={add.isPending || imageIds.length === 0}
            aria-busy={add.isPending || undefined}
          >
            {add.isPending ? <span className="spinner" aria-hidden /> : <UserRoundPlus />}
            {t("pm.addToModel")}
          </button>
        )}
      >
        {(close) => (
          <>
            <div className="menu-heading">
              <div className="h3">{t("pm.addToModel")}</div>
              <div className="tiny subtle">{t("pm.addToModelHint", { n: imageIds.length })}</div>
            </div>
            <div className="menu-sep" />
            {personas.isLoading ? (
              <div className="row" style={{ justifyContent: "center", padding: 12 }}>
                <Spinner />
              </div>
            ) : (
              <div style={{ maxHeight: 280, overflowY: "auto" }}>
                {items.map((p) => (
                  <MenuItem
                    key={p.id}
                    icon={<PersonaThumb persona={p} size={24} />}
                    disabled={p.image_count >= p.max_images}
                    onClick={() => {
                      close();
                      void run(p);
                    }}
                  >
                    <span className="truncate" style={{ display: "block" }}>
                      {p.name}
                    </span>
                    <span className="tiny subtle">
                      {t("pm.imageCount", { n: p.image_count, max: p.max_images })}
                    </span>
                  </MenuItem>
                ))}
                {items.length === 0 ? <div className="tiny subtle" style={{ padding: "6px 12px" }}>{t("pm.noneYet")}</div> : null}
              </div>
            )}
            <div className="menu-sep" />
            <MenuItem
              icon={<Plus />}
              onClick={() => {
                close();
                setCreating(true);
              }}
            >
              {t("pm.newEllipsis")}
            </MenuItem>
          </>
        )}
      </Popover>
      <CreatePersonaModal open={creating} onClose={() => setCreating(false)} onCreated={(p) => void run(p)} />
    </>
  );
}
