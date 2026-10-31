"use client";

import { Check, Plus, Search, Star, UserRound, X } from "lucide-react";
import Link from "next/link";
import { useState } from "react";

import { PersonaThumb } from "@/components/personas/shared";
import { Button, EmptyState, IconButton, Modal, Notice, Skeleton, Switch } from "@/components/ui";
import { useDebounced } from "@/lib/hooks";
import { useT } from "@/lib/i18n";
import { usePersonas } from "@/lib/queries";
import type { Persona, PersonaImage } from "@/lib/types";

/** What the Generate draft remembers about the selected AI model. */
export interface PersonaDraft {
  id: string;
  // Images the user switched off (new images of the AI model are included by default).
  excluded: string[];
  useRefs: boolean;
  useDescription: boolean;
}

export function newPersonaDraft(id: string): PersonaDraft {
  return { id, excluded: [], useRefs: true, useDescription: true };
}

/**
 * The AI model's images that go with the next request: the ones the user kept, primary
 * first, within what the endpoint accepts after the user's own references.
 */
export function attachedPersonaImages(
  persona: Persona | undefined,
  draft: PersonaDraft | null,
  maxReferences: number | null,
  manualCount: number,
): { candidates: PersonaImage[]; attached: PersonaImage[] } {
  if (!persona || !draft || !draft.useRefs || maxReferences === null) return { candidates: [], attached: [] };
  const candidates = persona.images.filter((img) => !draft.excluded.includes(img.id));
  const room = Math.max(0, maxReferences - manualCount);
  return { candidates, attached: candidates.slice(0, room) };
}

export function PersonaSection({
  value,
  persona,
  loading,
  hasPersonas,
  onChange,
  attached,
  maxReferences,
  modelName,
  providerName,
  manualCount,
}: {
  value: PersonaDraft | null;
  persona: Persona | undefined;
  loading: boolean;
  hasPersonas: boolean;
  onChange: (value: PersonaDraft | null) => void;
  attached: PersonaImage[];
  /** null when this model takes no reference images with the provider. */
  maxReferences: number | null;
  modelName: string;
  providerName: string;
  manualCount: number;
}) {
  const t = useT();
  const [picking, setPicking] = useState(false);

  const picker = (
    <PersonaPickerModal
      open={picking}
      onClose={() => setPicking(false)}
      value={value?.id ?? null}
      onSelect={(p) => onChange(p ? (p.id === value?.id ? value : { ...newPersonaDraft(p.id) }) : null)}
    />
  );

  if (loading && value) return <Skeleton style={{ height: 76 }} />;

  if (!value || !persona) {
    return (
      <>
        {hasPersonas ? (
          <button type="button" className="dropzone" onClick={() => setPicking(true)}>
            <UserRound />
            <div className="grow">
              <div style={{ color: "var(--text)", fontWeight: 500 }}>{t("pm.choose")}</div>
              <div className="tiny subtle">{t("pm.chooseHint")}</div>
            </div>
          </button>
        ) : (
          <Link href="/ai-models" className="dropzone">
            <Plus />
            <div className="grow">
              <div style={{ color: "var(--text)", fontWeight: 500 }}>{t("pm.createFirst")}</div>
              <div className="tiny subtle">{t("pm.chooseHint")}</div>
            </div>
          </Link>
        )}
        {picker}
      </>
    );
  }

  const attachedIds = new Set(attached.map((i) => i.id));
  const toggle = (img: PersonaImage) => {
    const excluded = value.excluded.includes(img.id)
      ? value.excluded.filter((id) => id !== img.id)
      : [...value.excluded, img.id];
    onChange({ ...value, excluded });
  };
  const candidates = persona.images.filter((img) => !value.excluded.includes(img.id));
  const limited = value.useRefs && maxReferences !== null ? candidates.length - attached.length : 0;
  const identity =
    attached.length === 0 ? "" : attached.length === 1 ? persona.identity_text.one : persona.identity_text.many;

  return (
    <div className="persona-selected">
      <div className="row" style={{ gap: 12 }}>
        <PersonaThumb persona={persona} size={44} />
        <div className="grow" style={{ minWidth: 0 }}>
          <Link href={`/ai-models/${persona.id}`} className="persona-name truncate">
            {persona.name}
          </Link>
          <div className="tiny subtle truncate">
            {t("pm.imageCount", { n: persona.image_count, max: persona.max_images })}
          </div>
        </div>
        <Button size="sm" variant="ghost" onClick={() => setPicking(true)}>
          {t("pm.change")}
        </Button>
        <IconButton label={t("pm.removeFromDraft")} onClick={() => onChange(null)}>
          <X />
        </IconButton>
      </div>

      {persona.images.length === 0 ? (
        <div className="hint">
          {t("pm.noImagesYet")}{" "}
          <Link href={`/ai-models/${persona.id}`} className="link">
            {t("pm.addImages")}
          </Link>
        </div>
      ) : maxReferences === null ? (
        <Notice>{t("pm.refsUnsupported", { model: modelName, provider: providerName })}</Notice>
      ) : (
        <div className="col" style={{ gap: 8 }}>
          <div className="row" style={{ gap: 10 }}>
            <Switch
              checked={value.useRefs}
              onChange={(useRefs) => onChange({ ...value, useRefs })}
              label={t("pm.attachRefs")}
            />
            <span className="small grow">{t("pm.attachRefs")}</span>
            {value.useRefs ? (
              <span className="tiny subtle">
                {t("pm.sentCount", { n: attached.length, total: persona.images.length })}
              </span>
            ) : null}
          </div>
          {value.useRefs ? (
            <div className="persona-strip" role="group" aria-label={t("pm.attachRefs")}>
              {persona.images.map((img) => {
                const off = value.excluded.includes(img.id);
                const sent = attachedIds.has(img.id);
                return (
                  <button
                    key={img.id}
                    type="button"
                    className="persona-ref"
                    aria-pressed={!off}
                    data-state={off ? "off" : sent ? "sent" : "limited"}
                    title={off ? t("pm.refOff") : sent ? t("pm.refSent") : t("pm.refLimited")}
                    onClick={() => toggle(img)}
                  >
                    <img src={img.thumb_url} alt="" loading="lazy" />
                    {img.is_primary ? (
                      <span className="persona-ref-star" aria-label={t("pm.primary")}>
                        <Star />
                      </span>
                    ) : null}
                    {sent ? (
                      <span className="persona-ref-check" aria-hidden>
                        <Check />
                      </span>
                    ) : null}
                  </button>
                );
              })}
            </div>
          ) : null}
          {value.useRefs && limited > 0 ? (
            <div className="hint">
              {manualCount > 0
                ? t("pm.limitWithOwn", { n: maxReferences, model: modelName, own: manualCount })
                : t("pm.limit", { n: maxReferences, model: modelName, provider: providerName })}
            </div>
          ) : null}
        </div>
      )}

      <div className="col" style={{ gap: 8 }}>
        <div className="row" style={{ gap: 10 }}>
          <Switch
            checked={value.useDescription}
            onChange={(useDescription) => onChange({ ...value, useDescription })}
            label={t("pm.useDescription")}
          />
          <span className="small grow">{t("pm.useDescription")}</span>
        </div>
        {value.useDescription ? (
          persona.prompt_text || identity ? (
            <div className="persona-prompt">
              <span className="tiny subtle">{t("pm.addedToPrompt")}</span>
              <div>{[persona.prompt_text, identity].filter(Boolean).join(" ")}</div>
            </div>
          ) : (
            <div className="hint">
              {t("pm.noDescriptionYet")}{" "}
              <Link href={`/ai-models/${persona.id}`} className="link">
                {t("pm.editProfile")}
              </Link>
            </div>
          )
        ) : null}
      </div>
      {picker}
    </div>
  );
}

// --- Picker ------------------------------------------------------------------------------------

export function PersonaPickerModal({
  open,
  onClose,
  value,
  onSelect,
}: {
  open: boolean;
  onClose: () => void;
  value: string | null;
  onSelect: (persona: Persona | null) => void;
}) {
  return open ? <PickerBody onClose={onClose} value={value} onSelect={onSelect} /> : null;
}

function PickerBody({
  onClose,
  value,
  onSelect,
}: {
  onClose: () => void;
  value: string | null;
  onSelect: (persona: Persona | null) => void;
}) {
  const t = useT();
  const [search, setSearch] = useState("");
  const q = useDebounced(search.trim(), 250);
  const list = usePersonas(q);
  const items = list.data?.items ?? [];

  const choose = (p: Persona | null) => {
    onSelect(p);
    onClose();
  };

  return (
    <Modal
      open
      wide
      onClose={onClose}
      title={t("pm.chooseTitle")}
      description={t("pm.chooseBody")}
      footer={
        <>
          <Link href="/ai-models" className="btn btn-ghost" onClick={onClose}>
            {t("pm.manage")}
          </Link>
          <span className="spacer" />
          {value ? (
            <Button variant="ghost" onClick={() => choose(null)}>
              {t("pm.useNone")}
            </Button>
          ) : null}
          <Button onClick={onClose}>{t("common.close")}</Button>
        </>
      }
    >
      <div className="col gap-16">
        <div className="input-wrap">
          <Search />
          <input
            className="input"
            value={search}
            autoFocus
            placeholder={t("pm.searchPlaceholder")}
            aria-label={t("common.search")}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>
        {list.isLoading ? (
          <div className="persona-pick-grid">
            {Array.from({ length: 6 }, (_, i) => (
              <Skeleton key={i} style={{ aspectRatio: "3 / 4" }} />
            ))}
          </div>
        ) : list.isError ? (
          <Notice tone="danger">{(list.error as Error).message}</Notice>
        ) : items.length === 0 ? (
          <EmptyState
            icon={q ? <Search /> : <UserRound />}
            title={q ? t("pm.noMatches") : t("pm.emptyTitle")}
            description={q ? undefined : t("pm.emptyBody")}
            action={
              q ? undefined : (
                <Link href="/ai-models" className="btn btn-primary btn-sm" onClick={onClose}>
                  <Plus /> {t("pm.new")}
                </Link>
              )
            }
          />
        ) : (
          <div className="persona-pick-grid" role="radiogroup" aria-label={t("pm.chooseTitle")}>
            {items.map((p) => (
              <button
                key={p.id}
                type="button"
                role="radio"
                aria-checked={p.id === value}
                className="persona-pick"
                onClick={() => choose(p)}
              >
                <span className="persona-pick-img">
                  {p.primary_image ? (
                    <img src={p.primary_image.thumb_url} alt="" loading="lazy" />
                  ) : (
                    <UserRound />
                  )}
                  {p.id === value ? (
                    <span className="persona-pick-check" aria-hidden>
                      <Check />
                    </span>
                  ) : null}
                </span>
                <span className="persona-pick-name truncate">{p.name}</span>
                <span className="tiny subtle truncate">
                  {t("pm.imageCount", { n: p.image_count, max: p.max_images })}
                </span>
              </button>
            ))}
          </div>
        )}
      </div>
    </Modal>
  );
}
