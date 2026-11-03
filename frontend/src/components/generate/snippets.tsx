"use client";

import { Bookmark, Plus, Trash2 } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

import { Button, IconButton, MenuItem, Popover, Spinner, Switch } from "@/components/ui";
import { ApiError } from "@/lib/api";
import { useT } from "@/lib/i18n";
import { useCreateSnippet, useDeleteSnippet, useSnippets } from "@/lib/queries";

/** Prompt snippets: saved pieces of text (lighting, outfit, place, …) for every AI model or
 * only for the chosen one; a click adds the text to the prompt. */
export function SnippetPicker({
  prompt,
  onInsert,
  personaId,
  personaName,
}: {
  prompt: string;
  onInsert: (text: string) => void;
  personaId?: string | null;
  personaName?: string | null;
}) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const snippets = useSnippets(open);
  const create = useCreateSnippet();
  const remove = useDeleteSnippet();
  const [name, setName] = useState("");
  const [text, setText] = useState("");
  const [onlyPersona, setOnlyPersona] = useState(false);
  const items = (snippets.data?.items ?? []).filter((s) => !s.persona_id || s.persona_id === personaId);

  const save = async () => {
    try {
      await create.mutateAsync({
        name: name.trim(),
        text: text.trim(),
        persona_id: onlyPersona && personaId ? personaId : null,
      });
      setName("");
      setText("");
      toast.success(t("sn.saved"));
    } catch (e) {
      toast.error(e instanceof ApiError ? e.message : t("errors.generic"));
    }
  };

  return (
    <Popover
      align="end"
      width={360}
      trigger={({ toggle, ref }) => (
        <span ref={ref}>
          <IconButton
            label={t("sn.title")}
            onClick={() => {
              setOpen(true);
              setText((current) => current || prompt.trim().slice(0, 1000));
              toggle();
            }}
          >
            <Bookmark />
          </IconButton>
        </span>
      )}
    >
      {(close) => (
        <>
          <div className="menu-heading">
            <div className="eyebrow">{t("sn.title")}</div>
            <div className="tiny subtle">{t("sn.hint")}</div>
          </div>
          {snippets.isLoading ? (
            <div className="row" style={{ padding: 14, justifyContent: "center" }}>
              <Spinner />
            </div>
          ) : items.length ? (
            <div style={{ maxHeight: 260, overflowY: "auto" }}>
              {items.map((s) => (
                <div key={s.id} className="row" style={{ gap: 4 }}>
                  <div className="grow" style={{ minWidth: 0 }}>
                    <MenuItem
                      onClick={() => {
                        onInsert(s.text);
                        close();
                      }}
                    >
                      <div className="small" style={{ fontWeight: 600 }}>
                        {s.name}
                        {s.persona_id && personaName ? <span className="tiny subtle"> · {personaName}</span> : null}
                      </div>
                      <div className="clamp-2 tiny subtle">{s.text}</div>
                    </MenuItem>
                  </div>
                  <IconButton label={t("common.delete")} onClick={() => remove.mutate(s.id)}>
                    <Trash2 />
                  </IconButton>
                </div>
              ))}
            </div>
          ) : (
            <div className="small subtle" style={{ padding: "8px 10px 12px" }}>
              {t("sn.empty")}
            </div>
          )}
          <div className="menu-sep" />
          <div className="col" style={{ gap: 8, padding: 10 }}>
            <div className="eyebrow">{t("sn.new")}</div>
            <input
              className="input"
              placeholder={t("sn.name")}
              aria-label={t("sn.name")}
              value={name}
              maxLength={80}
              onChange={(e) => setName(e.target.value)}
            />
            <textarea
              className="textarea"
              rows={3}
              placeholder={t("sn.text")}
              aria-label={t("sn.text")}
              value={text}
              maxLength={1000}
              onChange={(e) => setText(e.target.value)}
            />
            {personaId ? (
              <div className="row" style={{ gap: 8 }}>
                <Switch checked={onlyPersona} onChange={setOnlyPersona} label={t("sn.onlyPersona")} />
                <span className="small">{t("sn.onlyPersona")}{personaName ? ` (${personaName})` : ""}</span>
              </div>
            ) : null}
            <Button
              size="sm"
              variant="primary"
              icon={<Plus />}
              loading={create.isPending}
              disabled={!name.trim() || !text.trim()}
              onClick={save}
            >
              {t("sn.save")}
            </Button>
          </div>
        </>
      )}
    </Popover>
  );
}
