"use client";

import { MessageSquare, MessageSquarePlus, Pencil, Search, Trash2, X } from "lucide-react";
import { useEffect, useState } from "react";
import { toast } from "sonner";

import { ChatPanel } from "@/components/assistant/chat";
import { Button, IconButton, MenuItem, Modal, Popover, Skeleton, useConfirm } from "@/components/ui";
import { api } from "@/lib/api";
import { formatRelative } from "@/lib/format";
import { useDebounced } from "@/lib/hooks";
import { useI18n } from "@/lib/i18n";
import { aiKeys, useConversations } from "@/lib/queries";
import { useQueryClient } from "@tanstack/react-query";

export default function AssistantPage() {
  const { t, locale } = useI18n();
  const qc = useQueryClient();
  const confirm = useConfirm();
  const [search, setSearch] = useState("");
  const q = useDebounced(search.trim(), 300);
  const list = useConversations(q);
  const [active, setActive] = useState<string | null>(null);
  const [renaming, setRenaming] = useState<{ id: string; title: string } | null>(null);

  useEffect(() => {
    const id = new URLSearchParams(window.location.search).get("c");
    // eslint-disable-next-line react-hooks/set-state-in-effect -- deep link from the drawer
    if (id) setActive(id);
  }, []);

  const select = (id: string | null) => {
    setActive(id);
    window.history.replaceState(null, "", id ? `/assistant?c=${id}` : "/assistant");
  };

  const saveRename = async () => {
    if (!renaming) return;
    await api(`/api/ai/conversations/${renaming.id}`, {
      method: "PATCH",
      body: { title: renaming.title },
    });
    setRenaming(null);
    qc.invalidateQueries({ queryKey: ["ai", "conversations"] });
  };

  const remove = async (id: string) => {
    const ok = await confirm({
      title: t("chat.deleteTitle"),
      description: t("chat.deleteBody"),
      confirmLabel: t("common.delete"),
      danger: true,
    });
    if (!ok) return;
    await api(`/api/ai/conversations/${id}`, { method: "DELETE" });
    qc.invalidateQueries({ queryKey: ["ai", "conversations"] });
    qc.removeQueries({ queryKey: aiKeys.conversation(id) });
    if (active === id) select(null);
    toast.success(t("chat.deleted"));
  };

  const items = list.data?.items ?? [];

  return (
    <div className="assistant-layout">
      <aside className="assistant-list" aria-label={t("chat.history")}>
        <div className="row" style={{ padding: "18px 14px 10px" }}>
          <h1 className="h2 grow">{t("nav.assistant")}</h1>
          <Button size="sm" variant="primary" icon={<MessageSquarePlus />} onClick={() => select(null)}>
            {t("chat.new")}
          </Button>
        </div>
        <div style={{ padding: "0 14px 10px" }}>
          <div className="input-wrap">
            <Search />
            <input
              className="input"
              placeholder={t("chat.search")}
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              aria-label={t("chat.search")}
            />
            {search ? (
              <span className="input-action">
                <IconButton label={t("common.clear")} onClick={() => setSearch("")}>
                  <X />
                </IconButton>
              </span>
            ) : null}
          </div>
        </div>
        <div className="assistant-items">
          {list.isLoading ? (
            [0, 1, 2].map((i) => <Skeleton key={i} style={{ height: 44, margin: "4px 10px" }} />)
          ) : items.length === 0 ? (
            <div className="small subtle" style={{ padding: "12px 16px" }}>
              {q ? t("chat.noMatches") : t("chat.noConversations")}
            </div>
          ) : (
            items.map((c) => (
              <div
                key={c.id}
                className="conv-item"
                data-active={c.id === active || undefined}
                role="button"
                tabIndex={0}
                onClick={() => select(c.id)}
                onKeyDown={(e) => e.key === "Enter" && select(c.id)}
              >
                <MessageSquare size={15} className="subtle" />
                <div className="grow" style={{ minWidth: 0 }}>
                  <div className="truncate small" style={{ fontWeight: 500 }}>
                    {c.title || t("chat.untitled")}
                  </div>
                  <div className="tiny subtle">{formatRelative(c.updated_at, locale)}</div>
                </div>
                <span onClick={(e) => e.stopPropagation()}>
                  <Popover
                    align="end"
                    width={180}
                    trigger={({ toggle, ref }) => (
                      <span ref={ref}>
                        <IconButton label={t("chat.actions")} onClick={toggle}>
                          <Pencil />
                        </IconButton>
                      </span>
                    )}
                  >
                    {(close) => (
                      <>
                        <MenuItem
                          icon={<Pencil />}
                          onClick={() => {
                            close();
                            setRenaming({ id: c.id, title: c.title });
                          }}
                        >
                          {t("chat.rename")}
                        </MenuItem>
                        <MenuItem
                          icon={<Trash2 />}
                          danger
                          onClick={() => {
                            close();
                            void remove(c.id);
                          }}
                        >
                          {t("common.delete")}
                        </MenuItem>
                      </>
                    )}
                  </Popover>
                </span>
              </div>
            ))
          )}
        </div>
      </aside>
      <section className="assistant-main">
        <ChatPanel conversationId={active} onConversationChange={select} />
      </section>
      <Modal
        open={renaming !== null}
        onClose={() => setRenaming(null)}
        title={t("chat.rename")}
        footer={
          <>
            <Button variant="ghost" onClick={() => setRenaming(null)}>
              {t("common.cancel")}
            </Button>
            <Button variant="primary" onClick={saveRename}>
              {t("common.save")}
            </Button>
          </>
        }
      >
        <input
          className="input"
          value={renaming?.title ?? ""}
          maxLength={200}
          aria-label={t("chat.rename")}
          onChange={(e) => setRenaming((r) => (r ? { ...r, title: e.target.value } : r))}
          onKeyDown={(e) => e.key === "Enter" && saveRename()}
        />
      </Modal>
    </div>
  );
}
