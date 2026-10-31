"use client";

import { Maximize2, MessageSquarePlus, Sparkles, X } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useCallback, useEffect, useState } from "react";
import { createPortal } from "react-dom";

import { ChatPanel } from "@/components/assistant/chat";
import { IconButton } from "@/components/ui";
import { useT } from "@/lib/i18n";

const KEY = "studio:assistant-drawer";
const OPEN_EVENT = "studio:open-assistant";

/** Opens the assistant drawer from anywhere (e.g. the Generate page's "Ask" button). */
export function openAssistant() {
  window.dispatchEvent(new CustomEvent(OPEN_EVENT));
}

/** A floating assistant available on every page, aware of the current studio settings. */
export function AssistantDrawer() {
  const t = useT();
  const pathname = usePathname();
  const [open, setOpen] = useState(false);
  const [conversation, setConversation] = useState<string | null>(null);

  useEffect(() => {
    try {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- restore the drawer's chat
      setConversation(sessionStorage.getItem(KEY));
    } catch {
      /* storage blocked */
    }
    const onOpen = () => setOpen(true);
    window.addEventListener(OPEN_EVENT, onOpen);
    return () => window.removeEventListener(OPEN_EVENT, onOpen);
  }, []);

  const change = useCallback((id: string | null) => {
    setConversation(id);
    try {
      if (id) sessionStorage.setItem(KEY, id);
      else sessionStorage.removeItem(KEY);
    } catch {
      /* storage blocked */
    }
  }, []);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  if (pathname.startsWith("/assistant")) return null;

  return (
    <>
      {!open ? (
        <button type="button" className="assistant-fab" onClick={() => setOpen(true)} aria-label={t("chat.open")}>
          <Sparkles />
          <span>{t("nav.assistant")}</span>
        </button>
      ) : null}
      {open
        ? createPortal(
            <aside className="assistant-drawer" role="dialog" aria-label={t("nav.assistant")}>
              <div className="assistant-drawer-head">
                <Sparkles size={16} color="var(--accent)" />
                <div className="h3 grow">{t("nav.assistant")}</div>
                <IconButton label={t("chat.new")} onClick={() => change(null)}>
                  <MessageSquarePlus />
                </IconButton>
                <Link
                  href={conversation ? `/assistant?c=${conversation}` : "/assistant"}
                  className="icon-btn"
                  aria-label={t("chat.openFull")}
                  title={t("chat.openFull")}
                  onClick={() => setOpen(false)}
                >
                  <Maximize2 />
                </Link>
                <IconButton label={t("common.close")} onClick={() => setOpen(false)}>
                  <X />
                </IconButton>
              </div>
              <ChatPanel conversationId={conversation} onConversationChange={change} compact />
            </aside>,
            document.body,
          )
        : null}
    </>
  );
}
