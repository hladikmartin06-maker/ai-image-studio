"use client";

import { useQueryClient } from "@tanstack/react-query";
import {
  ArrowUp,
  Bot,
  CircleAlert,
  Eye,
  EyeOff,
  Gauge,
  Lightbulb,
  Plug,
  RefreshCw,
  SlidersHorizontal,
  Sparkles,
  Square,
  Wand2,
  Wrench,
} from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";

import { Markdown } from "@/components/assistant/markdown";
import { Badge, Button, Notice, Segmented, Select, Skeleton, Tooltip, cx } from "@/components/ui";
import { api, ApiError } from "@/lib/api";
import { streamChat, type SendBody } from "@/lib/chat-stream";
import { useI18n } from "@/lib/i18n";
import {
  aiKeys,
  useAIProviders,
  useAuthState,
  useCatalog,
  useConversation,
  useSettings,
  useUpdateSettings,
} from "@/lib/queries";
import { readStudioContext } from "@/lib/studio-bridge";
import type { AIMode, ChatMessage, Conversation } from "@/lib/types";

type QuickAction = NonNullable<SendBody["quick_action"]>;

function formatTokens(n: number, locale: string) {
  return n >= 1000 ? `${(n / 1000).toLocaleString(locale, { maximumFractionDigits: 1 })}k` : String(n);
}

function MessageMeta({ m }: { m: ChatMessage }) {
  const { t, locale } = useI18n();
  if (m.role !== "assistant" || m.status === "streaming") return null;
  if (m.source === "local") return <div className="chat-meta">{t("chat.local")}</div>;
  if (m.source === "cache") return <div className="chat-meta">{t("chat.cached")}</div>;
  return (
    <div className="chat-meta">
      <span>{m.meta?.served_by ?? m.model}</span>
      {m.tier ? <span>· {t(`ai.tier.${m.tier}`)}</span> : null}
      {m.input_tokens || m.output_tokens ? (
        <Tooltip
          label={t("chat.tokensTip", {
            input: m.input_tokens,
            cached: m.cached_tokens,
            output: m.output_tokens,
          })}
        >
          <span>
            · {m.usage_estimated ? "≈ " : ""}
            {formatTokens(m.input_tokens, locale)} → {formatTokens(m.output_tokens, locale)} {t("chat.tokens")}
            {m.cached_tokens ? ` (${t("chat.cachedShort", { n: formatTokens(m.cached_tokens, locale) })})` : ""}
          </span>
        </Tooltip>
      ) : null}
      {m.meta?.truncated ? <span>· {t("chat.truncated")}</span> : null}
      {m.status === "stopped" ? <span>· {t("chat.stopped")}</span> : null}
    </div>
  );
}

export function ChatPanel({
  conversationId,
  onConversationChange,
  compact,
}: {
  conversationId: string | null;
  onConversationChange: (id: string | null) => void;
  compact?: boolean;
}) {
  const { t, locale } = useI18n();
  const qc = useQueryClient();
  const pathname = usePathname();
  const router = useRouter();
  const auth = useAuthState();
  const catalog = useCatalog();
  const settings = useSettings();
  const updateSettings = useUpdateSettings();
  const providers = useAIProviders();
  const detail = useConversation(conversationId);
  const [input, setInput] = useState("");
  const [pending, setPending] = useState<ChatMessage[]>([]);
  const [busy, setBusy] = useState(false);
  const [lastSend, setLastSend] = useState<{ content: string; action?: QuickAction } | null>(null);
  const abort = useRef<AbortController | null>(null);
  const scroller = useRef<HTMLDivElement>(null);
  const prefs = settings.data;

  const connected = (providers.data?.providers ?? []).filter((p) => p.status !== "not_connected");
  const activeKey = providers.data?.active_provider ?? null;
  const active = connected.find((p) => p.key === activeKey);

  const messages = useMemo(() => {
    const saved = detail.data?.messages ?? [];
    const ids = new Set(saved.map((m) => m.id));
    return [...saved, ...pending.filter((m) => !ids.has(m.id))];
  }, [detail.data, pending]);

  useEffect(() => {
    const el = scroller.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages.length, pending]);

  useEffect(() => () => abort.current?.abort(), []);

  const contextLabel = useMemo(() => {
    if (!prefs?.ai_share_context) return null;
    const ctx = readStudioContext(auth.data?.user?.id, pathname);
    const model = catalog.data?.models.find((m) => m.key === ctx.model);
    if (!model) return null;
    const s = (ctx.settings ?? {}) as Record<string, unknown>;
    return [model.name, s.aspect_ratio && s.aspect_ratio !== "match" ? s.aspect_ratio : null, s.resolution]
      .filter(Boolean)
      .join(" · ");
    // Re-read whenever messages change (a send is a good moment) or the page changes.
  }, [prefs?.ai_share_context, auth.data, pathname, catalog.data, messages.length]); // eslint-disable-line react-hooks/exhaustive-deps

  const send = async (content: string, action?: QuickAction) => {
    const text = content.trim();
    if (!text || busy) return;
    if (!connected.length) {
      toast.error(t("chat.noProvider"));
      return;
    }
    setBusy(true);
    setLastSend({ content: text, action });
    let convId = conversationId;
    try {
      if (!convId) {
        const conv = await api<Conversation>("/api/ai/conversations", { body: {} });
        convId = conv.id;
        onConversationChange(conv.id);
      }
      setInput("");
      const controller = new AbortController();
      abort.current = controller;
      const context = prefs?.ai_share_context ? readStudioContext(auth.data?.user?.id, pathname) : null;
      await streamChat(
        convId,
        { content: text, quick_action: action, context },
        {
          onStart: (user, assistant) => setPending([user, assistant]),
          onDelta: (delta) =>
            setPending((list) =>
              list.map((m) => (m.role === "assistant" ? { ...m, content: m.content + delta } : m)),
            ),
          onError: (code, message) =>
            setPending((list) =>
              list.map((m) =>
                m.role === "assistant" ? { ...m, status: "error", error_code: code, error_message: message } : m,
              ),
            ),
          onDone: (message) => {
            if (message) setPending((list) => list.map((m) => (m.id === message.id ? message : m)));
          },
        },
        controller.signal,
      );
    } catch (e) {
      if ((e as Error).name === "AbortError") {
        setPending((list) => list.map((m) => (m.role === "assistant" ? { ...m, status: "stopped" } : m)));
      } else {
        const err = e as ApiError;
        toast.error(err.message, {
          action:
            err.code === "not_connected" || err.code === "model_not_found"
              ? { label: t("nav.integrations"), onClick: () => router.push("/integrations?tab=ai") }
              : undefined,
        });
        if (!pending.length) setInput(text);
      }
    } finally {
      abort.current = null;
      setBusy(false);
      if (convId) {
        // The server stores a stopped answer a moment later.
        window.setTimeout(async () => {
          await qc.invalidateQueries({ queryKey: aiKeys.conversation(convId!) });
          setPending([]);
        }, 350);
      }
      qc.invalidateQueries({ queryKey: ["ai", "conversations"] });
      qc.invalidateQueries({ queryKey: aiKeys.usage });
    }
  };

  const quick: { action: QuickAction; label: string; text: string; icon: React.ReactNode }[] = [
    { action: "improve_prompt", label: t("chat.qa.improve"), text: t("chat.qa.improveText"), icon: <Wand2 /> },
    { action: "suggest_settings", label: t("chat.qa.settings"), text: t("chat.qa.settingsText"), icon: <SlidersHorizontal /> },
    { action: "compare_models", label: t("chat.qa.model"), text: t("chat.qa.modelText"), icon: <Lightbulb /> },
    { action: "troubleshoot", label: t("chat.qa.trouble"), text: t("chat.qa.troubleText"), icon: <Wrench /> },
  ];

  const setMode = (mode: AIMode) =>
    updateSettings.mutate(
      { ai_mode: mode },
      { onSuccess: () => qc.invalidateQueries({ queryKey: aiKeys.providers }) },
    );

  if (providers.isLoading || settings.isLoading) {
    return (
      <div className="chat">
        <div className="chat-scroll" style={{ padding: 20 }}>
          <Skeleton style={{ height: 80 }} />
        </div>
      </div>
    );
  }

  return (
    <div className={cx("chat", compact && "compact")}>
      <div className="chat-toolbar">
        <Segmented
          label={t("ai.mode")}
          value={prefs?.ai_mode ?? "low"}
          onChange={setMode}
          options={[
            { value: "low", label: t("ai.mode.low"), hint: t("ai.mode.lowHint") },
            { value: "balanced", label: t("ai.mode.balanced"), hint: t("ai.mode.balancedHint") },
            { value: "high", label: t("ai.mode.high"), hint: t("ai.mode.highHint") },
          ]}
        />
        <span className="spacer" />
        {connected.length > 1 ? (
          <Select
            label={t("ai.provider")}
            value={activeKey ?? undefined}
            width={180}
            onChange={(v) =>
              updateSettings.mutate(
                { ai_provider: v },
                { onSuccess: () => qc.invalidateQueries({ queryKey: aiKeys.providers }) },
              )
            }
            options={connected.map((p) => ({ value: p.key, label: p.name }))}
          />
        ) : active ? (
          <span className="small subtle row gap-4">
            <Bot size={14} /> {active.name}
          </span>
        ) : null}
      </div>

      <div className="chat-scroll" ref={scroller}>
        {!connected.length ? (
          <div style={{ padding: 20 }}>
            <Notice tone="accent" icon={<Plug />}>
              <strong>{t("chat.connectTitle")}</strong>
              <div style={{ margin: "4px 0 10px" }}>{t("chat.connectBody")}</div>
              <Link href="/integrations?tab=ai" className="btn btn-sm btn-primary">
                {t("chat.openIntegrations")}
              </Link>
            </Notice>
          </div>
        ) : messages.length === 0 ? (
          <div className="chat-empty">
            <div className="empty-icon">
              <Sparkles />
            </div>
            <div className="h2">{t("chat.emptyTitle")}</div>
            <p className="muted small">{t("chat.emptyBody")}</p>
            <div className="chat-quick">
              {quick.map((q) => (
                <button key={q.action} type="button" className="chip" onClick={() => send(q.text, q.action)}>
                  {q.icon}
                  {q.label}
                </button>
              ))}
            </div>
          </div>
        ) : (
          <div className="chat-list">
            {detail.data?.has_summary ? <div className="chat-summary-note">{t("chat.summarized")}</div> : null}
            {messages.map((m) => (
              <div key={m.id} className={cx("chat-msg", m.role)}>
                {m.role === "user" ? (
                  <div className="chat-bubble">{m.content}</div>
                ) : (
                  <div className="chat-answer">
                    {m.content ? (
                      <Markdown text={m.content} streaming={m.status === "streaming"} />
                    ) : m.status === "streaming" ? (
                      <div className="typing" aria-label={t("chat.thinking")}>
                        <span />
                        <span />
                        <span />
                      </div>
                    ) : null}
                    {m.status === "error" ? (
                      <Notice tone="danger" icon={<CircleAlert />}>
                        <div>{m.error_message ?? t("errors.generic")}</div>
                        {lastSend && m === messages[messages.length - 1] ? (
                          <Button
                            size="sm"
                            style={{ marginTop: 8 }}
                            icon={<RefreshCw />}
                            onClick={() => send(lastSend.content, lastSend.action)}
                          >
                            {t("chat.retry")}
                          </Button>
                        ) : null}
                      </Notice>
                    ) : null}
                    <MessageMeta m={m} />
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      {connected.length ? (
        <div className="chat-composer">
          {messages.length > 0 ? (
            <div className="chat-quick inline">
              {quick.map((q) => (
                <button
                  key={q.action}
                  type="button"
                  className="chip"
                  disabled={busy}
                  onClick={() => send(q.text, q.action)}
                >
                  {q.icon}
                  {q.label}
                </button>
              ))}
            </div>
          ) : null}
          <div className="chat-input">
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={t("chat.placeholder")}
              rows={compact ? 2 : 3}
              maxLength={8000}
              aria-label={t("chat.placeholder")}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
                  e.preventDefault();
                  void send(input);
                }
              }}
            />
            <div className="row" style={{ padding: "0 8px 8px 12px" }}>
              <Tooltip label={prefs?.ai_share_context ? t("chat.contextOnTip") : t("chat.contextOffTip")}>
                <button
                  type="button"
                  className={cx("chip", "context-chip")}
                  aria-pressed={prefs?.ai_share_context}
                  onClick={() => updateSettings.mutate({ ai_share_context: !prefs?.ai_share_context })}
                >
                  {prefs?.ai_share_context ? <Eye size={13} /> : <EyeOff size={13} />}
                  {prefs?.ai_share_context ? (contextLabel ?? t("chat.contextOn")) : t("chat.contextOff")}
                </button>
              </Tooltip>
              <span className="spacer" />
              {active ? (
                <Tooltip label={t("chat.routeTip")}>
                  <Badge>
                    <Gauge size={12} />
                    {t(`ai.mode.${prefs?.ai_mode ?? "low"}`)}
                    {prefs?.ai_auto_route ? ` · ${t("chat.auto")}` : ""}
                  </Badge>
                </Tooltip>
              ) : null}
              {busy ? (
                <Button size="sm" icon={<Square />} onClick={() => abort.current?.abort()}>
                  {t("chat.stop")}
                </Button>
              ) : (
                <Button
                  size="sm"
                  variant="primary"
                  icon={<ArrowUp />}
                  disabled={!input.trim()}
                  onClick={() => send(input)}
                  aria-label={t("chat.send")}
                >
                  {t("chat.send")}
                </Button>
              )}
            </div>
          </div>
          {detail.data?.usage && (detail.data.usage.input_tokens || detail.data.usage.output_tokens) ? (
            <div className="tiny subtle" style={{ textAlign: "right", marginTop: 6 }}>
              {t("chat.convUsage", {
                input: formatTokens(detail.data.usage.input_tokens, locale),
                output: formatTokens(detail.data.usage.output_tokens, locale),
              })}
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}
