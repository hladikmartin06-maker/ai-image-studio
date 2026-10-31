"use client";

import { Check, Clipboard, SlidersHorizontal, Wand2 } from "lucide-react";
import { useRouter } from "next/navigation";
import { Fragment, type ReactNode } from "react";
import { toast } from "sonner";

import { copyText } from "@/components/images/actions";
import { Button } from "@/components/ui";
import { useT } from "@/lib/i18n";
import { applyToStudio, type StudioPatch } from "@/lib/studio-bridge";

// A small, safe Markdown renderer for assistant answers: it builds React elements (never
// HTML strings), supports paragraphs, headings, lists, bold, italics, inline code, links
// and fenced blocks, and turns ```prompt / ```studio-settings blocks into action cards.

type Block =
  | { type: "code"; lang: string; text: string; open: boolean }
  | { type: "heading"; level: number; text: string }
  | { type: "list"; ordered: boolean; items: string[] }
  | { type: "paragraph"; text: string };

function parse(source: string): Block[] {
  const lines = source.replace(/\r\n/g, "\n").split("\n");
  const blocks: Block[] = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const fence = line.match(/^```\s*([\w-]*)\s*$/);
    if (fence) {
      const body: string[] = [];
      i += 1;
      while (i < lines.length && !/^```\s*$/.test(lines[i])) body.push(lines[i++]);
      const open = i >= lines.length;
      i += 1;
      blocks.push({ type: "code", lang: fence[1].toLowerCase(), text: body.join("\n"), open });
      continue;
    }
    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    if (heading) {
      blocks.push({ type: "heading", level: heading[1].length, text: heading[2] });
      i += 1;
      continue;
    }
    if (/^\s*([-*•]|\d+[.)])\s+/.test(line)) {
      const ordered = /^\s*\d+[.)]\s+/.test(line);
      const items: string[] = [];
      while (i < lines.length && /^\s*([-*•]|\d+[.)])\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*([-*•]|\d+[.)])\s+/, ""));
        i += 1;
      }
      blocks.push({ type: "list", ordered, items });
      continue;
    }
    if (!line.trim()) {
      i += 1;
      continue;
    }
    const para: string[] = [];
    while (
      i < lines.length &&
      lines[i].trim() &&
      !/^```/.test(lines[i]) &&
      !/^#{1,4}\s/.test(lines[i]) &&
      !/^\s*([-*•]|\d+[.)])\s+/.test(lines[i])
    ) {
      para.push(lines[i]);
      i += 1;
    }
    blocks.push({ type: "paragraph", text: para.join("\n") });
  }
  return blocks;
}

function inline(text: string, keyBase: string): ReactNode[] {
  const out: ReactNode[] = [];
  const pattern = /(`[^`]+`|\*\*[^*]+\*\*|\*[^*\s][^*]*\*|\[[^\]]+\]\((https?:\/\/[^)\s]+)\))/g;
  let last = 0;
  let match: RegExpExecArray | null;
  let n = 0;
  while ((match = pattern.exec(text))) {
    if (match.index > last) out.push(text.slice(last, match.index));
    const token = match[0];
    const key = `${keyBase}-${n++}`;
    if (token.startsWith("`")) out.push(<code key={key}>{token.slice(1, -1)}</code>);
    else if (token.startsWith("**")) out.push(<strong key={key}>{token.slice(2, -2)}</strong>);
    else if (token.startsWith("[")) {
      const label = token.slice(1, token.indexOf("]"));
      out.push(
        <a key={key} href={match[2]} target="_blank" rel="noreferrer noopener">
          {label}
        </a>,
      );
    } else out.push(<em key={key}>{token.slice(1, -1)}</em>);
    last = match.index + token.length;
  }
  if (last < text.length) out.push(text.slice(last));
  const result: ReactNode[] = [];
  out.forEach((part, idx) => {
    if (typeof part !== "string") {
      result.push(part);
      return;
    }
    part.split("\n").forEach((piece, j, arr) => {
      result.push(piece);
      if (j < arr.length - 1) result.push(<br key={`${keyBase}-br-${idx}-${j}`} />);
    });
  });
  return result;
}

const SETTING_KEYS = ["model", "aspect_ratio", "resolution", "count", "seed", "thinking_mode"] as const;

function PromptCard({ text, streaming }: { text: string; streaming: boolean }) {
  const t = useT();
  const router = useRouter();
  const apply = () => {
    const result = applyToStudio({ prompt: text.trim() });
    toast.success(t("chat.promptApplied"));
    if (result === "pending") router.push("/generate");
  };
  return (
    <div className="chat-card">
      <div className="chat-card-head">
        <Wand2 size={14} />
        <span className="eyebrow grow">{t("chat.promptBlock")}</span>
      </div>
      <div className="chat-card-body prompt-text">{text}</div>
      {!streaming ? (
        <div className="row" style={{ gap: 6 }}>
          <Button size="sm" variant="primary" icon={<Check />} onClick={apply}>
            {t("chat.usePrompt")}
          </Button>
          <Button
            size="sm"
            variant="ghost"
            icon={<Clipboard />}
            onClick={async () => (await copyText(text.trim())) && toast.success(t("common.copied"))}
          >
            {t("common.copy")}
          </Button>
        </div>
      ) : null}
    </div>
  );
}

function SettingsCard({ text, streaming }: { text: string; streaming: boolean }) {
  const t = useT();
  const router = useRouter();
  let data: Record<string, unknown> | null = null;
  try {
    const parsed = JSON.parse(text);
    if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) data = parsed;
  } catch {
    data = null;
  }
  if (!data) {
    return <pre className="code-block">{text}</pre>;
  }
  const patch: StudioPatch = {};
  for (const key of SETTING_KEYS) {
    if (data[key] !== undefined && data[key] !== null) (patch as Record<string, unknown>)[key] = data[key];
  }
  const apply = () => {
    const result = applyToStudio(patch);
    toast.success(t("chat.settingsApplied"));
    if (result === "pending") router.push("/generate");
  };
  return (
    <div className="chat-card">
      <div className="chat-card-head">
        <SlidersHorizontal size={14} />
        <span className="eyebrow grow">{t("chat.settingsBlock")}</span>
      </div>
      <div className="chips">
        {Object.entries(patch).map(([k, v]) => (
          <span key={k} className="badge outline">
            {k.replace("_", " ")}: {String(v)}
          </span>
        ))}
      </div>
      {!streaming && Object.keys(patch).length ? (
        <div className="row">
          <Button size="sm" variant="primary" icon={<Check />} onClick={apply}>
            {t("chat.applySettings")}
          </Button>
        </div>
      ) : null}
    </div>
  );
}

export function Markdown({ text, streaming = false }: { text: string; streaming?: boolean }) {
  const blocks = parse(text);
  return (
    <div className="md">
      {blocks.map((block, i) => {
        const key = `b${i}`;
        if (block.type === "code") {
          const busy = streaming && block.open;
          if (block.lang === "prompt") return <PromptCard key={key} text={block.text} streaming={busy} />;
          if (block.lang === "studio-settings" || block.lang === "studio_settings")
            return busy ? (
              <pre key={key} className="code-block">
                {block.text}
              </pre>
            ) : (
              <SettingsCard key={key} text={block.text} streaming={busy} />
            );
          return (
            <pre key={key} className="code-block">
              {block.text}
            </pre>
          );
        }
        if (block.type === "heading") {
          return (
            <p key={key} className="md-heading">
              {inline(block.text, key)}
            </p>
          );
        }
        if (block.type === "list") {
          const List = block.ordered ? "ol" : "ul";
          return (
            <List key={key}>
              {block.items.map((item, j) => (
                <li key={`${key}-${j}`}>{inline(item, `${key}-${j}`)}</li>
              ))}
            </List>
          );
        }
        return <p key={key}>{inline(block.text, key).map((n, j) => <Fragment key={j}>{n}</Fragment>)}</p>;
      })}
    </div>
  );
}
