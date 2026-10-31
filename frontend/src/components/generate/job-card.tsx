"use client";

import { Ban, CircleAlert, Copy, Plug, RefreshCw, SlidersHorizontal, Timer } from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";
import { toast } from "sonner";

import { copyText, useImageActions } from "@/components/images/actions";
import { ImageTile } from "@/components/images/image-tile";
import { StatusBadge } from "@/components/images/status";
import { Button, IconButton, Notice, Skeleton } from "@/components/ui";
import { formatDuration, formatRelative } from "@/lib/format";
import { useI18n } from "@/lib/i18n";
import { useCancelGeneration, useRegenerate } from "@/lib/queries";
import type { Generation } from "@/lib/types";

function ratioOf(gen: Generation) {
  const first = gen.images[0];
  if (first) return first.width / first.height;
  const r = gen.settings.aspect_ratio;
  if (r && r !== "match" && r.includes(":")) {
    const [w, h] = r.split(":").map(Number);
    return w / h;
  }
  return 1;
}

function useElapsed(since: string | null, running: boolean) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!running) return;
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, [running]);
  if (!since) return 0;
  return Math.max(0, (now - new Date(since).getTime()) / 1000);
}

const CREDENTIAL_ERRORS = ["invalid_credentials", "insufficient_credits", "not_connected"];

export function JobCard({
  gen,
  onOpen,
  onLoad,
  showPrompts,
  blurNsfw,
}: {
  gen: Generation;
  onOpen: (gen: Generation, index: number) => void;
  onLoad: (gen: Generation) => void;
  showPrompts: boolean;
  blurNsfw: boolean;
}) {
  const { t, locale } = useI18n();
  const actions = useImageActions();
  const regenerate = useRegenerate();
  const cancel = useCancelGeneration();
  const running = gen.status === "queued" || gen.status === "processing";
  const elapsed = useElapsed(gen.started_at ?? gen.created_at, running);
  const ar = ratioOf(gen);
  const pending = running ? Math.max(0, gen.image_count - gen.images.length) : 0;

  const again = () =>
    regenerate.mutate(
      { id: gen.id },
      {
        onSuccess: () => toast.success(t("gen.started")),
        onError: (e) => toast.error((e as Error).message),
      },
    );

  return (
    <article className="card job" aria-busy={running}>
      <div className="job-head">
        <div className="grow" style={{ minWidth: 0 }}>
          <div className="row wrap" style={{ gap: 6, marginBottom: 6 }}>
            <StatusBadge status={gen.status} />
            <span className="small" style={{ fontWeight: 600 }}>
              {gen.model_name}
            </span>
            <span className="small subtle">· {gen.provider_name}</span>
            {gen.target_size ? <span className="small subtle">· {gen.target_size}</span> : null}
            {gen.reference_images.length ? (
              <span className="small subtle">· {t("gen.withRefs", { n: gen.reference_images.length })}</span>
            ) : null}
            {gen.persona?.name ? <span className="small subtle">· {gen.persona.name}</span> : null}
            {gen.comfy?.lora ? <span className="small subtle">· LoRA {gen.comfy.lora.name}</span> : null}
            {running && gen.tasks[0]?.progress?.queue_position ? (
              <span className="small subtle">· {t("jb.queue", { n: gen.tasks[0].progress.queue_position })}</span>
            ) : null}
            {running && gen.tasks[0]?.progress?.max ? (
              <span className="small subtle">
                · {Math.round((100 * (gen.tasks[0].progress.value ?? 0)) / gen.tasks[0].progress.max)}%
              </span>
            ) : null}
          </div>
          <div className={showPrompts ? "clamp-2" : "truncate"} style={{ color: "var(--text-2)" }}>
            {gen.prompt}
          </div>
        </div>
        <div className="row" style={{ gap: 2 }}>
          <span className="tiny subtle row gap-4" style={{ marginRight: 6 }}>
            <Timer size={13} />
            {running ? formatDuration(elapsed) : formatRelative(gen.created_at, locale)}
          </span>
          <IconButton
            label={t("common.copyPrompt")}
            onClick={async () => {
              if (await copyText(gen.prompt)) toast.success(t("common.promptCopied"));
            }}
          >
            <Copy />
          </IconButton>
          <IconButton label={t("gen.loadSettings")} onClick={() => onLoad(gen)}>
            <SlidersHorizontal />
          </IconButton>
          {gen.status === "queued" || (gen.provider === "comfyui" && gen.status === "processing") ? (
            <IconButton label={t("gen.cancel")} onClick={() => cancel.mutate(gen.id)}>
              <Ban />
            </IconButton>
          ) : (
            <IconButton label={t("images.regenerate")} disabled={running || regenerate.isPending} onClick={again}>
              <RefreshCw />
            </IconButton>
          )}
        </div>
      </div>

      {running ? <div className="progress" style={{ marginBottom: 12 }} /> : null}

      {gen.images.length || pending ? (
        <div className="job-grid" style={{ ["--ar" as string]: String(ar) }}>
          {gen.images.map((img, i) => (
            <ImageTile
              key={img.id}
              image={img}
              onOpen={() => onOpen(gen, i)}
              onFavorite={() => actions.toggleFavorite(img)}
              onDownload={() => actions.download(img)}
              showPrompt={false}
              blurNsfw={blurNsfw}
            />
          ))}
          {Array.from({ length: pending }, (_, i) => (
            <div className="job-slot" key={`slot-${i}`}>
              <Skeleton style={{ position: "absolute", inset: 0, borderRadius: 12 }} />
              <div className="job-slot-label">
                <span className="dot pulse" style={{ color: "var(--accent)" }} />
                {gen.status === "queued" ? t("status.queued") : t("gen.generating")}
              </div>
            </div>
          ))}
        </div>
      ) : null}

      {gen.error_message ? (
        <div style={{ marginTop: gen.images.length ? 12 : 0 }}>
          <Notice tone={gen.status === "failed" ? "danger" : "warning"} icon={<CircleAlert />}>
            <div>{gen.error_message}</div>
            {gen.error_code && CREDENTIAL_ERRORS.includes(gen.error_code) ? (
              <div style={{ marginTop: 8 }}>
                <Link href="/integrations" className="btn btn-sm">
                  <Plug /> {t("nav.integrations")}
                </Link>
              </div>
            ) : null}
          </Notice>
        </div>
      ) : null}

      {gen.status === "cancelled" ? (
        <div className="small subtle">{t("gen.cancelled")}</div>
      ) : null}
      {gen.status === "failed" && !running ? (
        <div className="row" style={{ marginTop: 10 }}>
          <Button size="sm" icon={<RefreshCw />} loading={regenerate.isPending} onClick={again}>
            {t("gen.tryAgain")}
          </Button>
        </div>
      ) : null}
    </article>
  );
}
