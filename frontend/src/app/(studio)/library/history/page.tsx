"use client";

import {
  ArrowDownUp,
  Copy,
  ImageIcon,
  ListOrdered,
  RefreshCw,
  Search,
  SlidersHorizontal,
  Trash2,
  X,
} from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

import { copyText } from "@/components/images/actions";
import { ImageViewer } from "@/components/images/image-viewer";
import { LibraryTabs } from "@/components/images/library-tabs";
import { StatusBadge } from "@/components/images/status";
import { PageHeader } from "@/components/layout/page-header";
import {
  Button,
  EmptyState,
  IconButton,
  Modal,
  Notice,
  Select,
  Skeleton,
  useConfirm,
} from "@/components/ui";
import { formatDate, formatDuration, formatRelative } from "@/lib/format";
import { useDebounced } from "@/lib/hooks";
import { useI18n } from "@/lib/i18n";
import {
  useCatalog,
  useDeleteGeneration,
  useGeneration,
  useGenerations,
  useRegenerate,
} from "@/lib/queries";
import type { Generation, StudioImage } from "@/lib/types";

function GenerationDetails({ id, onClose }: { id: string | null; onClose: () => void }) {
  const { t, locale } = useI18n();
  const router = useRouter();
  const gen = useGeneration(id);
  const regenerate = useRegenerate();
  const remove = useDeleteGeneration();
  const confirm = useConfirm();
  const [viewer, setViewer] = useState<number | null>(null);
  const g = gen.data;
  const running = g?.status === "queued" || g?.status === "processing";

  const again = () =>
    g &&
    regenerate.mutate(
      { id: g.id },
      {
        onSuccess: () =>
          toast.success(t("gen.started"), {
            action: { label: t("common.view"), onClick: () => router.push("/generate") },
          }),
        onError: (e) => toast.error((e as Error).message),
      },
    );

  const del = async () => {
    if (!g) return;
    const ok = await confirm({
      title: t("hist.deleteTitle"),
      description: t("hist.deleteBody", { n: g.images.length }),
      confirmLabel: t("common.delete"),
      danger: true,
    });
    if (!ok) return;
    remove.mutate(g.id, {
      onSuccess: () => {
        toast.success(t("hist.deleted"));
        onClose();
      },
      onError: (e) => toast.error((e as Error).message),
    });
  };

  return (
    <>
      <Modal
        open={id !== null && viewer === null}
        onClose={onClose}
        wide
        title={t("hist.detailsTitle")}
        description={g ? `${g.model_name} · ${g.provider_name} · ${formatDate(g.created_at, locale)}` : undefined}
        footer={
          g ? (
            <>
              <Button variant="danger" icon={<Trash2 />} onClick={del} disabled={running} loading={remove.isPending}>
                {t("common.delete")}
              </Button>
              <span className="spacer" />
              <Link href={`/generate?from=${g.id}`} className="btn">
                <SlidersHorizontal /> {t("hist.loadInStudio")}
              </Link>
              <Button variant="primary" icon={<RefreshCw />} onClick={again} loading={regenerate.isPending}>
                {t("images.regenerate")}
              </Button>
            </>
          ) : null
        }
      >
        {gen.isLoading || !g ? (
          gen.isError ? (
            <Notice tone="danger">{(gen.error as Error).message}</Notice>
          ) : (
            <Skeleton style={{ height: 240 }} />
          )
        ) : (
          <div className="col" style={{ gap: 18 }}>
            <div className="row">
              <StatusBadge status={g.status} />
              {g.duration_seconds !== null ? (
                <span className="small subtle">{formatDuration(g.duration_seconds)}</span>
              ) : null}
            </div>
            {g.error_message ? <Notice tone={g.status === "failed" ? "danger" : "warning"}>{g.error_message}</Notice> : null}
            <div>
              <div className="row" style={{ marginBottom: 8 }}>
                <div className="eyebrow grow">{t("viewer.prompt")}</div>
                <IconButton
                  label={t("common.copyPrompt")}
                  onClick={async () => {
                    if (await copyText(g.prompt)) toast.success(t("common.promptCopied"));
                  }}
                >
                  <Copy />
                </IconButton>
              </div>
              <div className="prompt-box">{g.prompt}</div>
            </div>
            {g.images.length ? (
              <div>
                <div className="eyebrow" style={{ marginBottom: 8 }}>
                  {t("hist.images", { n: g.images.length })}
                </div>
                <div className="refs" style={{ gridTemplateColumns: "repeat(6, 1fr)" }}>
                  {g.images.map((img, i) => (
                    <button
                      key={img.id}
                      type="button"
                      className="ref"
                      style={{ padding: 0, cursor: "zoom-in" }}
                      onClick={() => setViewer(i)}
                      aria-label={t("viewer.title")}
                    >
                      <img src={img.thumb_url} alt="" loading="lazy" />
                    </button>
                  ))}
                </div>
              </div>
            ) : null}
            {g.reference_images.length ? (
              <div>
                <div className="eyebrow" style={{ marginBottom: 8 }}>
                  {t("refs.title")}
                </div>
                <div className="refs" style={{ gridTemplateColumns: "repeat(8, 1fr)" }}>
                  {g.reference_images.map((r) => (
                    <div key={r.id} className="ref">
                      <img src={r.thumb_url} alt="" loading="lazy" />
                    </div>
                  ))}
                </div>
              </div>
            ) : null}
            <dl className="kv">
              <dt>{t("hist.id")}</dt>
              <dd className="mono">{g.id}</dd>
              <dt>{t("viewer.model")}</dt>
              <dd>{g.model_name}</dd>
              <dt>{t("viewer.provider")}</dt>
              <dd>{g.provider_name}</dd>
              <dt>{t("hist.endpoint")}</dt>
              <dd className="mono">{g.provider_model_id}</dd>
              <dt>{t("viewer.mode")}</dt>
              <dd>{g.mode === "image-edit" ? t("gen.modeEdit") : t("gen.modeText")}</dd>
              <dt>{t("gen.imageCount")}</dt>
              <dd>
                {g.images.length} / {g.image_count}
              </dd>
              <dt>{t("hist.requested")}</dt>
              <dd>{formatDate(g.created_at, locale)}</dd>
              <dt>{t("hist.completed")}</dt>
              <dd>{formatDate(g.completed_at, locale)}</dd>
              {g.target_size ? (
                <>
                  <dt>{t("hist.targetSize")}</dt>
                  <dd>{g.target_size}</dd>
                </>
              ) : null}
              <dt>{t("gen.seed")}</dt>
              <dd>{g.settings.seed ?? t("gen.seedRandom")}</dd>
            </dl>
            <div>
              <div className="eyebrow" style={{ marginBottom: 8 }}>
                {t("hist.request")}
              </div>
              <pre className="code-block">{JSON.stringify(g.request_payload, null, 2)}</pre>
              <div className="hint" style={{ marginTop: 6 }}>
                {t("hist.requestHint")}
              </div>
            </div>
            {g.tasks.length > 1 || g.tasks.some((tk) => tk.error_message) ? (
              <div>
                <div className="eyebrow" style={{ marginBottom: 8 }}>
                  {t("hist.requests")}
                </div>
                <div className="col" style={{ gap: 6 }}>
                  {g.tasks.map((tk) => (
                    <div key={tk.index} className="row small">
                      <span className="subtle" style={{ width: 24 }}>
                        #{tk.index + 1}
                      </span>
                      <span className="mono grow truncate">{tk.provider_task_id ?? "—"}</span>
                      <span className={tk.status === "failed" ? "badge danger" : "badge"}>{tk.status}</span>
                      {tk.error_message ? <span className="tiny subtle truncate" style={{ maxWidth: 260 }}>{tk.error_message}</span> : null}
                    </div>
                  ))}
                </div>
              </div>
            ) : null}
          </div>
        )}
      </Modal>
      {g ? (
        <ImageViewer
          images={g.images as StudioImage[]}
          index={viewer}
          onClose={() => setViewer(null)}
          onIndexChange={setViewer}
        />
      ) : null}
    </>
  );
}

export default function HistoryPage() {
  const { t, locale } = useI18n();
  const catalog = useCatalog();
  const [search, setSearch] = useState("");
  const [status, setStatus] = useState("any");
  const [model, setModel] = useState("any");
  const [provider, setProvider] = useState("any");
  const [sort, setSort] = useState<"newest" | "oldest">("newest");
  const [open, setOpen] = useState<string | null>(null);
  const q = useDebounced(search.trim(), 300);

  useEffect(() => {
    const id = new URLSearchParams(window.location.search).get("open");
    // eslint-disable-next-line react-hooks/set-state-in-effect -- deep link from the dashboard
    if (id) setOpen(id);
  }, []);

  const gens = useGenerations({
    q: q || undefined,
    status: status === "any" ? undefined : status === "active" ? "queued,processing" : status,
    model: model === "any" ? undefined : model,
    provider: provider === "any" ? undefined : provider,
    sort,
    limit: 30,
  });
  const items = useMemo(() => gens.data?.pages.flatMap((p) => p.items) ?? [], [gens.data]);
  const total = gens.data?.pages[0]?.total;

  const filtersActive = Boolean(q) || status !== "any" || model !== "any" || provider !== "any";

  return (
    <div className="page">
      <PageHeader
        title={t("hist.title")}
        subtitle={total === undefined ? t("hist.subtitle") : t("hist.count", { n: total.toLocaleString(locale) })}
        actions={<LibraryTabs />}
      />
      <div className="toolbar">
        <div className="input-wrap search">
          <Search />
          <input
            className="input"
            placeholder={t("hist.searchPlaceholder")}
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            aria-label={t("common.search")}
          />
          {search ? (
            <span className="input-action">
              <IconButton label={t("common.clear")} onClick={() => setSearch("")}>
                <X />
              </IconButton>
            </span>
          ) : null}
        </div>
        <Select
          label={t("lib.status")}
          value={status}
          onChange={setStatus}
          highlight={status !== "any"}
          options={[
            { value: "any", label: t("lib.statusAny") },
            { value: "active", label: t("hist.active") },
            { value: "completed", label: t("status.completed") },
            { value: "failed", label: t("status.failed") },
            { value: "cancelled", label: t("status.cancelled") },
          ]}
        />
        <Select
          label={t("lib.model")}
          value={model}
          onChange={setModel}
          highlight={model !== "any"}
          options={[
            { value: "any", label: t("hist.allModels") },
            ...(catalog.data?.models ?? []).map((m) => ({ value: m.key as string, label: m.name })),
          ]}
        />
        <Select
          label={t("lib.provider")}
          value={provider}
          onChange={setProvider}
          highlight={provider !== "any"}
          options={[
            { value: "any", label: t("hist.allProviders") },
            ...(catalog.data?.providers ?? []).map((p) => ({ value: p.key as string, label: p.name })),
          ]}
        />
        <span className="spacer" />
        <Select
          label={t("lib.sort")}
          icon={<ArrowDownUp />}
          value={sort}
          onChange={setSort}
          options={[
            { value: "newest", label: t("lib.sortNewest") },
            { value: "oldest", label: t("lib.sortOldest") },
          ]}
        />
      </div>

      {gens.isLoading ? (
        <div className="col">
          {Array.from({ length: 6 }, (_, i) => (
            <Skeleton key={i} style={{ height: 58 }} />
          ))}
        </div>
      ) : gens.isError ? (
        <Notice tone="danger">{(gens.error as Error).message}</Notice>
      ) : items.length === 0 ? (
        <EmptyState
          icon={<ListOrdered />}
          title={filtersActive ? t("lib.noMatches") : t("hist.emptyTitle")}
          description={filtersActive ? t("lib.noMatchesBody") : t("hist.emptyBody")}
        />
      ) : (
        <div className="card list">
          <div className="list-row list-head tiny subtle">
            <span>{t("hist.colImages")}</span>
            <span>{t("viewer.prompt")}</span>
            <span className="hide-md">{t("viewer.model")}</span>
            <span>{t("lib.status")}</span>
            <span className="hide-md">{t("hist.colWhen")}</span>
            <span />
          </div>
          {items.map((g: Generation) => (
            <div
              key={g.id}
              className="list-row"
              role="button"
              tabIndex={0}
              onClick={() => setOpen(g.id)}
              onKeyDown={(e) => (e.key === "Enter" || e.key === " ") && setOpen(g.id)}
            >
              <div className="strip">
                {g.images.slice(0, 3).map((img) => (
                  <img key={img.id} className="mini-thumb" src={img.thumb_url} alt="" loading="lazy" />
                ))}
                {g.images.length === 0 ? (
                  <span className="mini-thumb">
                    <ImageIcon />
                  </span>
                ) : null}
                {g.images.length > 3 ? <span className="tiny subtle" style={{ alignSelf: "center" }}>+{g.images.length - 3}</span> : null}
              </div>
              <div style={{ minWidth: 0 }}>
                <div className="truncate">{g.prompt}</div>
                <div className="tiny subtle truncate">
                  {g.error_message ?? `${g.images.length} / ${g.image_count} · ${g.target_size ?? g.mode}`}
                </div>
              </div>
              <div className="hide-md" style={{ minWidth: 0 }}>
                <div className="small truncate">{g.model_name}</div>
                <div className="tiny subtle">{g.provider_name}</div>
              </div>
              <div>
                <StatusBadge status={g.status} />
              </div>
              <div className="hide-md small subtle">{formatRelative(g.created_at, locale)}</div>
              <IconButton label={t("hist.open")} onClick={(e) => (e.stopPropagation(), setOpen(g.id))}>
                <SlidersHorizontal />
              </IconButton>
            </div>
          ))}
        </div>
      )}
      {gens.hasNextPage ? (
        <div className="row" style={{ justifyContent: "center", marginTop: 16 }}>
          <Button onClick={() => gens.fetchNextPage()} loading={gens.isFetchingNextPage}>
            {t("common.loadMore")}
          </Button>
        </div>
      ) : null}
      <GenerationDetails id={open} onClose={() => setOpen(null)} />
    </div>
  );
}
