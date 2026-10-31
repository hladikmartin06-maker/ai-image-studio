"use client";

import {
  ArrowDownUp,
  CalendarRange,
  CheckSquare,
  Download,
  Heart,
  HeartOff,
  ImageOff,
  LibraryBig,
  Search,
  Sparkles,
  Trash2,
  X,
} from "lucide-react";
import Link from "next/link";
import { type ReactNode, useCallback, useMemo, useState } from "react";
import { toast } from "sonner";

import { useImageActions } from "@/components/images/actions";
import { GridSkeleton, ImageGrid } from "@/components/images/image-grid";
import { AddToPersona } from "@/components/personas/shared";
import { ImageViewer } from "@/components/images/image-viewer";
import { StatusBadge } from "@/components/images/status";
import {
  Button,
  EmptyState,
  IconButton,
  MultiSelect,
  Notice,
  Select,
  useConfirm,
} from "@/components/ui";
import { downloadUrl } from "@/lib/api";
import { formatNumber, formatRelative } from "@/lib/format";
import { useDebounced } from "@/lib/hooks";
import { useI18n } from "@/lib/i18n";
import {
  type ImageFilters,
  useBulkImages,
  useCatalog,
  useGenerations,
  useImages,
  usePersonas,
  useSettings,
} from "@/lib/queries";
import type { ModelKey, ProviderKey, StudioImage } from "@/lib/types";

type DateRange = "any" | "today" | "7d" | "30d";
type StatusFilter = "any" | "complete" | "partial" | "processing" | "failed";

function since(range: DateRange): string | undefined {
  if (range === "any") return undefined;
  const d = new Date();
  d.setHours(0, 0, 0, 0);
  if (range === "7d") d.setDate(d.getDate() - 6);
  if (range === "30d") d.setDate(d.getDate() - 29);
  return d.toISOString();
}

export function ImageBrowser({
  favoritesOnly,
  header,
}: {
  favoritesOnly?: boolean;
  header: (total: number | undefined) => ReactNode;
}) {
  const { t, locale } = useI18n();
  const catalog = useCatalog();
  const settings = useSettings();
  const actions = useImageActions();
  const bulk = useBulkImages();
  const confirm = useConfirm();

  const [search, setSearch] = useState("");
  const [models, setModels] = useState<ModelKey[]>([]);
  const [providers, setProviders] = useState<ProviderKey[]>([]);
  const [range, setRange] = useState<DateRange>("any");
  const [onlyFavorites, setOnlyFavorites] = useState(false);
  const [status, setStatus] = useState<StatusFilter>("any");
  const [sort, setSort] = useState<ImageFilters["sort"]>(favoritesOnly ? "favorited" : "newest");
  const [scope, setScope] = useState<"library" | "unsaved">("library");
  const [persona, setPersona] = useState<string>("any");
  const personas = usePersonas();
  const [selection, setSelection] = useState<Set<string>>(new Set());
  const [selecting, setSelecting] = useState(false);
  const [lastSelected, setLastSelected] = useState<number | null>(null);
  const [viewer, setViewer] = useState<number | null>(null);
  const q = useDebounced(search.trim(), 300);

  const generationStatus = status === "processing" || status === "failed";
  const filters: ImageFilters = useMemo(
    () => ({
      q: q || undefined,
      model: models.length ? models.join(",") : undefined,
      provider: providers.length ? providers.join(",") : undefined,
      favorite: favoritesOnly || onlyFavorites ? true : undefined,
      status: status === "complete" || status === "partial" ? status : undefined,
      date_from: since(range),
      sort,
      scope: favoritesOnly ? "all" : scope,
      persona: persona === "any" ? undefined : persona,
      limit: 60,
    }),
    [q, models, providers, favoritesOnly, onlyFavorites, status, range, sort, scope, persona],
  );
  const images = useImages(filters, !generationStatus);
  const gens = useGenerations(
    {
      q: q || undefined,
      status: status === "processing" ? "queued,processing" : "failed",
      model: models.length === 1 ? models[0] : undefined,
      provider: providers.length === 1 ? providers[0] : undefined,
      persona: persona === "any" ? undefined : persona,
      limit: 40,
    },
    generationStatus,
  );

  const items = useMemo(() => images.data?.pages.flatMap((p) => p.items) ?? [], [images.data]);
  const total = generationStatus ? gens.data?.pages[0]?.total : images.data?.pages[0]?.total;
  const filtersActive =
    Boolean(q) ||
    models.length > 0 ||
    providers.length > 0 ||
    range !== "any" ||
    onlyFavorites ||
    status !== "any" ||
    persona !== "any";

  const resetFilters = () => {
    setSearch("");
    setModels([]);
    setProviders([]);
    setRange("any");
    setOnlyFavorites(false);
    setStatus("any");
    setPersona("any");
  };

  const select = useCallback(
    (img: StudioImage, index: number, shift: boolean) => {
      setSelection((prev) => {
        const next = new Set(prev);
        if (shift && lastSelected !== null) {
          const [a, b] = [Math.min(lastSelected, index), Math.max(lastSelected, index)];
          items.slice(a, b + 1).forEach((i) => next.add(i.id));
        } else if (next.has(img.id)) next.delete(img.id);
        else next.add(img.id);
        return next;
      });
      setLastSelected(index);
    },
    [items, lastSelected],
  );

  const ids = [...selection];
  const clear = () => {
    setSelection(new Set());
    setLastSelected(null);
    setSelecting(false);
  };

  const runBulk = async (action: "favorite" | "unfavorite" | "delete" | "save") => {
    if (action === "delete") {
      const ok = await confirm({
        title: t("lib.bulkDeleteTitle", { n: ids.length }),
        description: t("lib.bulkDeleteBody"),
        confirmLabel: t("common.delete"),
        danger: true,
      });
      if (!ok) return;
    }
    try {
      const res = await bulk.mutateAsync({ ids, action });
      toast.success(t(`lib.bulkDone.${action}`, { n: res.changed }));
      if (action === "delete" || (favoritesOnly && action === "unfavorite")) clear();
    } catch (e) {
      toast.error((e as Error).message);
    }
  };

  const downloadSelected = () => {
    if (ids.length > 100) {
      toast.error(t("lib.archiveLimit"));
      return;
    }
    downloadUrl(`/api/images/archive?ids=${ids.join(",")}`);
  };

  const modelOptions = (catalog.data?.models ?? []).map((m) => ({ value: m.key, label: m.name }));
  const providerOptions = (catalog.data?.providers ?? []).map((p) => ({ value: p.key, label: p.name }));

  return (
    <>
      {header(total)}
      <div className="toolbar">
        <div className="input-wrap search">
          <Search />
          <input
            className="input"
            placeholder={t("lib.searchPlaceholder")}
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
        <MultiSelect
          label={t("lib.model")}
          allLabel={t("common.all")}
          values={models}
          onChange={setModels}
          options={modelOptions}
        />
        {personas.data?.items.length ? (
          <Select
            label={t("pm.section")}
            value={persona}
            onChange={setPersona}
            highlight={persona !== "any"}
            options={[
              { value: "any", label: t("pm.filterAny") },
              ...personas.data.items.map((p) => ({ value: p.id, label: p.name })),
            ]}
          />
        ) : null}
        {!favoritesOnly ? (
          <>
            <MultiSelect
              label={t("lib.provider")}
              allLabel={t("common.all")}
              values={providers}
              onChange={setProviders}
              options={providerOptions}
            />
            <Select
              label={t("lib.date")}
              icon={<CalendarRange />}
              value={range}
              onChange={setRange}
              highlight={range !== "any"}
              options={[
                { value: "any", label: t("lib.dateAny") },
                { value: "today", label: t("common.today") },
                { value: "7d", label: t("lib.date7") },
                { value: "30d", label: t("lib.date30") },
              ]}
            />
            <Select
              label={t("lib.status")}
              value={status}
              onChange={(v) => {
                setStatus(v);
                clear();
              }}
              highlight={status !== "any"}
              options={[
                { value: "any", label: t("lib.statusAny") },
                { value: "complete", label: t("lib.statusComplete") },
                { value: "partial", label: t("lib.statusPartial") },
                { value: "processing", label: t("status.processing") },
                { value: "failed", label: t("status.failed") },
              ]}
            />
            <Button
              variant={onlyFavorites ? "primary" : "default"}
              icon={<Heart fill={onlyFavorites ? "currentColor" : "none"} />}
              onClick={() => setOnlyFavorites((v) => !v)}
              aria-pressed={onlyFavorites}
            >
              {t("nav.favorites")}
            </Button>
          </>
        ) : null}
        <span className="spacer" />
        <Select
          label={t("lib.sort")}
          icon={<ArrowDownUp />}
          value={sort}
          onChange={setSort}
          options={[
            ...(favoritesOnly ? [{ value: "favorited" as const, label: t("lib.sortFavorited") }] : []),
            { value: "newest" as const, label: t("lib.sortNewest") },
            { value: "oldest" as const, label: t("lib.sortOldest") },
          ]}
        />
        {!favoritesOnly && settings.data && !settings.data.auto_save ? (
          <Select
            label={t("lib.show")}
            icon={<LibraryBig />}
            value={scope}
            onChange={setScope}
            options={[
              { value: "library", label: t("lib.scopeLibrary") },
              { value: "unsaved", label: t("lib.scopeUnsaved") },
            ]}
          />
        ) : null}
        {!generationStatus ? (
          <Button
            variant={selecting || selection.size ? "primary" : "ghost"}
            icon={<CheckSquare />}
            aria-pressed={selecting}
            onClick={() => (selecting || selection.size ? clear() : setSelecting(true))}
          >
            {selecting || selection.size ? t("common.done") : t("common.select")}
          </Button>
        ) : null}
      </div>

      {generationStatus ? (
        gens.isLoading ? (
          <GridSkeleton count={6} />
        ) : (gens.data?.pages[0]?.items.length ?? 0) === 0 ? (
          <EmptyState icon={<ImageOff />} title={t("lib.noJobs")} description={t("lib.noJobsBody")} />
        ) : (
          <div className="col" style={{ gap: 10 }}>
            {gens.data!.pages.flatMap((p) => p.items).map((g) => (
              <Link key={g.id} href={`/library/history?open=${g.id}`} className="card activity-item" style={{ padding: 14 }}>
                <span className="mini-thumb">
                  <Sparkles />
                </span>
                <div className="grow" style={{ minWidth: 0 }}>
                  <div className="truncate">{g.prompt}</div>
                  <div className="tiny subtle">
                    {g.model_name} · {g.provider_name} · {formatRelative(g.created_at, locale)}
                    {g.error_message ? ` · ${g.error_message}` : ""}
                  </div>
                </div>
                <StatusBadge status={g.status} />
              </Link>
            ))}
          </div>
        )
      ) : images.isLoading ? (
        <GridSkeleton />
      ) : images.isError ? (
        <Notice tone="danger">{(images.error as Error).message}</Notice>
      ) : items.length === 0 ? (
        filtersActive ? (
          <EmptyState
            icon={<Search />}
            title={t("lib.noMatches")}
            description={t("lib.noMatchesBody")}
            action={<Button onClick={resetFilters}>{t("lib.resetFilters")}</Button>}
          />
        ) : favoritesOnly ? (
          <EmptyState
            icon={<Heart />}
            title={t("fav.emptyTitle")}
            description={t("fav.emptyBody")}
            action={
              <Link href="/library" className="btn btn-sm">
                {t("nav.library")}
              </Link>
            }
          />
        ) : (
          <EmptyState
            icon={<Sparkles />}
            title={scope === "unsaved" ? t("lib.noUnsaved") : t("lib.emptyTitle")}
            description={t("lib.emptyBody")}
            action={
              <Link href="/generate" className="btn btn-primary btn-sm">
                <Sparkles /> {t("gen.generate")}
              </Link>
            }
          />
        )
      ) : (
        <ImageGrid
          images={items}
          onOpen={setViewer}
          onFavorite={actions.toggleFavorite}
          onDownload={actions.download}
          selection={selection}
          selecting={selecting}
          onSelect={select}
          hasMore={images.hasNextPage}
          loadMore={() => !images.isFetchingNextPage && images.fetchNextPage()}
          loadingMore={images.isFetchingNextPage}
          showPrompts={settings.data?.show_prompts_on_cards}
          blurNsfw={settings.data?.blur_nsfw ?? true}
        />
      )}

      {selection.size > 0 ? (
        <div className="bulkbar" role="toolbar" aria-label={t("lib.bulkActions")}>
          <span className="small" style={{ fontWeight: 600, marginRight: 6 }}>
            {t("common.nSelected", { n: formatNumber(selection.size, locale) })}
          </span>
          <Button size="sm" variant="ghost" onClick={() => setSelection(new Set(items.map((i) => i.id)))}>
            {t("lib.selectAll", { n: items.length })}
          </Button>
          <Button size="sm" icon={<Download />} onClick={downloadSelected}>
            {t("common.download")}
          </Button>
          {scope === "unsaved" && !favoritesOnly ? (
            <Button size="sm" icon={<LibraryBig />} onClick={() => runBulk("save")}>
              {t("images.saveToLibrary")}
            </Button>
          ) : null}
          {!favoritesOnly ? (
            <Button size="sm" icon={<Heart />} onClick={() => runBulk("favorite")}>
              {t("images.favorite")}
            </Button>
          ) : null}
          <Button size="sm" icon={<HeartOff />} onClick={() => runBulk("unfavorite")}>
            {t("images.unfavorite")}
          </Button>
          <AddToPersona imageIds={ids} />
          <Button size="sm" variant="danger" icon={<Trash2 />} onClick={() => runBulk("delete")}>
            {t("common.delete")}
          </Button>
          <IconButton label={t("common.clearSelection")} onClick={clear}>
            <X />
          </IconButton>
        </div>
      ) : null}

      <ImageViewer images={items} index={viewer} onClose={() => setViewer(null)} onIndexChange={setViewer} />
    </>
  );
}
