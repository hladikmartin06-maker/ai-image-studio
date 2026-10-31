"use client";

import {
  Activity,
  ArrowRight,
  CalendarDays,
  Heart,
  ImageIcon,
  Images,
  Layers,
  Plug,
  Sparkles,
  Wallet,
} from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";
import { toast } from "sonner";

import { useImageActions } from "@/components/images/actions";
import { GridSkeleton, ImageGrid } from "@/components/images/image-grid";
import { ImageViewer } from "@/components/images/image-viewer";
import { StatusBadge } from "@/components/images/status";
import { UserMenu } from "@/components/layout/shell";
import { ProviderLogo } from "@/components/logos";
import { Button, EmptyState, Notice, Segmented, Select, Skeleton, Tooltip } from "@/components/ui";
import { ApiError } from "@/lib/api";
import { formatBytes, formatNumber, formatRelative } from "@/lib/format";
import { useI18n } from "@/lib/i18n";
import {
  useAuthState,
  useBalance,
  useCatalog,
  useCreateGeneration,
  useGenerations,
  useImages,
  useIntegrations,
  useSettings,
  useStats,
} from "@/lib/queries";
import type { ModelKey, Stats } from "@/lib/types";

function greetingKey(): "dash.morning" | "dash.afternoon" | "dash.evening" {
  const h = new Date().getHours();
  if (h < 12) return "dash.morning";
  if (h < 18) return "dash.afternoon";
  return "dash.evening";
}

function StatCard({
  icon,
  label,
  value,
  foot,
  loading,
}: {
  icon: React.ReactNode;
  label: string;
  value: React.ReactNode;
  foot?: React.ReactNode;
  loading?: boolean;
}) {
  return (
    <div className="card stat">
      <div className="stat-top">
        {icon}
        <span className="truncate">{label}</span>
      </div>
      {loading ? <Skeleton style={{ height: 28, width: 80 }} /> : <div className="stat-value">{value}</div>}
      <div className="stat-foot truncate">{loading ? null : foot}</div>
    </div>
  );
}

function ActivityBars({ stats }: { stats: Stats }) {
  const { locale, t } = useI18n();
  const max = Math.max(1, ...stats.activity.map((d) => d.images));
  return (
    <div>
      <div className="bars" role="img" aria-label={t("dash.last14")}>
        {stats.activity.map((d) => (
          <Tooltip
            key={d.date}
            label={`${new Date(d.date + "T12:00:00").toLocaleDateString(locale, { day: "numeric", month: "short" })} · ${d.images}`}
          >
            <div data-zero={d.images === 0 || undefined} style={{ height: `${Math.max(3, (d.images / max) * 100)}%` }} />
          </Tooltip>
        ))}
      </div>
      <div className="row tiny subtle" style={{ marginTop: 6 }}>
        <span className="grow">{t("dash.days14")}</span>
        <span>{t("common.today")}</span>
      </div>
    </div>
  );
}

function QuickGenerate() {
  const { t } = useI18n();
  const router = useRouter();
  const catalog = useCatalog();
  const settings = useSettings();
  const integrations = useIntegrations();
  const create = useCreateGeneration();
  const [model, setModel] = useState<ModelKey | null>(null);
  const [count, setCount] = useState<number | null>(null);
  const [prompt, setPrompt] = useState("");
  const connected = integrations.data?.integrations.filter((i) => i.status !== "not_connected") ?? [];
  const activeProvider = integrations.data?.active_provider ?? connected[0]?.key ?? null;
  const currentModel = model ?? settings.data?.default_model ?? "nano-banana-pro";
  const currentCount = count ?? settings.data?.default_image_count ?? 1;

  const submit = async () => {
    if (!prompt.trim()) {
      toast.error(t("gen.problemPrompt"));
      return;
    }
    if (!activeProvider) {
      toast.error(t("gen.problemNoProvider"), {
        action: { label: t("nav.integrations"), onClick: () => router.push("/integrations") },
      });
      return;
    }
    const prefs = settings.data;
    const info = catalog.data?.models.find((m) => m.key === currentModel);
    const variant = info?.variants.find((v) => v.provider === activeProvider && v.mode === "text-to-image");
    if (!variant) {
      toast.error(t("gen.problemNoText"));
      return;
    }
    // Defaults from Settings, only where this endpoint supports them.
    const s: Record<string, unknown> = {};
    if (variant.controls.includes("aspect_ratio") && prefs && variant.aspect_ratios.includes(prefs.default_aspect_ratio))
      s.aspect_ratio = prefs.default_aspect_ratio;
    if (variant.controls.includes("resolution") && prefs?.default_resolution) {
      const r = variant.resolutions.find((x) => x.toLowerCase() === prefs.default_resolution!.toLowerCase());
      if (r) s.resolution = r;
    }
    if (variant.controls.includes("output_format") && prefs) s.output_format = prefs.default_output_format;
    if (variant.controls.includes("thinking_mode") && prefs) s.thinking_mode = prefs.default_thinking_mode;
    try {
      await create.mutateAsync({
        model: currentModel,
        provider: activeProvider,
        prompt: prompt.trim(),
        count: currentCount,
        settings: s,
        reference_image_ids: [],
        client_request_id: crypto.randomUUID(),
      });
      setPrompt("");
      toast.success(t("gen.started"), {
        action: { label: t("common.view"), onClick: () => router.push("/generate") },
      });
    } catch (e) {
      toast.error((e as ApiError).message);
    }
  };

  return (
    <div className="card">
      <div className="card-head">
        <Sparkles size={16} color="var(--accent)" />
        <div className="h3 grow">{t("dash.quickGenerate")}</div>
        <Link href="/generate" className="btn btn-ghost btn-sm">
          {t("dash.fullStudio")} <ArrowRight />
        </Link>
      </div>
      <div className="card-body col" style={{ gap: 12 }}>
        <Select
          label={t("gen.model")}
          value={currentModel}
          onChange={setModel}
          width={340}
          options={(catalog.data?.models ?? []).map((m) => ({
            value: m.key,
            label: m.name,
            hint: t(`content.${m.content}`),
          }))}
        />
        <textarea
          className="textarea"
          rows={4}
          placeholder={t("gen.promptPlaceholder")}
          value={prompt}
          maxLength={catalog.data?.limits.max_prompt_chars ?? 4000}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) {
              e.preventDefault();
              void submit();
            }
          }}
          aria-label={t("gen.prompt")}
        />
        <div className="row">
          <Segmented
            label={t("gen.imageCount")}
            value={currentCount}
            onChange={setCount}
            options={[1, 2, 3, 4].map((n) => ({ value: n, label: String(n) }))}
          />
          <span className="spacer" />
          <Button variant="primary" loading={create.isPending} icon={<Sparkles />} onClick={submit}>
            {t("gen.generate")}
          </Button>
        </div>
        {!activeProvider && integrations.data ? (
          <Notice tone="accent" icon={<Plug />}>
            {t("gen.connectBody")}{" "}
            <Link href="/integrations" style={{ color: "var(--accent-text)" }}>
              {t("gen.openIntegrations")}
            </Link>
          </Notice>
        ) : null}
      </div>
    </div>
  );
}

function UsageOverview({ stats }: { stats: Stats | undefined }) {
  const { t, locale } = useI18n();
  const integrations = useIntegrations();
  const wavespeed = integrations.data?.integrations.find((i) => i.key === "wavespeed");
  const balance = useBalance("wavespeed", wavespeed?.status !== undefined && wavespeed.status !== "not_connected");
  const providers = integrations.data?.integrations ?? [];
  const totalGenerations = stats?.generations.total ?? 0;
  return (
    <div className="card">
      <div className="card-head">
        <Wallet size={16} color="var(--text-3)" />
        <div className="h3 grow">{t("dash.usage")}</div>
        <Link href="/integrations" className="btn btn-ghost btn-sm">
          {t("nav.integrations")}
        </Link>
      </div>
      <div className="card-body col" style={{ gap: 16 }}>
        {providers.map((p) => {
          const usage = stats?.providers.find((u) => u.provider === p.key);
          const share = totalGenerations ? ((usage?.generations ?? 0) / totalGenerations) * 100 : 0;
          return (
            <div key={p.key} className="col" style={{ gap: 8 }}>
              <div className="row">
                <ProviderLogo provider={p.key} small />
                <div className="grow">
                  <div className="small" style={{ fontWeight: 600 }}>
                    {p.name} {p.active ? <span className="badge accent" style={{ marginLeft: 6 }}>{t("int.active")}</span> : null}
                  </div>
                  <div className="tiny subtle">
                    {p.status === "not_connected"
                      ? t("int.notConnected")
                      : t("dash.providerUsage", { g: usage?.generations ?? 0, i: usage?.images ?? 0 })}
                  </div>
                </div>
                <div className="small" style={{ textAlign: "right" }}>
                  {p.key === "wavespeed" && p.status !== "not_connected" ? (
                    balance.data?.balance !== null && balance.data?.balance !== undefined ? (
                      <>
                        <div style={{ fontWeight: 600 }}>
                          {balance.data.balance.toLocaleString(locale, { style: "currency", currency: "USD" })}
                        </div>
                        <div className="tiny subtle">{t("dash.balance")}</div>
                      </>
                    ) : balance.isLoading ? (
                      <Skeleton style={{ width: 60, height: 16 }} />
                    ) : (
                      <span className="tiny subtle">{balance.data?.error ?? t("dash.balanceUnavailable")}</span>
                    )
                  ) : p.key === "atlas" && p.status !== "not_connected" ? (
                    <Tooltip label={t("dash.atlasNoBalance")}>
                      <span className="tiny subtle">{t("dash.balanceNotOffered")}</span>
                    </Tooltip>
                  ) : null}
                </div>
              </div>
              {p.status !== "not_connected" ? (
                <div className="meter" aria-label={t("dash.share")}>
                  <span style={{ width: `${share}%` }} />
                </div>
              ) : null}
            </div>
          );
        })}
        {stats ? <ActivityBars stats={stats} /> : <Skeleton style={{ height: 64 }} />}
      </div>
    </div>
  );
}

function GenerationActivity() {
  const { t, locale } = useI18n();
  const gens = useGenerations({ limit: 6 });
  const items = gens.data?.pages[0]?.items ?? [];
  return (
    <div className="card">
      <div className="card-head">
        <Activity size={16} color="var(--text-3)" />
        <div className="h3 grow">{t("dash.activity")}</div>
        <Link href="/library/history" className="btn btn-ghost btn-sm">
          {t("common.viewAll")}
        </Link>
      </div>
      <div className="card-body" style={{ paddingTop: 4, paddingBottom: 8 }}>
        {gens.isLoading ? (
          <div className="col" style={{ padding: "8px 0" }}>
            {[0, 1, 2].map((i) => (
              <Skeleton key={i} style={{ height: 40 }} />
            ))}
          </div>
        ) : items.length === 0 ? (
          <div className="small subtle" style={{ padding: "16px 0" }}>
            {t("dash.noActivity")}
          </div>
        ) : (
          items.map((g) => (
            <Link key={g.id} href={`/library/history?open=${g.id}`} className="activity-item">
              {g.images[0] ? (
                <img className="mini-thumb" src={g.images[0].thumb_url} alt="" loading="lazy" />
              ) : (
                <span className="mini-thumb">
                  <ImageIcon />
                </span>
              )}
              <div className="grow" style={{ minWidth: 0 }}>
                <div className="small truncate">{g.prompt}</div>
                <div className="tiny subtle">
                  {g.model_name} · {formatRelative(g.created_at, locale)}
                </div>
              </div>
              <StatusBadge status={g.status} />
            </Link>
          ))
        )}
      </div>
    </div>
  );
}

export default function DashboardPage() {
  const { t, locale } = useI18n();
  const auth = useAuthState();
  const settings = useSettings();
  const stats = useStats();
  const recent = useImages({ limit: 12, scope: "library" });
  const actions = useImageActions();
  const [viewer, setViewer] = useState<number | null>(null);
  const [today, setToday] = useState("");
  const s = stats.data;
  const images = recent.data?.pages[0]?.items ?? [];

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- client clock only
    setToday(
      new Date().toLocaleDateString(locale, { weekday: "long", day: "numeric", month: "long", year: "numeric" }),
    );
  }, [locale]);

  const firstName = auth.data?.user?.name.split(" ")[0] ?? "";
  const lastAt = s?.generations.last_at;

  return (
    <div className="page">
      <header className="page-header">
        <div className="grow">
          <div className="eyebrow row" style={{ marginBottom: 8, gap: 6 }}>
            <CalendarDays size={13} /> {today}
          </div>
          <h1 className="h1">{t(greetingKey(), { name: firstName })}</h1>
          <p className="row" style={{ gap: 8 }}>
            {s?.generations.active ? (
              <>
                <span className="dot pulse" style={{ color: "var(--accent)" }} />
                {t("dash.activeNow", { n: s.generations.active })}
              </>
            ) : lastAt ? (
              <>
                <span className="dot" style={{ color: "var(--success)" }} />
                {t("dash.lastActivity", { when: formatRelative(lastAt, locale) })}
              </>
            ) : (
              t("dash.welcomeLede")
            )}
          </p>
        </div>
        <div className="row">
          <Link href="/generate" className="btn btn-primary">
            <Sparkles /> {t("dash.quickGenerateBtn")}
          </Link>
          {auth.data?.user ? <UserMenu user={auth.data.user} /> : null}
        </div>
      </header>

      {stats.isError ? <Notice tone="danger">{(stats.error as Error).message}</Notice> : null}

      <div className="stats">
        <StatCard
          icon={<Images />}
          label={t("dash.totalImages")}
          value={formatNumber(s?.images_total ?? 0, locale)}
          foot={s ? t("dash.storageUsed", { size: formatBytes(s.storage_bytes, locale) }) : null}
          loading={stats.isLoading}
        />
        <StatCard
          icon={<CalendarDays />}
          label={t("dash.today")}
          value={formatNumber(s?.images_today ?? 0, locale)}
          foot={s ? t("dash.requestsToday", { n: s.generations.today }) : null}
          loading={stats.isLoading}
        />
        <StatCard
          icon={<Heart />}
          label={t("dash.favorites")}
          value={formatNumber(s?.favorites_total ?? 0, locale)}
          foot={
            <Link href="/favorites" style={{ color: "var(--text-2)" }}>
              {t("dash.openFavorites")}
            </Link>
          }
          loading={stats.isLoading}
        />
        <StatCard
          icon={<Layers />}
          label={t("dash.requests")}
          value={formatNumber(s?.generations.total ?? 0, locale)}
          foot={s ? t("dash.requestsFoot", { ok: s.generations.completed, failed: s.generations.failed }) : null}
          loading={stats.isLoading}
        />
        <StatCard
          icon={<Plug />}
          label={t("dash.providerUsageTitle")}
          value={
            s && s.providers.length
              ? s.providers
                  .slice()
                  .sort((a, b) => b.generations - a.generations)
                  .map((p) => (p.provider === "wavespeed" ? "WaveSpeed" : "Atlas"))[0]
              : "—"
          }
          foot={
            s && s.providers.length
              ? s.providers.map((p) => `${p.provider === "wavespeed" ? "WaveSpeed" : "Atlas"} ${p.generations}`).join(" · ")
              : t("dash.noUsage")
          }
          loading={stats.isLoading}
        />
      </div>

      <div className="dash-grid">
        <div className="col" style={{ gap: 20 }}>
          <div className="card">
            <div className="card-head">
              <Images size={16} color="var(--text-3)" />
              <div className="h3 grow">{t("dash.recent")}</div>
              <Link href="/library" className="btn btn-ghost btn-sm">
                {t("common.viewAll")} <ArrowRight />
              </Link>
            </div>
            <div className="card-body">
              {recent.isLoading ? (
                <GridSkeleton count={8} />
              ) : images.length === 0 ? (
                <EmptyState
                  icon={<Sparkles />}
                  title={t("dash.emptyTitle")}
                  description={t("dash.emptyBody")}
                  action={
                    <Link href="/generate" className="btn btn-primary btn-sm">
                      <Sparkles /> {t("gen.generate")}
                    </Link>
                  }
                />
              ) : (
                <ImageGrid
                  images={images}
                  onOpen={setViewer}
                  onFavorite={actions.toggleFavorite}
                  onDownload={actions.download}
                  showPrompts={settings.data?.show_prompts_on_cards}
                  blurNsfw={settings.data?.blur_nsfw ?? true}
                />
              )}
            </div>
          </div>
          <GenerationActivity />
        </div>
        <div className="col" style={{ gap: 20 }}>
          <QuickGenerate />
          <UsageOverview stats={s} />
        </div>
      </div>
      <ImageViewer images={images} index={viewer} onClose={() => setViewer(null)} onIndexChange={setViewer} />
    </div>
  );
}
