"use client";

import {
  Activity,
  ArrowRight,
  CalendarDays,
  SquareUserRound,
  ImageIcon,
  Images,
  Layers,
  Plug,
  Sparkles,
  Wallet,
} from "lucide-react";
import Link from "next/link";
import { useEffect, useState } from "react";

import { useImageActions } from "@/components/images/actions";
import { GridSkeleton, ImageGrid } from "@/components/images/image-grid";
import { ImageViewer } from "@/components/images/image-viewer";
import { StatusBadge } from "@/components/images/status";
import { UserMenu } from "@/components/layout/shell";
import { ProviderLogo } from "@/components/logos";
import { EmptyState, Notice, Skeleton, Tooltip } from "@/components/ui";
import { formatBytes, formatNumber, formatRelative } from "@/lib/format";
import { useI18n } from "@/lib/i18n";
import {
  useAuthState,
  useBalance,
  useGenerations,
  useImages,
  useIntegrations,
  usePersonas,
  useSettings,
  useStats,
} from "@/lib/queries";
import type { Stats } from "@/lib/types";

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

/** Where a generation ran: the two image providers or the user's own ComfyUI. */
function providerName(key: string) {
  return key === "wavespeed" ? "WaveSpeed" : key === "atlas" ? "Atlas Cloud" : key === "comfyui" ? "ComfyUI" : key;
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
            <div className="bar" data-zero={d.images === 0 || undefined} style={{ height: `${Math.max(3, (d.images / max) * 100)}%` }} />
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
  const personas = usePersonas();
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
            <Sparkles /> {t("gen.generate")}
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
          icon={<SquareUserRound />}
          label={t("nav.aiModels")}
          value={formatNumber(personas.data?.total ?? 0, locale)}
          foot={
            <Link href="/ai-models" style={{ color: "var(--text-2)" }}>
              {t("dash.openAiModels")}
            </Link>
          }
          loading={personas.isLoading}
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
              ? providerName(
                  s.providers.slice().sort((a, b) => b.generations - a.generations)[0].provider,
                )
              : "—"
          }
          foot={
            s && s.providers.length
              ? s.providers.map((p) => `${providerName(p.provider)} ${p.generations}`).join(" · ")
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
          <UsageOverview stats={s} />
        </div>
      </div>
      <ImageViewer images={images} index={viewer} onClose={() => setViewer(null)} onIndexChange={setViewer} />
    </div>
  );
}
