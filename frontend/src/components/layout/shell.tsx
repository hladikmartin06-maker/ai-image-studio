"use client";

import { useQueryClient } from "@tanstack/react-query";
import {
  MessagesSquare,
  Images,
  LayoutDashboard,
  LogOut,
  PanelLeftClose,
  PanelLeftOpen,
  Plug,
  Settings,
  Sparkles,
  SquareUserRound,
  UserRound,
  Wand,
} from "lucide-react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { type ReactNode, useCallback, useEffect, useRef } from "react";
import { toast } from "sonner";

import { AssistantDrawer } from "@/components/assistant/drawer";
import { BrandMark, ProviderLogo } from "@/components/logos";
import { IconButton, MenuItem, Popover, Spinner, Tooltip } from "@/components/ui";
import { isVideo } from "@/lib/types";
import { api } from "@/lib/api";
import { useLiveGenerations } from "@/lib/events";
import { useT } from "@/lib/i18n";
import type { MessageKey } from "@/lib/i18n";
import {
  keys,
  useAuthState,
  useGenerations,
  useIntegrations,
  useSettings,
  useUpdateSettings,
} from "@/lib/queries";
import type { AuthState, Generation, User } from "@/lib/types";

type NavItem = { href: string; label: MessageKey; icon: typeof LayoutDashboard };

const NAV: { group: MessageKey | null; items: NavItem[] }[] = [
  {
    group: null,
    items: [
      { href: "/dashboard", label: "nav.dashboard", icon: LayoutDashboard },
      { href: "/generate", label: "nav.generate", icon: Sparkles },
      { href: "/ai-models", label: "nav.aiModels", icon: SquareUserRound },
      { href: "/loras", label: "nav.lora", icon: Wand },
      { href: "/library", label: "nav.library", icon: Images },
      { href: "/assistant", label: "nav.assistant", icon: MessagesSquare },
    ],
  },
  {
    group: "nav.system",
    items: [
      { href: "/integrations", label: "nav.integrations", icon: Plug },
      { href: "/settings", label: "nav.settings", icon: Settings },
    ],
  },
];

export function useSignOut() {
  const qc = useQueryClient();
  const router = useRouter();
  return useCallback(async () => {
    await api("/api/auth/logout", { body: {} }).catch(() => undefined);
    qc.clear();
    router.replace("/welcome");
  }, [qc, router]);
}

export function UserMenu({ user, align = "end" }: { user: User; align?: "start" | "end" }) {
  const t = useT();
  const router = useRouter();
  const signOut = useSignOut();
  const autoLogin = Boolean(useAuthState().data?.auto_login);
  const initials = user.name
    .split(/\s+/)
    .map((p) => p[0])
    .join("")
    .slice(0, 2)
    .toUpperCase();
  return (
    <Popover
      align={align}
      width={240}
      trigger={({ toggle, ref }) => (
        <button
          ref={ref}
          type="button"
          onClick={toggle}
          className="btn btn-ghost"
          aria-label={t("account.menu")}
          style={{ paddingLeft: 6, gap: 10 }}
        >
          <span
            className="mini-thumb"
            style={{
              width: 28,
              height: 28,
              borderRadius: 999,
              fontSize: 11,
              fontWeight: 700,
              color: "var(--accent-text)",
              background: "var(--accent-soft)",
              borderColor: "transparent",
            }}
          >
            {initials || <UserRound />}
          </span>
          <span className="truncate" style={{ maxWidth: 140 }}>
            {user.name}
          </span>
        </button>
      )}
    >
      {(close) => (
        <>
          <div className="menu-heading">
            <div className="h3 truncate">{user.name}</div>
            <div className="tiny subtle truncate">{user.email}</div>
          </div>
          <div className="menu-sep" />
          <MenuItem
            icon={<UserRound />}
            onClick={() => {
              close();
              router.push("/settings?section=account");
            }}
          >
            {t("account.profile")}
          </MenuItem>
          <MenuItem
            icon={<Settings />}
            onClick={() => {
              close();
              router.push("/settings");
            }}
          >
            {t("nav.settings")}
          </MenuItem>
          {autoLogin ? null : (
            <>
              <div className="menu-sep" />
              <MenuItem icon={<LogOut />} onClick={signOut}>
                {t("account.signOut")}
              </MenuItem>
            </>
          )}
        </>
      )}
    </Popover>
  );
}

function Sidebar({ activeJobs }: { activeJobs: number }) {
  const t = useT();
  const pathname = usePathname();
  const settings = useSettings();
  const update = useUpdateSettings();
  const integrations = useIntegrations();
  const compact = settings.data?.sidebar === "compact";
  const active = integrations.data?.integrations.find((i) => i.active);

  return (
    <aside className="sidebar">
      <div className="brand">
        <BrandMark />
        <div className="brand-text">
          <div className="brand-name">AI Image Studio</div>
          <div className="brand-sub">{t("brand.tagline")}</div>
        </div>
      </div>
      <nav className="nav" aria-label={t("nav.main")}>
        {NAV.flatMap(({ group, items }) => [
          group ? (
            <div key={group} className="nav-label eyebrow">
              {t(group)}
            </div>
          ) : null,
          ...items.map(({ href, label, icon: Icon }) => {
          const current = pathname === href || pathname.startsWith(`${href}/`);
          const link = (
            <Link
              key={href}
              href={href}
              className="nav-item"
              aria-current={current ? "page" : undefined}
              aria-label={compact ? t(label) : undefined}
            >
              <Icon />
              <span className="nav-text">{t(label)}</span>
              {href === "/generate" && activeJobs > 0 ? (
                <span className="nav-badge" title={t("nav.activeJobs", { n: activeJobs })}>
                  {activeJobs}
                </span>
              ) : null}
            </Link>
          );
          return compact ? (
            <Tooltip key={href} label={t(label)} side="right">
              {link}
            </Tooltip>
          ) : (
            link
          );
          }),
        ])}
      </nav>
      <div className="sidebar-foot">
        <Link href="/integrations" className="provider-pill" aria-label={t("nav.provider")}>
          {active ? (
            <>
              <ProviderLogo provider={active.key} small />
              <div className="sidebar-foot-text grow">
                <div className="tiny subtle">{t("nav.activeProvider")}</div>
                <div className="small truncate" style={{ fontWeight: 500 }}>
                  {active.name}
                </div>
              </div>
            </>
          ) : (
            <>
              <span className="logo sm" style={{ color: "var(--warning)" }}>
                <Plug />
              </span>
              <div className="sidebar-foot-text grow">
                <div className="small" style={{ fontWeight: 500 }}>
                  {t("nav.noProvider")}
                </div>
                <div className="tiny subtle">{t("nav.connectHint")}</div>
              </div>
            </>
          )}
        </Link>
        <div className="row" style={{ justifyContent: compact ? "center" : "flex-end" }}>
          <IconButton
            label={compact ? t("nav.expand") : t("nav.collapse")}
            tooltipSide="right"
            onClick={() => update.mutate({ sidebar: compact ? "expanded" : "compact" })}
          >
            {compact ? <PanelLeftOpen /> : <PanelLeftClose />}
          </IconButton>
        </div>
      </div>
    </aside>
  );
}

export function StudioShell({ children }: { children: ReactNode }) {
  const t = useT();
  const router = useRouter();
  const qc = useQueryClient();
  const auth = useAuthState();
  const user = auth.data?.user;
  const settings = useSettings(Boolean(user));

  const autoStarted = useRef(false);
  useEffect(() => {
    if (!auth.data || auth.data.user) return;
    if (!auth.data.auto_login) {
      router.replace("/welcome");
      return;
    }
    // No sign-in screen: open the account on this Mac and stay on the same page.
    if (autoStarted.current) return;
    autoStarted.current = true;
    api<{ user: User }>("/api/auth/auto", { body: { language: navigator.language.startsWith("sk") ? "sk" : "en" } })
      .then((res) => {
        qc.setQueryData(keys.auth, { has_users: true, user: res.user, auto_login: true });
        qc.removeQueries({ queryKey: keys.settings });
      })
      .catch(() => router.replace("/welcome"))
      .finally(() => {
        autoStarted.current = false;
      });
  }, [auth.data, router, qc]);

  useEffect(() => {
    const onUnauth = () => {
      qc.setQueryData(keys.auth, (prev: AuthState | undefined) => ({
        has_users: prev?.has_users ?? true,
        auto_login: prev?.auto_login,
        user: null,
      }));
    };
    window.addEventListener("studio:unauthenticated", onUnauth);
    return () => window.removeEventListener("studio:unauthenticated", onUnauth);
  }, [qc]);

  const onFinished = useCallback(
    (gen: Generation) => {
      if (gen.status === "completed") {
        toast.success(t(gen.images.some(isVideo) ? "toast.videoDone" : "toast.generationDone", { n: gen.images.length }), {
          description: gen.prompt.slice(0, 90),
          action: { label: t("common.view"), onClick: () => router.push("/generate") },
        });
      } else if (gen.status === "failed") {
        toast.error(t("toast.generationFailed"), { description: gen.error_message ?? undefined });
      }
    },
    [router, t],
  );
  useLiveGenerations(Boolean(user), onFinished);
  const active = useGenerations({ active: true, limit: 20 }, Boolean(user));
  const activeJobs = active.data?.pages[0]?.total ?? 0;

  if (!user) {
    return (
      <div style={{ minHeight: "100vh", display: "grid", placeItems: "center" }}>
        <Spinner size={22} />
      </div>
    );
  }
  return (
    <div className="shell" data-sidebar={settings.data?.sidebar ?? "expanded"}>
      <Sidebar activeJobs={activeJobs} />
      <main className="main">{children}</main>
      <AssistantDrawer />
    </div>
  );
}
