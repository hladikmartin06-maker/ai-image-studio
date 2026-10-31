"use client";

import { useQueryClient } from "@tanstack/react-query";
import {
  Copy,
  HardDrive,
  LogOut,
  MessagesSquare,
  Monitor,
  Moon,
  Palette,
  Shield,
  SlidersHorizontal,
  Sparkles,
  Sun,
  Trash2,
  UserRound,
} from "lucide-react";
import Link from "next/link";
import { type FormEvent, type ReactNode, useEffect, useState } from "react";
import { toast } from "sonner";

import { copyText } from "@/components/images/actions";
import { PageHeader } from "@/components/layout/page-header";
import { useSignOut } from "@/components/layout/shell";
import {
  Button,
  Field,
  Notice,
  Segmented,
  Select,
  Skeleton,
  Switch,
  useConfirm,
} from "@/components/ui";
import { api, ApiError } from "@/lib/api";
import { formatBytes, formatDate, formatRelative } from "@/lib/format";
import { useI18n } from "@/lib/i18n";
import type { MessageKey } from "@/lib/i18n";
import {
  keys,
  useAIProviders,
  useAuthState,
  useCatalog,
  useIntegrations,
  useSessions,
  useSettings,
  useStorage,
  useUpdateProfile,
  useUpdateSettings,
} from "@/lib/queries";
import type { Preferences } from "@/lib/types";

type Section = "general" | "appearance" | "generation" | "assistant" | "storage" | "account";

const SECTIONS: { key: Section; label: MessageKey; icon: ReactNode }[] = [
  { key: "general", label: "set.general", icon: <SlidersHorizontal /> },
  { key: "appearance", label: "set.appearance", icon: <Palette /> },
  { key: "generation", label: "set.generation", icon: <Sparkles /> },
  { key: "assistant", label: "set.assistant", icon: <MessagesSquare /> },
  { key: "storage", label: "set.storage", icon: <HardDrive /> },
  { key: "account", label: "set.account", icon: <UserRound /> },
];

function Row({ title, description, children }: { title: string; description?: string; children: ReactNode }) {
  return (
    <div className="setting-row">
      <div className="grow">
        <div>{title}</div>
        {description ? <div>{description}</div> : null}
      </div>
      <div className="setting-control">{children}</div>
    </div>
  );
}

function Group({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="setting-group">
      <div className="eyebrow" style={{ margin: "0 0 10px 4px" }}>
        {title}
      </div>
      <div className="card">{children}</div>
    </section>
  );
}

function usePref() {
  const { t } = useI18n();
  const update = useUpdateSettings();
  return (patch: Partial<Preferences> | { default_resolution: "auto" }) =>
    update.mutate(patch, {
      onSuccess: () => toast.success(t("set.saved"), { duration: 1500 }),
      onError: (e) => toast.error((e as Error).message),
    });
}

function GeneralSection({ prefs }: { prefs: Preferences }) {
  const { t } = useI18n();
  const catalog = useCatalog();
  const integrations = useIntegrations();
  const save = usePref();
  const connected = integrations.data?.integrations.filter((i) => i.status !== "not_connected") ?? [];
  return (
    <>
      <Group title={t("set.general")}>
        <Row title={t("set.theme")} description={t("set.themeHint")}>
          <Segmented
            label={t("set.theme")}
            value={prefs.theme}
            onChange={(v) => save({ theme: v })}
            options={[
              { value: "dark", label: <><Moon /> {t("set.dark")}</> },
              { value: "light", label: <><Sun /> {t("set.light")}</> },
              { value: "system", label: <><Monitor /> {t("set.system")}</> },
            ]}
          />
        </Row>
        <Row title={t("set.language")} description={t("set.languageHint")}>
          <Segmented
            label={t("set.language")}
            value={prefs.language}
            onChange={(v) => save({ language: v })}
            options={[
              { value: "en", label: "English" },
              { value: "sk", label: "Slovenčina" },
            ]}
          />
        </Row>
      </Group>
      <Group title={t("set.defaults")}>
        <Row title={t("set.defaultModel")} description={t("set.defaultModelHint")}>
          <Select
            label={t("set.defaultModel")}
            value={prefs.default_model}
            onChange={(v) => save({ default_model: v })}
            width={220}
            options={(catalog.data?.models ?? []).map((m) => ({ value: m.key, label: m.name }))}
          />
        </Row>
        <Row title={t("set.defaultProvider")} description={t("set.defaultProviderHint")}>
          {connected.length ? (
            <Select
              label={t("set.defaultProvider")}
              value={prefs.active_provider ?? undefined}
              onChange={(v) => save({ active_provider: v })}
              width={220}
              options={connected.map((i) => ({ value: i.key, label: i.name }))}
            />
          ) : (
            <span className="small subtle">{t("set.noProviders")}</span>
          )}
        </Row>
        <Row title={t("set.defaultCount")}>
          <Segmented
            label={t("set.defaultCount")}
            value={prefs.default_image_count}
            onChange={(v) => save({ default_image_count: v })}
            options={[1, 2, 3, 4].map((n) => ({ value: n, label: String(n) }))}
          />
        </Row>
        <Row title={t("set.defaultAspect")} description={t("set.defaultAspectHint")}>
          <Select
            label={t("set.defaultAspect")}
            value={prefs.default_aspect_ratio}
            onChange={(v) => save({ default_aspect_ratio: v })}
            width={140}
            options={["1:1", "3:2", "2:3", "4:3", "3:4", "16:9", "9:16"].map((r) => ({ value: r, label: r }))}
          />
        </Row>
      </Group>
    </>
  );
}

function AppearanceSection({ prefs }: { prefs: Preferences }) {
  const { t } = useI18n();
  const save = usePref();
  return (
    <Group title={t("set.appearance")}>
      <Row title={t("set.theme")}>
        <Segmented
          label={t("set.theme")}
          value={prefs.theme}
          onChange={(v) => save({ theme: v })}
          options={[
            { value: "dark", label: <><Moon /> {t("set.dark")}</> },
            { value: "light", label: <><Sun /> {t("set.light")}</> },
            { value: "system", label: <><Monitor /> {t("set.system")}</> },
          ]}
        />
      </Row>
      <Row title={t("set.sidebar")} description={t("set.sidebarHint")}>
        <Segmented
          label={t("set.sidebar")}
          value={prefs.sidebar}
          onChange={(v) => save({ sidebar: v })}
          options={[
            { value: "expanded", label: t("set.expanded") },
            { value: "compact", label: t("set.compact") },
          ]}
        />
      </Row>
      <Row title={t("set.density")} description={t("set.densityHint")}>
        <Segmented
          label={t("set.density")}
          value={prefs.grid_density}
          onChange={(v) => save({ grid_density: v })}
          options={[
            { value: "compact", label: t("set.densityCompact") },
            { value: "comfortable", label: t("set.densityComfortable") },
            { value: "large", label: t("set.densityLarge") },
          ]}
        />
      </Row>
      <Row title={t("set.showPrompts")} description={t("set.showPromptsHint")}>
        <Switch label={t("set.showPrompts")} checked={prefs.show_prompts_on_cards} onChange={(v) => save({ show_prompts_on_cards: v })} />
      </Row>
      <Row title={t("set.blurNsfw")} description={t("set.blurNsfwHint")}>
        <Switch label={t("set.blurNsfw")} checked={prefs.blur_nsfw} onChange={(v) => save({ blur_nsfw: v })} />
      </Row>
      <Row title={t("set.reduceMotion")} description={t("set.reduceMotionHint")}>
        <Switch label={t("set.reduceMotion")} checked={prefs.reduce_motion} onChange={(v) => save({ reduce_motion: v })} />
      </Row>
    </Group>
  );
}

function GenerationSection({ prefs }: { prefs: Preferences }) {
  const { t } = useI18n();
  const save = usePref();
  return (
    <>
      <Group title={t("set.genDefaults")}>
        <Row title={t("set.defaultResolution")} description={t("set.defaultResolutionHint")}>
          <Segmented
            label={t("set.defaultResolution")}
            value={prefs.default_resolution ?? "auto"}
            onChange={(v) => save(v === "auto" ? { default_resolution: "auto" } : { default_resolution: v as "1K" | "2K" | "4K" })}
            options={[
              { value: "auto", label: t("set.modelDefault") },
              { value: "1K", label: "1K" },
              { value: "2K", label: "2K" },
              { value: "4K", label: "4K" },
            ]}
          />
        </Row>
        <Row title={t("set.defaultFormat")} description={t("set.defaultFormatHint")}>
          <Segmented
            label={t("set.defaultFormat")}
            value={prefs.default_output_format}
            onChange={(v) => save({ default_output_format: v })}
            options={[
              { value: "png", label: "PNG" },
              { value: "jpeg", label: "JPEG" },
            ]}
          />
        </Row>
        <Row title={t("gen.thinkingMode")} description={t("set.thinkingHint")}>
          <Switch label={t("gen.thinkingMode")} checked={prefs.default_thinking_mode} onChange={(v) => save({ default_thinking_mode: v })} />
        </Row>
      </Group>
      <Group title={t("set.prompts")}>
        <Row title={t("set.rememberPrompt")} description={t("set.rememberPromptHint")}>
          <Switch label={t("set.rememberPrompt")} checked={prefs.remember_prompt} onChange={(v) => save({ remember_prompt: v })} />
        </Row>
        <Row title={t("set.clearAfter")} description={t("set.clearAfterHint")}>
          <Switch label={t("set.clearAfter")} checked={prefs.clear_prompt_after_generate} onChange={(v) => save({ clear_prompt_after_generate: v })} />
        </Row>
      </Group>
      <Group title={t("set.savingHistory")}>
        <Row title={t("set.autoSave")} description={t("set.autoSaveHint")}>
          <Switch label={t("set.autoSave")} checked={prefs.auto_save} onChange={(v) => save({ auto_save: v })} />
        </Row>
        <Row title={t("set.keepFailed")} description={t("set.keepFailedHint")}>
          <Select
            label={t("set.keepFailed")}
            value={prefs.keep_failed_days}
            onChange={(v) => save({ keep_failed_days: v })}
            width={180}
            options={[
              { value: 0, label: t("set.keepForever") },
              { value: 7, label: t("set.keepDays", { n: 7 }) },
              { value: 30, label: t("set.keepDays", { n: 30 }) },
              { value: 90, label: t("set.keepDays", { n: 90 }) },
            ]}
          />
        </Row>
      </Group>
    </>
  );
}

function AssistantSection({ prefs }: { prefs: Preferences }) {
  const { t } = useI18n();
  const save = usePref();
  const providers = useAIProviders();
  const connected = providers.data?.providers.filter((p) => p.status !== "not_connected") ?? [];
  return (
    <Group title={t("set.assistant")}>
      <Row title={t("set.aiProvider")} description={t("set.aiProviderHint")}>
        {connected.length ? (
          <Select
            label={t("set.aiProvider")}
            value={providers.data?.active_provider ?? undefined}
            onChange={(v) => save({ ai_provider: v })}
            width={220}
            options={connected.map((p) => ({ value: p.key, label: p.name }))}
          />
        ) : (
          <Link href="/integrations?tab=ai" className="btn btn-sm">
            {t("chat.openIntegrations")}
          </Link>
        )}
      </Row>
      <Row title={t("ai.mode")} description={t("ai.modeHint")}>
        <Segmented
          label={t("ai.mode")}
          value={prefs.ai_mode}
          onChange={(v) => save({ ai_mode: v })}
          options={[
            { value: "low", label: t("ai.mode.low") },
            { value: "balanced", label: t("ai.mode.balanced") },
            { value: "high", label: t("ai.mode.high") },
          ]}
        />
      </Row>
      <Row title={t("ai.autoRoute")} description={t("ai.autoRouteHint")}>
        <Switch label={t("ai.autoRoute")} checked={prefs.ai_auto_route} onChange={(v) => save({ ai_auto_route: v })} />
      </Row>
      <Row title={t("set.aiContext")} description={t("set.aiContextHint")}>
        <Switch label={t("set.aiContext")} checked={prefs.ai_share_context} onChange={(v) => save({ ai_share_context: v })} />
      </Row>
    </Group>
  );
}

function StorageSection({ prefs }: { prefs: Preferences }) {
  const { t, locale } = useI18n();
  const qc = useQueryClient();
  const storage = useStorage();
  const confirm = useConfirm();
  const save = usePref();
  const s = storage.data;

  const cleanup = async (action: "unused_references" | "failed_generations" | "unsaved_images") => {
    const ok = await confirm({
      title: t(`set.cleanup.${action}`),
      description: t(`set.cleanup.${action}.body`),
      confirmLabel: t("common.delete"),
      danger: true,
    });
    if (!ok) return;
    try {
      const res = await api<{ removed: number }>("/api/storage/cleanup", { body: { action } });
      toast.success(t("set.cleanupDone", { n: res.removed }));
      qc.invalidateQueries({ queryKey: keys.storage });
      qc.invalidateQueries({ queryKey: keys.generations });
      qc.invalidateQueries({ queryKey: keys.images });
      qc.invalidateQueries({ queryKey: keys.stats });
    } catch (e) {
      toast.error((e as Error).message);
    }
  };

  const total = s?.bytes_total ?? 0;
  const parts = s
    ? ([
        ["images", s.bytes.images, "var(--accent)"],
        ["thumbs", s.bytes.thumbs, "var(--success)"],
        ["refs", s.bytes.refs, "var(--warning)"],
      ] as const)
    : [];

  return (
    <>
      <Group title={t("set.storageUsage")}>
        <div className="card-body" style={{ paddingTop: 18 }}>
          {!s ? (
            <Skeleton style={{ height: 70 }} />
          ) : (
            <div className="col" style={{ gap: 12 }}>
              <div className="row">
                <div className="stat-value" style={{ fontSize: 24 }}>
                  {formatBytes(total, locale)}
                </div>
                <span className="small subtle">{t("set.storageCounts", { images: s.images, refs: s.reference_images })}</span>
              </div>
              <div className="meter" style={{ height: 8 }}>
                {parts.map(([k, v, color]) => (
                  <span key={k} style={{ width: `${total ? (v / total) * 100 : 0}%`, background: color }} />
                ))}
              </div>
              <div className="row wrap small" style={{ gap: 16 }}>
                {parts.map(([k, v, color]) => (
                  <span key={k} className="row" style={{ gap: 6 }}>
                    <span className="dot" style={{ color }} /> {t(`set.part.${k}`)} · {formatBytes(v, locale)}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>
        <Row title={t("set.location")} description={s?.location}>
          <Button
            size="sm"
            icon={<Copy />}
            disabled={!s}
            onClick={async () => s && (await copyText(s.location)) && toast.success(t("common.copied"))}
          >
            {t("common.copy")}
          </Button>
        </Row>
      </Group>
      <Group title={t("set.downloads")}>
        <Row title={t("set.downloadName")} description={t("set.downloadNameHint")}>
          <Select
            label={t("set.downloadName")}
            value={prefs.download_name}
            onChange={(v) => save({ download_name: v })}
            width={220}
            options={[
              { value: "prompt", label: t("set.namePrompt") },
              { value: "date", label: t("set.nameDate") },
              { value: "id", label: t("set.nameId") },
            ]}
          />
        </Row>
        <Row title={t("set.downloadQuality")} description={t("set.downloadQualityHint")}>
          <span className="badge success">{t("set.original")}</span>
        </Row>
      </Group>
      <Group title={t("set.management")}>
        <Row title={t("set.cleanup.unused_references")} description={t("set.cleanupRefsHint", { n: s?.reference_images ?? 0 })}>
          <Button size="sm" variant="danger" icon={<Trash2 />} disabled={!s?.reference_images} onClick={() => cleanup("unused_references")}>
            {t("common.delete")}
          </Button>
        </Row>
        <Row title={t("set.cleanup.failed_generations")} description={t("set.cleanupFailedHint", { n: s?.failed_generations ?? 0 })}>
          <Button size="sm" variant="danger" icon={<Trash2 />} disabled={!s?.failed_generations} onClick={() => cleanup("failed_generations")}>
            {t("common.delete")}
          </Button>
        </Row>
        <Row title={t("set.cleanup.unsaved_images")} description={t("set.cleanupUnsavedHint", { n: s?.unsaved_images ?? 0 })}>
          <Button size="sm" variant="danger" icon={<Trash2 />} disabled={!s?.unsaved_images} onClick={() => cleanup("unsaved_images")}>
            {t("common.delete")}
          </Button>
        </Row>
      </Group>
    </>
  );
}

function AccountSection() {
  const { t, locale } = useI18n();
  const qc = useQueryClient();
  const auth = useAuthState();
  const sessions = useSessions();
  const profile = useUpdateProfile();
  const signOut = useSignOut();
  const user = auth.data?.user;
  const [name, setName] = useState(user?.name ?? "");
  const [email, setEmail] = useState(user?.email ?? "");
  const [current, setCurrent] = useState("");
  const [next, setNext] = useState("");
  const [pwError, setPwError] = useState<string | null>(null);
  const [pwBusy, setPwBusy] = useState(false);

  const saveProfile = (e: FormEvent) => {
    e.preventDefault();
    profile.mutate(
      { name, email },
      {
        onSuccess: () => toast.success(t("set.profileSaved")),
        onError: (err) => toast.error((err as Error).message),
      },
    );
  };

  const changePassword = async (e: FormEvent) => {
    e.preventDefault();
    setPwError(null);
    if (next.length < 8) {
      setPwError(t("auth.passwordTooShort"));
      return;
    }
    setPwBusy(true);
    try {
      await api("/api/account/password", { body: { current_password: current, new_password: next } });
      toast.success(t("set.passwordChanged"));
      setCurrent("");
      setNext("");
      qc.invalidateQueries({ queryKey: keys.sessions });
    } catch (err) {
      setPwError(err instanceof ApiError ? err.message : t("errors.generic"));
    } finally {
      setPwBusy(false);
    }
  };

  const revoke = async (id?: string) => {
    try {
      if (id) await api(`/api/account/sessions/${id}`, { method: "DELETE" });
      else await api("/api/account/sessions/revoke-others", { body: {} });
      qc.invalidateQueries({ queryKey: keys.sessions });
      toast.success(t("set.sessionsRevoked"));
    } catch (err) {
      toast.error((err as Error).message);
    }
  };

  if (!user) return <Skeleton style={{ height: 200 }} />;
  const others = sessions.data?.sessions.filter((s) => !s.current) ?? [];

  return (
    <>
      <Group title={t("set.profile")}>
        <form className="card-body col" style={{ gap: 14, paddingTop: 18 }} onSubmit={saveProfile}>
          <div className="row wrap" style={{ gap: 14, alignItems: "flex-start" }}>
            <div className="grow" style={{ minWidth: 220 }}>
              <Field label={t("auth.name")} htmlFor="profile-name">
                <input id="profile-name" className="input" value={name} onChange={(e) => setName(e.target.value)} />
              </Field>
            </div>
            <div className="grow" style={{ minWidth: 220 }}>
              <Field label={t("auth.email")} htmlFor="profile-email">
                <input id="profile-email" type="email" className="input" value={email} onChange={(e) => setEmail(e.target.value)} />
              </Field>
            </div>
          </div>
          <div className="row">
            <span className="tiny subtle grow">{t("set.memberSince", { date: formatDate(user.created_at, locale, false) })}</span>
            <Button
              type="submit"
              variant="primary"
              loading={profile.isPending}
              disabled={!name.trim() || !email.trim() || (name === user.name && email === user.email)}
            >
              {t("common.save")}
            </Button>
          </div>
        </form>
      </Group>
      {/* No sign-in screen (auto sign-in): password, sessions and sign out are not needed. */}
      {auth.data?.auto_login ? null : (
        <>
          <Group title={t("set.security")}>
            <form className="card-body col" style={{ gap: 14, paddingTop: 18 }} onSubmit={changePassword}>
              <div className="row wrap" style={{ gap: 14, alignItems: "flex-start" }}>
                <div className="grow" style={{ minWidth: 220 }}>
                  <Field label={t("set.currentPassword")} htmlFor="pw-current">
                    <input id="pw-current" type="password" className="input" autoComplete="current-password" value={current} onChange={(e) => setCurrent(e.target.value)} />
                  </Field>
                </div>
                <div className="grow" style={{ minWidth: 220 }}>
                  <Field label={t("set.newPassword")} htmlFor="pw-new" hint={t("auth.passwordHint")}>
                    <input id="pw-new" type="password" className="input" autoComplete="new-password" value={next} onChange={(e) => setNext(e.target.value)} />
                  </Field>
                </div>
              </div>
              {pwError ? <Notice tone="danger">{pwError}</Notice> : null}
              <div className="row">
                <span className="tiny subtle grow">{t("set.passwordNote")}</span>
                <Button type="submit" loading={pwBusy} disabled={!current || !next} icon={<Shield />}>
                  {t("set.changePassword")}
                </Button>
              </div>
            </form>
            <div className="setting-row" style={{ alignItems: "flex-start" }}>
              <div className="grow">
                <div>{t("set.sessions")}</div>
                <div>{t("set.sessionsHint")}</div>
                <div className="col" style={{ gap: 8, marginTop: 12 }}>
                  {(sessions.data?.sessions ?? []).map((s) => (
                    <div key={s.id} className="row small">
                      <Monitor size={14} className="subtle" />
                      <span className="truncate grow" title={s.user_agent}>
                        {s.user_agent.split(") ")[0].replace("Mozilla/5.0 (", "") || t("set.unknownDevice")}
                      </span>
                      <span className="tiny subtle">{formatRelative(s.last_seen_at, locale)}</span>
                      {s.current ? (
                        <span className="badge accent">{t("set.thisSession")}</span>
                      ) : (
                        <Button size="sm" variant="ghost" onClick={() => revoke(s.id)}>
                          {t("set.revoke")}
                        </Button>
                      )}
                    </div>
                  ))}
                </div>
              </div>
              <div className="setting-control">
                <Button size="sm" disabled={!others.length} onClick={() => revoke()}>
                  {t("set.revokeOthers")}
                </Button>
              </div>
            </div>
          </Group>
          <Group title={t("set.signOutTitle")}>
            <Row title={t("account.signOut")} description={t("set.signOutHint")}>
              <Button variant="danger" icon={<LogOut />} onClick={signOut}>
                {t("account.signOut")}
              </Button>
            </Row>
          </Group>
        </>
      )}
    </>
  );
}

export default function SettingsPage() {
  const { t } = useI18n();
  const settings = useSettings();
  const [section, setSection] = useState<Section>("general");

  useEffect(() => {
    const s = new URLSearchParams(window.location.search).get("section") as Section | null;
    // eslint-disable-next-line react-hooks/set-state-in-effect -- deep link (?section=account)
    if (s && SECTIONS.some((x) => x.key === s)) setSection(s);
  }, []);

  const prefs = settings.data;
  return (
    <div className="page">
      <PageHeader title={t("set.title")} subtitle={t("set.subtitle")} />
      <div className="settings">
        <nav className="settings-nav" aria-label={t("set.title")}>
          {SECTIONS.map((s) => (
            <button
              key={s.key}
              type="button"
              className="nav-item"
              aria-current={section === s.key ? "page" : undefined}
              onClick={() => {
                setSection(s.key);
                window.history.replaceState(null, "", `/settings?section=${s.key}`);
              }}
            >
              {s.icon}
              <span>{t(s.label)}</span>
            </button>
          ))}
        </nav>
        <div>
          {settings.isError ? <Notice tone="danger">{(settings.error as Error).message}</Notice> : null}
          {!prefs ? (
            <Skeleton style={{ height: 320 }} />
          ) : section === "general" ? (
            <GeneralSection prefs={prefs} />
          ) : section === "appearance" ? (
            <AppearanceSection prefs={prefs} />
          ) : section === "generation" ? (
            <GenerationSection prefs={prefs} />
          ) : section === "assistant" ? (
            <AssistantSection prefs={prefs} />
          ) : section === "storage" ? (
            <StorageSection prefs={prefs} />
          ) : (
            <AccountSection />
          )}
        </div>
      </div>
    </div>
  );
}
