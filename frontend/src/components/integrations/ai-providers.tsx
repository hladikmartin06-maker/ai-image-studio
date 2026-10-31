"use client";

import { useQueryClient } from "@tanstack/react-query";
import {
  BarChart3,
  Bot,
  CircleDot,
  ExternalLink,
  Gauge,
  KeyRound,
  PlugZap,
  RefreshCw,
  Settings2,
  ShieldCheck,
  Unplug,
} from "lucide-react";
import { type FormEvent, useState } from "react";
import { toast } from "sonner";

import { Badge, Button, Field, Modal, Notice, Segmented, Select, Skeleton, Switch, useConfirm } from "@/components/ui";
import { api, ApiError } from "@/lib/api";
import { formatNumber, formatRelative } from "@/lib/format";
import { useI18n } from "@/lib/i18n";
import { aiKeys, keys, useAIProviders, useAIUsage, useUpdateSettings } from "@/lib/queries";
import type { AIProvider, AIProviderList, AITier } from "@/lib/types";

const TIERS: AITier[] = ["low", "balanced", "high"];

function KeyDialog({
  provider,
  presets,
  open,
  onClose,
}: {
  provider: AIProvider;
  presets: AIProviderList["presets"];
  open: boolean;
  onClose: () => void;
}) {
  const { t } = useI18n();
  const qc = useQueryClient();
  const [key, setKey] = useState("");
  const [baseUrl, setBaseUrl] = useState(provider.base_url ?? "");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const updating = provider.status !== "not_connected";

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      await api(`/api/ai/providers/${provider.key}`, {
        method: "PUT",
        body: { api_key: key.trim(), base_url: provider.custom_base_url ? baseUrl.trim() : null },
      });
      await qc.invalidateQueries({ queryKey: aiKeys.providers });
      qc.invalidateQueries({ queryKey: keys.settings });
      toast.success(t("int.connected", { name: provider.name }));
      setKey("");
      onClose();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : t("errors.generic"));
    } finally {
      setBusy(false);
    }
  };

  const canSave = provider.custom_base_url
    ? baseUrl.trim().length > 8 && (key.trim().length >= 8 || provider.key_optional || updating)
    : key.trim().length >= 8;

  return (
    <Modal
      open={open}
      onClose={onClose}
      icon={
        <span className="logo">
          <Bot />
        </span>
      }
      title={updating ? t("int.updateTitle", { name: provider.name }) : t("int.connectTitle", { name: provider.name })}
      description={t("ai.keyLede")}
    >
      <form onSubmit={submit} className="col" style={{ gap: 16, paddingBottom: 14 }}>
        {provider.custom_base_url ? (
          <Field label={t("ai.baseUrl")} htmlFor="ai-base" hint={t("ai.baseUrlHint")}>
            <input
              id="ai-base"
              className="input mono"
              placeholder="https://openrouter.ai/api/v1"
              value={baseUrl}
              onChange={(e) => setBaseUrl(e.target.value)}
            />
            <div className="chips" style={{ marginTop: 8 }}>
              {presets.map((p) => (
                <button
                  key={p.base_url}
                  type="button"
                  className="chip"
                  aria-pressed={baseUrl === p.base_url}
                  onClick={() => setBaseUrl(p.base_url)}
                >
                  {p.name}
                </button>
              ))}
            </div>
          </Field>
        ) : null}
        <Field
          label={t("int.apiKey")}
          htmlFor="ai-key"
          error={error}
          hint={
            updating
              ? t("int.currentKey", { key: provider.masked_key ?? "—" })
              : provider.key_optional
                ? t("ai.keyOptional")
                : t("ai.keyHint")
          }
          aside={
            provider.keys_url ? (
              <a href={provider.keys_url} target="_blank" rel="noreferrer" className="tiny row gap-4" style={{ color: "var(--accent-text)" }}>
                {t("int.getKey")} <ExternalLink size={12} />
              </a>
            ) : null
          }
        >
          <input
            id="ai-key"
            className="input mono"
            type="password"
            autoComplete="off"
            spellCheck={false}
            value={key}
            onChange={(e) => setKey(e.target.value)}
            placeholder={updating ? (provider.masked_key ?? "") : t("int.keyPlaceholder")}
          />
        </Field>
        <Notice icon={<ShieldCheck />}>{t("int.securityNote")}</Notice>
        <div className="row" style={{ justifyContent: "flex-end" }}>
          <Button variant="ghost" onClick={onClose}>
            {t("common.cancel")}
          </Button>
          <Button type="submit" variant="primary" loading={busy} disabled={!canSave}>
            {t("int.validateSave")}
          </Button>
        </div>
      </form>
    </Modal>
  );
}

function TierModels({ provider }: { provider: AIProvider }) {
  const { t } = useI18n();
  const qc = useQueryClient();
  const set = async (tier: AITier, model: string) => {
    try {
      await api(`/api/ai/providers/${provider.key}/models`, {
        method: "PATCH",
        body: { [tier]: model === "__auto" ? null : model },
      });
      qc.invalidateQueries({ queryKey: aiKeys.providers });
    } catch (e) {
      toast.error((e as Error).message);
    }
  };
  return (
    <div className="col" style={{ gap: 8 }}>
      {TIERS.map((tier) => {
        const current = provider.tiers[tier];
        const auto = current.source !== "user";
        return (
          <div key={tier} className="row" style={{ gap: 10 }}>
            <div style={{ width: 120 }}>
              <div className="small" style={{ fontWeight: 500 }}>
                {t(`ai.tier.${tier}`)}
              </div>
              <div className="tiny subtle">{auto ? t("ai.auto") : t("ai.chosen")}</div>
            </div>
            <div className="grow" style={{ minWidth: 0 }}>
              <Select
                label={t(`ai.tier.${tier}`)}
                value={auto ? "__auto" : (current.model ?? "__auto")}
                onChange={(v) => set(tier, v)}
                options={[
                  {
                    value: "__auto",
                    label: current.model && auto ? t("ai.autoModel", { model: current.model }) : t("ai.auto"),
                  },
                  ...provider.models.map((m) => ({ value: m, label: m })),
                ]}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function ProviderCard({
  provider,
  presets,
}: {
  provider: AIProvider;
  presets: AIProviderList["presets"];
}) {
  const { t, locale } = useI18n();
  const qc = useQueryClient();
  const confirm = useConfirm();
  const update = useUpdateSettings();
  const [dialog, setDialog] = useState(false);
  const [testing, setTesting] = useState(false);
  const connected = provider.status !== "not_connected";

  const test = async () => {
    setTesting(true);
    try {
      const res = await api<{ ok: boolean; message: string | null }>(`/api/ai/providers/${provider.key}/test`, { body: {} });
      if (res.ok) toast.success(t("ai.testOk", { name: provider.name }));
      else toast.error(res.message ?? t("errors.generic"));
      qc.invalidateQueries({ queryKey: aiKeys.providers });
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setTesting(false);
    }
  };

  const disconnect = async () => {
    const ok = await confirm({
      title: t("int.disconnectTitle", { name: provider.name }),
      description: t("ai.disconnectBody"),
      confirmLabel: t("int.disconnect"),
      danger: true,
    });
    if (!ok) return;
    await api(`/api/ai/providers/${provider.key}`, { method: "DELETE" });
    qc.invalidateQueries({ queryKey: aiKeys.providers });
    qc.invalidateQueries({ queryKey: keys.settings });
    toast.success(t("int.disconnected", { name: provider.name }));
  };

  return (
    <div className="card provider-card" data-active={provider.active || undefined}>
      <div className="row" style={{ gap: 14, alignItems: "flex-start" }}>
        <span className="logo">
          <Bot />
        </span>
        <div className="grow">
          <div className="row wrap" style={{ gap: 8 }}>
            <div className="h2">{provider.name}</div>
            {provider.status === "connected" ? (
              <Badge tone="success">
                <span className="dot" /> {t("int.statusConnected")}
              </Badge>
            ) : provider.status === "error" ? (
              <Badge tone="danger">
                <span className="dot" /> {t("int.statusError")}
              </Badge>
            ) : (
              <Badge>{t("int.notConnected")}</Badge>
            )}
            {provider.active ? <Badge tone="accent">{t("int.active")}</Badge> : null}
          </div>
          <div className="small muted" style={{ marginTop: 4 }}>
            {t(`ai.desc.${provider.key}`)}
          </div>
        </div>
      </div>

      {connected ? (
        <>
          <dl className="kv">
            {provider.masked_key ? (
              <>
                <dt>{t("int.apiKey")}</dt>
                <dd className="mono">{provider.masked_key}</dd>
              </>
            ) : null}
            {provider.base_url ? (
              <>
                <dt>{t("ai.baseUrl")}</dt>
                <dd className="mono">{provider.base_url}</dd>
              </>
            ) : null}
            <dt>{t("ai.models")}</dt>
            <dd>{t("ai.modelCount", { n: provider.models.length })}</dd>
            <dt>{t("int.lastChecked")}</dt>
            <dd>{formatRelative(provider.last_checked_at, locale)}</dd>
          </dl>
          <TierModels provider={provider} />
        </>
      ) : (
        <div className="small subtle">{t("ai.notConnectedBody")}</div>
      )}

      {provider.status === "error" && provider.last_error ? <Notice tone="danger">{provider.last_error}</Notice> : null}

      <div className="row wrap" style={{ marginTop: "auto" }}>
        {connected ? (
          <>
            <Button icon={<Settings2 />} onClick={() => setDialog(true)}>
              {t("int.configure")}
            </Button>
            <Button icon={<PlugZap />} loading={testing} onClick={test}>
              {t("ai.refreshModels")}
            </Button>
            {!provider.active ? (
              <Button
                icon={<CircleDot />}
                onClick={() =>
                  update.mutate(
                    { ai_provider: provider.key },
                    {
                      onSuccess: () => {
                        qc.invalidateQueries({ queryKey: aiKeys.providers });
                        toast.success(t("int.nowActive", { name: provider.name }));
                      },
                    },
                  )
                }
              >
                {t("int.makeActive")}
              </Button>
            ) : null}
            <span className="spacer" />
            <Button variant="ghost" icon={<Unplug />} onClick={disconnect}>
              {t("int.disconnect")}
            </Button>
          </>
        ) : (
          <>
            <Button variant="primary" icon={<KeyRound />} onClick={() => setDialog(true)}>
              {t("int.connect")}
            </Button>
            <a className="btn btn-ghost" href={provider.docs_url} target="_blank" rel="noreferrer">
              {t("int.docs")} <ExternalLink />
            </a>
          </>
        )}
      </div>
      <KeyDialog key={String(dialog)} provider={provider} presets={presets} open={dialog} onClose={() => setDialog(false)} />
    </div>
  );
}

function UsageCard() {
  const { t, locale } = useI18n();
  const usage = useAIUsage();
  const u = usage.data;
  return (
    <div className="card">
      <div className="card-head">
        <BarChart3 size={16} color="var(--text-3)" />
        <div className="h3 grow">{t("ai.usageTitle")}</div>
        <span className="tiny subtle">{t("ai.usageHint")}</span>
      </div>
      <div className="card-body">
        {usage.isLoading ? (
          <Skeleton style={{ height: 80 }} />
        ) : !u || u.requests === 0 ? (
          <div className="small subtle">{t("ai.noUsage")}</div>
        ) : (
          <div className="col" style={{ gap: 14 }}>
            <div className="row wrap" style={{ gap: 24 }}>
              {[
                [t("ai.requests"), formatNumber(u.requests, locale)],
                [t("ai.inputTokens"), formatNumber(u.input_tokens, locale)],
                [t("ai.cachedTokens"), formatNumber(u.cached_tokens, locale)],
                [t("ai.outputTokens"), formatNumber(u.output_tokens, locale)],
                [t("ai.freeAnswers"), formatNumber(u.free_answers, locale)],
              ].map(([label, value]) => (
                <div key={label}>
                  <div className="tiny subtle">{label}</div>
                  <div className="stat-value" style={{ fontSize: 22 }}>
                    {value}
                  </div>
                </div>
              ))}
            </div>
            <table className="matrix">
              <thead>
                <tr>
                  <th>{t("ai.model")}</th>
                  <th>{t("ai.requests")}</th>
                  <th>{t("ai.inputTokens")}</th>
                  <th>{t("ai.outputTokens")}</th>
                </tr>
              </thead>
              <tbody>
                {u.by_model.map((row) => (
                  <tr key={`${row.provider}-${row.model}`}>
                    <td>
                      <span className="mono small">{row.model}</span>
                      <div className="tiny subtle">{row.provider}</div>
                    </td>
                    <td className="small">{formatNumber(row.requests, locale)}</td>
                    <td className="small">
                      {formatNumber(row.input_tokens, locale)}
                      {row.cached_tokens ? <span className="subtle"> ({t("chat.cachedShort", { n: formatNumber(row.cached_tokens, locale) })})</span> : null}
                    </td>
                    <td className="small">{formatNumber(row.output_tokens, locale)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}

export function AIProvidersTab() {
  const { t } = useI18n();
  const qc = useQueryClient();
  const providers = useAIProviders();
  const update = useUpdateSettings();
  const data = providers.data;

  const save = (patch: Parameters<typeof update.mutate>[0]) =>
    update.mutate(patch, { onSuccess: () => qc.invalidateQueries({ queryKey: aiKeys.providers }) });

  return (
    <>
      <div className="card" style={{ marginBottom: 16 }}>
        <div className="setting-row">
          <div className="grow">
            <div className="row gap-4" style={{ fontWeight: 500 }}>
              <Gauge size={15} /> {t("ai.mode")}
            </div>
            <div>{t("ai.modeHint")}</div>
          </div>
          <div className="setting-control">
            <Segmented
              label={t("ai.mode")}
              value={data?.mode ?? "low"}
              onChange={(v) => save({ ai_mode: v })}
              options={[
                { value: "low", label: t("ai.mode.low") },
                { value: "balanced", label: t("ai.mode.balanced") },
                { value: "high", label: t("ai.mode.high") },
              ]}
            />
          </div>
        </div>
        <div className="setting-row">
          <div className="grow">
            <div className="row gap-4" style={{ fontWeight: 500 }}>
              <RefreshCw size={15} /> {t("ai.autoRoute")}
            </div>
            <div>{t("ai.autoRouteHint")}</div>
          </div>
          <div className="setting-control">
            <Switch label={t("ai.autoRoute")} checked={data?.auto_route ?? true} onChange={(v) => save({ ai_auto_route: v })} />
          </div>
        </div>
      </div>
      {providers.isError ? <Notice tone="danger">{(providers.error as Error).message}</Notice> : null}
      <div className="providers">
        {providers.isLoading
          ? [0, 1, 2, 3].map((i) => <Skeleton key={i} style={{ height: 240, borderRadius: 14 }} />)
          : data?.providers.map((p) => <ProviderCard key={p.key} provider={p} presets={data.presets} />)}
      </div>
      <UsageCard />
    </>
  );
}
