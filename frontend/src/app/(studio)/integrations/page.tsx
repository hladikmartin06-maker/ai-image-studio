"use client";

import { useQueryClient } from "@tanstack/react-query";
import {
  Check,
  CircleCheck,
  CircleDot,
  ExternalLink,
  KeyRound,
  Minus,
  PlugZap,
  Settings2,
  ShieldCheck,
  Unplug,
} from "lucide-react";
import { type FormEvent, useState } from "react";
import { toast } from "sonner";

import { PageHeader } from "@/components/layout/page-header";
import { ProviderLogo } from "@/components/logos";
import { Badge, Button, Field, Modal, Notice, Skeleton, useConfirm } from "@/components/ui";
import { api, ApiError } from "@/lib/api";
import { formatRelative } from "@/lib/format";
import { useI18n } from "@/lib/i18n";
import { keys, useBalance, useCatalog, useIntegrations } from "@/lib/queries";
import type { Integration, ProviderKey } from "@/lib/types";

function KeyDialog({
  integration,
  open,
  onClose,
}: {
  integration: Integration;
  open: boolean;
  onClose: () => void;
}) {
  const { t } = useI18n();
  const qc = useQueryClient();
  const [key, setKey] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const updating = integration.status !== "not_connected";

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      const res = await api<Integration>(`/api/integrations/${integration.key}`, {
        method: "PUT",
        body: { api_key: key.trim() },
      });
      await qc.invalidateQueries({ queryKey: keys.integrations });
      qc.invalidateQueries({ queryKey: keys.settings });
      qc.invalidateQueries({ queryKey: keys.balance(integration.key) });
      if (res.status === "error") toast.warning(res.last_error ?? t("int.savedWithError"));
      else toast.success(t("int.connected", { name: integration.name }));
      setKey("");
      onClose();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : t("errors.generic"));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal
      open={open}
      onClose={() => {
        setKey("");
        setError(null);
        onClose();
      }}
      icon={<ProviderLogo provider={integration.key} />}
      title={updating ? t("int.updateTitle", { name: integration.name }) : t("int.connectTitle", { name: integration.name })}
      description={t("int.keyLede")}
    >
      <form onSubmit={submit} className="col" style={{ gap: 16, paddingBottom: 14 }}>
        <Field
          label={t("int.apiKey")}
          htmlFor="api-key"
          hint={updating ? t("int.currentKey", { key: integration.masked_key ?? "" }) : t("int.keyHint")}
          error={error}
          aside={
            <a href={integration.keys_url} target="_blank" rel="noreferrer" className="tiny row gap-4" style={{ color: "var(--accent-text)" }}>
              {t("int.getKey")} <ExternalLink size={12} />
            </a>
          }
        >
          <input
            id="api-key"
            className="input mono"
            type="password"
            autoComplete="off"
            spellCheck={false}
            value={key}
            onChange={(e) => setKey(e.target.value)}
            placeholder={updating ? integration.masked_key ?? "" : t("int.keyPlaceholder")}
            aria-invalid={Boolean(error) || undefined}
          />
        </Field>
        <Notice icon={<ShieldCheck />}>{t("int.securityNote")}</Notice>
        <div className="row" style={{ justifyContent: "flex-end" }}>
          <Button variant="ghost" onClick={onClose}>
            {t("common.cancel")}
          </Button>
          <Button type="submit" variant="primary" loading={busy} disabled={key.trim().length < 8}>
            {t("int.validateSave")}
          </Button>
        </div>
      </form>
    </Modal>
  );
}

function ProviderCard({ integration, onActivate }: { integration: Integration; onActivate: () => void }) {
  const { t, locale } = useI18n();
  const qc = useQueryClient();
  const confirm = useConfirm();
  const [dialog, setDialog] = useState(false);
  const [testing, setTesting] = useState(false);
  const [disconnecting, setDisconnecting] = useState(false);
  const connected = integration.status !== "not_connected";
  const balance = useBalance(integration.key, connected && integration.supports_balance);

  const test = async () => {
    setTesting(true);
    try {
      const res = await api<{ ok: boolean; message: string }>(`/api/integrations/${integration.key}/test`, {
        body: {},
      });
      if (res.ok) toast.success(t("int.testOk", { name: integration.name }));
      else toast.error(res.message);
      await qc.invalidateQueries({ queryKey: keys.integrations });
      qc.invalidateQueries({ queryKey: keys.balance(integration.key) });
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setTesting(false);
    }
  };

  const disconnect = async () => {
    const ok = await confirm({
      title: t("int.disconnectTitle", { name: integration.name }),
      description: t("int.disconnectBody"),
      confirmLabel: t("int.disconnect"),
      danger: true,
    });
    if (!ok) return;
    setDisconnecting(true);
    try {
      await api(`/api/integrations/${integration.key}`, { method: "DELETE" });
      await qc.invalidateQueries({ queryKey: keys.integrations });
      qc.invalidateQueries({ queryKey: keys.settings });
      toast.success(t("int.disconnected", { name: integration.name }));
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setDisconnecting(false);
    }
  };

  const statusBadge =
    integration.status === "connected" ? (
      <Badge tone="success">
        <span className="dot" /> {t("int.statusConnected")}
      </Badge>
    ) : integration.status === "error" ? (
      <Badge tone="danger">
        <span className="dot" /> {t("int.statusError")}
      </Badge>
    ) : (
      <Badge>
        <span className="dot" style={{ opacity: 0.5 }} /> {t("int.notConnected")}
      </Badge>
    );

  return (
    <div className="card provider-card" data-active={integration.active || undefined}>
      <div className="row" style={{ gap: 14, alignItems: "flex-start" }}>
        <ProviderLogo provider={integration.key} />
        <div className="grow">
          <div className="row wrap" style={{ gap: 8 }}>
            <div className="h2">{integration.name}</div>
            {statusBadge}
            {integration.active ? <Badge tone="accent">{t("int.active")}</Badge> : null}
          </div>
          <div className="small muted" style={{ marginTop: 4 }}>
            {t(`int.desc.${integration.key}`)}
          </div>
        </div>
      </div>

      {connected ? (
        <dl className="kv">
          <dt>{t("int.apiKey")}</dt>
          <dd className="mono">{integration.masked_key}</dd>
          <dt>{t("int.lastChecked")}</dt>
          <dd>{formatRelative(integration.last_checked_at, locale)}</dd>
          <dt>{t("dash.balance")}</dt>
          <dd>
            {!integration.supports_balance ? (
              <span className="subtle">{t("dash.atlasNoBalance")}</span>
            ) : balance.isLoading ? (
              <Skeleton style={{ width: 70, height: 16 }} />
            ) : balance.data?.balance !== null && balance.data?.balance !== undefined ? (
              balance.data.balance.toLocaleString(locale, { style: "currency", currency: "USD" })
            ) : (
              <span className="subtle">{balance.data?.error ?? t("dash.balanceUnavailable")}</span>
            )}
          </dd>
        </dl>
      ) : (
        <div className="small subtle">{t("int.notConnectedBody")}</div>
      )}

      {integration.status === "error" && integration.last_error ? (
        <Notice tone="danger">{integration.last_error}</Notice>
      ) : null}

      <div className="row wrap" style={{ marginTop: "auto" }}>
        {connected ? (
          <>
            <Button icon={<Settings2 />} onClick={() => setDialog(true)}>
              {t("int.configure")}
            </Button>
            <Button icon={<PlugZap />} loading={testing} onClick={test}>
              {t("int.test")}
            </Button>
            {!integration.active ? (
              <Button icon={<CircleDot />} onClick={onActivate}>
                {t("int.makeActive")}
              </Button>
            ) : null}
            <span className="spacer" />
            <Button variant="ghost" icon={<Unplug />} loading={disconnecting} onClick={disconnect}>
              {t("int.disconnect")}
            </Button>
          </>
        ) : (
          <>
            <Button variant="primary" icon={<KeyRound />} onClick={() => setDialog(true)}>
              {t("int.connect")}
            </Button>
            <a className="btn btn-ghost" href={integration.docs_url} target="_blank" rel="noreferrer">
              {t("int.docs")} <ExternalLink />
            </a>
          </>
        )}
      </div>
      <KeyDialog integration={integration} open={dialog} onClose={() => setDialog(false)} />
    </div>
  );
}

export default function IntegrationsPage() {
  const { t } = useI18n();
  const qc = useQueryClient();
  const integrations = useIntegrations();
  const catalog = useCatalog();

  const activate = async (provider: ProviderKey) => {
    try {
      await api(`/api/integrations/${provider}/activate`, { body: {} });
      await qc.invalidateQueries({ queryKey: keys.integrations });
      qc.invalidateQueries({ queryKey: keys.settings });
      toast.success(t("int.nowActive", { name: provider === "wavespeed" ? "WaveSpeed" : "Atlas Cloud" }));
    } catch (e) {
      toast.error((e as Error).message);
    }
  };

  return (
    <div className="page">
      <PageHeader title={t("int.title")} subtitle={t("int.subtitle")} />
      {integrations.isError ? <Notice tone="danger">{(integrations.error as Error).message}</Notice> : null}
      <div className="providers">
        {integrations.isLoading
          ? [0, 1].map((i) => <Skeleton key={i} style={{ height: 260, borderRadius: 14 }} />)
          : integrations.data?.integrations.map((i) => (
              <ProviderCard key={i.key} integration={i} onActivate={() => activate(i.key)} />
            ))}
      </div>

      <div className="card">
        <div className="card-head">
          <CircleCheck size={16} color="var(--text-3)" />
          <div className="h3 grow">{t("int.matrixTitle")}</div>
          <span className="tiny subtle">{t("int.matrixHint")}</span>
        </div>
        <div style={{ overflowX: "auto" }}>
          <table className="matrix">
            <thead>
              <tr>
                <th>{t("viewer.model")}</th>
                <th>{t("int.content")}</th>
                {(catalog.data?.providers ?? []).map((p) => (
                  <th key={p.key}>{p.name}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {(catalog.data?.models ?? []).map((m) => (
                <tr key={m.key}>
                  <td>
                    <div style={{ fontWeight: 600 }}>{m.name}</div>
                    <div className="tiny subtle">{m.family}</div>
                  </td>
                  <td>
                    <Badge tone={m.content === "sfw" ? undefined : "accent"}>{t(`content.${m.content}`)}</Badge>
                  </td>
                  {(catalog.data?.providers ?? []).map((p) => {
                    const variants = m.variants.filter((v) => v.provider === p.key);
                    return (
                      <td key={p.key}>
                        {variants.length ? (
                          <div className="col" style={{ gap: 4 }}>
                            {variants.map((v) => (
                              <a
                                key={v.mode}
                                href={v.docs_url}
                                target="_blank"
                                rel="noreferrer"
                                className="row small"
                                style={{ gap: 6 }}
                                title={v.model_id}
                              >
                                <Check size={14} className="yes" />
                                {v.mode === "image-edit" ? t("gen.modeEdit") : t("gen.modeText")}
                                <span className="mono subtle" style={{ fontSize: 11 }}>
                                  {v.model_id}
                                </span>
                              </a>
                            ))}
                          </div>
                        ) : (
                          <Minus size={14} className="no" />
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
