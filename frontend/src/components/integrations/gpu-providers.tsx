"use client";

import { useQueryClient } from "@tanstack/react-query";
import { Cpu, ExternalLink, KeyRound, PlugZap, Settings2, ShieldCheck, Unplug } from "lucide-react";
import { type FormEvent, useState } from "react";
import { toast } from "sonner";

import { Badge, Button, Field, Modal, Notice, Skeleton, useConfirm } from "@/components/ui";
import { api, ApiError } from "@/lib/api";
import { formatRelative } from "@/lib/format";
import { useI18n } from "@/lib/i18n";
import { gpuKeys, useGPUProviders } from "@/lib/queries";
import type { GPUProvider } from "@/lib/types";

function CredentialsDialog({ provider, open, onClose }: { provider: GPUProvider; open: boolean; onClose: () => void }) {
  const { t } = useI18n();
  const qc = useQueryClient();
  const [values, setValues] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const updating = provider.status !== "not_connected";

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    setBusy(true);
    try {
      await api(`/api/gpu/providers/${provider.key}`, { method: "PUT", body: values });
      await qc.invalidateQueries({ queryKey: gpuKeys.providers });
      qc.invalidateQueries({ queryKey: ["gpu", provider.key] });
      toast.success(t("int.connected", { name: provider.name }));
      onClose();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : t("errors.generic"));
    } finally {
      setBusy(false);
    }
  };

  const ready = provider.credential_fields.every((f) => (values[f] ?? "").trim().length >= 6);

  return (
    <Modal
      open={open}
      onClose={onClose}
      icon={
        <span className="logo">
          <Cpu />
        </span>
      }
      title={updating ? t("int.updateTitle", { name: provider.name }) : t("int.connectTitle", { name: provider.name })}
      description={t(`gpuint.lede.${provider.key}`)}
    >
      <form onSubmit={submit} className="col" style={{ gap: 16, paddingBottom: 14 }}>
        {provider.credential_fields.map((field, i) => (
          <Field
            key={field}
            label={t(`gpuint.field.${field}`)}
            htmlFor={`gpu-${field}`}
            error={i === provider.credential_fields.length - 1 ? error : null}
            aside={
              i === 0 ? (
                <a href={provider.keys_url} target="_blank" rel="noreferrer" className="tiny row gap-4" style={{ color: "var(--accent-text)" }}>
                  {t("int.getKey")} <ExternalLink size={12} />
                </a>
              ) : null
            }
          >
            <input
              id={`gpu-${field}`}
              className="input mono"
              type={field === "token_id" ? "text" : "password"}
              autoComplete="off"
              spellCheck={false}
              value={values[field] ?? ""}
              onChange={(e) => setValues((v) => ({ ...v, [field]: e.target.value }))}
            />
          </Field>
        ))}
        <Notice icon={<ShieldCheck />}>{t("int.securityNote")}</Notice>
        <div className="row" style={{ justifyContent: "flex-end" }}>
          <Button variant="ghost" onClick={onClose}>
            {t("common.cancel")}
          </Button>
          <Button type="submit" variant="primary" loading={busy} disabled={!ready}>
            {t("int.validateSave")}
          </Button>
        </div>
      </form>
    </Modal>
  );
}

function GPUCard({ provider }: { provider: GPUProvider }) {
  const { t, locale } = useI18n();
  const qc = useQueryClient();
  const confirm = useConfirm();
  const [dialog, setDialog] = useState(false);
  const [testing, setTesting] = useState(false);
  const connected = provider.status !== "not_connected";

  const test = async () => {
    setTesting(true);
    try {
      const res = await api<{ ok: boolean; message: string | null }>(`/api/gpu/providers/${provider.key}/test`, { body: {} });
      if (res.ok) toast.success(t("int.testOk", { name: provider.name }));
      else toast.error(res.message ?? t("errors.generic"));
      qc.invalidateQueries({ queryKey: gpuKeys.providers });
    } catch (e) {
      toast.error((e as Error).message);
    } finally {
      setTesting(false);
    }
  };

  const disconnect = async () => {
    const ok = await confirm({
      title: t("int.disconnectTitle", { name: provider.name }),
      description: t("gpuint.disconnectBody"),
      confirmLabel: t("int.disconnect"),
      danger: true,
    });
    if (!ok) return;
    try {
      await api(`/api/gpu/providers/${provider.key}`, { method: "DELETE" });
    } catch (err) {
      toast.error(err instanceof ApiError ? err.message : t("errors.generic"));
      return;
    }
    qc.invalidateQueries({ queryKey: gpuKeys.providers });
    toast.success(t("int.disconnected", { name: provider.name }));
  };

  return (
    <div className="card provider-card">
      <div className="row" style={{ gap: 14, alignItems: "flex-start" }}>
        <span className="logo">
          <Cpu />
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
          </div>
          <div className="small muted" style={{ marginTop: 4 }}>
            {t(`gpuint.desc.${provider.key}`)}
          </div>
        </div>
      </div>
      {connected ? (
        <dl className="kv">
          <dt>{t("gpuint.credentials")}</dt>
          <dd className="mono">{provider.masked_key}</dd>
          <dt>{t("int.lastChecked")}</dt>
          <dd>{formatRelative(provider.last_checked_at, locale)}</dd>
        </dl>
      ) : (
        <div className="small subtle">{t("gpuint.notConnectedBody")}</div>
      )}
      {provider.status === "error" && provider.last_error ? <Notice tone="danger">{provider.last_error}</Notice> : null}
      <div className="row wrap" style={{ marginTop: "auto" }}>
        {connected ? (
          <>
            <Button icon={<Settings2 />} onClick={() => setDialog(true)}>
              {t("int.configure")}
            </Button>
            <Button icon={<PlugZap />} loading={testing} onClick={test}>
              {t("int.test")}
            </Button>
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
      <CredentialsDialog key={String(dialog)} provider={provider} open={dialog} onClose={() => setDialog(false)} />
    </div>
  );
}

export function GPUProvidersTab() {
  const providers = useGPUProviders();
  return (
    <>
      {providers.isError ? <Notice tone="danger">{(providers.error as Error).message}</Notice> : null}
      <div className="providers">
        {providers.isLoading
          ? [0, 1].map((i) => <Skeleton key={i} style={{ height: 220, borderRadius: 14 }} />)
          : providers.data?.providers.map((p) => <GPUCard key={p.key} provider={p} />)}
      </div>
    </>
  );
}
