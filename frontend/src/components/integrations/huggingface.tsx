"use client";

import { CircleCheck, CircleDashed, ExternalLink, KeyRound, RefreshCw, ShieldCheck, Unplug } from "lucide-react";
import { type FormEvent, useState } from "react";
import { toast } from "sonner";

import { Badge, Button, Field, Notice, Skeleton, useConfirm } from "@/components/ui";
import { ApiError } from "@/lib/api";
import { useT } from "@/lib/i18n";
import type { MessageKey } from "@/lib/i18n";
import { useHfStatus, useRefreshHfStatus, useSaveSecret } from "@/lib/queries";
import type { HfStatus } from "@/lib/types";

/** Short status for other pages (LoRA training): what is still missing, if anything. */
export function hfProblem(status: HfStatus | undefined, variant: "raw" | "turbo"): MessageKey | null {
  if (!status) return null;
  if (!status.set) return "hf.needToken";
  if (status.token === "invalid") return "hf.tokenInvalid";
  const repo = status.repos.find((r) => r.variant === variant);
  if (repo?.access === "needs_license") return "hf.needLicense";
  return null;
}

function TokenForm({ onSaved }: { onSaved?: () => void }) {
  const t = useT();
  const save = useSaveSecret();
  const [value, setValue] = useState("");
  const [error, setError] = useState<string | null>(null);

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    try {
      await save.mutateAsync({ name: "huggingface", value: value.trim() });
      setValue("");
      toast.success(t("hf.saved"));
      onSaved?.();
    } catch (err) {
      setError(err instanceof ApiError ? err.message : t("errors.generic"));
    }
  };

  return (
    <form onSubmit={submit} className="col gap-12">
      <Field
        label={t("hf.tokenLabel")}
        htmlFor="hf-token"
        error={error}
        aside={
          <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer" className="tiny row gap-4" style={{ color: "var(--accent-text)" }}>
            {t("hf.getToken")} <ExternalLink size={12} />
          </a>
        }
      >
        <div className="row" style={{ gap: 8 }}>
          <input
            id="hf-token"
            className="input mono grow"
            type="password"
            autoComplete="off"
            spellCheck={false}
            placeholder="hf_…"
            value={value}
            onChange={(e) => setValue(e.target.value)}
          />
          <Button type="submit" variant="primary" disabled={value.trim().length < 8} loading={save.isPending}>
            {t("common.save")}
          </Button>
        </div>
      </Field>
      <div className="hint">{t("hf.tokenHint")}</div>
    </form>
  );
}

export function HuggingFaceTab() {
  const t = useT();
  const confirm = useConfirm();
  const status = useHfStatus();
  const refresh = useRefreshHfStatus();
  const save = useSaveSecret();
  const s = status.data;

  const check = async () => {
    try {
      const res = await refresh.mutateAsync();
      if (res.ready) toast.success(t("hf.allReady"));
    } catch (e) {
      toast.error((e as Error).message);
    }
  };

  const remove = async () => {
    const ok = await confirm({
      title: t("hf.removeTitle"),
      description: t("hf.removeBody"),
      confirmLabel: t("int.disconnect"),
      danger: true,
    });
    if (ok) save.mutate({ name: "huggingface", value: null });
  };

  if (status.isLoading) return <Skeleton style={{ height: 320, borderRadius: 14 }} />;
  if (status.isError || !s) return <Notice tone="danger">{(status.error as Error)?.message ?? t("errors.generic")}</Notice>;

  return (
    <div className="hf-layout">
      <section className="card provider-card">
        <div className="row" style={{ gap: 14, alignItems: "flex-start" }}>
          <span className="logo">
            <KeyRound />
          </span>
          <div className="grow">
            <div className="row wrap" style={{ gap: 8 }}>
              <div className="h2">Hugging Face</div>
              {!s.set ? (
                <Badge>{t("int.notConnected")}</Badge>
              ) : s.token === "invalid" ? (
                <Badge tone="danger">
                  <span className="dot" /> {t("hf.badgeInvalid")}
                </Badge>
              ) : s.ready ? (
                <Badge tone="success">
                  <span className="dot" /> {t("hf.badgeReady")}
                </Badge>
              ) : (
                <Badge tone="warning">
                  <span className="dot" /> {t("hf.badgeAlmost")}
                </Badge>
              )}
            </div>
            <div className="small muted" style={{ marginTop: 4 }}>
              {t("hf.lede")}
            </div>
          </div>
        </div>

        <ol className="hf-steps">
          <li>
            <div className="row" style={{ gap: 8 }}>
              <span className="step-num">1</span>
              <span className="h3 grow">{t("hf.step1")}</span>
              {s.set && s.token !== "invalid" ? <CircleCheck size={18} color="var(--success)" /> : null}
            </div>
            {s.set ? (
              <div className="row wrap" style={{ gap: 8, marginTop: 8 }}>
                <span className="small mono">{s.masked}</span>
                {s.account ? <span className="small subtle">· {t("hf.account", { name: s.account })}</span> : null}
                <span className="spacer" />
                <Button size="sm" variant="ghost" icon={<Unplug />} onClick={remove}>
                  {t("int.disconnect")}
                </Button>
              </div>
            ) : null}
            {!s.set || s.token === "invalid" ? (
              <div style={{ marginTop: 10 }}>
                {s.token === "invalid" ? <Notice tone="danger">{t("hf.tokenInvalid")}</Notice> : null}
                <TokenForm />
              </div>
            ) : null}
          </li>
          <li>
            <div className="row" style={{ gap: 8 }}>
              <span className="step-num">2</span>
              <span className="h3 grow">{t("hf.step2")}</span>
              {s.ready ? <CircleCheck size={18} color="var(--success)" /> : null}
            </div>
            <div className="hint" style={{ marginTop: 6 }}>{t("hf.step2Hint")}</div>
            <div className="col" style={{ gap: 8, marginTop: 10 }}>
              {s.repos.map((r) => (
                <div key={r.repo} className="hf-repo">
                  {r.access === "granted" ? <CircleCheck size={16} color="var(--success)" /> : <CircleDashed size={16} />}
                  <div className="grow" style={{ minWidth: 0 }}>
                    <div className="small" style={{ fontWeight: 600 }}>{r.name}</div>
                    <div className="tiny subtle mono truncate">{r.repo}</div>
                  </div>
                  {r.access === "granted" ? (
                    <Badge tone="success">{t("hf.access.granted")}</Badge>
                  ) : (
                    <a className="btn btn-sm btn-primary" href={r.url} target="_blank" rel="noreferrer">
                      {t("hf.acceptLicense")} <ExternalLink />
                    </a>
                  )}
                </div>
              ))}
            </div>
            {s.set ? (
              <div className="row wrap" style={{ gap: 8, marginTop: 12 }}>
                <Button size="sm" icon={<RefreshCw />} loading={refresh.isPending} onClick={check}>
                  {t("hf.checkAgain")}
                </Button>
                {s.token === "unknown" ? <span className="tiny subtle">{t("hf.unreachable")}</span> : null}
              </div>
            ) : (
              <div className="tiny subtle" style={{ marginTop: 10 }}>{t("hf.tokenFirst")}</div>
            )}
          </li>
        </ol>
        <Notice icon={<ShieldCheck />}>{t("hf.security")}</Notice>
      </section>
    </div>
  );
}
