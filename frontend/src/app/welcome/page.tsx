"use client";

import { useQueryClient } from "@tanstack/react-query";
import { ArrowRight, Lock, Mail, UserRound } from "lucide-react";
import { useRouter } from "next/navigation";
import { type FormEvent, useCallback, useEffect, useRef, useState } from "react";

import { BrandMark } from "@/components/logos";
import { Button, Field, Notice, Segmented, Spinner } from "@/components/ui";
import { api, ApiError } from "@/lib/api";
import { useI18n } from "@/lib/i18n";
import { keys, useAuthState } from "@/lib/queries";
import type { User } from "@/lib/types";

export default function WelcomePage() {
  const { t, lang } = useI18n();
  const router = useRouter();
  const qc = useQueryClient();
  const auth = useAuthState();
  const [mode, setMode] = useState<"signin" | "signup" | null>(null);
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [autoError, setAutoError] = useState<string | null>(null);
  const autoStarted = useRef(false);

  useEffect(() => {
    if (auth.data?.user) router.replace("/dashboard");
  }, [auth.data, router]);

  // No sign-in screen: open the account on this Mac right away.
  const autoLogin = useCallback(async () => {
    setAutoError(null);
    try {
      const res = await api<{ user: User }>("/api/auth/auto", { body: { language: lang } });
      qc.setQueryData(keys.auth, { has_users: true, user: res.user, auto_login: true });
      qc.removeQueries({ queryKey: keys.settings });
      router.replace("/dashboard");
    } catch (err) {
      setAutoError(err instanceof ApiError ? err.message : t("errors.generic"));
    }
  }, [lang, qc, router, t]);

  useEffect(() => {
    if (!auth.data?.auto_login || auth.data.user || autoStarted.current) return;
    autoStarted.current = true;
    void autoLogin();
  }, [auth.data, autoLogin]);

  const effectiveMode = mode ?? (auth.data?.has_users ? "signin" : "signup");

  const submit = async (e: FormEvent) => {
    e.preventDefault();
    setError(null);
    if (effectiveMode === "signup" && password.length < 8) {
      setError(t("auth.passwordTooShort"));
      return;
    }
    setBusy(true);
    try {
      const path = effectiveMode === "signup" ? "/api/auth/register" : "/api/auth/login";
      const body =
        effectiveMode === "signup" ? { name, email, password, language: lang } : { email, password };
      const res = await api<{ user: User }>(path, { body });
      qc.setQueryData(keys.auth, { has_users: true, user: res.user });
      qc.removeQueries({ queryKey: keys.settings });
      router.replace("/dashboard");
    } catch (err) {
      setError(err instanceof ApiError ? err.message : t("errors.generic"));
    } finally {
      setBusy(false);
    }
  };

  if (auth.isLoading || (auth.data?.auto_login && !autoError)) {
    return (
      <div style={{ minHeight: "100vh", display: "grid", placeItems: "center" }}>
        <Spinner size={22} />
      </div>
    );
  }
  if (auth.data?.auto_login && autoError) {
    return (
      <div style={{ minHeight: "100vh", display: "grid", placeItems: "center", padding: 24 }}>
        <div className="col gap-12" style={{ maxWidth: 420 }}>
          <Notice tone="danger">{autoError}</Notice>
          <Button variant="primary" onClick={() => void autoLogin()}>
            {t("auth.tryAgain")}
          </Button>
        </div>
      </div>
    );
  }

  return (
    <div className="auth">
      <section className="auth-art">
        <div className="row" style={{ position: "relative", zIndex: 1, gap: 12 }}>
          <BrandMark />
          <div className="brand-name">AI Image Studio</div>
        </div>
        <div style={{ position: "relative", zIndex: 1, maxWidth: 460 }}>
          <div className="eyebrow" style={{ marginBottom: 14 }}>
            {t("auth.eyebrow")}
          </div>
          <h1 className="h1" style={{ fontSize: 40, lineHeight: 1.1 }}>
            {t("auth.headline")}
          </h1>
          <p className="muted" style={{ fontSize: 15, marginTop: 16 }}>
            {t("auth.lede")}
          </p>
          <div className="row wrap" style={{ marginTop: 24, gap: 6 }}>
            {["Nano Banana Pro", "Seedream", "Wan 2.7", "Wan 2.7 Pro"].map((m) => (
              <span key={m} className="badge outline">
                {m}
              </span>
            ))}
          </div>
        </div>
        <div className="tiny subtle" style={{ position: "relative", zIndex: 1 }}>
          {t("auth.localNote")}
        </div>
      </section>
      <section className="auth-form">
        <form onSubmit={submit} noValidate>
          <div>
            <h2 className="h1" style={{ fontSize: 24 }}>
              {effectiveMode === "signup" ? t("auth.createTitle") : t("auth.signInTitle")}
            </h2>
            <p className="muted" style={{ margin: "6px 0 0" }}>
              {effectiveMode === "signup" ? t("auth.createLede") : t("auth.signInLede")}
            </p>
          </div>
          <Segmented
            block
            label={t("auth.mode")}
            value={effectiveMode}
            onChange={(v) => {
              setMode(v);
              setError(null);
            }}
            options={[
              { value: "signin", label: t("auth.signIn") },
              { value: "signup", label: t("auth.createAccount") },
            ]}
          />
          {effectiveMode === "signup" ? (
            <Field label={t("auth.name")} htmlFor="name">
              <div className="input-wrap">
                <UserRound />
                <input
                  id="name"
                  className="input"
                  autoComplete="name"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  required
                />
              </div>
            </Field>
          ) : null}
          <Field label={t("auth.email")} htmlFor="email">
            <div className="input-wrap">
              <Mail />
              <input
                id="email"
                type="email"
                className="input"
                autoComplete="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
              />
            </div>
          </Field>
          <Field
            label={t("auth.password")}
            htmlFor="password"
            hint={effectiveMode === "signup" ? t("auth.passwordHint") : undefined}
          >
            <div className="input-wrap">
              <Lock />
              <input
                id="password"
                type="password"
                className="input"
                autoComplete={effectiveMode === "signup" ? "new-password" : "current-password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
              />
            </div>
          </Field>
          {error ? <Notice tone="danger">{error}</Notice> : null}
          <Button
            type="submit"
            variant="primary"
            size="lg"
            block
            loading={busy}
            disabled={!email || !password || (effectiveMode === "signup" && !name)}
          >
            {effectiveMode === "signup" ? t("auth.createAccount") : t("auth.signIn")}
            <ArrowRight />
          </Button>
        </form>
      </section>
    </div>
  );
}
