"use client";

import { createContext, useCallback, useContext, useMemo, type ReactNode } from "react";

import { en, type MessageKey } from "@/lib/i18n/en";
import { sk } from "@/lib/i18n/sk";

export type Language = "en" | "sk";

const dictionaries: Record<Language, Record<MessageKey, string>> = { en, sk };

type Vars = Record<string, string | number>;

interface I18n {
  lang: Language;
  locale: string;
  t: (key: MessageKey, vars?: Vars) => string;
}

const I18nContext = createContext<I18n | null>(null);

function format(template: string, vars?: Vars) {
  if (!vars) return template;
  return template.replace(/\{(\w+)\}/g, (_, name: string) =>
    name in vars ? String(vars[name]) : `{${name}}`,
  );
}

export function I18nProvider({ lang, children }: { lang: Language; children: ReactNode }) {
  const t = useCallback(
    (key: MessageKey, vars?: Vars) => format(dictionaries[lang][key] ?? en[key] ?? key, vars),
    [lang],
  );
  const value = useMemo(
    () => ({ lang, locale: lang === "sk" ? "sk-SK" : "en-US", t }),
    [lang, t],
  );
  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

export function useI18n(): I18n {
  const ctx = useContext(I18nContext);
  if (!ctx) throw new Error("useI18n outside I18nProvider");
  return ctx;
}

export function useT() {
  return useI18n().t;
}

export function guessLanguage(): Language {
  if (typeof navigator === "undefined") return "en";
  const lang = navigator.language?.toLowerCase() ?? "";
  return lang.startsWith("sk") || lang.startsWith("cs") ? "sk" : "en";
}

export type { MessageKey };
