"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { type ReactNode, useEffect, useState } from "react";
import { Toaster } from "sonner";

import { ConfirmProvider } from "@/components/ui";
import { ApiError } from "@/lib/api";
import { guessLanguage, I18nProvider, type Language } from "@/lib/i18n";
import { useAuthState, useSettings } from "@/lib/queries";

const LOCAL_KEY = "studio:appearance";

interface Appearance {
  theme: "dark" | "light" | "system";
  density: string;
  motion: boolean;
  language: Language;
}

function readLocal(): Partial<Appearance> {
  try {
    return JSON.parse(localStorage.getItem(LOCAL_KEY) ?? "{}");
  } catch {
    return {};
  }
}

function applyAppearance(a: Appearance) {
  const root = document.documentElement;
  const dark =
    a.theme === "dark" ||
    (a.theme === "system" && window.matchMedia("(prefers-color-scheme: dark)").matches);
  root.dataset.theme = dark ? "dark" : "light";
  root.dataset.density = a.density;
  root.dataset.motion = a.motion ? "reduced" : "full";
  root.lang = a.language;
  try {
    localStorage.setItem(LOCAL_KEY, JSON.stringify(a));
  } catch {
    /* private mode */
  }
}

/** Applies theme, density and language from the user's settings (or local fallback). */
function AppearanceBridge({ children }: { children: ReactNode }) {
  const auth = useAuthState();
  const signedIn = Boolean(auth.data?.user);
  const settings = useSettings(signedIn);
  const prefs = signedIn ? settings.data : undefined;
  const [fallback, setFallback] = useState<Appearance>({
    theme: "dark",
    density: "comfortable",
    motion: false,
    language: "en",
  });

  useEffect(() => {
    const local = readLocal();
    // eslint-disable-next-line react-hooks/set-state-in-effect -- read once from the browser
    setFallback({
      theme: local.theme ?? "dark",
      density: local.density ?? "comfortable",
      motion: local.motion ?? false,
      language: local.language ?? guessLanguage(),
    });
  }, []);

  const appearance: Appearance = prefs
    ? {
        theme: prefs.theme,
        density: prefs.grid_density,
        motion: prefs.reduce_motion,
        language: prefs.language,
      }
    : fallback;

  useEffect(() => {
    applyAppearance(appearance);
    if (appearance.theme !== "system") return;
    const mq = window.matchMedia("(prefers-color-scheme: dark)");
    const onChange = () => applyAppearance(appearance);
    mq.addEventListener("change", onChange);
    return () => mq.removeEventListener("change", onChange);
  }, [appearance.theme, appearance.density, appearance.motion, appearance.language]); // eslint-disable-line react-hooks/exhaustive-deps

  return <I18nProvider lang={appearance.language}>{children}</I18nProvider>;
}

export function Providers({ children }: { children: ReactNode }) {
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            refetchOnWindowFocus: false,
            retry: (count, error) => {
              if (error instanceof ApiError && error.status > 0 && error.status < 500) return false;
              return count < 2;
            },
          },
        },
      }),
  );
  return (
    <QueryClientProvider client={client}>
      <AppearanceBridge>
        <ConfirmProvider>
          {children}
          <Toaster position="bottom-right" theme="system" closeButton duration={4500} />
        </ConfirmProvider>
      </AppearanceBridge>
    </QueryClientProvider>
  );
}
