import type { ProviderKey } from "@/lib/types";

/** The studio's own mark: an aperture with a spark. */
export function BrandMark() {
  return (
    <span className="brand-mark" aria-hidden>
      <svg viewBox="0 0 24 24" fill="none">
        <defs>
          <linearGradient id="bm" x1="4" y1="3" x2="20" y2="21" gradientUnits="userSpaceOnUse">
            <stop stopColor="#D9D1FF" />
            <stop offset="1" stopColor="#8F7CFF" />
          </linearGradient>
        </defs>
        <path
          d="M12 2.5C12.7 7.6 16.4 11.3 21.5 12C16.4 12.7 12.7 16.4 12 21.5C11.3 16.4 7.6 12.7 2.5 12C7.6 11.3 11.3 7.6 12 2.5Z"
          fill="url(#bm)"
        />
      </svg>
    </span>
  );
}

/**
 * Neutral monogram marks for the two providers. They are drawn here (not the providers'
 * trademarked logos), so the app ships no third-party artwork.
 */
export function ProviderLogo({ provider, small }: { provider: ProviderKey; small?: boolean }) {
  const cls = small ? "logo sm" : "logo";
  if (provider === "wavespeed") {
    return (
      <span
        className={cls}
        style={{ background: "linear-gradient(145deg, #1b2240, #0f1322)" }}
        aria-hidden
      >
        <svg viewBox="0 0 24 24" fill="none" stroke="#9fb4ff" strokeWidth="2" strokeLinecap="round">
          <path d="M2.5 13c2.2 0 2.2-5 4.4-5s2.2 8 4.4 8 2.2-10 4.4-10 2.2 7 4.4 7" />
        </svg>
      </span>
    );
  }
  return (
    <span
      className={cls}
      style={{ background: "linear-gradient(145deg, #2b2033, #151018)" }}
      aria-hidden
    >
      <svg viewBox="0 0 24 24" fill="none" stroke="#e7b6ff" strokeWidth="1.8" strokeLinecap="round">
        <circle cx="12" cy="12" r="8.5" />
        <path d="M3.5 12h17M12 3.5c2.6 2.6 2.6 14.4 0 17M12 3.5c-2.6 2.6-2.6 14.4 0 17" />
      </svg>
    </span>
  );
}
