export function formatBytes(bytes: number, locale = "en-US") {
  if (!bytes) return "0 B";
  const units = ["B", "KB", "MB", "GB", "TB"];
  const i = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  const value = bytes / 1024 ** i;
  return `${value.toLocaleString(locale, { maximumFractionDigits: value < 10 && i > 0 ? 1 : 0 })} ${units[i]}`;
}

export function formatDate(iso: string | null | undefined, locale: string, withTime = true) {
  if (!iso) return "—";
  const date = new Date(iso);
  return date.toLocaleString(locale, {
    day: "numeric",
    month: "short",
    year: date.getFullYear() === new Date().getFullYear() ? undefined : "numeric",
    ...(withTime ? { hour: "2-digit", minute: "2-digit" } : {}),
  });
}

export function formatRelative(iso: string | null | undefined, locale: string) {
  if (!iso) return "—";
  const diff = (new Date(iso).getTime() - Date.now()) / 1000;
  const rtf = new Intl.RelativeTimeFormat(locale, { numeric: "auto" });
  const abs = Math.abs(diff);
  if (abs < 45) return rtf.format(Math.round(diff), "second");
  if (abs < 45 * 60) return rtf.format(Math.round(diff / 60), "minute");
  if (abs < 22 * 3600) return rtf.format(Math.round(diff / 3600), "hour");
  if (abs < 26 * 86400) return rtf.format(Math.round(diff / 86400), "day");
  return formatDate(iso, locale, false);
}

export function formatDuration(seconds: number | null | undefined) {
  if (seconds === null || seconds === undefined) return "—";
  if (seconds < 60) return `${Math.max(1, Math.round(seconds))} s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m} min ${s.toString().padStart(2, "0")} s`;
}

export function formatNumber(value: number, locale: string) {
  return value.toLocaleString(locale);
}

export function aspectLabel(width: number, height: number) {
  const gcd = (a: number, b: number): number => (b ? gcd(b, a % b) : a);
  const d = gcd(width, height);
  const w = width / d;
  const h = height / d;
  if (w > 50 || h > 50) {
    const r = width / height;
    const known: [string, number][] = [
      ["1:1", 1],
      ["4:3", 4 / 3],
      ["3:4", 3 / 4],
      ["3:2", 3 / 2],
      ["2:3", 2 / 3],
      ["16:9", 16 / 9],
      ["9:16", 9 / 16],
      ["21:9", 21 / 9],
      ["4:5", 4 / 5],
      ["5:4", 5 / 4],
    ];
    const best = known.reduce((a, b) => (Math.abs(b[1] - r) < Math.abs(a[1] - r) ? b : a));
    return `≈ ${best[0]}`;
  }
  return `${w}:${h}`;
}
