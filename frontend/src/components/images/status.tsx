"use client";

import { Badge } from "@/components/ui";
import { useT } from "@/lib/i18n";
import type { GenerationStatus } from "@/lib/types";

const TONES: Record<GenerationStatus, "accent" | "success" | "danger" | "warning" | undefined> = {
  queued: undefined,
  processing: "accent",
  completed: "success",
  failed: "danger",
  cancelled: "warning",
};

export function StatusBadge({ status }: { status: GenerationStatus }) {
  const t = useT();
  return (
    <Badge tone={TONES[status]}>
      <span className={status === "processing" || status === "queued" ? "dot pulse" : "dot"} />
      {t(`status.${status}`)}
    </Badge>
  );
}
