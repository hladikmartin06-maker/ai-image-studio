"use client";

import { ExternalLink } from "lucide-react";
import { useCallback, useMemo } from "react";

import { formatMoney } from "@/components/studio/common";
import { useConfirm } from "@/components/ui";
import { api } from "@/lib/api";
import { useI18n } from "@/lib/i18n";
import type { PriceEstimate } from "@/lib/types";

/** What the confirmation shows: the total (an estimate), per image, and where it comes from. */
function PriceSummary({ estimate }: { estimate: PriceEstimate }) {
  const { t, locale } = useI18n();
  const known = estimate.total !== null;
  return (
    <div className="col" style={{ gap: 10, marginTop: 8 }}>
      <div className="small">
        {t(estimate.mode.endsWith("-video") ? "price.whatVideo" : "price.what", {
          model: estimate.model_name,
          provider: estimate.provider_name,
          n: estimate.images,
        })}
      </div>
      <div className="price-total">
        <span className="price-amount">{known ? `≈ ${formatMoney(estimate.total, locale)}` : t("price.unknown")}</span>
        {known && estimate.images > 1 ? (
          <span className="small subtle">
            {t("price.perImage", { price: formatMoney(estimate.per_image, locale) })}
          </span>
        ) : null}
      </div>
      <div className="hint">
        {estimate.error
          ? t("price.error", { error: estimate.error })
          : !estimate.supported
            ? t("price.noApi", { provider: estimate.provider_name })
            : estimate.source === "list_price"
              ? t("price.fromList", { provider: estimate.provider_name })
              : known
                ? t("price.fromEstimate", { provider: estimate.provider_name })
                : t("price.notListed", { provider: estimate.provider_name })}
      </div>
      {!known && estimate.pricing_url ? (
        <a className="small row" style={{ gap: 4 }} href={estimate.pricing_url} target="_blank" rel="noreferrer">
          {t("price.openPricing", { provider: estimate.provider_name })} <ExternalLink size={13} />
        </a>
      ) : null}
    </div>
  );
}

/**
 * Asks for the price first (the provider's own API, through the studio) and starts only after
 * the user confirms it. Errors of the check (e.g. a refused prompt) are thrown to the caller.
 */
export function usePriceConfirm() {
  const { t } = useI18n();
  const confirm = useConfirm();
  // false = cancelled; otherwise the confirmed total (null when no price was known).
  const ask = useCallback(
    async (estimate: PriceEstimate): Promise<false | { price: number | null }> => {
      const ok = await confirm({
        title: t("price.title"),
        description: <PriceSummary estimate={estimate} />,
        confirmLabel: t("price.start"),
      });
      return ok ? { price: estimate.total } : false;
    },
    [confirm, t],
  );
  /** A new request (the same body as POST /api/generations). */
  const forRequest = useCallback(
    async (body: Record<string, unknown>) =>
      ask(await api<PriceEstimate>("/api/generations/estimate", { body })),
    [ask],
  );
  /** Repeating a past hosted request. */
  const forRepeat = useCallback(
    async (generationId: string, count?: number) =>
      ask(await api<PriceEstimate>(`/api/generations/${generationId}/estimate`, { body: { count } })),
    [ask],
  );
  return useMemo(() => ({ forRequest, forRepeat }), [forRequest, forRepeat]);
}
