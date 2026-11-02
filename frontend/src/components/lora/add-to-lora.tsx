"use client";

import { Plus, Wand2 } from "lucide-react";
import { useRouter } from "next/navigation";
import { toast } from "sonner";

import { MenuItem, Popover, Spinner } from "@/components/ui";
import { ApiError } from "@/lib/api";
import { useT } from "@/lib/i18n";
import { useAddToLoraDataset, useLoras } from "@/lib/queries";
import type { LoraSummary } from "@/lib/types";

/** "Add to LoRA": copies images into a LoRA's photos (its dataset), e.g. the best results of
 * Generate, so the next training (or "Continue training") learns from them. */
export function AddToLora({ imageIds, compact = false }: { imageIds: string[]; compact?: boolean }) {
  const t = useT();
  const router = useRouter();
  const loras = useLoras();
  const add = useAddToLoraDataset();
  const items = (loras.data?.items ?? []).filter((l) => l.source === "trained");

  const run = async (lora: LoraSummary) => {
    try {
      const res = await add.mutateAsync({ loraId: lora.id, imageIds });
      const open = { label: t("common.view"), onClick: () => router.push(`/loras/${lora.id}`) };
      if (res.added) toast.success(t("lo.addedTo", { n: res.added, name: lora.name }), { action: open });
      else toast.info(t("lo.alreadyIn", { name: lora.name }), { action: open });
    } catch (e) {
      toast.error(e instanceof ApiError ? e.message : t("errors.generic"));
    }
  };

  return (
    <Popover
      width={280}
      align={compact ? "end" : "start"}
      trigger={({ toggle, ref }) =>
        compact ? (
          <button
            ref={ref}
            type="button"
            className="icon-btn on-media"
            aria-label={t("lo.addToLora")}
            title={t("lo.addToLora")}
            onClick={toggle}
            disabled={add.isPending || imageIds.length === 0}
          >
            {add.isPending ? <span className="spinner" aria-hidden /> : <Wand2 />}
          </button>
        ) : (
          <button
            ref={ref}
            type="button"
            className="btn btn-sm"
            onClick={toggle}
            disabled={add.isPending || imageIds.length === 0}
            aria-busy={add.isPending || undefined}
          >
            {add.isPending ? <span className="spinner" aria-hidden /> : <Wand2 />}
            {t("lo.addToLora")}
          </button>
        )
      }
    >
      {(close) => (
        <>
          <div className="menu-heading">
            <div className="h3">{t("lo.addToLora")}</div>
            <div className="tiny subtle">{t("lo.addToLoraHint", { n: imageIds.length })}</div>
          </div>
          <div className="menu-sep" />
          {loras.isLoading ? (
            <div className="row" style={{ justifyContent: "center", padding: 12 }}>
              <Spinner />
            </div>
          ) : (
            <div style={{ maxHeight: 280, overflowY: "auto" }}>
              {items.map((l) => (
                <MenuItem
                  key={l.id}
                  icon={<Wand2 />}
                  onClick={() => {
                    close();
                    void run(l);
                  }}
                >
                  <span className="truncate" style={{ display: "block" }}>
                    {l.name}
                  </span>
                  <span className="tiny subtle">{t("lo.photoCount", { n: l.dataset_count ?? 0 })}</span>
                </MenuItem>
              ))}
              {items.length === 0 ? <div className="tiny subtle" style={{ padding: "6px 12px" }}>{t("lo.noneYet")}</div> : null}
            </div>
          )}
          <div className="menu-sep" />
          <MenuItem
            icon={<Plus />}
            onClick={() => {
              close();
              router.push("/loras");
            }}
          >
            {t("lo.newLoraEllipsis")}
          </MenuItem>
        </>
      )}
    </Popover>
  );
}
