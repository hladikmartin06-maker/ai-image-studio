"use client";

import { useRouter } from "next/navigation";
import { toast } from "sonner";

import { useConfirm } from "@/components/ui";
import { useI18n } from "@/lib/i18n";
import { useDeletePersona, useDuplicatePersona } from "@/lib/queries";
import type { Persona } from "@/lib/types";

/** Open in Generate, duplicate and delete — shared by the AI model list and detail pages. */
export function usePersonaActions() {
  const { t } = useI18n();
  const router = useRouter();
  const confirm = useConfirm();
  const duplicate = useDuplicatePersona();
  const remove = useDeletePersona();

  const openInGenerate = (p: Persona) => router.push(`/generate?ai_model=${p.id}`);

  const copy = (p: Persona) =>
    duplicate.mutate(p.id, {
      onSuccess: (created) =>
        toast.success(t("pm.duplicated", { name: created.name }), {
          action: { label: t("common.view"), onClick: () => router.push(`/ai-models/${created.id}`) },
        }),
      onError: (e) => toast.error((e as Error).message),
    });

  const destroy = async (p: Persona, after?: () => void) => {
    const ok = await confirm({
      title: t("pm.deleteTitle", { name: p.name }),
      description: t("pm.deleteBody", { n: p.image_count }),
      confirmLabel: t("common.delete"),
      danger: true,
    });
    if (!ok) return;
    try {
      await remove.mutateAsync(p.id);
      toast.success(t("pm.deleted", { name: p.name }));
      after?.();
    } catch (e) {
      toast.error((e as Error).message);
    }
  };

  return { openInGenerate, copy, destroy, duplicating: duplicate.isPending };
}
