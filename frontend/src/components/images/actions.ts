"use client";

import { useQueryClient } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { useCallback } from "react";
import { toast } from "sonner";

import { usePriceConfirm } from "@/components/generate/price-confirm";
import { useConfirm } from "@/components/ui";
import { ApiError, downloadUrl } from "@/lib/api";
import { useT } from "@/lib/i18n";
import {
  patchImageEverywhere,
  removeGenerationEverywhere,
  useDeleteGeneration,
  useDeleteImage,
  useRegenerate,
  useSaveToLibrary,
} from "@/lib/queries";
import { enhanceInStudio } from "@/lib/studio-bridge";
import type { Generation, StudioImage } from "@/lib/types";

export async function copyText(text: string) {
  try {
    await navigator.clipboard.writeText(text);
    return true;
  } catch {
    // Fallback for browsers that block the async clipboard on http://localhost.
    const area = document.createElement("textarea");
    area.value = text;
    area.style.position = "fixed";
    area.style.opacity = "0";
    document.body.appendChild(area);
    area.select();
    const ok = document.execCommand("copy");
    area.remove();
    return ok;
  }
}

// How long "Undo" is offered after a quick delete; the delete is sent only then.
const UNDO_MS = 6000;

/** One-click delete (no confirmation): the item disappears at once and is deleted after a
 * few seconds unless "Undo" is pressed. Closing the app before that keeps it. */
export function useQuickDelete() {
  const t = useT();
  const qc = useQueryClient();
  const removeImage = useDeleteImage();
  const removeGeneration = useDeleteGeneration();

  const later = useCallback(
    (hide: () => void, commit: () => Promise<unknown>, message: string) => {
      hide();
      let undone = false;
      const restore = () => {
        qc.invalidateQueries({ queryKey: ["generations"] });
        qc.invalidateQueries({ queryKey: ["images"] });
      };
      const timer = window.setTimeout(() => {
        if (undone) return;
        commit().catch((e: Error) => {
          toast.error(e.message);
          restore();
        });
      }, UNDO_MS);
      toast(message, {
        duration: UNDO_MS,
        action: {
          label: t("common.undo"),
          onClick: () => {
            undone = true;
            window.clearTimeout(timer);
            restore();
          },
        },
      });
    },
    [qc, t],
  );

  const deleteImageQuick = useCallback(
    (img: StudioImage) =>
      later(
        () => patchImageEverywhere(qc, img.id, () => null),
        () => removeImage.mutateAsync(img.id),
        t("images.deleted"),
      ),
    [later, qc, removeImage, t],
  );

  const deleteGenerationQuick = useCallback(
    (gen: Generation) =>
      later(
        () => removeGenerationEverywhere(qc, gen.id),
        () => removeGeneration.mutateAsync(gen.id),
        t("gen.deleted"),
      ),
    [later, qc, removeGeneration, t],
  );

  return { deleteImageQuick, deleteGenerationQuick };
}

export function useImageActions() {
  const t = useT();
  const router = useRouter();
  const confirm = useConfirm();
  const remove = useDeleteImage();
  const regenerate = useRegenerate();
  const priceConfirm = usePriceConfirm();
  const save = useSaveToLibrary();

  const download = useCallback((img: StudioImage) => downloadUrl(img.download_url), []);

  const copyPrompt = useCallback(
    async (img: StudioImage) => {
      if (await copyText(img.prompt)) toast.success(t("common.promptCopied"));
      else toast.error(t("errors.copy"));
    },
    [t],
  );

  const deleteImage = useCallback(
    async (img: StudioImage) => {
      const ok = await confirm({
        title: t("images.deleteTitle"),
        description: t("images.deleteBody"),
        confirmLabel: t("common.delete"),
        danger: true,
      });
      if (!ok) return false;
      try {
        await remove.mutateAsync(img.id);
        toast.success(t("images.deleted"));
        return true;
      } catch (e) {
        toast.error((e as Error).message);
        return false;
      }
    },
    [confirm, remove, t],
  );

  const regenerateImage = useCallback(
    async (img: StudioImage) => {
      try {
        // Paid models: the price first; it runs only after the user confirms it.
        if (img.provider !== "comfyui" && !(await priceConfirm.forRepeat(img.generation_id, 1))) return;
        await regenerate.mutateAsync({ id: img.generation_id, count: 1 });
        toast.success(t("images.regenerating"), {
          action: { label: t("common.view"), onClick: () => router.push("/generate") },
        });
      } catch (e) {
        const err = e as ApiError;
        toast.error(err.message, {
          action:
            err.code === "not_connected" || err.code === "invalid_credentials"
              ? { label: t("nav.integrations"), onClick: () => router.push("/integrations") }
              : undefined,
        });
      }
    },
    [priceConfirm, regenerate, router, t],
  );

  const saveToLibrary = useCallback(
    (img: StudioImage) =>
      save.mutate(img.id, {
        onSuccess: () => toast.success(t("images.saved")),
        onError: (e) => toast.error((e as Error).message),
      }),
    [save, t],
  );

  const openInStudio = useCallback(
    (img: StudioImage) => router.push(`/generate?from=${img.generation_id}`),
    [router],
  );

  // Generate › Enhance photo with this image (switches there when that page is open).
  const enhanceImage = useCallback(
    (img: StudioImage) => {
      if (enhanceInStudio({ kind: "image", id: img.id, thumb_url: img.thumb_url }) === "pending") router.push("/generate");
    },
    [router],
  );

  return {
    download,
    enhanceImage,
    copyPrompt,
    deleteImage,
    regenerateImage,
    saveToLibrary,
    openInStudio,
    regenerating: regenerate.isPending,
  };
}
