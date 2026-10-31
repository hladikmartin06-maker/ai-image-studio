"use client";

import { useRouter } from "next/navigation";
import { useCallback } from "react";
import { toast } from "sonner";

import { useConfirm } from "@/components/ui";
import { ApiError, downloadUrl } from "@/lib/api";
import { useT } from "@/lib/i18n";
import { useDeleteImage, useRegenerate, useSaveToLibrary, useToggleFavorite } from "@/lib/queries";
import type { StudioImage } from "@/lib/types";

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

export function useImageActions() {
  const t = useT();
  const router = useRouter();
  const confirm = useConfirm();
  const favorite = useToggleFavorite();
  const remove = useDeleteImage();
  const regenerate = useRegenerate();
  const save = useSaveToLibrary();

  const toggleFavorite = useCallback(
    (img: StudioImage) => {
      favorite.mutate(
        { id: img.id, value: !img.is_favorite },
        {
          onSuccess: (res) =>
            toast.success(res.is_favorite ? t("images.addedFavorite") : t("images.removedFavorite")),
          onError: (e) => toast.error((e as Error).message),
        },
      );
    },
    [favorite, t],
  );

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
    [regenerate, router, t],
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

  return {
    toggleFavorite,
    download,
    copyPrompt,
    deleteImage,
    regenerateImage,
    saveToLibrary,
    openInStudio,
    regenerating: regenerate.isPending,
  };
}
