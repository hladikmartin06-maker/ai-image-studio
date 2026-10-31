"use client";

import { ImageBrowser } from "@/components/images/image-browser";
import { PageHeader } from "@/components/layout/page-header";
import { useI18n } from "@/lib/i18n";

export default function FavoritesPage() {
  const { t, locale } = useI18n();
  return (
    <div className="page">
      <ImageBrowser
        favoritesOnly
        header={(total) => (
          <PageHeader
            title={t("fav.title")}
            subtitle={
              total === undefined ? t("fav.subtitle") : t("fav.count", { n: total.toLocaleString(locale) })
            }
          />
        )}
      />
    </div>
  );
}
