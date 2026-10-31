"use client";

import { ImageBrowser } from "@/components/images/image-browser";
import { LibraryTabs } from "@/components/images/library-tabs";
import { PageHeader } from "@/components/layout/page-header";
import { useI18n } from "@/lib/i18n";

export default function LibraryPage() {
  const { t, locale } = useI18n();
  return (
    <div className="page">
      <ImageBrowser
        header={(total) => (
          <PageHeader
            title={t("lib.title")}
            subtitle={
              total === undefined ? t("lib.subtitle") : t("lib.count", { n: total.toLocaleString(locale) })
            }
            actions={<LibraryTabs />}
          />
        )}
      />
    </div>
  );
}
