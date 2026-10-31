"use client";

import { useEffect, useState } from "react";

import { ImageBrowser } from "@/components/images/image-browser";
import { LibraryTabs } from "@/components/images/library-tabs";
import { PersonaPhotos } from "@/components/images/persona-photos";
import { PageHeader } from "@/components/layout/page-header";
import { Segmented } from "@/components/ui";
import { useI18n } from "@/lib/i18n";

type Source = "generated" | "models";

export default function LibraryPage() {
  const { t, locale } = useI18n();
  const [source, setSource] = useState<Source>("generated");

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- deep link (?source=models)
    if (new URLSearchParams(window.location.search).get("source") === "models") setSource("models");
  }, []);

  const change = (value: Source) => {
    setSource(value);
    window.history.replaceState(null, "", value === "models" ? "/library?source=models" : "/library");
  };

  const sourceSwitch = (
    <Segmented
      label={t("lib.source")}
      value={source}
      onChange={change}
      options={[
        { value: "generated", label: t("lib.sourceGenerated") },
        { value: "models", label: t("lib.sourceModels") },
      ]}
    />
  );
  const header = (total: number | undefined) => (
    <PageHeader
      title={t("lib.title")}
      subtitle={total === undefined ? t("lib.subtitle") : t("lib.count", { n: total.toLocaleString(locale) })}
      actions={<LibraryTabs />}
    />
  );

  return (
    <div className="page">
      {source === "generated" ? (
        <ImageBrowser header={header} toolbarStart={sourceSwitch} />
      ) : (
        <PersonaPhotos header={header} toolbarStart={sourceSwitch} />
      )}
    </div>
  );
}
