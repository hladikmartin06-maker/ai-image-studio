"use client";

import { Bot, Cpu, ImageIcon, KeyRound } from "lucide-react";
import { useEffect, useState } from "react";

import { AIProvidersTab } from "@/components/integrations/ai-providers";
import { GPUProvidersTab } from "@/components/integrations/gpu-providers";
import { HuggingFaceTab } from "@/components/integrations/huggingface";
import { ImageProvidersTab } from "@/components/integrations/image-providers";
import { PageHeader } from "@/components/layout/page-header";
import { Segmented } from "@/components/ui";
import { useT } from "@/lib/i18n";

type Tab = "images" | "ai" | "gpu" | "hf";

export default function IntegrationsPage() {
  const t = useT();
  const [tab, setTab] = useState<Tab>("images");

  useEffect(() => {
    const value = new URLSearchParams(window.location.search).get("tab");
    // eslint-disable-next-line react-hooks/set-state-in-effect -- deep link (?tab=ai|gpu|hf)
    if (value === "ai" || value === "gpu" || value === "hf" || value === "images") setTab(value);
  }, []);

  const change = (value: Tab) => {
    setTab(value);
    window.history.replaceState(null, "", value === "images" ? "/integrations" : `/integrations?tab=${value}`);
  };

  return (
    <div className="page">
      <PageHeader
        title={t("int.title")}
        subtitle={
          tab === "ai" ? t("ai.subtitle") : tab === "gpu" ? t("gpuint.subtitle") : tab === "hf" ? t("hf.subtitle") : t("int.subtitle")
        }
        actions={
          <Segmented
            label={t("int.sections")}
            value={tab}
            onChange={change}
            options={[
              { value: "images", label: (<><ImageIcon /> {t("int.tabImages")}</>) },
              { value: "ai", label: (<><Bot /> {t("int.tabAI")}</>) },
              { value: "gpu", label: (<><Cpu /> {t("int.tabGPU")}</>) },
              { value: "hf", label: (<><KeyRound /> Hugging Face</>) },
            ]}
          />
        }
      />
      {tab === "images" ? (
        <ImageProvidersTab />
      ) : tab === "ai" ? (
        <AIProvidersTab />
      ) : tab === "gpu" ? (
        <GPUProvidersTab />
      ) : (
        <HuggingFaceTab />
      )}
    </div>
  );
}
