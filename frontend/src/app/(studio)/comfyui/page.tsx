"use client";

import { ExternalLink, Plus, Workflow } from "lucide-react";
import { useState } from "react";

import { AddConnectionModal, ConnectionCard } from "@/components/comfy/connections";
import { WorkflowsCard } from "@/components/comfy/workflows";
import { PageHeader } from "@/components/layout/page-header";
import { Button, EmptyState, Notice, Skeleton } from "@/components/ui";
import { formatBytes } from "@/lib/format";
import { useI18n } from "@/lib/i18n";
import { useComfyCatalog, useComfyConnections } from "@/lib/queries";

export default function ComfyPage() {
  const { t, locale } = useI18n();
  const conns = useComfyConnections();
  const catalog = useComfyCatalog();
  const [adding, setAdding] = useState(false);
  const k2 = catalog.data?.krea2;

  return (
    <div className="page">
      <PageHeader
        title="ComfyUI"
        subtitle={t("cf.subtitle")}
        actions={
          <Button variant="primary" icon={<Plus />} onClick={() => setAdding(true)}>
            {t("cf.add")}
          </Button>
        }
      />
      <div className="comfy-layout">
        <div className="col gap-16">
          {conns.isLoading ? (
            <Skeleton style={{ height: 240 }} />
          ) : conns.isError ? (
            <Notice tone="danger">{(conns.error as Error).message}</Notice>
          ) : !conns.data?.items.length ? (
            <EmptyState
              icon={<Workflow />}
              title={t("cf.emptyTitle")}
              description={t("cf.emptyBody")}
              action={
                <Button variant="primary" size="sm" icon={<Plus />} onClick={() => setAdding(true)}>
                  {t("cf.add")}
                </Button>
              }
            />
          ) : (
            <div className="comfy-cards">
              {conns.data.items.map((c) => (
                <ConnectionCard key={c.id} conn={c} />
              ))}
            </div>
          )}
          <WorkflowsCard />
        </div>
        <aside className="col gap-16">
          <section className="card card-pad col gap-12">
            <div className="eyebrow">{t("k2.about")}</div>
            <div className="h3">Krea 2</div>
            {k2 ? (
              <>
                <dl className="kv">
                  <dt>{t("k2.architecture")}</dt>
                  <dd>{String(k2.architecture.type)}</dd>
                  <dt>{t("cf.textEncoder")}</dt>
                  <dd>{String(k2.architecture.text_encoder)}</dd>
                  <dt>VAE</dt>
                  <dd>{String(k2.architecture.vae)}</dd>
                  <dt>{t("k2.minComfy")}</dt>
                  <dd>{String(k2.architecture.min_comfyui)}</dd>
                  <dt>{t("k2.loraFormat")}</dt>
                  <dd>{String(k2.architecture.lora_format)}</dd>
                </dl>
                {k2.variants.map((v) => (
                  <div key={v.key} className="k2-variant">
                    <div className="small" style={{ fontWeight: 600 }}>{v.name}</div>
                    <div className="tiny subtle">{v.description}</div>
                    <div className="tiny">
                      {t("k2.defaults", { steps: v.steps, cfg: v.cfg, tiers: v.tiers.join(" / ") })}
                    </div>
                    <div className="tiny subtle">{v.settings_source}</div>
                    {v.known_files.map((f) => (
                      <a key={f.filename} className="tiny link row" style={{ gap: 4 }} href={f.url} target="_blank" rel="noreferrer">
                        {f.filename} {f.size_gb ? `· ${formatBytes(f.size_gb * 1e9, locale)}` : ""} <ExternalLink size={11} />
                      </a>
                    ))}
                    {!v.known_files.length ? <div className="tiny subtle">{t("k2.rawFiles")}</div> : null}
                  </div>
                ))}
                <div className="tiny subtle">{String(k2.architecture.license)}</div>
                <Notice>{t("k2.loraRule")}</Notice>
              </>
            ) : (
              <Skeleton style={{ height: 200 }} />
            )}
          </section>
        </aside>
      </div>
      <AddConnectionModal open={adding} onClose={() => setAdding(false)} />
    </div>
  );
}
