"use client";

import { FileUp, Library, Search, Sparkles, X } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { PageHeader } from "@/components/layout/page-header";
import { ImportLoraModal } from "@/components/lora/create";
import { StatusPill } from "@/components/studio/common";
import { Badge, Button, EmptyState, IconButton, Notice, Skeleton } from "@/components/ui";
import { formatBytes, formatRelative } from "@/lib/format";
import { useDebounced } from "@/lib/hooks";
import { useI18n } from "@/lib/i18n";
import { useLoras } from "@/lib/queries";

export default function MyLorasPage() {
  const { t, locale } = useI18n();
  const router = useRouter();
  const [search, setSearch] = useState("");
  const q = useDebounced(search.trim(), 250);
  const loras = useLoras(q);
  const [importing, setImporting] = useState(false);
  const items = loras.data?.items ?? [];

  return (
    <div className="page">
      <PageHeader
        title={t("nav.myLoras")}
        subtitle={t("ml.subtitle")}
        actions={
          <>
            <Button icon={<FileUp />} onClick={() => setImporting(true)}>{t("ls.importLora")}</Button>
            <Link href="/lora-studio" className="btn btn-primary">
              <Sparkles /> {t("ls.createNew")}
            </Link>
          </>
        }
      />
      <div className="toolbar">
        <div className="input-wrap search">
          <Search />
          <input className="input" value={search} placeholder={t("ml.search")} onChange={(e) => setSearch(e.target.value)} aria-label={t("common.search")} />
          {search ? (
            <span className="input-action">
              <IconButton label={t("common.clear")} onClick={() => setSearch("")}>
                <X />
              </IconButton>
            </span>
          ) : null}
        </div>
      </div>
      {loras.isLoading ? (
        <Skeleton style={{ height: 240 }} />
      ) : loras.isError ? (
        <Notice tone="danger">{(loras.error as Error).message}</Notice>
      ) : !items.length ? (
        <EmptyState icon={<Library />} title={q ? t("ml.noMatches") : t("ml.empty")} description={q ? undefined : t("ml.emptyBody")} />
      ) : (
        <div className="lora-grid">
          {items.map((l) => {
            const p = l.preferred_version;
            const sample = p?.samples.at(-1);
            return (
              <Link key={l.id} href={`/loras/${l.id}`} className="card lora-card">
                {sample ? <img className="lora-cover" src={sample.url} alt="" loading="lazy" /> : null}
                <div className="row" style={{ gap: 8 }}>
                  <span className="h3 grow truncate">{l.name}</span>
                  {p ? <Badge tone="success">v{p.number}</Badge> : l.latest_version ? <StatusPill status={l.latest_version.status} /> : <Badge>{t("ls.draft")}</Badge>}
                </div>
                <div className="tiny subtle">
                  <span className="mono">{l.trigger_word}</span> · Krea 2 {l.base_variant.toUpperCase()} · {t(l.source === "imported" ? "ls.importedBadge" : "ml.trained")}
                </div>
                <div className="row wrap" style={{ gap: 4 }}>
                  {l.persona_name ? <Badge tone="accent">{l.persona_name}</Badge> : null}
                  {l.tags.map((tag) => (
                    <Badge key={tag}>{tag}</Badge>
                  ))}
                </div>
                <div className="tiny subtle">
                  {p?.file_size ? `${formatBytes(p.file_size, locale)} · ` : ""}
                  {formatRelative(l.updated_at, locale)}
                </div>
              </Link>
            );
          })}
        </div>
      )}
      {importing ? <ImportLoraModal onClose={() => setImporting(false)} onCreated={(l) => router.push(`/loras/${l.id}`)} /> : null}
    </div>
  );
}
