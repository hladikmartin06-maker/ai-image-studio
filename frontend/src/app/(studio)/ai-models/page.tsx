"use client";

import { CopyPlus, Ellipsis, Images, Pencil, Plus, Search, Sparkles, Trash2, UserRound, X } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { PageHeader } from "@/components/layout/page-header";
import { usePersonaActions } from "@/components/personas/actions";
import { CreatePersonaModal } from "@/components/personas/shared";
import { Button, EmptyState, IconButton, MenuItem, Notice, Popover, Skeleton } from "@/components/ui";
import { formatRelative } from "@/lib/format";
import { useDebounced } from "@/lib/hooks";
import { useI18n } from "@/lib/i18n";
import { usePersonas } from "@/lib/queries";
import type { Persona } from "@/lib/types";

function PersonaCard({ persona }: { persona: Persona }) {
  const { t, locale } = useI18n();
  const router = useRouter();
  const actions = usePersonaActions();
  const href = `/ai-models/${persona.id}`;
  return (
    <article className="persona-card">
      <Link href={href} className="persona-card-cover" aria-label={persona.name}>
        {persona.primary_image ? (
          <img src={persona.primary_image.thumb_url} alt="" loading="lazy" />
        ) : (
          <span className="persona-card-empty">
            <UserRound />
          </span>
        )}
        <span className="persona-card-count">
          <Images /> {persona.image_count}
        </span>
      </Link>
      <div className="persona-card-body">
        <div className="row" style={{ gap: 6 }}>
          <Link href={href} className="persona-name grow truncate">
            {persona.name}
          </Link>
          <Popover
            align="end"
            width={220}
            trigger={({ toggle, ref }) => (
              <button ref={ref} type="button" className="icon-btn" aria-label={t("pm.actions")} onClick={toggle}>
                <Ellipsis />
              </button>
            )}
          >
            {(close) => (
              <>
                <MenuItem
                  icon={<Sparkles />}
                  onClick={() => {
                    close();
                    actions.openInGenerate(persona);
                  }}
                >
                  {t("pm.useInGenerate")}
                </MenuItem>
                <MenuItem
                  icon={<Pencil />}
                  onClick={() => {
                    close();
                    router.push(href);
                  }}
                >
                  {t("pm.edit")}
                </MenuItem>
                <MenuItem
                  icon={<CopyPlus />}
                  onClick={() => {
                    close();
                    actions.copy(persona);
                  }}
                >
                  {t("pm.duplicate")}
                </MenuItem>
                <div className="menu-sep" />
                <MenuItem
                  icon={<Trash2 />}
                  danger
                  onClick={() => {
                    close();
                    void actions.destroy(persona);
                  }}
                >
                  {t("common.delete")}
                </MenuItem>
              </>
            )}
          </Popover>
        </div>
        <div className="tiny subtle clamp-2" style={{ minHeight: 32 }}>
          {persona.description || t("pm.noDescription")}
        </div>
        <div className="tiny subtle">
          {persona.generation_count
            ? t("pm.generations", { n: persona.generation_count })
            : t("pm.notUsedYet")}
          {persona.last_used_at ? ` · ${formatRelative(persona.last_used_at, locale)}` : ""}
        </div>
      </div>
    </article>
  );
}

export default function AIModelsPage() {
  const { t } = useI18n();
  const router = useRouter();
  const [creating, setCreating] = useState(false);
  const [search, setSearch] = useState("");
  const q = useDebounced(search.trim(), 250);
  const list = usePersonas(q);
  const items = list.data?.items ?? [];

  return (
    <div className="page">
      <PageHeader
        title={t("pm.title")}
        subtitle={t("pm.subtitle")}
        actions={
          <Button variant="primary" icon={<Plus />} onClick={() => setCreating(true)}>
            {t("pm.new")}
          </Button>
        }
      />
      {items.length > 0 || q ? (
        <div className="toolbar">
          <div className="input-wrap search">
            <Search />
            <input
              className="input"
              placeholder={t("pm.searchPlaceholder")}
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              aria-label={t("common.search")}
            />
            {search ? (
              <span className="input-action">
                <IconButton label={t("common.clear")} onClick={() => setSearch("")}>
                  <X />
                </IconButton>
              </span>
            ) : null}
          </div>
          <span className="spacer" />
          <span className="small subtle">{t("pm.count", { n: list.data?.total ?? 0 })}</span>
        </div>
      ) : null}

      {list.isLoading ? (
        <div className="persona-grid">
          {Array.from({ length: 8 }, (_, i) => (
            <Skeleton key={i} style={{ aspectRatio: "3 / 4.6", borderRadius: 14 }} />
          ))}
        </div>
      ) : list.isError ? (
        <Notice tone="danger">{(list.error as Error).message}</Notice>
      ) : items.length === 0 ? (
        q ? (
          <EmptyState
            icon={<Search />}
            title={t("pm.noMatches")}
            action={<Button onClick={() => setSearch("")}>{t("common.clear")}</Button>}
          />
        ) : (
          <EmptyState
            icon={<UserRound />}
            title={t("pm.emptyTitle")}
            description={t("pm.emptyBody")}
            action={
              <Button variant="primary" size="sm" icon={<Plus />} onClick={() => setCreating(true)}>
                {t("pm.new")}
              </Button>
            }
          />
        )
      ) : (
        <div className="persona-grid">
          {items.map((p) => (
            <PersonaCard key={p.id} persona={p} />
          ))}
        </div>
      )}

      <CreatePersonaModal
        open={creating}
        onClose={() => setCreating(false)}
        onCreated={(p) => router.push(`/ai-models/${p.id}`)}
      />
    </div>
  );
}
