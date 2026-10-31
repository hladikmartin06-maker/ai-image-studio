"use client";

import { Download, FileJson, Trash2, Upload } from "lucide-react";
import { useRef } from "react";
import { toast } from "sonner";

import { Badge, Button, EmptyState, IconButton, Notice, Skeleton, useConfirm } from "@/components/ui";
import { downloadUrl } from "@/lib/api";
import { formatRelative } from "@/lib/format";
import { useI18n } from "@/lib/i18n";
import { useComfyWorkflows, useDeleteWorkflow, useImportWorkflow } from "@/lib/queries";

export function WorkflowsCard() {
  const { t, locale } = useI18n();
  const list = useComfyWorkflows();
  const importer = useImportWorkflow();
  const remove = useDeleteWorkflow();
  const confirm = useConfirm();
  const input = useRef<HTMLInputElement>(null);

  const onFile = async (file: File) => {
    if (file.size > 5 * 1024 * 1024) {
      toast.error(t("wf.tooLarge"));
      return;
    }
    try {
      const data = JSON.parse(await file.text());
      const res = await importer.mutateAsync({ name: file.name.replace(/\.json$/i, ""), family: "krea2", data });
      if (res.format === "ui") toast.warning(t("wf.uiImported"));
      else toast.success(res.runnable ? t("wf.imported") : t("wf.importedNoPrompt"));
      if (res.removed_secrets.length) toast.info(t("wf.secretsRemoved", { n: res.removed_secrets.length }));
    } catch (e) {
      toast.error(e instanceof SyntaxError ? t("wf.notJson") : (e as Error).message);
    }
  };

  return (
    <section className="card">
      <div className="card-head">
        <FileJson size={16} />
        <div className="grow">
          <div className="h3">{t("wf.title")}</div>
          <div className="tiny subtle">{t("wf.subtitle")}</div>
        </div>
        <input ref={input} type="file" accept="application/json,.json" hidden onChange={(e) => {
          const f = e.target.files?.[0];
          e.target.value = "";
          if (f) void onFile(f);
        }} />
        <Button size="sm" icon={<Upload />} onClick={() => input.current?.click()} loading={importer.isPending}>
          {t("wf.import")}
        </Button>
      </div>
      <div className="card-body col gap-12">
        <div className="wf-row">
          <div className="grow">
            <div className="small" style={{ fontWeight: 600 }}>{t("wf.builtin")}</div>
            <div className="tiny subtle">{t("wf.builtinHint")}</div>
          </div>
          <Button size="sm" variant="ghost" icon={<Download />} onClick={() => downloadUrl("/api/comfy/workflows/builtin-krea2/export?variant=turbo")}>
            Turbo
          </Button>
          <Button size="sm" variant="ghost" icon={<Download />} onClick={() => downloadUrl("/api/comfy/workflows/builtin-krea2/export?variant=raw")}>
            RAW
          </Button>
        </div>
        {list.isLoading ? (
          <Skeleton style={{ height: 60 }} />
        ) : list.isError ? (
          <Notice tone="danger">{(list.error as Error).message}</Notice>
        ) : !list.data?.items.length ? (
          <EmptyState icon={<FileJson />} title={t("wf.empty")} description={t("wf.emptyBody")} />
        ) : (
          list.data.items.map((wf) => (
            <div key={wf.id} className="wf-row">
              <div className="grow" style={{ minWidth: 0 }}>
                <div className="row" style={{ gap: 6 }}>
                  <span className="small truncate" style={{ fontWeight: 600 }}>{wf.name}</span>
                  {wf.runnable ? <Badge tone="success">{t("wf.runnable")}</Badge> : <Badge tone="warning">{wf.has_api ? t("wf.noBindings") : t("wf.uiOnly")}</Badge>}
                </div>
                <div className="tiny subtle truncate">
                  {wf.node_types.length ? t("wf.nodes", { n: wf.node_types.length }) : ""} · {formatRelative(wf.updated_at, locale)}
                </div>
              </div>
              {wf.has_api ? (
                <Button size="sm" variant="ghost" icon={<Download />} onClick={() => downloadUrl(`/api/comfy/workflows/${wf.id}/export?format=api`)}>
                  API
                </Button>
              ) : null}
              {wf.has_ui ? (
                <Button size="sm" variant="ghost" icon={<Download />} onClick={() => downloadUrl(`/api/comfy/workflows/${wf.id}/export?format=ui`)}>
                  UI
                </Button>
              ) : null}
              <IconButton
                label={t("common.delete")}
                onClick={async () => {
                  if (await confirm({ title: t("wf.deleteTitle", { name: wf.name }), confirmLabel: t("common.delete"), danger: true }))
                    remove.mutate(wf.id);
                }}
              >
                <Trash2 />
              </IconButton>
            </div>
          ))
        )}
      </div>
    </section>
  );
}
