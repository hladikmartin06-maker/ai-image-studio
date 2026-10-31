"use client";

import { CloudCog, Laptop, Play, Plus, Settings2, Square } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

import { type AddMode, AddConnectionModal, ConnectionCard } from "@/components/comfy/connections";
import { WorkflowsCard } from "@/components/comfy/workflows";
import { StatusPill } from "@/components/studio/common";
import { Button, IconButton, Modal, Select, useConfirm } from "@/components/ui";
import { ApiError } from "@/lib/api";
import { useT } from "@/lib/i18n";
import { useComfyConnections, useComfyPower } from "@/lib/queries";
import type { ComfyConnection } from "@/lib/types";

/** Where Krea 2 runs, chosen right on the Generate page: this Mac or a rented GPU. */
export function ComfyConnectionBar({
  selected,
  onSelect,
}: {
  selected: ComfyConnection | undefined;
  onSelect: (id: string) => void;
}) {
  const t = useT();
  const confirm = useConfirm();
  const conns = useComfyConnections();
  const power = useComfyPower();
  const [adding, setAdding] = useState<AddMode | null>(null);
  const [managing, setManaging] = useState(false);
  const items = conns.data?.items ?? [];
  const deployed = selected?.auth_type === "agent";

  const start = async (c: ComfyConnection) => {
    const ok = await confirm({
      title: t("gpu.billingTitle"),
      description: t("gpu.billingBody", { provider: c.gpu_provider === "modal" ? "Modal" : "RunPod" }),
      confirmLabel: t("cf.start"),
    });
    if (!ok) return;
    try {
      await power.mutateAsync({ id: c.id, action: "start" });
    } catch (e) {
      toast.error(e instanceof ApiError ? e.message : t("errors.generic"));
    }
  };
  const stop = async (c: ComfyConnection) => {
    try {
      await power.mutateAsync({ id: c.id, action: "stop" });
      toast.success(t("cf.stopped"));
    } catch (e) {
      toast.error(e instanceof ApiError ? e.message : t("errors.generic"));
    }
  };

  const modals = (
    <>
      <AddConnectionModal open={adding !== null} mode={adding ?? "local"} onClose={() => setAdding(null)} />
      {managing ? (
        <Modal open wide onClose={() => setManaging(false)} title={t("cf.manageTitle")} description={t("cf.manageBody")}>
          <div className="col gap-16">
            {items.map((c) => (
              <ConnectionCard key={c.id} conn={c} />
            ))}
            <WorkflowsCard />
          </div>
        </Modal>
      ) : null}
    </>
  );

  if (!items.length) {
    return (
      <div className="connect-box">
        <div className="h3">{t("k2.connectTitle")}</div>
        <div className="small subtle">{t("k2.connectBody")}</div>
        <div className="connect-choices">
          <button type="button" className="model-card" onClick={() => setAdding("local")}>
            <span className="row" style={{ gap: 8 }}>
              <Laptop size={18} />
              <span className="name">{t("cf.mode.local")}</span>
            </span>
            <span className="desc">{t("cf.choiceLocal")}</span>
          </button>
          <button type="button" className="model-card" onClick={() => setAdding("deploy")}>
            <span className="row" style={{ gap: 8 }}>
              <CloudCog size={18} />
              <span className="name">{t("cf.mode.deploy")}</span>
            </span>
            <span className="desc">{t("cf.choiceDeploy")}</span>
          </button>
        </div>
        {modals}
      </div>
    );
  }

  return (
    <div className="col" style={{ gap: 8 }}>
      <div className="row" style={{ gap: 8 }}>
        <div className="grow" style={{ minWidth: 0 }}>
          <Select
            label={t("cf.runsOn")}
            value={selected?.id ?? null}
            onChange={onSelect}
            options={items.map((c) => ({ value: c.id, label: c.name }))}
          />
        </div>
        {selected ? <StatusPill status={selected.status} /> : null}
        <IconButton label={t("cf.manageTitle")} onClick={() => setManaging(true)}>
          <Settings2 />
        </IconButton>
        <IconButton label={t("cf.add")} onClick={() => setAdding("local")}>
          <Plus />
        </IconButton>
      </div>
      {selected && deployed ? (
        <div className="row" style={{ gap: 8 }}>
          {selected.status === "stopped" || selected.status === "error" ? (
            <Button size="sm" icon={<Play />} loading={power.isPending} onClick={() => start(selected)}>
              {t("cf.startGpu")}
            </Button>
          ) : (
            <Button size="sm" icon={<Square />} loading={power.isPending} onClick={() => stop(selected)}>
              {t("cf.stopGpu")}
            </Button>
          )}
          <span className="tiny subtle">
            {selected.status === "deploying" ? t(`cf.stage.${selected.deploy.stage ?? "starting"}`) : t("cf.gpuBills")}
          </span>
        </div>
      ) : null}
      {selected?.last_error && selected.status !== "ready" ? <div className="error-text tiny">{selected.last_error}</div> : null}
      {modals}
    </div>
  );
}
