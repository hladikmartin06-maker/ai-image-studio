"use client";

import { Play, Plus, RotateCcw, ScrollText, Settings2, Square, Trash2 } from "lucide-react";
import { useState } from "react";
import { toast } from "sonner";

import { type AddMode, AddConnectionModal, ConnectForm, ConnectionCard } from "@/components/comfy/connections";
import { WorkflowsCard } from "@/components/comfy/workflows";
import { formatElapsed, formatMoney, StatusPill } from "@/components/studio/common";
import { Button, IconButton, Modal, Notice, Select, useConfirm } from "@/components/ui";
import { ApiError } from "@/lib/api";
import { useI18n } from "@/lib/i18n";
import type { MessageKey } from "@/lib/i18n";
import { useComfyConnections, useComfyPower, useRemoveComfy } from "@/lib/queries";
import type { ComfyConnection } from "@/lib/types";

/** Spent so far and time left in the GPU's session (estimate). */
function SessionLine({ conn }: { conn: ComfyConnection }) {
  const { t, locale } = useI18n();
  const s = conn.session;
  if (!s || s.per_hour == null) return null;
  return (
    <span className="tiny subtle" style={{ display: "block" }}>
      {t("cf.sessionLine", {
        rate: formatMoney(s.per_hour, locale),
        spent: formatMoney(s.spent ?? 0, locale),
        left: formatElapsed(s.seconds_left),
      })}
    </span>
  );
}

function secondsSince(iso: string | undefined) {
  return iso ? Math.max(0, (Date.now() - new Date(iso).getTime()) / 1000) : null;
}

/** Step 1 on Generate › Krea 2: where it runs. Pick a GPU (or this Mac), connect, and see
 * the setup progress until ComfyUI is ready. */
export function ComfyConnectionBar({
  selected,
  onSelect,
}: {
  selected: ComfyConnection | undefined;
  onSelect: (id: string) => void;
}) {
  const { t, locale } = useI18n();
  const confirm = useConfirm();
  const conns = useComfyConnections();
  const power = useComfyPower();
  const remove = useRemoveComfy();
  const [adding, setAdding] = useState<AddMode | null>(null);
  const [managing, setManaging] = useState(false);
  const [log, setLog] = useState(false);
  const items = conns.data?.items ?? [];
  const deployed = selected?.auth_type === "agent";

  const run = async (fn: () => Promise<unknown>, done?: string) => {
    try {
      await fn();
      if (done) toast.success(done);
    } catch (e) {
      toast.error(e instanceof ApiError ? e.message : t("errors.generic"));
    }
  };
  const start = async (c: ComfyConnection) => {
    const ok = await confirm({
      title: t("gpu.billingTitle"),
      description: t("gpu.billingBody", { provider: c.gpu_provider === "modal" ? "Modal" : "RunPod" }),
      confirmLabel: t("cf.startGpu"),
    });
    if (ok) await run(() => power.mutateAsync({ id: c.id, action: "start" }));
  };
  const onRemove = async (c: ComfyConnection) => {
    const agent = c.auth_type === "agent";
    const ok = await confirm({
      title: t("cf.removeTitle", { name: c.name }),
      description: agent ? t("cf.removeDeployedBody") : t("cf.removeBody"),
      confirmLabel: agent ? t("cf.removeAndTerminate") : t("common.delete"),
      danger: true,
    });
    if (ok) await run(() => remove.mutateAsync({ id: c.id, terminate: agent }), t("cf.removed"));
  };

  const modals = (
    <>
      <AddConnectionModal open={adding !== null} mode={adding ?? "deploy"} onClose={() => setAdding(null)} />
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
      {log && selected ? (
        <Modal open wide onClose={() => setLog(false)} title={t("cf.logs")} description={selected.name}>
          <pre className="log-box">{selected.deploy.log_tail || t("cf.noLogs")}</pre>
        </Modal>
      ) : null}
    </>
  );

  if (!items.length) {
    return (
      <div className="connect-box">
        <div className="h3">{t("k2.connectTitle")}</div>
        <div className="small subtle">{t("k2.connectBody")}</div>
        <ConnectForm initialMode="deploy" />
        {modals}
      </div>
    );
  }

  const d = selected?.deploy ?? {};
  // starting: the GPU boots and the studio agent comes up; booting: ComfyUI starts (the first
  // start copies it onto the GPU's disk).
  const stage = d.stage === "booting" ? "booting" : d.stage === "creating" ? "creating" : "starting";
  const since = secondsSince(d.started_at);

  return (
    <div className="col" style={{ gap: 10 }}>
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
        <IconButton label={t("cf.add")} onClick={() => setAdding("deploy")}>
          <Plus />
        </IconButton>
      </div>

      {selected?.status === "deploying" ? (
        <div className="setup-box">
          <div className="row" style={{ gap: 8 }}>
            <span className="dot pulse" style={{ color: "var(--accent)" }} />
            <span className="small grow" style={{ fontWeight: 600 }}>
              {t(stage === "booting" ? "cf.setupBooting" : stage === "creating" ? "cf.setupCreating" : "cf.setupStarting")}
            </span>
            {since !== null ? <span className="tiny subtle">{formatElapsed(since)}</span> : null}
          </div>
          <ol className="setup-steps tiny">
            <li data-state={stage === "booting" ? "done" : "active"}>{t("cf.setupStep1")}</li>
            <li data-state={stage === "booting" ? "active" : "todo"}>{t("cf.setupStep2")}</li>
            <li data-state="todo">{t("cf.setupStep3")}</li>
          </ol>
          <div className="tiny subtle">{t(stage === "creating" ? "cf.setupCreatingHint" : "cf.setupHint")}</div>
          {selected.session ? <SessionLine conn={selected} /> : null}
          <div className="row wrap" style={{ gap: 8 }}>
            <Button size="sm" icon={<ScrollText />} onClick={() => setLog(true)}>
              {t("cf.logs")}
            </Button>
            {deployed ? (
              <Button
                size="sm"
                variant="ghost"
                icon={<Square />}
                loading={power.isPending}
                onClick={() => run(() => power.mutateAsync({ id: selected.id, action: "stop" }), t("cf.stopped"))}
              >
                {t("cf.stopGpu")}
              </Button>
            ) : null}
          </div>
        </div>
      ) : null}

      {selected?.status === "error" ? (
        <Notice tone="danger">
          <div>{selected.last_error ?? t("errors.generic")}</div>
          <div className="row wrap" style={{ gap: 8, marginTop: 10 }}>
            {deployed ? (
              <Button size="sm" variant="primary" icon={<RotateCcw />} loading={power.isPending} onClick={() => start(selected)}>
                {t("cf.tryAgain")}
              </Button>
            ) : null}
            {deployed ? (
              <Button size="sm" icon={<ScrollText />} onClick={() => setLog(true)}>
                {t("cf.logs")}
              </Button>
            ) : null}
            <Button size="sm" variant="ghost" icon={<Trash2 />} onClick={() => onRemove(selected)}>
              {t("common.delete")}
            </Button>
          </div>
        </Notice>
      ) : null}

      {selected?.status === "stopped" && deployed && d.stopped_by && d.stopped_by !== "user" ? (
        <div className="tiny subtle">
          {t(`cf.stoppedBy.${d.stopped_by}` as MessageKey)}
        </div>
      ) : null}
      {selected?.status === "stopped" && deployed ? (
        <div className="row" style={{ gap: 8 }}>
          <Button size="sm" variant="primary" icon={<Play />} loading={power.isPending} onClick={() => start(selected)}>
            {t("cf.startGpu")}
          </Button>
          <span className="tiny subtle">{t("cf.stoppedHint")}</span>
        </div>
      ) : null}

      {selected?.status === "ready" ? (
        <div className="row wrap" style={{ gap: 8 }}>
          <span className="tiny subtle grow">
            {[
              selected.info.devices?.[0]?.name?.replace(/^cuda:\d+\s*/, "").split(":")[0],
              deployed && !selected.session && d.cost_per_hour ? `${formatMoney(d.cost_per_hour, locale)}/h` : null,
              deployed ? t("cf.gpuBills") : null,
            ]
              .filter(Boolean)
              .join(" · ")}
            {deployed && selected.session ? <SessionLine conn={selected} /> : null}
          </span>
          {deployed ? (
            <Button
              size="sm"
              variant="ghost"
              icon={<Square />}
              loading={power.isPending}
              onClick={() => run(() => power.mutateAsync({ id: selected.id, action: "stop" }), t("cf.stopped"))}
            >
              {t("cf.stopGpu")}
            </Button>
          ) : null}
        </div>
      ) : null}
      {modals}
    </div>
  );
}
