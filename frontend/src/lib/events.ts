"use client";

import { useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef } from "react";

import { keys, upsertGeneration } from "@/lib/queries";
import type { Generation } from "@/lib/types";

type Notify = (gen: Generation) => void;

/**
 * Live job updates. Opens one server-sent event stream (/api/events) for the signed-in
 * user and writes each update into the query cache, so every page shows it immediately.
 * When the stream cannot be opened, active jobs are polled every 4 s instead.
 */
export function useLiveGenerations(enabled: boolean, onFinished?: Notify) {
  const qc = useQueryClient();
  const finished = useRef(onFinished);
  useEffect(() => {
    finished.current = onFinished;
  }, [onFinished]);

  useEffect(() => {
    if (!enabled) return;
    let source: EventSource | null = null;
    let pollTimer: number | undefined;
    let invalidateTimer: number | undefined;
    let trainingTimer: number | undefined;
    let failures = 0;
    let closed = false;
    const lastStatus = new Map<string, string>();

    const scheduleInvalidate = () => {
      window.clearTimeout(invalidateTimer);
      invalidateTimer = window.setTimeout(() => {
        qc.invalidateQueries({ queryKey: keys.images });
        qc.invalidateQueries({ queryKey: keys.stats });
        qc.invalidateQueries({ queryKey: ["generations", "list"] });
      }, 400);
    };

    const handle = (gen: Generation) => {
      upsertGeneration(qc, gen);
      const before = lastStatus.get(gen.id);
      lastStatus.set(gen.id, gen.status);
      const done = gen.status === "completed" || gen.status === "failed";
      if (done && before && before !== gen.status) finished.current?.(gen);
      scheduleInvalidate();
    };

    const poll = async () => {
      try {
        const res = await fetch("/api/generations?active=true&limit=20", {
          credentials: "same-origin",
        });
        if (res.ok) {
          const data = (await res.json()) as { items: Generation[] };
          const activeIds = new Set(data.items.map((g) => g.id));
          data.items.forEach(handle);
          // Jobs that left the active list have finished: fetch their final state once.
          for (const [id, status] of lastStatus) {
            if ((status === "queued" || status === "processing") && !activeIds.has(id)) {
              const r = await fetch(`/api/generations/${id}`, { credentials: "same-origin" });
              if (r.ok) handle((await r.json()) as Generation);
            }
          }
        }
      } catch {
        /* offline: try again on the next tick */
      }
      if (!closed) pollTimer = window.setTimeout(poll, 4000);
    };

    const connect = () => {
      if (closed) return;
      source = new EventSource("/api/events", { withCredentials: true });
      source.addEventListener("ready", () => {
        failures = 0;
        window.clearTimeout(pollTimer);
        pollTimer = undefined;
        // Catch up on anything that changed while disconnected.
        qc.invalidateQueries({ queryKey: keys.generations });
      });
      source.addEventListener("training", () => {
        // LoRA training progress: refresh the LoRA pages and the job list.
        window.clearTimeout(trainingTimer);
        trainingTimer = window.setTimeout(() => {
          qc.invalidateQueries({ queryKey: ["loras"] });
          qc.invalidateQueries({ queryKey: ["jobs"] });
        }, 600);
      });
      source.addEventListener("generation", (event) => {
        try {
          handle(JSON.parse((event as MessageEvent).data) as Generation);
        } catch {
          /* ignore malformed event */
        }
      });
      source.onerror = () => {
        failures += 1;
        if (failures >= 3 && source) {
          source.close();
          source = null;
          if (pollTimer === undefined) poll();
          // Try the stream again later.
          window.setTimeout(() => {
            failures = 0;
            connect();
          }, 30_000);
        }
      };
    };

    connect();
    return () => {
      closed = true;
      source?.close();
      window.clearTimeout(pollTimer);
      window.clearTimeout(invalidateTimer);
      window.clearTimeout(trainingTimer);
    };
  }, [enabled, qc]);
}

