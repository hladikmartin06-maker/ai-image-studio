"use client";

import { useSyncExternalStore } from "react";

/**
 * Images and generations the user just deleted. They stay hidden in every list even when a
 * refetch (live updates, polling of running jobs) still returns them before the server has
 * deleted them — so a deleted photo never flashes back. "Undo" or a failed delete shows
 * them again.
 */
const hidden = new Set<string>();
const listeners = new Set<() => void>();
let version = 0;

function changed() {
  version += 1;
  for (const listener of listeners) listener();
}

export function hideDeleted(id: string) {
  if (hidden.has(id)) return;
  hidden.add(id);
  changed();
}

export function showAgain(id: string) {
  if (!hidden.delete(id)) return;
  changed();
}

export function isHidden(id: string) {
  return hidden.has(id);
}

export function hiddenCount() {
  return hidden.size;
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  return () => {
    listeners.delete(listener);
  };
}

/** Changes whenever the set changes, so lists filter again. */
export function useHiddenVersion() {
  return useSyncExternalStore(
    subscribe,
    () => version,
    () => 0,
  );
}
