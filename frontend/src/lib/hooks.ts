"use client";

import { useEffect, useState } from "react";

/** The value, updated only after it stopped changing for `ms` (search inputs). */
export function useDebounced<T>(value: T, ms = 300) {
  const [current, setCurrent] = useState(value);
  useEffect(() => {
    const id = window.setTimeout(() => setCurrent(value), ms);
    return () => window.clearTimeout(id);
  }, [value, ms]);
  return current;
}
