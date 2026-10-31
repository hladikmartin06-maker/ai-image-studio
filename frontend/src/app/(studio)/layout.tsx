import type { ReactNode } from "react";

import { StudioShell } from "@/components/layout/shell";

export default function StudioLayout({ children }: { children: ReactNode }) {
  return <StudioShell>{children}</StudioShell>;
}
