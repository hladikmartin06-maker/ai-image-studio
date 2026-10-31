"use client";

import { Images, ListOrdered } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

import { useT } from "@/lib/i18n";

export function LibraryTabs() {
  const t = useT();
  const pathname = usePathname();
  const tabs = [
    { href: "/library", label: t("lib.tabImages"), icon: <Images /> },
    { href: "/library/history", label: t("lib.tabHistory"), icon: <ListOrdered /> },
  ];
  return (
    <nav className="segmented" aria-label={t("lib.sections")}>
      {tabs.map((tab) => (
        <Link
          key={tab.href}
          href={tab.href}
          aria-current={pathname === tab.href ? "page" : undefined}
          data-pressed={pathname === tab.href || undefined}
        >
          {tab.icon}
          {tab.label}
        </Link>
      ))}
    </nav>
  );
}
