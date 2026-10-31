import type { Metadata, Viewport } from "next";
import type { ReactNode } from "react";

import { Providers } from "@/components/providers";

import "./globals.css";

export const metadata: Metadata = {
  title: "AI Image Studio",
  description: "Generate, organise and manage AI images with WaveSpeed and Atlas Cloud.",
  icons: { icon: "/icon.svg" },
};

export const viewport: Viewport = {
  themeColor: "#08080a",
  colorScheme: "dark light",
};

// Applies the saved theme before the first paint, so there is no flash of the wrong theme.
const themeScript = `(function(){try{var a=JSON.parse(localStorage.getItem("studio:appearance")||"{}");var t=a.theme||"dark";if(t==="system"){t=matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light"}var r=document.documentElement;r.dataset.theme=t;r.dataset.density=a.density||"comfortable";r.dataset.motion=a.motion?"reduced":"full";if(a.language)r.lang=a.language}catch(e){}})();`;

export default function RootLayout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" data-theme="dark" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeScript }} />
      </head>
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
