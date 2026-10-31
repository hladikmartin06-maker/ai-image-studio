import path from "node:path";

import type { NextConfig } from "next";

// The FastAPI backend. The browser only talks to Next.js; /api/* is forwarded to the
// backend on 127.0.0.1, so provider API keys never reach the browser.
const BACKEND_URL = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

const nextConfig: NextConfig = {
  reactStrictMode: true,
  poweredByHeader: false,
  // Compression would buffer the live event stream (/api/events).
  compress: false,
  experimental: {
    // Reference uploads pass through the proxy (backend limit: 10 MB per image).
    proxyClientMaxBodySize: "12mb",
    // Archive downloads and slow provider uploads may take longer than the 30 s default.
    proxyTimeout: 10 * 60 * 1000,
  },
  turbopack: {
    // Pin the root, so a lockfile in a parent folder is never picked up.
    root: path.resolve(__dirname),
  },
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${BACKEND_URL}/api/:path*` }];
  },
  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "X-Frame-Options", value: "DENY" },
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "same-origin" },
        ],
      },
    ];
  },
};

export default nextConfig;
