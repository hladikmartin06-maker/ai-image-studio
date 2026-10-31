import { NextResponse, type NextRequest } from "next/server";

// The studio is only for this computer. A web page on another site could point its own
// domain at 127.0.0.1 ("DNS rebinding") and then call the local studio; such requests
// arrive with that foreign host name and are refused here.
const ALLOWED_HOSTS = new Set(["localhost:3000", "127.0.0.1:3000", "[::1]:3000"]);

export function proxy(request: NextRequest) {
  const host = request.headers.get("host") ?? "";
  if (!ALLOWED_HOSTS.has(host)) {
    return new NextResponse("This studio only accepts requests from this computer.", {
      status: 403,
    });
  }
  return NextResponse.next();
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|icon.svg).*)"],
};
