// The only way the browser talks to the backend: same-origin /api, proxied by Next.js.
// The session cookie is httpOnly; API keys are never sent back to the browser.

export class ApiError extends Error {
  status: number;
  code: string;

  constructor(status: number, code: string, message: string) {
    super(message);
    this.status = status;
    this.code = code;
  }
}

type Json = Record<string, unknown> | unknown[];

export async function api<T>(
  path: string,
  options: { method?: string; body?: Json | FormData; signal?: AbortSignal } = {},
): Promise<T> {
  const headers: Record<string, string> = { "x-studio-client": "web" };
  let body: BodyInit | undefined;
  if (options.body instanceof FormData) {
    body = options.body;
  } else if (options.body !== undefined) {
    headers["content-type"] = "application/json";
    body = JSON.stringify(options.body);
  }
  let response: Response;
  try {
    response = await fetch(path, {
      method: options.method ?? (body ? "POST" : "GET"),
      headers,
      body,
      credentials: "same-origin",
      signal: options.signal,
    });
  } catch (error) {
    if ((error as Error).name === "AbortError") throw error;
    throw new ApiError(0, "network_error", "The studio server is not reachable.");
  }
  if (!response.ok) {
    let code = `http_${response.status}`;
    let message = response.statusText || "Request failed";
    try {
      const data = await response.json();
      if (data?.error) {
        code = data.error.code ?? code;
        message = data.error.message ?? message;
      }
    } catch {
      if (response.status >= 500) message = "The studio server is not reachable.";
    }
    if (response.status === 401 && typeof window !== "undefined" && code === "unauthenticated") {
      window.dispatchEvent(new CustomEvent("studio:unauthenticated"));
    }
    throw new ApiError(response.status, code, message);
  }
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export function query(params: Record<string, string | number | boolean | null | undefined>) {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params)) {
    if (value === undefined || value === null || value === "") continue;
    search.set(key, String(value));
  }
  const text = search.toString();
  return text ? `?${text}` : "";
}

/** Starts a browser download of a same-origin URL (the session cookie is sent). */
export function downloadUrl(url: string) {
  const a = document.createElement("a");
  a.href = url;
  a.download = "";
  a.rel = "noopener";
  document.body.appendChild(a);
  a.click();
  a.remove();
}
