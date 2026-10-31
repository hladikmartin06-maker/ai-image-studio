// Streaming assistant answers: POST, then read server-sent events from the response body.

import { ApiError } from "@/lib/api";
import type { ChatMessage, StudioContextPayload } from "@/lib/types";

export interface SendBody {
  content: string;
  context?: StudioContextPayload | null;
  quick_action?: "improve_prompt" | "suggest_settings" | "troubleshoot" | "compare_models";
  provider?: string | null;
  mode?: "low" | "balanced" | "high";
  image_ids?: string[];
}

export interface StreamHandlers {
  onStart: (user: ChatMessage, assistant: ChatMessage) => void;
  onDelta: (text: string) => void;
  onError: (code: string, message: string) => void;
  onDone: (message: ChatMessage | null) => void;
}

export async function streamChat(
  conversationId: string,
  body: SendBody,
  handlers: StreamHandlers,
  signal: AbortSignal,
): Promise<void> {
  const response = await fetch(`/api/ai/conversations/${conversationId}/messages`, {
    method: "POST",
    headers: { "content-type": "application/json", "x-studio-client": "web" },
    body: JSON.stringify(body),
    credentials: "same-origin",
    signal,
  });
  if (!response.ok || !response.body) {
    let code = `http_${response.status}`;
    let message = "The assistant is not reachable.";
    try {
      const data = await response.json();
      code = data?.error?.code ?? code;
      message = data?.error?.message ?? message;
    } catch {
      /* not JSON */
    }
    throw new ApiError(response.status, code, message);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let finished = false;
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let index: number;
    while ((index = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, index);
      buffer = buffer.slice(index + 2);
      let name = "";
      let data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) name = line.slice(7);
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      if (!name || !data) continue;
      const payload = JSON.parse(data);
      if (name === "start") handlers.onStart(payload.user_message, payload.assistant_message);
      else if (name === "delta") handlers.onDelta(payload.text);
      else if (name === "error") handlers.onError(payload.code, payload.message);
      else if (name === "done") {
        finished = true;
        handlers.onDone(payload.message?.id ? payload.message : null);
      }
    }
  }
  if (!finished) handlers.onDone(null);
}
