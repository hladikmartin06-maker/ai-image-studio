// Connects the assistant with the Generate page: the page publishes what the user has set
// up (for context-aware answers) and receives prompts/settings the user applies from chat.

import type { ReferenceImage, StudioContextPayload } from "@/lib/types";

export interface StudioPatch {
  prompt?: string;
  model?: string;
  aspect_ratio?: string;
  resolution?: string;
  count?: number;
  seed?: number | string;
  thinking_mode?: boolean;
  // "Animate": these photos become the references (the first one is the video's first frame).
  refs?: ReferenceImage[];
  // Switch Generate to the online models.
  engine?: "hosted";
}

const PENDING_KEY = "studio:pending-apply";

let current: StudioContextPayload | null = null;
let applyHandler: ((patch: StudioPatch) => void) | null = null;

export function publishStudioContext(ctx: StudioContextPayload | null) {
  current = ctx;
}

/** The live context from the Generate page, else the saved draft of the signed-in user. */
export function readStudioContext(userId: string | undefined, page: string): StudioContextPayload {
  if (current) return { ...current, page };
  if (userId) {
    try {
      const draft = JSON.parse(localStorage.getItem(`studio:draft:${userId}`) ?? "null");
      if (draft) {
        return {
          page,
          model: draft.model,
          provider: draft.provider,
          settings: draft.settings ?? {},
          prompt: draft.prompt,
          reference_count: Array.isArray(draft.refs) ? draft.refs.length : 0,
          count: draft.count,
        };
      }
    } catch {
      /* no draft */
    }
  }
  return { page };
}

export function registerApplyHandler(handler: (patch: StudioPatch) => void) {
  applyHandler = handler;
  return () => {
    if (applyHandler === handler) applyHandler = null;
  };
}

/** Applies at once when the Generate page is open; otherwise it is applied when it opens. */
export function applyToStudio(patch: StudioPatch): "applied" | "pending" {
  if (applyHandler) {
    applyHandler(patch);
    return "applied";
  }
  try {
    sessionStorage.setItem(PENDING_KEY, JSON.stringify(patch));
  } catch {
    /* storage blocked */
  }
  return "pending";
}

export function takePendingApply(): StudioPatch | null {
  try {
    const raw = sessionStorage.getItem(PENDING_KEY);
    if (!raw) return null;
    sessionStorage.removeItem(PENDING_KEY);
    return JSON.parse(raw) as StudioPatch;
  } catch {
    return null;
  }
}

// "Enhance photo" from an image viewer: opens Generate › Enhance photo with that image.
export interface EnhanceRequest {
  kind: "reference" | "image";
  id: string;
  thumb_url: string;
  // "advanced": opens Generate › Advanced (edit this photo) instead of Enhance photo.
  target?: "enhance" | "advanced";
}

const ENHANCE_KEY = "studio:pending-enhance";
let enhanceHandler: ((photo: EnhanceRequest) => void) | null = null;

export function registerEnhanceHandler(handler: (photo: EnhanceRequest) => void) {
  enhanceHandler = handler;
  return () => {
    if (enhanceHandler === handler) enhanceHandler = null;
  };
}

/** At once when the Generate page is open; otherwise when it opens. */
export function enhanceInStudio(photo: EnhanceRequest): "applied" | "pending" {
  if (enhanceHandler) {
    enhanceHandler(photo);
    return "applied";
  }
  try {
    sessionStorage.setItem(ENHANCE_KEY, JSON.stringify(photo));
  } catch {
    /* storage blocked */
  }
  return "pending";
}

export function takePendingEnhance(): EnhanceRequest | null {
  try {
    const raw = sessionStorage.getItem(ENHANCE_KEY);
    if (!raw) return null;
    sessionStorage.removeItem(ENHANCE_KEY);
    return JSON.parse(raw) as EnhanceRequest;
  } catch {
    return null;
  }
}
