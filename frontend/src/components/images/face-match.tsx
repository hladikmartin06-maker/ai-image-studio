"use client";

import { ScanFace } from "lucide-react";
import { useState } from "react";

import { Button } from "@/components/ui";
import { api, ApiError } from "@/lib/api";
import { useT } from "@/lib/i18n";

export interface FaceResult {
  image_id: string;
  face: boolean;
  score: number | null;
  match: boolean | null;
}
interface CompareResponse {
  reference: { kind: "persona" | "lora"; name: string; photos: number; faces: number };
  threshold: number;
  items: FaceResult[];
}

export async function compareFaces(imageIds: string[], target: { persona_id?: string; lora_id?: string }) {
  return api<CompareResponse>("/api/faces/compare", { body: { image_ids: imageIds, ...target } });
}

/** The score as people read it: "78 %" (cosine similarity of the face features). */
export function FaceBadge({ result }: { result: FaceResult | undefined }) {
  const t = useT();
  if (!result) return null;
  if (!result.face || result.score === null) return <span className="badge">{t("fm.noFace")}</span>;
  return (
    <span className={`badge ${result.match ? "success" : "danger"}`} title={t(result.match ? "fm.same" : "fm.different")}>
      {t("fm.score", { n: Math.max(0, Math.round(result.score * 100)) })}
    </span>
  );
}

/** Image viewer: how close the face is to the AI model's photos (or the LoRA's photos). Runs
 * on this Mac with OpenCV's face models (downloaded once). */
export function FaceMatch({ imageId, target, name }: { imageId: string; target: { persona_id?: string; lora_id?: string }; name: string }) {
  const t = useT();
  const [state, setState] = useState<{ busy: boolean; result?: FaceResult; error?: string; faces?: number }>({ busy: false });
  const run = async () => {
    setState({ busy: true });
    try {
      const res = await compareFaces([imageId], target);
      setState({ busy: false, result: res.items[0], faces: res.reference.faces });
    } catch (e) {
      setState({ busy: false, error: e instanceof ApiError ? e.message : t("errors.generic") });
    }
  };
  return (
    <div className="col" style={{ gap: 8 }}>
      <div className="row" style={{ gap: 8 }}>
        <Button size="sm" icon={<ScanFace />} loading={state.busy} onClick={run}>
          {t("fm.check", { name })}
        </Button>
        <FaceBadge result={state.result} />
      </div>
      {state.result?.score !== null && state.result ? (
        <div className="tiny subtle">{t(state.result.match ? "fm.same" : "fm.different")} · {t("fm.hint", { n: state.faces ?? 0 })}</div>
      ) : null}
      {state.error ? <div className="error-text small">{state.error}</div> : null}
    </div>
  );
}
