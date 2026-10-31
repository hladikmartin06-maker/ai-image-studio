"use client";

import { ArrowRight, Images, MessagesSquare, Plug, Sparkles, Wand2 } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { toast } from "sonner";

import {
  type DraftSettings,
  ModelPicker,
  PromptEditor,
  ReferenceImages,
  SettingsControls,
} from "@/components/generate/controls";
import { openAssistant } from "@/components/assistant/drawer";
import { ComfyPanel } from "@/components/comfy/generate-panel";
import {
  attachedPersonaImages,
  newPersonaDraft,
  type PersonaDraft,
  PersonaSection,
} from "@/components/personas/persona-picker";
import { JobCard } from "@/components/generate/job-card";
import { ImageViewer } from "@/components/images/image-viewer";
import { ProviderLogo } from "@/components/logos";
import { Button, EmptyState, IconButton, Notice, Segmented, Skeleton, Spinner } from "@/components/ui";
import { api, ApiError } from "@/lib/api";
import { useT } from "@/lib/i18n";
import {
  publishStudioContext,
  registerApplyHandler,
  type StudioPatch,
  takePendingApply,
} from "@/lib/studio-bridge";
import {
  useAuthState,
  useCatalog,
  useCreateGeneration,
  useGenerations,
  useIntegrations,
  usePersonas,
  useSettings,
} from "@/lib/queries";
import type {
  Generation,
  ModelInfo,
  ModelKey,
  Preferences,
  ProviderKey,
  ReferenceImage,
  StudioImage,
  Variant,
} from "@/lib/types";

interface Draft {
  model: ModelKey;
  provider: ProviderKey | null;
  prompt: string;
  count: number;
  settings: DraftSettings;
  refs: ReferenceImage[];
  // The saved AI model (character) whose description and references go with the request.
  persona: PersonaDraft | null;
}

/** The AI model part of a past generation, for "load these settings". */
function personaFromGeneration(gen: Generation): { persona: PersonaDraft | null; refs: ReferenceImage[] } {
  const used = gen.persona?.id ? gen.persona : null;
  if (!used?.id) return { persona: null, refs: gen.reference_images };
  const fromPersona = new Set(used.reference_ids);
  return {
    persona: {
      ...newPersonaDraft(used.id),
      useRefs: used.references_used,
      useDescription: used.description_used,
    },
    refs: gen.reference_images.filter((r) => !fromPersona.has(r.id)),
  };
}

function pickVariant(model: ModelInfo | undefined, provider: ProviderKey | null, withRefs: boolean) {
  if (!model || !provider) return undefined;
  return model.variants.find(
    (v) => v.provider === provider && v.mode === (withRefs ? "image-edit" : "text-to-image"),
  );
}

/** Keeps the draft settings valid for the endpoint that will run them. */
function fitSettings(variant: Variant, s: DraftSettings, prefs: Preferences | undefined): DraftSettings {
  const out: DraftSettings = { ...s };
  const ratios = variant.allow_match_input ? ["match", ...variant.aspect_ratios] : variant.aspect_ratios;
  if (variant.controls.includes("aspect_ratio")) {
    if (!out.aspect_ratio || !ratios.includes(out.aspect_ratio)) {
      const preferred = prefs?.default_aspect_ratio;
      out.aspect_ratio =
        variant.mode === "image-edit" && variant.allow_match_input
          ? "match"
          : preferred && ratios.includes(preferred)
            ? preferred
            : (variant.default_aspect_ratio ?? ratios[0]);
    }
  } else delete out.aspect_ratio;
  if (variant.controls.includes("resolution")) {
    const match = (r?: string | null) =>
      r ? variant.resolutions.find((x) => x.toLowerCase() === r.toLowerCase()) : undefined;
    out.resolution =
      match(out.resolution) ?? match(prefs?.default_resolution) ?? variant.default_resolution ?? variant.resolutions[0];
  } else delete out.resolution;
  if (variant.controls.includes("output_format")) {
    if (!out.output_format || !variant.output_formats.includes(out.output_format)) {
      out.output_format =
        prefs && variant.output_formats.includes(prefs.default_output_format)
          ? prefs.default_output_format
          : variant.output_formats[0];
    }
  } else delete out.output_format;
  if (!variant.controls.includes("seed")) delete out.seed;
  if (variant.controls.includes("thinking_mode")) {
    if (out.thinking_mode === undefined) out.thinking_mode = prefs?.default_thinking_mode ?? true;
  } else delete out.thinking_mode;
  return out;
}

function toPayload(settings: DraftSettings) {
  const out: Record<string, unknown> = {};
  if (settings.aspect_ratio) out.aspect_ratio = settings.aspect_ratio;
  if (settings.resolution) out.resolution = settings.resolution;
  if (settings.output_format) out.output_format = settings.output_format;
  if (settings.seed !== undefined && settings.seed !== "") out.seed = Number(settings.seed);
  if (settings.thinking_mode !== undefined) out.thinking_mode = settings.thinking_mode;
  return out;
}

function newRequestId() {
  return typeof crypto !== "undefined" && "randomUUID" in crypto
    ? crypto.randomUUID()
    : `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

export default function GeneratePage() {
  const t = useT();
  const router = useRouter();
  const auth = useAuthState();
  const catalog = useCatalog();
  const settings = useSettings();
  const integrations = useIntegrations();
  const create = useCreateGeneration();
  const results = useGenerations({ limit: 10 });
  const personas = usePersonas();
  const prefs = settings.data;
  const userId = auth.data?.user?.id;
  const draftKey = userId ? `studio:draft:${userId}` : null;

  const connected = useMemo(
    () =>
      (integrations.data?.integrations ?? [])
        .filter((i) => i.status !== "not_connected")
        .map((i) => i.key),
    [integrations.data],
  );

  const [draft, setDraft] = useState<Draft | null>(null);
  // Hosted models (WaveSpeed / Atlas) or Krea 2 on a ComfyUI connection.
  const [engine, setEngineState] = useState<"hosted" | "comfy">(() => {
    if (typeof window === "undefined") return "hosted";
    if (new URLSearchParams(window.location.search).get("lora")) return "comfy";
    try {
      return localStorage.getItem("studio:engine") === "comfy" ? "comfy" : "hosted";
    } catch {
      return "hosted";
    }
  });
  const [loraFromUrl] = useState<string | null>(() =>
    typeof window === "undefined" ? null : new URLSearchParams(window.location.search).get("lora"),
  );
  const [comfyLoad, setComfyLoad] = useState<Generation | null>(null);
  const setEngine = useCallback((value: "hosted" | "comfy") => {
    setEngineState(value);
    try {
      localStorage.setItem("studio:engine", value);
    } catch {
      /* storage blocked */
    }
  }, []);
  const [viewer, setViewer] = useState<{ images: StudioImage[]; index: number } | null>(null);
  const [touched, setTouched] = useState(false);
  const submitting = useRef(false);

  // Initial draft: ?from=<generation id>, else the saved draft, else the defaults.
  useEffect(() => {
    if (draft || !prefs || !integrations.data || !catalog.data) return;
    const defaults: Draft = {
      model: prefs.default_model,
      provider: integrations.data.active_provider ?? connected[0] ?? null,
      prompt: "",
      count: prefs.default_image_count,
      settings: {},
      refs: [],
      persona: null,
    };
    const params = new URLSearchParams(window.location.search);
    const from = params.get("from");
    // ?ai_model=<id>: "Use in Generate" from the AI Models page.
    const chosenPersona = params.get("ai_model");
    if (from) {
      api<Generation>(`/api/generations/${from}`)
        .then((gen) => {
          if (gen.provider === "comfyui") {
            setComfyLoad(gen);
            setEngine("comfy");
            setDraft(defaults);
            router.replace("/generate");
            return;
          }
          setDraft({
            model: gen.model_key,
            provider: connected.includes(gen.provider as ProviderKey)
              ? (gen.provider as ProviderKey)
              : defaults.provider,
            prompt: gen.prompt,
            count: gen.image_count,
            settings: {
              ...gen.settings,
              seed: gen.settings.seed !== undefined ? String(gen.settings.seed) : "",
            },
            ...personaFromGeneration(gen),
          });
          router.replace("/generate");
        })
        .catch(() => setDraft(defaults));
      return;
    }
    let saved: Partial<Draft> | null = null;
    if (prefs.remember_prompt && draftKey) {
      try {
        saved = JSON.parse(localStorage.getItem(draftKey) ?? "null");
      } catch {
        saved = null;
      }
    }
    const model = catalog.data.models.some((m) => m.key === saved?.model) ? saved!.model! : defaults.model;
    const provider =
      saved?.provider && connected.includes(saved.provider) ? saved.provider : defaults.provider;
    // eslint-disable-next-line react-hooks/set-state-in-effect -- one-time initialisation
    setDraft({
      ...defaults,
      ...(saved ?? {}),
      model,
      provider,
      count: Math.min(4, Math.max(1, saved?.count ?? defaults.count)),
      settings: saved?.settings ?? {},
      refs: saved?.refs ?? [],
      persona: chosenPersona
        ? saved?.persona?.id === chosenPersona
          ? saved.persona
          : newPersonaDraft(chosenPersona)
        : (saved?.persona ?? null),
    });
    if (chosenPersona) router.replace("/generate");
  }, [draft, prefs, integrations.data, catalog.data, connected, draftKey, router, setEngine]);

  // Remember the draft (prompt persistence).
  useEffect(() => {
    if (!draft || !draftKey || !prefs?.remember_prompt) return;
    try {
      localStorage.setItem(draftKey, JSON.stringify(draft));
    } catch {
      /* storage full or blocked */
    }
  }, [draft, draftKey, prefs?.remember_prompt]);

  const model = catalog.data?.models.find((m) => m.key === draft?.model);
  const provider = draft?.provider ?? null;
  const editVariant = pickVariant(model, provider, true);
  // The selected AI model counts only while it still exists.
  const persona = draft?.persona ? personas.data?.items.find((p) => p.id === draft.persona?.id) : undefined;
  const personaDraft = persona ? (draft?.persona ?? null) : null;
  const { attached: personaRefs } = attachedPersonaImages(
    persona,
    personaDraft,
    editVariant ? editVariant.max_reference_images : null,
    draft?.refs.length ?? 0,
  );
  const referenceTotal = (draft?.refs.length ?? 0) + personaRefs.length;
  const withRefs = referenceTotal > 0;
  const variant = pickVariant(model, provider, withRefs);
  const maxPrompt = catalog.data?.limits.max_prompt_chars ?? 4000;
  const maxImages = catalog.data?.limits.max_images ?? 4;

  const fitted = variant && draft ? fitSettings(variant, draft.settings, prefs) : undefined;

  // What the assistant sees when the user asks about the current setup.
  useEffect(() => {
    if (!draft) return;
    publishStudioContext({
      page: "generate",
      model: draft.model,
      provider: draft.provider,
      settings: fitted ? toPayload(fitted) : {},
      prompt: draft.prompt,
      reference_count: referenceTotal,
      count: draft.count,
      ...(persona && personaDraft
        ? {
            ai_model: persona.name,
            ai_model_text: personaDraft.useDescription ? persona.prompt_text : "",
            ai_model_references: personaRefs.length,
          }
        : {}),
    });
  });
  useEffect(() => () => publishStudioContext(null), []);

  const update = useCallback((patch: Partial<Draft>) => {
    setDraft((d) => (d ? { ...d, ...patch } : d));
  }, []);

  // Prompts and settings applied from the assistant.
  const applyPatch = useCallback(
    (patch: StudioPatch) => {
      setDraft((d) => {
        if (!d) return d;
        const next: Draft = { ...d, settings: { ...d.settings } };
        if (typeof patch.prompt === "string") next.prompt = patch.prompt;
        if (patch.model && catalog.data?.models.some((m) => m.key === patch.model)) {
          next.model = patch.model as ModelKey;
        }
        if (typeof patch.count === "number") next.count = Math.min(4, Math.max(1, Math.round(patch.count)));
        if (typeof patch.aspect_ratio === "string") next.settings.aspect_ratio = patch.aspect_ratio;
        if (typeof patch.resolution === "string") next.settings.resolution = patch.resolution;
        if (patch.seed !== undefined && patch.seed !== null) next.settings.seed = String(patch.seed);
        if (typeof patch.thinking_mode === "boolean") next.settings.thinking_mode = patch.thinking_mode;
        return next;
      });
    },
    [catalog.data],
  );
  useEffect(() => registerApplyHandler(applyPatch), [applyPatch]);
  useEffect(() => {
    if (!draft) return;
    const pending = takePendingApply();
    // Applies a prompt/settings the user chose in the assistant before this page was open.
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (pending) applyPatch(pending);
  }, [Boolean(draft)]); // eslint-disable-line react-hooks/exhaustive-deps

  const problems: string[] = [];
  if (draft) {
    if (!connected.length) problems.push(t("gen.problemNoProvider"));
    else if (!provider) problems.push(t("gen.problemPickProvider"));
    else if (model && !model.providers.includes(provider))
      problems.push(t("gen.problemModelProvider", { model: model.name, provider: provider === "wavespeed" ? "WaveSpeed" : "Atlas Cloud" }));
    else if (!variant)
      problems.push(
        withRefs ? t("gen.problemNoEdit", { model: model?.name ?? "" }) : t("gen.problemNoText"),
      );
    if (variant && draft.refs.length > variant.max_reference_images)
      problems.push(t("refs.limit", { n: variant.max_reference_images }));
    if (draft.prompt.length > maxPrompt) problems.push(t("gen.problemTooLong", { n: maxPrompt }));
    if (fitted?.seed && !/^-?\d+$/.test(fitted.seed)) problems.push(t("gen.problemSeed"));
  }
  const promptMissing = !draft?.prompt.trim();

  const submit = async () => {
    if (!draft || submitting.current || create.isPending) return;
    setTouched(true);
    if (!connected.length) {
      toast.error(t("gen.problemNoProvider"), {
        action: { label: t("nav.integrations"), onClick: () => router.push("/integrations") },
      });
      return;
    }
    if (promptMissing) {
      toast.error(t("gen.problemPrompt"));
      return;
    }
    if (problems.length || !variant || !fitted) {
      toast.error(problems[0] ?? t("errors.generic"));
      return;
    }
    submitting.current = true;
    try {
      await create.mutateAsync({
        model: draft.model,
        provider,
        prompt: draft.prompt.trim(),
        count: draft.count,
        settings: toPayload(fitted),
        reference_image_ids: draft.refs.map((r) => r.id),
        client_request_id: newRequestId(),
        persona:
          persona && personaDraft
            ? {
                id: persona.id,
                reference_image_ids: personaRefs.map((r) => r.id),
                use_references: personaDraft.useRefs && personaRefs.length > 0,
                use_description: personaDraft.useDescription,
              }
            : null,
      });
      if (prefs?.clear_prompt_after_generate) update({ prompt: "" });
      setTouched(false);
    } catch (e) {
      const err = e as ApiError;
      toast.error(err.message, {
        action:
          err.code === "not_connected"
            ? { label: t("nav.integrations"), onClick: () => router.push("/integrations") }
            : undefined,
      });
    } finally {
      submitting.current = false;
    }
  };

  const loadFrom = (gen: Generation) => {
    if (gen.provider === "comfyui") {
      setComfyLoad(gen);
      setEngine("comfy");
      toast.success(t("gen.loaded"));
      return;
    }
    setEngine("hosted");
    update({
      model: gen.model_key,
      provider: connected.includes(gen.provider as ProviderKey) ? (gen.provider as ProviderKey) : provider,
      prompt: gen.prompt,
      count: gen.image_count,
      settings: { ...gen.settings, seed: gen.settings.seed !== undefined ? String(gen.settings.seed) : "" },
      ...personaFromGeneration(gen),
    });
    toast.success(t("gen.loaded"));
  };

  const items = results.data?.pages.flatMap((p) => p.items) ?? [];
  const activeCount = items.filter((g) => g.status === "queued" || g.status === "processing").length;

  if (!draft || !catalog.data || !prefs) {
    return (
      <div className="studio">
        <div className="studio-controls" style={{ padding: 22, gap: 16 }}>
          <Skeleton style={{ height: 200 }} />
          <Skeleton style={{ height: 160, marginTop: 16 }} />
          <Skeleton style={{ height: 220, marginTop: 16 }} />
        </div>
        <div className="studio-results">
          <Skeleton style={{ height: 320 }} />
        </div>
      </div>
    );
  }

  const resultsSection = (
      <section className="studio-results" aria-label={t("gen.results")}>
      <div className="row" style={{ marginBottom: 18 }}>
        <div className="grow">
          <h2 className="h2">{t("gen.results")}</h2>
          <div className="small subtle">
            {activeCount ? t("gen.activeNow", { n: activeCount }) : t("gen.resultsHint")}
          </div>
        </div>
        <Link href="/library/history" className="btn btn-ghost btn-sm">
          {t("gen.fullHistory")} <ArrowRight />
        </Link>
      </div>
      {results.isLoading ? (
        <div className="col gap-16">
          <Skeleton style={{ height: 260 }} />
          <Skeleton style={{ height: 260 }} />
        </div>
      ) : results.isError ? (
        <Notice tone="danger">{(results.error as Error).message}</Notice>
      ) : items.length === 0 ? (
        <EmptyState
          icon={<Wand2 />}
          title={t("gen.emptyTitle")}
          description={t("gen.emptyBody")}
          action={
            <Link href="/library" className="btn btn-sm">
              <Images /> {t("nav.library")}
            </Link>
          }
        />
      ) : (
        <>
          {items.map((gen) => (
            <JobCard
              key={gen.id}
              gen={gen}
              showPrompts={prefs.show_prompts_on_cards}
              blurNsfw={prefs.blur_nsfw}
              onOpen={(g, index) => setViewer({ images: g.images, index })}
              onLoad={loadFrom}
            />
          ))}
          {results.hasNextPage ? (
            <div className="row" style={{ justifyContent: "center" }}>
              <Button onClick={() => results.fetchNextPage()} loading={results.isFetchingNextPage}>
                {t("common.loadMore")}
              </Button>
            </div>
          ) : null}
        </>
      )}
      {results.isFetching && !results.isLoading ? (
        <div className="row tiny subtle" style={{ justifyContent: "center", marginTop: 12 }}>
          <Spinner size={12} />
        </div>
      ) : null}
    </section>
  );

  const engineSwitch = (
    <Segmented
      label={t("k2.engine")}
      value={engine}
      block
      onChange={setEngine}
      options={[
        { value: "hosted", label: t("k2.engineHosted") },
        { value: "comfy", label: t("k2.engineComfy") },
      ]}
    />
  );
  const viewerNode = viewer ? (
    <ViewerFromGeneration
      viewer={viewer}
      items={items}
      onClose={() => setViewer(null)}
      onIndex={(index: number) => setViewer((v) => (v ? { ...v, index } : v))}
    />
  ) : null;

  if (engine === "comfy") {
    return (
      <div className="studio">
        <ComfyPanel
          key={comfyLoad?.id ?? "draft"}
          storageKey={userId ? `studio:comfy-draft:${userId}` : null}
          initial={comfyLoad}
          loraFromUrl={loraFromUrl}
          header={
            <div className="col gap-12">
              <div className="row">
                <h1 className="h2 grow">{t("gen.title")}</h1>
              </div>
              {engineSwitch}
            </div>
          }
        />
        {resultsSection}
        {viewerNode}
      </div>
    );
  }

  return (
    <div className="studio">
      <section className="studio-controls" aria-label={t("gen.controls")}>
        <div className="studio-scroll">
          <div className="row">
            <h1 className="h2 grow">{t("gen.title")}</h1>
            <IconButton label={t("chat.askAboutSetup")} onClick={openAssistant}>
              <MessagesSquare />
            </IconButton>
            {connected.length > 1 ? (
              <Segmented
                label={t("gen.provider")}
                value={provider}
                onChange={(p) => update({ provider: p })}
                options={connected.map((p) => ({
                  value: p,
                  label: (
                    <>
                      <ProviderLogo provider={p} small />
                      {p === "wavespeed" ? "WaveSpeed" : "Atlas"}
                    </>
                  ),
                }))}
              />
            ) : provider ? (
              <span className="row small subtle" style={{ gap: 8 }}>
                <ProviderLogo provider={provider} small />
                {provider === "wavespeed" ? "WaveSpeed" : "Atlas Cloud"}
              </span>
            ) : null}
          </div>

          {engineSwitch}

          {!connected.length ? (
            <Notice tone="accent" icon={<Plug />}>
              <strong>{t("gen.connectTitle")}</strong>
              <div style={{ margin: "4px 0 10px" }}>{t("gen.connectBody")}</div>
              <Link href="/integrations" className="btn btn-sm btn-primary">
                {t("gen.openIntegrations")} <ArrowRight />
              </Link>
            </Notice>
          ) : null}

          <div>
            <div className="section-title">
              <span className="eyebrow grow">{t("gen.model")}</span>
            </div>
            <ModelPicker
              models={catalog.data.models}
              value={draft.model}
              connected={connected}
              onChange={(m) => update({ model: m })}
            />
          </div>

          <div>
            <div className="section-title">
              <span className="eyebrow grow">{t("pm.section")}</span>
              <Link href="/ai-models" className="tiny subtle">
                {t("pm.manage")}
              </Link>
            </div>
            <PersonaSection
              value={personaDraft ?? draft.persona}
              persona={persona}
              loading={personas.isLoading}
              hasPersonas={(personas.data?.total ?? 0) > 0}
              onChange={(value) => update({ persona: value })}
              attached={personaRefs}
              maxReferences={editVariant ? editVariant.max_reference_images : null}
              modelName={model?.name ?? ""}
              providerName={provider === "atlas" ? "Atlas Cloud" : provider === "wavespeed" ? "WaveSpeed" : ""}
              manualCount={draft.refs.length}
            />
          </div>

          <div>
            <div className="section-title">
              <span className="eyebrow grow">{t("gen.prompt")}</span>
            </div>
            <PromptEditor
              value={draft.prompt}
              onChange={(p) => update({ prompt: p })}
              max={maxPrompt}
              onSubmit={submit}
            />
            {touched && promptMissing ? <div className="error-text" style={{ marginTop: 6 }}>{t("gen.problemPrompt")}</div> : null}
          </div>

          {editVariant ? (
            <div>
              <div className="section-title">
                <span className="eyebrow grow">{t("refs.title")}</span>
                <span className="tiny subtle">
                  {personaRefs.length
                    ? t("pm.refCounter", {
                        n: referenceTotal,
                        max: editVariant.max_reference_images,
                        own: personaRefs.length,
                        name: persona?.name ?? "",
                      })
                    : `${draft.refs.length}/${editVariant.max_reference_images}`}
                </span>
              </div>
              <ReferenceImages
                refs={draft.refs}
                max={editVariant.max_reference_images}
                onChange={(refs) => update({ refs })}
              />
            </div>
          ) : provider && model ? (
            <div className="hint">{t("refs.unsupported", { model: model.name })}</div>
          ) : null}

          {variant && fitted ? (
            <div>
              <div className="section-title">
                <span className="eyebrow grow">{t("gen.settings")}</span>
                <span className="tiny subtle">{variant.mode === "image-edit" ? t("gen.modeEdit") : t("gen.modeText")}</span>
              </div>
              <SettingsControls
                variant={variant}
                value={fitted}
                onChange={(patch) => update({ settings: { ...fitted, ...patch } })}
                count={draft.count}
                onCount={(n) => update({ count: n })}
                maxImages={Math.min(maxImages, variant.max_images)}
              />
            </div>
          ) : null}

          {problems.length && connected.length ? (
            <Notice tone="warning">
              {problems.map((p) => (
                <div key={p}>{p}</div>
              ))}
            </Notice>
          ) : null}
        </div>
        <div className="studio-foot">
          <Button
            variant="primary"
            size="lg"
            block
            loading={create.isPending}
            disabled={create.isPending || (connected.length > 0 && problems.length > 0)}
            onClick={submit}
            icon={<Sparkles />}
          >
            {create.isPending
              ? t("gen.submitting")
              : draft.count > 1
                ? t("gen.generateN", { n: draft.count })
                : t("gen.generate")}
            <span className="kbd" aria-hidden>
              ⌘↵
            </span>
          </Button>
        </div>
      </section>

      {resultsSection}

      {viewerNode}
    </div>
  );
}

/** Keeps the viewer in sync with live updates of the generation (favorites, deletes). */
function ViewerFromGeneration({
  viewer,
  items,
  onClose,
  onIndex,
}: {
  viewer: { images: StudioImage[]; index: number };
  items: Generation[];
  onClose: () => void;
  onIndex: (i: number) => void;
}) {
  const genId = viewer.images[0]?.generation_id;
  const live = items.find((g) => g.id === genId)?.images ?? viewer.images;
  return <ImageViewer images={live} index={viewer.index} onClose={onClose} onIndexChange={onIndex} />;
}
