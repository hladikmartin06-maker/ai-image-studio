"use client";

import {
  type InfiniteData,
  type QueryClient,
  useInfiniteQuery,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";
import { useCallback } from "react";

import { api, query } from "@/lib/api";
import { hiddenCount, hideDeleted, isHidden, showAgain, useHiddenVersion } from "@/lib/pending-deletes";
import type {
  AIProviderList,
  AIUsage,
  AuthState,
  Balance,
  Catalog,
  Conversation,
  ConversationDetail,
  EnhanceBody,
  Generation,
  GPUProvider,
  GpuEndpoint,
  GpuType,
  GpuUsage,
  GpuWorkload,
  IntegrationList,
  PriceList,
  ComfyCatalog,
  ComfyConnection,
  ComfyWorkflow,
  JobItem,
  LoraCatalog,
  LoraDetail,
  LoraItem,
  LoraSummary,
  LoraTestGroup,
  LoraVersion,
  Page,
  Persona,
  PersonaList,
  Preferences,
  ReferenceImage,
  SecretInfo,
  HfStatus,
  SessionInfo,
  TrainingConfig,
  Stats,
  StorageInfo,
  StudioImage,
  User,
} from "@/lib/types";

export const keys = {
  auth: ["auth"] as const,
  catalog: ["catalog"] as const,
  settings: ["settings"] as const,
  integrations: ["integrations"] as const,
  balance: (p: string) => ["balance", p] as const,
  prices: ["prices"] as const,
  stats: ["stats"] as const,
  images: ["images"] as const,
  imageList: (filters: ImageFilters) => ["images", "list", filters] as const,
  image: (id: string) => ["images", "one", id] as const,
  generations: ["generations"] as const,
  generationList: (filters: GenerationFilters) => ["generations", "list", filters] as const,
  generation: (id: string) => ["generations", "one", id] as const,
  prompts: ["prompts"] as const,
  references: ["references"] as const,
  storage: ["storage"] as const,
  sessions: ["sessions"] as const,
};

// --- auth ------------------------------------------------------------------------------

export function useAuthState() {
  return useQuery({
    queryKey: keys.auth,
    queryFn: () => api<AuthState>("/api/auth/state"),
    staleTime: 60_000,
  });
}

export function useSessions() {
  return useQuery({
    queryKey: keys.sessions,
    queryFn: () => api<{ sessions: SessionInfo[] }>("/api/account/sessions"),
  });
}

export function useUpdateProfile() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { name?: string; email?: string }) =>
      api<{ user: User }>("/api/account", { method: "PATCH", body }),
    onSuccess: () => qc.invalidateQueries({ queryKey: keys.auth }),
  });
}

// --- catalog & settings ------------------------------------------------------------------

export function useCatalog() {
  return useQuery({
    queryKey: keys.catalog,
    queryFn: () => api<Catalog>("/api/catalog"),
    staleTime: Infinity,
  });
}

export function useSettings(enabled = true) {
  return useQuery({
    queryKey: keys.settings,
    queryFn: () => api<Preferences>("/api/settings"),
    staleTime: 5 * 60_000,
    enabled,
  });
}

export function useUpdateSettings() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (patch: Partial<Preferences> | { default_resolution: "auto" }) =>
      api<Preferences>("/api/settings", { method: "PATCH", body: patch }),
    onMutate: async (patch) => {
      await qc.cancelQueries({ queryKey: keys.settings });
      const previous = qc.getQueryData<Preferences>(keys.settings);
      if (previous) {
        const next = { ...previous, ...patch } as Preferences;
        if ((patch as { default_resolution?: string }).default_resolution === "auto") {
          next.default_resolution = null;
        }
        qc.setQueryData(keys.settings, next);
      }
      return { previous };
    },
    onError: (_e, _p, ctx) => {
      if (ctx?.previous) qc.setQueryData(keys.settings, ctx.previous);
    },
    onSuccess: (data) => {
      qc.setQueryData(keys.settings, data);
      qc.invalidateQueries({ queryKey: keys.integrations });
    },
  });
}

// --- integrations ------------------------------------------------------------------------

export function useIntegrations() {
  return useQuery({
    queryKey: keys.integrations,
    queryFn: () => api<IntegrationList>("/api/integrations"),
    staleTime: 30_000,
  });
}

export function useBalance(provider: string, enabled: boolean) {
  return useQuery({
    queryKey: keys.balance(provider),
    queryFn: () => api<Balance>(`/api/integrations/${provider}/balance`),
    enabled,
    staleTime: 60_000,
    retry: false,
  });
}

/** List prices of the hosted models (from the providers' price APIs, where they exist). */
export function usePrices(enabled: boolean) {
  return useQuery({
    queryKey: keys.prices,
    queryFn: () => api<PriceList>("/api/prices"),
    enabled,
    staleTime: 10 * 60_000,
    retry: false,
  });
}

// --- stats -----------------------------------------------------------------------------

export function useStats() {
  const tz = typeof Intl !== "undefined" ? Intl.DateTimeFormat().resolvedOptions().timeZone : "UTC";
  return useQuery({
    queryKey: keys.stats,
    queryFn: () => api<Stats>(`/api/stats${query({ tz })}`),
    staleTime: 15_000,
  });
}

export function useStorage() {
  return useQuery({
    queryKey: keys.storage,
    queryFn: () => api<StorageInfo>("/api/storage"),
  });
}

// --- images ----------------------------------------------------------------------------

export interface ImageFilters {
  q?: string;
  // Only images or only videos (Wan 3.0).
  media?: "image" | "video";
  model?: string;
  provider?: string;
  favorite?: boolean;
  status?: "complete" | "partial";
  date_from?: string;
  date_to?: string;
  scope?: "library" | "all" | "unsaved";
  sort?: "newest" | "oldest" | "favorited";
  generation_id?: string;
  persona?: string;
  lora_version?: string;
  limit?: number;
}

/** Leaves out what the user just deleted (a refetch may still return it for a moment). */
function withoutDeleted<T extends { id: string }>(
  data: InfiniteData<Page<T>>,
  inner?: (item: T) => T,
): InfiniteData<Page<T>> {
  if (!hiddenCount()) return data;
  return {
    ...data,
    pages: data.pages.map((page) => {
      const items = page.items.filter((item) => !isHidden(item.id)).map((item) => (inner ? inner(item) : item));
      return { ...page, items, total: page.total - (page.items.length - items.length) };
    }),
  };
}

function withoutDeletedImages(gen: Generation): Generation {
  return gen.images.some((img) => isHidden(img.id))
    ? { ...gen, images: gen.images.filter((img) => !isHidden(img.id)) }
    : gen;
}

export function useImages(filters: ImageFilters, enabled = true) {
  const deleted = useHiddenVersion();
  const select = useCallback(
    (data: InfiniteData<Page<StudioImage>>) => withoutDeleted(data),
    // The version is what changes when an item is hidden or shown again.
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [deleted],
  );
  return useInfiniteQuery({
    queryKey: keys.imageList(filters),
    queryFn: ({ pageParam, signal }) =>
      api<Page<StudioImage>>(
        `/api/images${query({ ...filters, cursor: pageParam as string | undefined })}`,
        { signal },
      ),
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (last) => last.next_cursor ?? undefined,
    enabled,
    staleTime: 10_000,
    select,
  });
}

/** Applies a change to one image in every cached list and detail. */
export function patchImageEverywhere(
  qc: QueryClient,
  id: string,
  change: (img: StudioImage) => StudioImage | null,
) {
  qc.setQueriesData<InfiniteData<Page<StudioImage>>>({ queryKey: ["images", "list"] }, (data) => {
    if (!data) return data;
    return {
      ...data,
      pages: data.pages.map((page) => {
        const items = page.items
          .map((img) => (img.id === id ? change(img) : img))
          .filter((img): img is StudioImage => img !== null);
        return { ...page, items, total: page.total - (page.items.length - items.length) };
      }),
    };
  });
  qc.setQueriesData<InfiniteData<Page<Generation>>>(
    { queryKey: ["generations", "list"] },
    (data) => {
      if (!data) return data;
      return {
        ...data,
        pages: data.pages.map((page) => ({
          ...page,
          items: page.items.map((gen) => ({
            ...gen,
            images: gen.images
              .map((img) => (img.id === id ? change(img) : img))
              .filter((img): img is StudioImage => img !== null),
          })),
        })),
      };
    },
  );
}

export function useSaveToLibrary() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) =>
      api<StudioImage>(`/api/images/${id}`, { method: "PATCH", body: { in_library: true } }),
    onSuccess: (img) => {
      patchImageEverywhere(qc, img.id, () => img);
      qc.invalidateQueries({ queryKey: keys.images });
    },
  });
}

export function useDeleteImage() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api(`/api/images/${id}`, { method: "DELETE" }),
    onMutate: (id) => hideDeleted(id),
    onError: (_e, id) => showAgain(id),
    onSuccess: (_d, id) => {
      patchImageEverywhere(qc, id, () => null);
      qc.invalidateQueries({ queryKey: keys.images });
      qc.invalidateQueries({ queryKey: keys.generations });
      qc.invalidateQueries({ queryKey: keys.stats });
      qc.invalidateQueries({ queryKey: keys.storage });
    },
  });
}

export function useBulkImages() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: { ids: string[]; action: "favorite" | "unfavorite" | "delete" | "save" }) =>
      api<{ changed: number }>("/api/images/bulk", { body }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: keys.images });
      qc.invalidateQueries({ queryKey: keys.generations });
      qc.invalidateQueries({ queryKey: keys.stats });
      qc.invalidateQueries({ queryKey: keys.storage });
    },
  });
}

// --- generations -------------------------------------------------------------------------

export interface GenerationFilters {
  q?: string;
  status?: string;
  model?: string;
  provider?: string;
  sort?: "newest" | "oldest";
  active?: boolean;
  persona?: string;
  limit?: number;
}

export function useGenerations(filters: GenerationFilters, enabled = true) {
  const deleted = useHiddenVersion();
  const select = useCallback(
    (data: InfiniteData<Page<Generation>>) => withoutDeleted(data, withoutDeletedImages),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [deleted],
  );
  return useInfiniteQuery({
    queryKey: keys.generationList(filters),
    queryFn: ({ pageParam, signal }) =>
      api<Page<Generation>>(
        `/api/generations${query({ ...filters, cursor: pageParam as string | undefined })}`,
        { signal },
      ),
    initialPageParam: undefined as string | undefined,
    getNextPageParam: (last) => last.next_cursor ?? undefined,
    enabled,
    staleTime: 10_000,
    select,
  });
}

export function useGeneration(id: string | null) {
  return useQuery({
    queryKey: keys.generation(id ?? "none"),
    queryFn: () => api<Generation>(`/api/generations/${id}`),
    enabled: Boolean(id),
  });
}

export interface CreateGenerationBody {
  model: string;
  provider?: string | null;
  prompt: string;
  count: number;
  settings: Record<string, unknown>;
  reference_image_ids: string[];
  client_request_id: string;
  // "18+ content" on Wan 2.7 / Wan 2.7 Pro.
  adult?: boolean;
  persona?: {
    id: string;
    reference_image_ids: string[];
    use_references: boolean;
    use_description: boolean;
  } | null;
}

/** Puts a generation (new or updated) into every cached generation list. */
/** Takes a generation out of the cached lists at once (deleted, or about to be). */
export function removeGenerationEverywhere(qc: QueryClient, id: string) {
  qc.setQueriesData<InfiniteData<Page<Generation>>>({ queryKey: ["generations", "list"] }, (data) => {
    if (!data) return data;
    return {
      ...data,
      pages: data.pages.map((page) => {
        const items = page.items.filter((gen) => gen.id !== id);
        return { ...page, items, total: page.total - (page.items.length - items.length) };
      }),
    };
  });
}

export function upsertGeneration(qc: QueryClient, gen: Generation) {
  qc.setQueryData(keys.generation(gen.id), gen);
  qc.setQueriesData<InfiniteData<Page<Generation>>>(
    { queryKey: ["generations", "list"] },
    (data) => {
      if (!data) return data;
      let found = false;
      const pages = data.pages.map((page) => ({
        ...page,
        items: page.items.map((item) => {
          if (item.id !== gen.id) return item;
          found = true;
          return gen;
        }),
      }));
      return { ...data, pages: found ? pages : data.pages };
    },
  );
}

export function useCreateGeneration() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: CreateGenerationBody) =>
      api<Generation>("/api/generations", { body: body as unknown as Record<string, unknown> }),
    onSuccess: (gen) => {
      upsertGeneration(qc, gen);
      qc.invalidateQueries({ queryKey: keys.generations });
      qc.invalidateQueries({ queryKey: keys.stats });
      qc.invalidateQueries({ queryKey: keys.prompts });
      if (gen.persona) qc.invalidateQueries({ queryKey: personaKeys.all });
    },
  });
}

export function useRegenerate() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, count }: { id: string; count?: number }) =>
      api<Generation>(`/api/generations/${id}/regenerate`, {
        body: { count, client_request_id: crypto.randomUUID() },
      }),
    onSuccess: (gen) => {
      upsertGeneration(qc, gen);
      qc.invalidateQueries({ queryKey: keys.generations });
      qc.invalidateQueries({ queryKey: keys.stats });
    },
  });
}

export function useCancelGeneration() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api<Generation>(`/api/generations/${id}/cancel`, { body: {} }),
    onSuccess: (gen) => upsertGeneration(qc, gen),
    onSettled: () => qc.invalidateQueries({ queryKey: keys.generations }),
  });
}

export function useDeleteGeneration() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api(`/api/generations/${id}`, { method: "DELETE" }),
    onMutate: (id) => hideDeleted(id),
    onError: (_e, id) => showAgain(id),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: keys.generations });
      qc.invalidateQueries({ queryKey: keys.images });
      qc.invalidateQueries({ queryKey: keys.stats });
      qc.invalidateQueries({ queryKey: keys.storage });
    },
  });
}

export function usePromptHistory(enabled: boolean) {
  return useQuery({
    queryKey: keys.prompts,
    queryFn: () =>
      api<{ items: { prompt: string; last_used: string }[] }>("/api/prompts/history?limit=40"),
    enabled,
    staleTime: 30_000,
  });
}

// --- references --------------------------------------------------------------------------

export function useUploadReference() {
  return useMutation({
    mutationFn: (file: File) => {
      const form = new FormData();
      form.append("file", file);
      return api<ReferenceImage>("/api/reference-images", { body: form });
    },
  });
}

// --- AI models (saved characters) -------------------------------------------------------------

export const personaKeys = {
  all: ["personas"] as const,
  list: (q: string) => ["personas", "list", q] as const,
  one: (id: string) => ["personas", "one", id] as const,
};

export function usePersonas(q = "", enabled = true) {
  return useQuery({
    queryKey: personaKeys.list(q),
    queryFn: ({ signal }) => api<PersonaList>(`/api/personas${query({ q: q || undefined })}`, { signal }),
    enabled,
    staleTime: 15_000,
    placeholderData: (prev) => prev,
  });
}

export function usePersona(id: string | null) {
  return useQuery({
    queryKey: personaKeys.one(id ?? ""),
    queryFn: () => api<Persona>(`/api/personas/${id}`),
    enabled: Boolean(id),
  });
}

/** Writes a changed AI model into its detail query and every cached list. */
function storePersona(qc: QueryClient, persona: Persona) {
  qc.setQueryData(personaKeys.one(persona.id), persona);
  qc.setQueriesData<PersonaList>({ queryKey: ["personas", "list"] }, (data) =>
    data ? { ...data, items: data.items.map((p) => (p.id === persona.id ? persona : p)) } : data,
  );
  qc.invalidateQueries({ queryKey: ["personas", "list"] });
}

export interface PersonaBody {
  name?: string;
  description?: string;
  notes?: string;
  appearance?: Partial<Persona["appearance"]>;
  adult_confirmed?: boolean;
  cover_reference_id?: string | null;
  image_order?: string[];
}

export function useCreatePersona() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: PersonaBody) =>
      api<Persona>("/api/personas", { body: body as Record<string, unknown> }),
    onSuccess: (persona) => {
      qc.setQueryData(personaKeys.one(persona.id), persona);
      qc.invalidateQueries({ queryKey: personaKeys.all });
    },
  });
}

export function useUpdatePersona(id: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: PersonaBody) =>
      api<Persona>(`/api/personas/${id}`, { method: "PATCH", body: body as Record<string, unknown> }),
    onSuccess: (persona) => storePersona(qc, persona),
  });
}

export function useDuplicatePersona() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api<Persona>(`/api/personas/${id}/duplicate`, { body: {} }),
    onSuccess: () => qc.invalidateQueries({ queryKey: personaKeys.all }),
  });
}

export function useDeletePersona() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api(`/api/personas/${id}`, { method: "DELETE" }),
    onSuccess: (_, id) => {
      qc.removeQueries({ queryKey: personaKeys.one(id) });
      qc.invalidateQueries({ queryKey: personaKeys.all });
      qc.invalidateQueries({ queryKey: keys.storage });
    },
  });
}

export function useUploadPersonaImage(id: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (file: File) => {
      const form = new FormData();
      form.append("file", file);
      return api<{ persona: Persona; added: number }>(`/api/personas/${id}/images`, { body: form });
    },
    onSuccess: (res) => {
      storePersona(qc, res.persona);
      qc.invalidateQueries({ queryKey: keys.storage });
    },
  });
}

export function useAddPersonaImages() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({
      id,
      libraryImageIds = [],
      referenceImageIds = [],
    }: {
      id: string;
      libraryImageIds?: string[];
      referenceImageIds?: string[];
    }) =>
      api<{ persona: Persona; added: number }>(`/api/personas/${id}/images/add`, {
        body: { library_image_ids: libraryImageIds, reference_image_ids: referenceImageIds },
      }),
    onSuccess: (res) => {
      storePersona(qc, res.persona);
      qc.invalidateQueries({ queryKey: keys.storage });
    },
  });
}

export function useRemovePersonaImage(id: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (referenceId: string) =>
      api<Persona>(`/api/personas/${id}/images/${referenceId}`, { method: "DELETE" }),
    onSuccess: (persona) => {
      storePersona(qc, persona);
      qc.invalidateQueries({ queryKey: keys.storage });
    },
  });
}

// --- AI assistant ---------------------------------------------------------------------------

export const aiKeys = {
  providers: ["ai", "providers"] as const,
  conversations: (q: string) => ["ai", "conversations", q] as const,
  conversation: (id: string) => ["ai", "conversation", id] as const,
  usage: ["ai", "usage"] as const,
};

export function useAIProviders() {
  return useQuery({
    queryKey: aiKeys.providers,
    queryFn: () => api<AIProviderList>("/api/ai/providers"),
    staleTime: 30_000,
  });
}

export function useConversations(q: string) {
  return useQuery({
    queryKey: aiKeys.conversations(q),
    queryFn: () => api<{ items: Conversation[] }>(`/api/ai/conversations${query({ q })}`),
    staleTime: 10_000,
  });
}

export function useConversation(id: string | null) {
  return useQuery({
    queryKey: aiKeys.conversation(id ?? "none"),
    queryFn: () => api<ConversationDetail>(`/api/ai/conversations/${id}`),
    enabled: Boolean(id),
  });
}

export function useAIUsage() {
  return useQuery({
    queryKey: aiKeys.usage,
    queryFn: () => api<AIUsage>("/api/ai/usage?days=30"),
    staleTime: 30_000,
  });
}

// --- GPU compute ----------------------------------------------------------------------------

export const gpuKeys = {
  providers: ["gpu", "providers"] as const,
  types: (p: string) => ["gpu", p, "types"] as const,
  workloads: (p: string) => ["gpu", p, "workloads"] as const,
  endpoints: (p: string) => ["gpu", p, "endpoints"] as const,
  usage: (p: string) => ["gpu", p, "usage"] as const,
};

export function useGPUProviders() {
  return useQuery({
    queryKey: gpuKeys.providers,
    queryFn: () => api<{ providers: GPUProvider[] }>("/api/gpu/providers"),
    staleTime: 30_000,
  });
}

/** GPU types with prices; for RunPod also what is free right now for this purpose, so the
 * list is refreshed often. */
export function useGpuTypes(provider: string, enabled: boolean, purpose?: "comfyui" | "training") {
  return useQuery({
    queryKey: [...gpuKeys.types(provider), purpose ?? "any"],
    queryFn: () => api<{ items: GpuType[] }>(`/api/gpu/${provider}/gpu-types${purpose ? `?purpose=${purpose}` : ""}`),
    enabled,
    staleTime: 30_000,
    refetchInterval: enabled ? 60_000 : false,
  });
}

export function useGpuWorkloads(provider: string, enabled: boolean) {
  return useQuery({
    queryKey: gpuKeys.workloads(provider),
    queryFn: () =>
      api<{ items: GpuWorkload[]; supports_stop_start: boolean }>(`/api/gpu/${provider}/workloads`),
    enabled,
    // Controlled polling while something is starting; otherwise refresh on demand.
    refetchInterval: (q) =>
      q.state.data?.items.some((w) => w.status === "starting" || w.status === "created")
        ? 10_000
        : false,
  });
}

export function useGpuEndpoints(provider: string, enabled: boolean) {
  return useQuery({
    queryKey: gpuKeys.endpoints(provider),
    queryFn: () => api<{ items: GpuEndpoint[]; supported: boolean }>(`/api/gpu/${provider}/endpoints`),
    enabled,
  });
}

export function useGpuUsage(provider: string, enabled: boolean) {
  return useQuery({
    queryKey: gpuKeys.usage(provider),
    queryFn: () => api<GpuUsage>(`/api/gpu/${provider}/usage`),
    enabled,
    staleTime: 15_000,
  });
}

// --- ComfyUI --------------------------------------------------------------------------------

export const comfyKeys = {
  catalog: ["comfy", "catalog"] as const,
  connections: ["comfy", "connections"] as const,
  workflows: ["comfy", "workflows"] as const,
};

export function useComfyCatalog() {
  return useQuery({
    queryKey: comfyKeys.catalog,
    queryFn: () => api<ComfyCatalog>("/api/comfy/catalog"),
    staleTime: Infinity,
  });
}

export function useComfyConnections(enabled = true) {
  return useQuery({
    queryKey: comfyKeys.connections,
    queryFn: () => api<{ items: ComfyConnection[] }>("/api/comfy/connections"),
    enabled,
    // Deployments and model downloads advance in the background: follow them.
    refetchInterval: (q) =>
      q.state.data?.items.some((c) => c.status === "deploying" || c.deploy?.model_job?.status === "running")
        ? 4000
        : false,
  });
}

export function useComfyWorkflows() {
  return useQuery({
    queryKey: comfyKeys.workflows,
    queryFn: () => api<{ items: ComfyWorkflow[] }>("/api/comfy/workflows"),
  });
}

function useComfyMutation<TVars, TResult = unknown>(
  fn: (vars: TVars) => Promise<TResult>,
  key: readonly string[] = comfyKeys.connections,
) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: fn,
    onSuccess: () => qc.invalidateQueries({ queryKey: key }),
  });
}

export function useAddComfy() {
  return useComfyMutation((body: Record<string, unknown>) =>
    api<ComfyConnection>("/api/comfy/connections", { body }),
  );
}

export function useUpdateComfy() {
  return useComfyMutation(({ id, ...body }: { id: string } & Record<string, unknown>) =>
    api<ComfyConnection>(`/api/comfy/connections/${id}`, { method: "PATCH", body }),
  );
}

export function useTestComfy() {
  return useComfyMutation((id: string) =>
    api<ComfyConnection>(`/api/comfy/connections/${id}/test`, { body: {} }),
  );
}

export function useRemoveComfy() {
  return useComfyMutation(({ id, terminate }: { id: string; terminate: boolean }) =>
    api(`/api/comfy/connections/${id}?terminate=${terminate}`, { method: "DELETE" }),
  );
}

export function useDeployComfy() {
  return useComfyMutation((body: Record<string, unknown>) =>
    api<ComfyConnection>("/api/comfy/deployments", { body }),
  );
}

export function useComfyPower() {
  return useComfyMutation(({ id, action }: { id: string; action: "start" | "stop" }) =>
    api<ComfyConnection>(`/api/comfy/connections/${id}/${action}`, {
      body: action === "start" ? { confirm_billing: true } : {},
    }),
  );
}

export function useInstallKrea2Model() {
  return useComfyMutation(({ id, ...body }: { id: string; variant: "turbo" | "raw" | "pgupscaler"; filename?: string | null }) =>
    api<ComfyConnection>(`/api/comfy/connections/${id}/models/install`, { body }),
  );
}

export function useComfyModelDownload() {
  return useMutation({
    mutationFn: ({ id, ...body }: { id: string; url: string; folder: string; filename: string }) =>
      api<{ job: { id: string; status: string } }>(`/api/comfy/connections/${id}/models/download`, {
        body,
      }),
  });
}

export function useComfyModelUpload() {
  return useComfyMutation(({ id, upload_id, folder }: { id: string; upload_id: string; folder: string }) =>
    api<ComfyConnection>(`/api/comfy/connections/${id}/models/upload`, { body: { upload_id, folder } }),
  );
}

export function useImportWorkflow() {
  return useComfyMutation(
    (body: { name: string; family: string | null; data: Record<string, unknown> }) =>
      api<ComfyWorkflow & { format: string; removed_secrets: string[] }>("/api/comfy/workflows", {
        body,
      }),
    comfyKeys.workflows,
  );
}

export function useDeleteWorkflow() {
  return useComfyMutation(
    (id: string) => api(`/api/comfy/workflows/${id}`, { method: "DELETE" }),
    comfyKeys.workflows,
  );
}

export interface ComfyGenerateBody {
  connection_id: string;
  variant: "turbo" | "raw";
  model_file?: string | null;
  workflow_id?: string | null;
  prompt: string;
  negative_prompt?: string;
  aspect_ratio: string;
  resolution: string;
  seed?: number | null;
  steps?: number | null;
  cfg?: number | null;
  count: number;
  lora?: { version_id: string; strength: number; insert_trigger: boolean } | null;
  persona?: { id: string; use_description: boolean } | null;
  // Image to image (built-in workflow): the photo and how much of it is redrawn.
  image?: { kind: "reference" | "image"; id: string } | null;
  denoise?: number;
  // 18+ content (adults only): explicit prompts are refused when off.
  adult?: boolean;
  client_request_id: string;
}

/** Enhance photo (PGUpscaler) on a ComfyUI connection. */
export function useEnhancePhoto() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: EnhanceBody) =>
      api<Generation>("/api/comfy/enhance", { body: body as unknown as Record<string, unknown> }),
    onSuccess: (gen) => {
      upsertGeneration(qc, gen);
      qc.invalidateQueries({ queryKey: keys.generations });
      qc.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
}

export function useComfyGenerate() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: ComfyGenerateBody) =>
      api<Generation>("/api/comfy/generate", { body: body as unknown as Record<string, unknown> }),
    onSuccess: (gen) => {
      upsertGeneration(qc, gen);
      qc.invalidateQueries({ queryKey: keys.generations });
      qc.invalidateQueries({ queryKey: keys.prompts });
      qc.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
}

// --- LoRA studio ----------------------------------------------------------------------------

export const loraKeys = {
  all: ["loras"] as const,
  catalog: ["loras", "catalog"] as const,
  list: (q: string, usable: boolean) => ["loras", "list", q, usable] as const,
  one: (id: string) => ["loras", "one", id] as const,
  tests: (id: string) => ["loras", "tests", id] as const,
};

export function useLoraCatalog() {
  return useQuery({ queryKey: loraKeys.catalog, queryFn: () => api<LoraCatalog>("/api/loras/catalog") });
}

export function useLoras(q = "", usable = false, enabled = true) {
  return useQuery({
    queryKey: loraKeys.list(q, usable),
    queryFn: () =>
      api<{ items: LoraSummary[]; total: number }>(`/api/loras${query({ q: q || undefined, usable })}`),
    enabled,
    placeholderData: (prev) => prev,
    refetchInterval: (qy) => (qy.state.data?.items.some((l) => l.active) ? 5000 : false),
  });
}

export function useLora(id: string | null) {
  return useQuery({
    queryKey: loraKeys.one(id ?? ""),
    queryFn: () => api<LoraDetail>(`/api/loras/${id}`),
    enabled: Boolean(id),
    refetchInterval: (q) =>
      q.state.data?.versions.some((v) =>
        ["queued", "preparing", "uploading", "running", "downloading"].includes(v.status),
      )
        ? 4000
        : false,
  });
}

function useLoraMutation<TVars, TResult>(
  id: string,
  fn: (vars: TVars) => Promise<TResult>,
  pick?: (result: TResult) => LoraDetail | null,
) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: fn,
    onSuccess: (result) => {
      const detail = pick ? pick(result) : null;
      if (detail) qc.setQueryData(loraKeys.one(id), detail);
      qc.invalidateQueries({ queryKey: loraKeys.all });
    },
  });
}

export function useCreateLora() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: Record<string, unknown>) => api<LoraDetail>("/api/loras", { body }),
    onSuccess: () => qc.invalidateQueries({ queryKey: loraKeys.all }),
  });
}

export function useImportLora() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: Record<string, unknown>) => api<LoraDetail>("/api/loras/import", { body }),
    onSuccess: () => qc.invalidateQueries({ queryKey: loraKeys.all }),
  });
}

export function useUpdateLora(id: string) {
  return useLoraMutation(
    id,
    (body: Record<string, unknown>) => api<LoraDetail>(`/api/loras/${id}`, { method: "PATCH", body }),
    (r) => r,
  );
}

export function useDeleteLora() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api(`/api/loras/${id}`, { method: "DELETE" }),
    onSuccess: () => qc.invalidateQueries({ queryKey: loraKeys.all }),
  });
}

export function useUploadDatasetImage(id: string) {
  return useLoraMutation(
    id,
    (file: File) => {
      const form = new FormData();
      form.append("file", file);
      return api<{ added: number; lora: LoraDetail }>(`/api/loras/${id}/dataset/upload`, { body: form });
    },
    (r) => r.lora,
  );
}

export function useAddDatasetImages(id: string) {
  return useLoraMutation(
    id,
    (body: { library_image_ids?: string[]; reference_image_ids?: string[]; persona_id?: string }) =>
      api<{ added: number; lora: LoraDetail }>(`/api/loras/${id}/dataset/add`, { body }),
    (r) => r.lora,
  );
}

/** Copies library images into any LoRA's photos (from Generate, a viewer or the library). */
export function useAddToLoraDataset() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ loraId, imageIds }: { loraId: string; imageIds: string[] }) =>
      api<{ added: number; lora: LoraDetail }>(`/api/loras/${loraId}/dataset/add`, {
        body: { library_image_ids: imageIds },
      }),
    onSuccess: (r, { loraId }) => {
      qc.setQueryData(loraKeys.one(loraId), r.lora);
      qc.invalidateQueries({ queryKey: loraKeys.all });
    },
  });
}

export function useUpdateDatasetItem(id: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ itemId, ...body }: { itemId: string } & Record<string, unknown>) =>
      api<LoraItem>(`/api/loras/${id}/dataset/${itemId}`, { method: "PATCH", body }),
    onSuccess: (item) => {
      qc.setQueryData<LoraDetail>(loraKeys.one(id), (d) =>
        d ? { ...d, dataset: d.dataset.map((i) => (i.id === item.id ? item : i)) } : d,
      );
      qc.invalidateQueries({ queryKey: loraKeys.one(id) });
    },
  });
}

export function useBulkDataset(id: string) {
  return useLoraMutation(
    id,
    (body: { item_ids: string[]; action: string }) =>
      api<LoraDetail>(`/api/loras/${id}/dataset/bulk`, { body }),
    (r) => r,
  );
}

export function useCaptionItem(id: string) {
  return useMutation({
    mutationFn: ({ itemId, tier, overwrite }: { itemId: string; tier: string; overwrite: boolean }) =>
      api<LoraItem>(`/api/loras/${id}/dataset/${itemId}/caption`, { body: { tier, overwrite } }),
  });
}

export interface TrainBody {
  preset: string;
  config: TrainingConfig;
  provider: string;
  gpu_type: string;
  gpu_count: number;
  cloud_type: string;
  image?: string | null;
  auto_cleanup: boolean;
  notes?: string;
  client_request_id: string;
  confirm_billing: boolean;
}

export function useStartTraining(id: string) {
  return useLoraMutation(id, (body: TrainBody) =>
    api<LoraVersion>(`/api/loras/${id}/versions`, { body: body as unknown as Record<string, unknown> }),
  );
}

export function useVersionAction(id: string) {
  return useLoraMutation(
    id,
    ({ versionId, action, body }: { versionId: string; action: string; body?: Record<string, unknown> }) => {
      const base = `/api/loras/${id}/versions/${versionId}`;
      if (action === "delete") return api(base, { method: "DELETE" });
      if (action === "notes") return api(base, { method: "PATCH", body });
      return api(`${base}/${action}`, { body: body ?? {} });
    },
  );
}

export function useLoraLogs(id: string, versionId: string | null, live: boolean) {
  return useQuery({
    queryKey: ["loras", "logs", id, versionId],
    queryFn: () =>
      api<{ text: string; next_offset: number }>(`/api/loras/${id}/versions/${versionId}/logs`),
    enabled: Boolean(versionId),
    refetchInterval: live ? 5000 : false,
  });
}

export function useLoraTests(id: string) {
  return useQuery({
    queryKey: loraKeys.tests(id),
    queryFn: () => api<{ items: LoraTestGroup[] }>(`/api/loras/${id}/tests`),
    refetchInterval: (q) =>
      q.state.data?.items.some((g) => g.runs.some((r) => r.status === "queued" || r.status === "processing"))
        ? 3000
        : false,
  });
}

export function useRunLoraTest(id: string) {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (body: Record<string, unknown>) =>
      api<{ group: string; count: number }>(`/api/loras/${id}/tests`, { body }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: loraKeys.tests(id) });
      qc.invalidateQueries({ queryKey: keys.generations });
    },
  });
}

// --- jobs & secrets -------------------------------------------------------------------------

export function useJobs(filters: { type?: string; status?: string; provider?: string }) {
  return useQuery({
    queryKey: ["jobs", filters],
    queryFn: () =>
      api<{ items: JobItem[]; counts: Record<string, number> }>(`/api/jobs${query(filters)}`),
    refetchInterval: 5000,
    placeholderData: (prev) => prev,
  });
}

export function useSecrets() {
  return useQuery({ queryKey: ["secrets"], queryFn: () => api<{ items: SecretInfo[] }>("/api/secrets") });
}

export function useSaveSecret() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ name, value }: { name: string; value: string | null }) =>
      value === null
        ? api(`/api/secrets/${name}`, { method: "DELETE" })
        : api(`/api/secrets/${name}`, { method: "PUT", body: { value } }),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["secrets"] });
      qc.invalidateQueries({ queryKey: ["hf-status"] });
      qc.invalidateQueries({ queryKey: loraKeys.catalog });
    },
  });
}

/** The Hugging Face token's account and access to the gated Krea 2 repositories. */
export function useHfStatus(enabled = true) {
  return useQuery({
    queryKey: ["hf-status"],
    queryFn: () => api<HfStatus>("/api/secrets/huggingface/status"),
    enabled,
    staleTime: 60_000,
  });
}

export function useRefreshHfStatus() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: () => api<HfStatus>("/api/secrets/huggingface/status?refresh=true"),
    onSuccess: (data) => qc.setQueryData(["hf-status"], data),
  });
}
