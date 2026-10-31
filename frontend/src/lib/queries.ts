"use client";

import {
  type InfiniteData,
  type QueryClient,
  useInfiniteQuery,
  useMutation,
  useQuery,
  useQueryClient,
} from "@tanstack/react-query";

import { api, query } from "@/lib/api";
import type {
  AIProviderList,
  AIUsage,
  AuthState,
  Balance,
  Catalog,
  Conversation,
  ConversationDetail,
  Generation,
  GPUProvider,
  GpuEndpoint,
  GpuType,
  GpuUsage,
  GpuWorkload,
  IntegrationList,
  Page,
  Persona,
  PersonaList,
  Preferences,
  ReferenceImage,
  SessionInfo,
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
  limit?: number;
}

export function useImages(filters: ImageFilters, enabled = true) {
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

export function useToggleFavorite() {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: ({ id, value }: { id: string; value: boolean }) =>
      api<StudioImage>(`/api/images/${id}`, { method: "PATCH", body: { is_favorite: value } }),
    onMutate: ({ id, value }) => {
      patchImageEverywhere(qc, id, (img) => ({
        ...img,
        is_favorite: value,
        favorited_at: value ? new Date().toISOString() : null,
      }));
    },
    onSettled: () => {
      qc.invalidateQueries({ queryKey: keys.images });
      qc.invalidateQueries({ queryKey: keys.stats });
    },
  });
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
  persona?: {
    id: string;
    reference_image_ids: string[];
    use_references: boolean;
    use_description: boolean;
  } | null;
}

/** Puts a generation (new or updated) into every cached generation list. */
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

export function useGpuTypes(provider: string, enabled: boolean) {
  return useQuery({
    queryKey: gpuKeys.types(provider),
    queryFn: () => api<{ items: GpuType[] }>(`/api/gpu/${provider}/gpu-types`),
    enabled,
    staleTime: 5 * 60_000,
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
