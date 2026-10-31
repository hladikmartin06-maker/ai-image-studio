// Shapes returned by the FastAPI backend (see backend/app/services/serializers.py).

export type ProviderKey = "wavespeed" | "atlas";
export type ModelKey = "nano-banana-pro" | "seedream" | "wan-2.7" | "wan-2.7-pro";
export type Mode = "text-to-image" | "image-edit";
export type GenerationStatus = "queued" | "processing" | "completed" | "failed" | "cancelled";
export type Control =
  | "aspect_ratio"
  | "resolution"
  | "output_format"
  | "seed"
  | "thinking_mode"
  | "negative_prompt";

export interface User {
  id: string;
  name: string;
  email: string;
  created_at: string;
}

export interface AuthState {
  has_users: boolean;
  user: User | null;
}

export interface ProviderInfo {
  key: ProviderKey;
  name: string;
  description: string;
  website: string;
  keys_url: string;
  docs_url: string;
  supports_balance: boolean;
}

export interface Variant {
  provider: ProviderKey;
  mode: Mode;
  model_id: string;
  docs_url: string;
  controls: Control[];
  size_mode: "aspect+resolution" | "pixels" | "tier";
  aspect_ratios: string[];
  default_aspect_ratio: string | null;
  allow_match_input: boolean;
  resolutions: string[];
  default_resolution: string | null;
  output_formats: string[];
  seed_min: number | null;
  seed_max: number | null;
  max_reference_images: number;
  min_reference_images: number;
  max_images: number;
  batched: boolean;
  notes: string[];
  pixel_presets: Record<string, Record<string, [number, number]>>;
}

export interface ModelInfo {
  key: ModelKey;
  name: string;
  family: string;
  description: string;
  content: "sfw" | "sfw+nsfw";
  providers: ProviderKey[];
  variants: Variant[];
}

export interface Catalog {
  providers: ProviderInfo[];
  models: ModelInfo[];
  limits: { max_prompt_chars: number; max_images: number };
}

export interface Preferences {
  theme: "dark" | "light" | "system";
  language: "en" | "sk";
  default_model: ModelKey;
  active_provider: ProviderKey | null;
  default_image_count: number;
  default_aspect_ratio: string;
  sidebar: "expanded" | "compact";
  grid_density: "compact" | "comfortable" | "large";
  reduce_motion: boolean;
  show_prompts_on_cards: boolean;
  blur_nsfw: boolean;
  default_resolution: "1K" | "2K" | "4K" | null;
  default_output_format: "png" | "jpeg";
  default_thinking_mode: boolean;
  remember_prompt: boolean;
  clear_prompt_after_generate: boolean;
  auto_save: boolean;
  keep_failed_days: 0 | 7 | 30 | 90;
  download_name: "prompt" | "id" | "date";
  ai_provider: string | null;
  ai_mode: "low" | "balanced" | "high";
  ai_auto_route: boolean;
  ai_share_context: boolean;
}

export interface Integration extends ProviderInfo {
  status: "not_connected" | "connected" | "error";
  masked_key: string | null;
  last_error: string | null;
  last_checked_at: string | null;
  connected_at: string | null;
  active: boolean;
}

export interface IntegrationList {
  active_provider: ProviderKey | null;
  integrations: Integration[];
}

export interface GenerationSettings {
  aspect_ratio?: string;
  resolution?: string;
  output_format?: string;
  seed?: number;
  thinking_mode?: boolean;
}

export interface StudioImage {
  id: string;
  generation_id: string;
  index: number;
  url: string;
  thumb_url: string;
  download_url: string;
  width: number;
  height: number;
  mime_type: string;
  size_bytes: number;
  nsfw: boolean | null;
  in_library: boolean;
  is_favorite: boolean;
  favorited_at: string | null;
  created_at: string;
  model_key: ModelKey;
  model_name: string;
  provider: ProviderKey;
  provider_name: string;
  mode: Mode;
  prompt: string;
  settings: GenerationSettings;
  seed: number | null;
  status: GenerationStatus;
  persona_id: string | null;
  persona_name: string | null;
}

export interface ProviderTask {
  index: number;
  status: "pending" | "submitting" | "processing" | "completed" | "failed";
  images_requested: number;
  provider_task_id: string | null;
  error_code: string | null;
  error_message: string | null;
}

export interface ReferenceImage {
  id: string;
  url: string;
  thumb_url: string;
  original_name?: string;
  mime_type?: string;
  width?: number;
  height?: number;
  size_bytes?: number;
  created_at?: string;
}

export interface Generation {
  id: string;
  model_key: ModelKey;
  model_name: string;
  provider: ProviderKey;
  provider_name: string;
  mode: Mode;
  provider_model_id: string;
  prompt: string;
  settings: GenerationSettings;
  target_size: string | null;
  request_payload: Record<string, unknown>;
  reference_image_ids: string[];
  reference_images: ReferenceImage[];
  image_count: number;
  status: GenerationStatus;
  error_code: string | null;
  error_message: string | null;
  source_generation_id: string | null;
  final_prompt: string | null;
  persona: GenerationPersona | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  duration_seconds: number | null;
  tasks: ProviderTask[];
  images: StudioImage[];
}

/** The AI model (saved character) used for a generation; id is null once it was deleted. */
export interface GenerationPersona {
  id: string | null;
  name: string | null;
  reference_ids: string[];
  description_used: boolean;
  references_used: boolean;
  references_skipped: number | "unsupported" | null;
}

// --- AI models (saved characters) ------------------------------------------------------------

export interface PersonaAppearance {
  age: number | null;
  gender: string;
  ethnicity: string;
  skin: string;
  eyes: string;
  hair_color: string;
  hairstyle: string;
  face: string;
  body: string;
  height: string;
  style: string;
  distinctive: string;
  extra: string;
}

export interface PersonaImage extends ReferenceImage {
  is_primary: boolean;
}

export interface Persona {
  id: string;
  name: string;
  description: string;
  notes: string;
  appearance: PersonaAppearance;
  primary_image: PersonaImage | null;
  images: PersonaImage[];
  image_count: number;
  max_images: number;
  /** Exactly the text added to the prompt (without the identity line for references). */
  prompt_text: string;
  /** Added after prompt_text when reference images are attached. */
  identity_text: { one: string; many: string };
  generation_count: number;
  last_used_at: string | null;
  created_at: string;
  updated_at: string;
}

export interface PersonaList {
  items: Persona[];
  total: number;
  max_images: number;
}

export interface Page<T> {
  items: T[];
  next_cursor: string | null;
  total: number;
}

export interface Stats {
  images_total: number;
  images_today: number;
  favorites_total: number;
  storage_bytes: number;
  generations: {
    total: number;
    completed: number;
    failed: number;
    active: number;
    today: number;
    last_at: string | null;
  };
  providers: {
    provider: ProviderKey;
    generations: number;
    images: number;
    failed: number;
    last_used_at: string | null;
  }[];
  models: { model: ModelKey; generations: number; images: number }[];
  activity: { date: string; images: number }[];
}

export interface Balance {
  supported: boolean;
  balance: number | null;
  currency: string | null;
  connected?: boolean;
  error?: string;
}

export interface StorageInfo {
  location: string;
  bytes: { images: number; thumbs: number; refs: number };
  bytes_total: number;
  images: number;
  unsaved_images: number;
  favorites: number;
  reference_images: number;
  persona_reference_images: number;
  failed_generations: number;
}

export interface SessionInfo {
  id: string;
  user_agent: string;
  created_at: string;
  last_seen_at: string;
  expires_at: string;
  current: boolean;
}

// --- AI assistant ---------------------------------------------------------------------------

export type AIProviderKey = "anthropic" | "openai" | "xai" | "gemini" | "openai_compatible";
export type AIMode = "low" | "balanced" | "high";
export type AITier = "low" | "balanced" | "high";

export interface AIProvider {
  key: AIProviderKey;
  name: string;
  description: string;
  keys_url: string;
  docs_url: string;
  custom_base_url: boolean;
  key_optional: boolean;
  default_base_url: string | null;
  status: "not_connected" | "connected" | "error";
  masked_key: string | null;
  base_url: string | null;
  last_error: string | null;
  last_checked_at: string | null;
  models: string[];
  tiers: Record<AITier, { model: string | null; source: "user" | "preferred" | "matched" | "fallback" | "none" }>;
  active: boolean;
}

export interface AIProviderList {
  active_provider: AIProviderKey | null;
  mode: AIMode;
  auto_route: boolean;
  share_context: boolean;
  presets: { name: string; base_url: string }[];
  providers: AIProvider[];
}

export interface ChatMessage {
  id: string;
  conversation_id: string;
  role: "user" | "assistant";
  content: string;
  status: "complete" | "streaming" | "stopped" | "error";
  provider: string | null;
  model: string | null;
  tier: AITier | null;
  source: "api" | "local" | "cache";
  input_tokens: number;
  output_tokens: number;
  cached_tokens: number;
  usage_estimated: boolean;
  error_code: string | null;
  error_message: string | null;
  meta: { complexity?: string; route?: string; mode?: AIMode; truncated?: boolean; served_by?: string; quick_action?: string; context?: boolean };
  created_at: string;
}

export interface Conversation {
  id: string;
  title: string;
  has_summary: boolean;
  created_at: string;
  updated_at: string;
}

export interface ConversationDetail extends Conversation {
  summary: string;
  messages: ChatMessage[];
  usage: { input_tokens: number; output_tokens: number; cached_tokens: number };
}

export interface AIUsage {
  days: number;
  requests: number;
  input_tokens: number;
  output_tokens: number;
  cached_tokens: number;
  free_answers: number;
  summaries: number;
  by_model: { provider: string; model: string; requests: number; input_tokens: number; output_tokens: number; cached_tokens: number }[];
  by_tier: Record<string, number>;
}

export interface StudioContextPayload {
  page?: string;
  model?: string;
  provider?: string | null;
  settings?: Record<string, unknown>;
  prompt?: string;
  reference_count?: number;
  count?: number;
  ai_model?: string;
  ai_model_text?: string;
  ai_model_references?: number;
}

// --- GPU compute ----------------------------------------------------------------------------

export type GPUProviderKey = "runpod" | "modal";

export interface GPUProvider {
  key: GPUProviderKey;
  name: string;
  description: string;
  credential_fields: ("api_key" | "token_id" | "token_secret")[];
  keys_url: string;
  docs_url: string;
  dashboard_url: string;
  supports_endpoints: boolean;
  supports_stop_start: boolean;
  default_image: string;
  status: "not_connected" | "connected" | "error";
  masked_key: string | null;
  last_error: string | null;
  last_checked_at: string | null;
}

export interface GpuType {
  id: string;
  name: string;
  memory_gb: number | null;
  price_secure: number | null;
  price_community: number | null;
  available: boolean | null;
}

export interface GpuWorkload {
  id: string;
  name: string;
  kind: "pod" | "sandbox";
  status: string;
  gpu_type: string;
  gpu_count: number;
  image: string;
  cost_per_hour: number | null;
  urls: string[];
  created_at: string | null;
  dashboard_url: string | null;
  can_start: boolean;
  can_stop: boolean;
  can_terminate: boolean;
  from_studio: boolean;
}

export interface GpuEndpoint {
  id: string;
  name: string;
  gpu_types: string[];
  workers_min: number;
  workers_max: number;
  url: string | null;
  health: { jobs?: Record<string, number>; workers?: Record<string, number> } | null;
}

export interface GpuUsage {
  balance: number | null;
  spend_per_hour: number | null;
  source: string | null;
  error?: string;
  studio_workloads: number;
  studio_running: number;
  studio_gpu_hours: number;
  recent: { id: string; name: string; gpu_type: string; gpu_count: number; status: string; created_at: string | null; stopped_at: string | null }[];
}
