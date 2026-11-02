// Shapes returned by the FastAPI backend (see backend/app/services/serializers.py).

export type ProviderKey = "wavespeed" | "atlas";
export type GenerationProvider = ProviderKey | "comfyui";
export type ModelKey =
  | "nano-banana-pro"
  | "seedream"
  | "wan-2.7"
  | "wan-2.7-pro"
  | "krea-2-turbo"
  | "krea-2-raw";
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
  // No sign-in screen: the studio opens the first account on this Mac by itself.
  auto_login?: boolean;
}

export interface ProviderInfo {
  key: ProviderKey;
  name: string;
  description: string;
  website: string;
  keys_url: string;
  docs_url: string;
  supports_balance: boolean;
  // A price API exists (WaveSpeed); otherwise the price is on the provider's pricing page.
  supports_prices: boolean;
  pricing_url: string | null;
}

/** List prices (USD) of the hosted models on each connected provider. */
export interface PriceList {
  currency: "USD";
  providers: Partial<
    Record<
      ProviderKey,
      {
        supported: boolean;
        pricing_url: string | null;
        error: string | null;
        prices: { model: ModelKey; mode: Mode; model_id: string; price: number | null }[];
      }
    >
  >;
}

/** The price of one request before it is sent (an estimate from the provider's own API). */
export interface PriceEstimate {
  provider: ProviderKey;
  provider_name: string;
  model: ModelKey;
  model_name: string;
  mode: Mode;
  images: number;
  requests: number;
  currency: "USD";
  total: number | null;
  per_image: number | null;
  source: "estimate" | "list_price" | null;
  supported: boolean;
  pricing_url: string | null;
  error: string | null;
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
  // ComfyUI runs
  width?: number;
  height?: number;
  steps?: number;
  cfg?: number;
  sampler?: string;
  scheduler?: string;
  negative_prompt?: string;
  // Enhance photo (PGUpscaler)
  strength?: number;
  face_only?: boolean;
  upscale?: number;
  source_width?: number;
  source_height?: number;
  face_target?: string;
  // Krea 2 image to image
  denoise?: number;
  // Krea 2 "18+ content" switch
  adult?: boolean;
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
  provider: GenerationProvider;
  provider_name: string;
  mode: Mode;
  prompt: string;
  settings: GenerationSettings;
  seed: number | null;
  status: GenerationStatus;
  persona_id: string | null;
  persona_name: string | null;
  comfy: GenerationComfy | null;
}

export interface ProviderTask {
  index: number;
  status: "pending" | "submitting" | "processing" | "completed" | "failed";
  images_requested: number;
  provider_task_id: string | null;
  error_code: string | null;
  error_message: string | null;
  progress?: { state?: string; queue_position?: number | null; value?: number; max?: number };
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
  provider: GenerationProvider;
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
  comfy: GenerationComfy | null;
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
  meta: {
    complexity?: string;
    route?: string;
    mode?: AIMode;
    truncated?: boolean;
    served_by?: string;
    quick_action?: string;
    context?: boolean;
    images?: { id: string; url: string; thumb_url: string }[];
  };
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
  manufacturer?: string | null;
  // Modal: the workspace's live rates ("api") or its list prices ("list", an estimate), plus
  // the CPU and memory of the studio's sandbox per hour.
  price_estimate?: boolean;
  price_extra?: number | null;
  price_source?: "api" | "list" | null;
  // RunPod: free machines right now per cloud (for the requested purpose), else null.
  stock_secure?: "NONE" | "LOW" | "MEDIUM" | "HIGH" | null;
  stock_community?: "NONE" | "LOW" | "MEDIUM" | "HIGH" | null;
  cuda_secure?: string[];
  cuda_community?: string[];
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

// --- ComfyUI ------------------------------------------------------------------------------

export type ComfyKind = "local" | "remote" | "runpod" | "modal";

export interface ComfyDevice {
  name: string;
  type: string;
  vram_total: number | null;
  vram_free: number | null;
}

export interface AgentGpu {
  name: string;
  utilization: number | null;
  memory_used_mb: number | null;
  memory_total_mb: number | null;
  temperature: number | null;
  driver: string;
}

export interface AgentHealth {
  ok: boolean;
  gpus: AgentGpu[];
  ram_total: number | null;
  ram_available: number | null;
  load_1m: number | null;
  cpu_count: number | null;
  disk_free: number | null;
  disk_total: number | null;
}

export interface Krea2VariantState {
  name: string;
  installed: boolean;
  files: { filename: string; precision: string }[];
}

export interface Krea2Detection {
  supported: boolean;
  missing_nodes: string[];
  variants: Record<"turbo" | "raw", Krea2VariantState>;
  text_encoders: string[];
  vaes: string[];
  loras: string[];
  ready: boolean;
  missing: string[];
}

export interface ComfyInfo {
  version?: string | null;
  python?: string | null;
  pytorch?: string | null;
  os?: string | null;
  ram_total?: number | null;
  ram_free?: number | null;
  devices?: ComfyDevice[];
  node_count?: number;
  custom_node_packs?: string[];
  files?: Record<string, string[]>;
  krea2?: Krea2Detection;
  enhance?: EnhanceDetection;
  agent?: AgentHealth | null;
  checked_at?: string;
}

export interface ComfyDeploy {
  stage?: "creating" | "starting" | "booting" | "installing" | "ready" | "stopped" | "error";
  provider?: "runpod" | "modal";
  gpu_type?: string;
  gpu_count?: number;
  cloud_type?: string;
  image?: string | null;
  volume_gb?: number;
  session_minutes?: number;
  spend_limit?: number | null;
  cost_is_estimate?: boolean;
  spent_total?: number;
  // The RunPod ComfyUI image and driver CUDA versions the GPU was rented with.
  cuda?: string[];
  setup?: number;
  auto_stop_minutes?: number | null;
  agent_url?: string | null;
  started_at?: string;
  ready_at?: string;
  cost_per_hour?: number | null;
  log_tail?: string;
  stopped_by?: string;
  installing_at?: string;
  agent_at?: string;
  // Krea 2 model download on the GPU (after connecting).
  model_job?: {
    id: string;
    // A Krea 2 variant, or "pgupscaler" (Enhance photo).
    variant: "turbo" | "raw" | "pgupscaler";
    filename: string;
    files: string[];
    status: "running" | "completed" | "failed";
    progress: number;
    // The file being downloaded right now.
    current?: string | null;
    error: string | null;
    started_at: string;
  } | null;
}

export interface ComfyConnection {
  id: string;
  name: string;
  kind: ComfyKind;
  base_url: string | null;
  auth_type: "none" | "bearer" | "basic" | "agent";
  masked_secret: string | null;
  models_dir: string | null;
  gpu_provider: "runpod" | "modal" | null;
  workload_id: string | null;
  status: "new" | "ready" | "error" | "deploying" | "stopped";
  last_error: string | null;
  last_checked_at: string | null;
  info: ComfyInfo;
  deploy: ComfyDeploy;
  // The current session (estimate): spent so far and time left, from the start request and
  // the GPU's hourly rate.
  session: {
    per_hour: number | null;
    spent: number | null;
    seconds: number;
    seconds_left: number;
    spend_limit: number | null;
  } | null;
  file_capability: "agent" | "folder" | "none";
  created_at: string;
}

export interface Krea2Variant {
  key: "turbo" | "raw";
  model_key: string;
  name: string;
  description: string;
  steps: number;
  cfg: number;
  sampler: string;
  scheduler: string;
  negative_prompt: boolean;
  tiers: string[];
  settings_source: string;
  known_files: { filename: string; folder: string; precision: string; size_gb: number | null; url: string }[];
}

/** Enhance photo (PGUpscaler) files and nodes on a ComfyUI. */
export interface EnhanceDetection {
  files: { folder: string; filename: string; size: number; installed: boolean }[];
  missing_files: string[];
  missing_nodes: string[];
  ready: boolean;
}

export interface EnhanceCatalog {
  name: string;
  model_key: string;
  repo: string;
  template: string;
  files: { folder: string; filename: string; size: number }[];
  total_bytes: number;
  strengths: number[];
  upscales: number[];
  default_strength: number;
  default_upscale: number;
  min_vram_gb: number;
  min_vram_gb_3x: number;
  skin_prompt: string;
  face_target: string;
  max_prompt: number;
  max_face_target: number;
}

export interface EnhanceBody {
  connection_id: string;
  image: { kind: "reference" | "image"; id: string };
  strength: number;
  face_only: boolean;
  upscale: number;
  seed: number | null;
  prompt: string;
  face_target: string;
  client_request_id: string;
}

export interface ComfyCatalog {
  krea2: {
    architecture: Record<string, string | string[]>;
    variants: Krea2Variant[];
    text_encoder: { filename: string; size_gb: number | null; url: string };
    vae: { filename: string; size_gb: number | null; url: string };
    aspect_ratios: string[];
    sizes: Record<string, Record<string, [number, number]>>;
    seed_max: number;
    img2img: { denoise_default: number; denoise_min: number; denoise_max: number };
    content: "sfw" | "sfw+nsfw";
  };
  enhance: EnhanceCatalog;
  deploy_images: Record<string, string>;
  model_folders: string[];
}

export interface ComfyWorkflow {
  id: string;
  name: string;
  family: string | null;
  has_api: boolean;
  has_ui: boolean;
  bindings: Record<string, unknown>;
  node_types: string[];
  runnable: boolean;
  created_at: string;
  updated_at: string;
}

export interface GenerationComfy {
  connection_id: string | null;
  connection_name: string | null;
  variant: "turbo" | "raw" | null;
  model_file: string | null;
  workflow: { id: string | null; name: string } | null;
  // "krea2", or "pgupscaler" for Enhance photo (then `input` is the photo it started from).
  family?: string | null;
  input?: { kind: "reference" | "image"; id: string; thumb_url: string; url: string; width: number; height: number } | null;
  lora: {
    lora_id: string;
    version_id: string;
    name: string;
    version: number;
    trigger_word: string;
    strength: number;
    file: string;
  } | null;
  test_group: string | null;
  test_label: string | null;
  test_prompt?: string | null;
  cancel_requested: boolean;
}

// --- LoRA studio -----------------------------------------------------------------------------

export interface LoraItem {
  id: string;
  reference_id: string;
  url: string;
  thumb_url: string;
  original_name: string;
  mime_type: string | null;
  width: number | null;
  height: number | null;
  size_bytes: number | null;
  caption: string;
  caption_source: "manual" | "ai" | "empty";
  split: "train" | "validation";
  excluded: boolean;
  crop: { x: number; y: number; w: number; h: number } | null;
  position: number;
}

export interface LoraWarning {
  code: string;
  level: "error" | "warning" | "info";
  n?: number;
  items?: string[];
  pairs?: string[][];
}

export interface LoraAnalysis {
  total: number;
  used: number;
  train: number;
  validation: number;
  excluded: number;
  captioned: number;
  min_edge: number | null;
  median_edge: number | null;
  aspect_buckets: Record<string, number>;
  warnings: LoraWarning[];
  ready: boolean;
}

export interface TrainingConfig {
  base_variant: "raw" | "turbo";
  rank: number;
  alpha: number;
  learning_rate: number;
  optimizer: string;
  lr_scheduler: string;
  steps: number;
  batch_size: number;
  gradient_accumulation: number;
  gradient_checkpointing: boolean;
  mixed_precision: "bf16" | "fp16";
  resolutions: number[];
  quantize: boolean;
  low_vram: boolean;
  timestep_type: string;
  caption_dropout_rate: number;
  diff_output_preservation: boolean;
  diff_output_preservation_multiplier: number;
  save_every: number;
  max_checkpoints: number;
  sample_every: number;
  previews: "during" | "end";
  // Continue an earlier version (its LoRA is the starting point; its size is kept).
  continue_from?: string | null;
  validation_prompts: string[];
  seed: number;
  save_dtype: "float16" | "bf16";
}

export type TrainingStatus =
  | "queued"
  | "preparing"
  | "uploading"
  | "running"
  | "downloading"
  | "completed"
  | "failed"
  | "cancelled"
  | "interrupted";

export interface LoraVersion {
  id: string;
  lora_id: string;
  number: number;
  status: TrainingStatus;
  stage_detail: string;
  preset: string;
  config: Partial<TrainingConfig> & { image?: string | null; cloud_type?: string; continue_from_number?: number };
  gpu_provider: "runpod" | "modal" | null;
  gpu_type: string;
  gpu_count: number;
  workload_id: string | null;
  auto_cleanup: boolean;
  cost_per_hour: number | null;
  cost_estimate: number | null;
  cost_final: number | null;
  step: number | null;
  total_steps: number | null;
  loss: [number, number][];
  gpu: AgentGpu | null;
  samples: { file: string; step: number | null; index: number | null; url: string }[];
  file_url: string | null;
  file_size: number | null;
  file_sha256: string | null;
  checkpoints: { step: number; size: number | null; url: string }[];
  deployments: Record<string, { filename: string; at: string }>;
  dataset_size: number;
  source: string;
  notes: string;
  error_code: string | null;
  error_message: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
  gpu_started_at: string | null;
  gpu_stopped_at: string | null;
}

export interface LoraSummary {
  id: string;
  name: string;
  trigger_word: string;
  class_word: string;
  architecture: string;
  base_variant: "raw" | "turbo";
  source: "trained" | "imported";
  persona_id: string | null;
  persona_name: string | null;
  caption_template: string;
  tags: string[];
  notes: string;
  preferred_version_id: string | null;
  preferred_version: LoraVersion | null;
  latest_version: LoraVersion | null;
  usable_versions: { id: string; number: number; preferred: boolean }[];
  active: boolean;
  version_count: number;
  dataset_count: number | null;
  created_at: string;
  updated_at: string;
}

export interface LoraDetail extends LoraSummary {
  dataset: LoraItem[];
  analysis: LoraAnalysis;
  versions: LoraVersion[];
}

export interface LoraPreset {
  key: "character_identity" | "photorealistic" | "fast_experiment" | "high_quality";
  name: string;
  summary: string;
  tradeoff: string;
  config: TrainingConfig;
}

export interface LoraArchitecture {
  key: string;
  name: string;
  trainer: string;
  trainer_image: string;
  variants: Record<
    "raw" | "turbo",
    { name: string; name_or_path: string; gated: boolean; usable_on: string[]; note: string }
  >;
  resolutions: number[];
  optimizers: string[];
  schedulers: string[];
  timestep_types: string[];
  unavailable: Record<string, string>;
  vram_quantized_gb: number;
  vram_full_gb: number;
  vram_no_checkpointing_gb: number;
  bucketing: string;
}

export interface LoraCatalog {
  architectures: LoraArchitecture[];
  presets: LoraPreset[];
  defaults: TrainingConfig;
  gpu_providers: { key: "runpod" | "modal"; name: string; connected: boolean }[];
  hf_token_set: boolean;
}

export interface LoraTestGroup {
  group: string;
  created_at: string | null;
  runs: Generation[];
}

// --- Jobs ----------------------------------------------------------------------------------

export type JobStatus =
  | "queued"
  | "preparing"
  | "running"
  | "completed"
  | "failed"
  | "cancelled"
  | "interrupted";

export interface JobItem {
  id: string;
  type: "generation" | "training" | "deployment";
  name: string;
  model: string;
  model_key: string;
  lora: string | null;
  provider: string | null;
  provider_name: string;
  gpu: string | null;
  status: JobStatus;
  stage?: string;
  stage_detail?: string;
  queue_position: number | null;
  progress: number | null;
  step?: number | null;
  total_steps?: number | null;
  created_at: string | null;
  started_at: string | null;
  completed_at: string | null;
  elapsed_seconds: number | null;
  cost: number | null;
  cost_is_final?: boolean;
  error: string | null;
  link: string;
  artifacts: number;
}

export type HfAccess = "granted" | "needs_license" | "invalid_token" | "not_found" | "unknown";

export interface HfStatus {
  set: boolean;
  masked: string | null;
  token: "valid" | "invalid" | "unknown" | null;
  // Hugging Face's own reason when the token is not valid or could not be checked.
  token_message: string | null;
  account: string | null;
  ready: boolean;
  repos: {
    repo: string;
    variant: "raw" | "turbo";
    name: string;
    url: string;
    access: HfAccess;
    // Hugging Face's own reason when access is not granted.
    message: string | null;
  }[];
}

export interface SecretInfo {
  name: string;
  set: boolean;
  masked: string | null;
  updated_at: string | null;
}
