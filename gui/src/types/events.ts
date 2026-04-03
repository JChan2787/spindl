/**
 * Socket.IO Event Type Definitions
 * Contract between frontend and orchestrator
 */

// Agent states matching Python AgentState enum
export type AgentState =
  | "idle"
  | "listening"
  | "user_speaking"
  | "processing"
  | "system_speaking";

// ============================================
// Server -> Client Events (Orchestrator emits)
// ============================================

export interface StateChangedEvent {
  from: AgentState;
  to: AgentState;
  trigger: string;
  timestamp: string; // ISO8601
}

export interface TranscriptionEvent {
  text: string;
  duration: number;
  is_final: boolean;
  input_modality?: "voice" | "text" | "stimulus";
}

// NANO-037 Phase 2: Activated codex entry info for GUI display
export interface ActivatedCodexEntry {
  name: string;
  keys: string[];
  activation_method: "keyword" | "regex";
}

// NANO-044: Retrieved memory info for GUI display
export interface RetrievedMemory {
  content_preview: string;
  collection: "global" | "general" | "flashcards" | "summaries" | string;
  distance: number;
  score?: number; // NANO-107: composite retrieval score (higher = better)
}

export interface ResponseEvent {
  text: string;
  is_final: boolean;
  activated_codex_entries?: ActivatedCodexEntry[];
  retrieved_memories?: RetrievedMemory[];
  reasoning?: string; // NANO-042: Thinking/reasoning content from LLM
  stimulus_source?: string; // NANO-056: "patience" | "custom" | null
  emotion?: string; // NANO-094: Classified emotion mood string
  emotion_confidence?: number; // NANO-094: Normalized confidence (0.0-1.0)
  tts_text?: string; // NANO-109: TTS-safe text with formatting stripped
}

export interface TTSStatusEvent {
  status: "started" | "completed" | "interrupted";
  duration?: number;
}

// NANO-069: Real-time audio output level for portrait visualization
export interface AudioLevelEvent {
  level: number;
}

// NANO-073b: Real-time mic input level for voice overlay visualization
export interface MicLevelEvent {
  level: number;
}

export interface TokenUsageEvent {
  prompt: number;
  completion: number;
  total: number;
  max: number;
  percent: number;
}

export interface HealthStatusEvent {
  stt: boolean;
  tts: boolean;
  llm: boolean;
  vlm: boolean;
  embedding?: boolean;
  mic?: "ok" | "restarting" | "down";
}

export interface ContextUpdatedEvent {
  sources: string[];
}

export interface PipelineErrorEvent {
  stage: "stt" | "llm" | "tts";
  error_type: string;
  message: string;
}

// NANO-025 Phase 3: Prompt Inspector types

// NANO-045b: Per-block token data
export interface BlockTokenData {
  id: string;
  label: string;
  section: string | null;
  tokens: number;
  content?: string;
}

export interface TokenBreakdown {
  total: number;
  prompt: number;
  completion: number;
  system: number;
  user: number;
  sections: {
    agent: number;
    context: number;
    rules: number;
    conversation: number;
  };
  blocks?: BlockTokenData[]; // NANO-045b: per-block data (absent in legacy mode)
}

export interface PromptMessage {
  role: "system" | "user" | "assistant";
  content: string;
}

export interface PromptSnapshotEvent {
  messages: PromptMessage[];
  token_breakdown: TokenBreakdown;
  input_modality: string;
  state_trigger: string | null;
  timestamp: string;
}

export interface ProviderInfo {
  name: string;
  config: Record<string, unknown>;
}

export interface ConfigLoadedEvent {
  persona: {
    id: string;
    name: string;
    voice?: string;
  };
  providers: {
    llm: ProviderInfo;
    tts: ProviderInfo;
    stt: ProviderInfo;
    vlm?: ProviderInfo;
    embedding?: { base_url: string; enabled: boolean };
  };
  settings: {
    vad: {
      threshold: number;
      min_speech_ms: number;
      min_silence_ms: number;
      speech_pad_ms: number;
    };
    pipeline: {
      summarization_threshold: number;
      budget_strategy: string;
    };
    memory?: {
      top_k: number;
      relevance_threshold: number | null;
      enabled: boolean;
    };
    prompt?: {
      rag_prefix: string;
      rag_suffix: string;
      codex_prefix: string;
      codex_suffix: string;
      example_dialogue_prefix: string;
      example_dialogue_suffix: string;
    };
    generation?: {
      temperature: number;
      max_tokens: number;
      top_p: number;
      repeat_penalty: number;
      repeat_last_n: number;
      frequency_penalty: number;
      presence_penalty: number;
    };
    // NANO-056: Stimuli system config
    stimuli?: {
      enabled: boolean;
      patience_enabled: boolean;
      patience_seconds: number;
      patience_prompt: string;
    };
    // NANO-065a: Tools runtime config
    tools?: {
      master_enabled: boolean;
      tools: Record<string, { enabled: boolean }>;
    };
    // NANO-065b: LLM provider runtime state
    llm?: {
      provider: string;
      model: string;
      context_size: number | null;
      available_providers: string[];
    };
    // NANO-093/094: Avatar bridge config
    avatar?: {
      enabled: boolean;
      emotion_classifier: "classifier" | "off";
      show_emotion_in_chat: boolean;
      emotion_confidence_threshold: number;
    };
  };
}

export interface SessionInfo {
  filepath: string;
  persona: string;
  timestamp: string;
  turn_count: number;
  visible_count: number;
  file_size: number;
}

export interface SessionListEvent {
  sessions: SessionInfo[];
  active_session: string | null;
}

export interface Turn {
  turn_id: number;
  uuid: string;
  role: "user" | "assistant" | "summary";
  content: string;
  timestamp: string;
  hidden: boolean;
}

export interface SessionDetailEvent {
  filepath: string;
  turns: Turn[];
}

export interface SessionResumedEvent {
  filepath: string;
  success: boolean;
  error?: string;
}

export interface SessionDeletedEvent {
  filepath: string;
  success: boolean;
  error?: string;
}

// NANO-071: Create New Session
export interface SessionCreatedEvent {
  success: boolean;
  filepath?: string;
  error?: string;
}

// NANO-043 Phase 4: Session Summary
export interface SessionSummaryGeneratedEvent {
  filepath: string;
  success: boolean;
  summary_preview?: string;
  error?: string;
}

// NANO-073a + NANO-075: Chat History (with metadata for hydration survival)
export interface ChatHistoryTurn {
  role: "user" | "assistant";
  text: string;
  timestamp: string;
  // NANO-075: Metadata fields (optional — absent in pre-075 JSONL)
  input_modality?: string; // user turns: "VOICE", "TEXT", "stimulus"
  reasoning?: string; // assistant turns
  stimulus_source?: string; // assistant turns: "patience", "custom"
  activated_codex_entries?: ActivatedCodexEntry[]; // assistant turns
  retrieved_memories?: RetrievedMemory[]; // assistant turns
  // NANO-094: Emotion classifier metadata
  emotion?: string; // assistant turns
  emotion_confidence?: number; // assistant turns
}

export interface ChatHistoryEvent {
  turns: ChatHistoryTurn[];
}

export interface PersonaChangedEvent {
  persona_id: string;
  restart_required: boolean;
}

// NANO-077: Character switch failure
export interface PersonaChangeFailedEvent {
  error: string;
}

export interface VADConfigUpdatedEvent {
  threshold: number;
  min_speech_ms: number;
  min_silence_ms: number;
  speech_pad_ms: number;
}

export interface PipelineConfigUpdatedEvent {
  summarization_threshold: number;
  budget_strategy: string;
}

export interface MemoryConfigUpdatedEvent {
  top_k: number;
  relevance_threshold: number | null;
  dedup_threshold: number | null;
  reflection_interval: number;
  reflection_prompt: string | null;
  reflection_system_message: string | null;
  reflection_delimiter: string;
  enabled: boolean;
}

// NANO-045d + NANO-052 follow-up: Prompt injection wrapper config
export interface PromptConfigUpdatedEvent {
  rag_prefix: string;
  rag_suffix: string;
  codex_prefix: string;
  codex_suffix: string;
  example_dialogue_prefix: string;
  example_dialogue_suffix: string;
  persisted: boolean;
}

// NANO-053, NANO-108: Generation parameters
export interface GenerationParamsUpdatedEvent {
  temperature: number;
  max_tokens: number;
  top_p: number;
  repeat_penalty: number;
  repeat_last_n: number;
  frequency_penalty: number;
  presence_penalty: number;
  persisted: boolean;
}

export interface SetGenerationParamsPayload {
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
}

// NANO-065a: Runtime tools toggle
export interface ToolInfo {
  enabled: boolean;
  label: string;
}

export interface ToolsConfigUpdatedEvent {
  master_enabled: boolean;
  tools: Record<string, ToolInfo>;
  persisted?: boolean;
}

export interface SetToolsConfigPayload {
  master_enabled?: boolean;
  tools?: Record<string, { enabled: boolean }>;
}

// NANO-065b: Runtime LLM provider/model swap
export interface LLMConfigUpdatedEvent {
  provider: string;
  model: string;
  context_size: number | null;
  available_providers: string[];
  persisted?: boolean;
  success?: boolean;
  error?: string;
}

export interface SetLLMProviderPayload {
  provider: string;
  config: Record<string, unknown>;
}

// NANO-065c: Runtime VLM provider swap (extended NANO-079: unified toggle)
export interface VLMConfigUpdatedEvent {
  provider: string;
  available_providers: string[];
  healthy: boolean;
  unified_vlm?: boolean;
  persisted?: boolean;
  success?: boolean;
  error?: string;
  cloud_config?: {
    api_key: string;
    model: string;
    base_url: string;
  };
}

export interface SetVLMProviderPayload {
  provider: string;
  config: Record<string, unknown>;
}

// NANO-065b Enhancement: Dashboard Local LLM Launch (extended NANO-079: mmproj for unified VLM)
export interface LocalLLMConfig {
  executable_path?: string;
  model_path?: string;
  mmproj_path?: string;
  host?: string;
  port?: number;
  gpu_layers?: number;
  context_size?: number;
  device?: string;
  tensor_split?: string;
  extra_args?: string;
  timeout?: number;
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  reasoning_format?: string;
  reasoning_budget?: number;
}

export interface LaunchLLMServerPayload {
  config: LocalLLMConfig;
}

export interface LLMServerLaunchedEvent {
  success: boolean | null; // null = "launching" acknowledgment
  status?: "launching";
  already_running?: boolean;
  persisted?: boolean;
  error?: string;
}

export interface LocalLLMConfigEvent {
  config: LocalLLMConfig;
  server_running: boolean;
}

// NANO-079: Dashboard VLM Launch — All Four Cases
export interface LocalVLMConfig {
  model_type?: string;
  executable_path?: string;
  model_path?: string;
  mmproj_path?: string;
  host?: string;
  port?: number;
  gpu_layers?: number;
  context_size?: number;
  device?: string;
  tensor_split?: string;
  extra_args?: string;
  timeout?: number;
  max_tokens?: number;
}

export interface LaunchVLMServerPayload {
  config: LocalVLMConfig;
}

export interface VLMServerLaunchedEvent {
  success: boolean | null; // null = "launching" acknowledgment
  status?: "launching";
  already_running?: boolean;
  persisted?: boolean;
  error?: string;
}

export interface LocalVLMConfigEvent {
  config: LocalVLMConfig;
  server_running: boolean;
}

export interface OpenRouterModel {
  id: string;
  name: string;
  context_length: number | null;
}

export interface OpenRouterModelsEvent {
  models?: OpenRouterModel[];
  error?: string;
}

// NANO-056: Stimuli system events
// NANO-110: Tauri app build status (first-time build notification)
export interface TauriBuildStatusEvent {
  app: string; // "avatar", "subtitle", "stream_deck", "all"
  status: "building" | "ready" | "failed";
  message: string;
  progress?: number; // current crate count
  total?: number; // total crate count (0 if unknown)
}

// NANO-110: Tauri app install check result
export interface TauriInstallStatusEvent {
  avatar: boolean;
  subtitle: boolean;
  stream_deck: boolean;
}

// NANO-110: Addressing-others context
export interface AddressingContext {
  id: string;
  label: string;
  prompt: string;
}

// NANO-110: Addressing-others state broadcast
export interface AddressingOthersStateEvent {
  active: boolean;
  context_id: string | null;
}

export interface StimuliConfigUpdatedEvent {
  enabled: boolean;
  patience_enabled: boolean;
  patience_seconds: number;
  patience_prompt: string;
  // Twitch integration (NANO-056b)
  twitch_enabled: boolean;
  twitch_channel: string;
  twitch_app_id: string;
  twitch_app_secret: string;
  twitch_buffer_size: number;
  twitch_max_message_length: number;
  twitch_prompt_template: string;
  twitch_has_credentials: boolean;
  // NANO-110: Addressing-others contexts
  addressing_others_contexts: AddressingContext[];
  persisted: boolean;
}

export interface PatienceProgressEvent {
  elapsed: number;
  total: number;
  progress: number; // 0.0–1.0
  blocked?: boolean; // True when timer is paused (playback or typing)
  blocked_reason?: "playback" | "typing" | null;
}

// NANO-056b: Twitch module status
export interface TwitchStatusEvent {
  connected: boolean;
  channel: string;
  buffer_count: number;
  recent_messages: string[];
}

export interface StimulusFiredEvent {
  source: string;
  prompt_text: string;
  elapsed_seconds: number;
}

export interface SetStimuliConfigPayload {
  enabled?: boolean;
  patience_enabled?: boolean;
  patience_seconds?: number;
  patience_prompt?: string;
  // Twitch integration (NANO-056b)
  twitch_enabled?: boolean;
  twitch_channel?: string;
  twitch_app_id?: string;
  twitch_app_secret?: string;
  twitch_buffer_size?: number;
  twitch_max_message_length?: number;
  twitch_prompt_template?: string;
  // NANO-110: Addressing-others contexts
  addressing_others_contexts?: AddressingContext[];
}

// NANO-060b: VTubeStudio events
export interface VTSExpression {
  file: string;
  name: string;
  active: boolean;
}

export interface VTSStatusEvent {
  connected: boolean;
  authenticated: boolean;
  enabled: boolean;
  model_name: string | null;
  hotkeys: string[];
  expressions: VTSExpression[];
}

export interface VTSConfigUpdatedEvent {
  enabled: boolean;
  host: string;
  port: number;
  persisted: boolean;
}

export interface VTSHotkeysEvent {
  hotkeys: string[];
}

export interface VTSExpressionsEvent {
  expressions: VTSExpression[];
}

export interface VTSHotkeyTriggeredEvent {
  name: string;
}

export interface VTSExpressionTriggeredEvent {
  file: string;
  active: boolean;
}

export interface VTSMoveTriggeredEvent {
  preset: string;
}

export interface SetVTSConfigPayload {
  enabled?: boolean;
  host?: string;
  port?: number;
}

export interface RequestVTSStatusPayload {
  // Empty — request current VTS connection status
}

// NANO-093/094: Avatar bridge events
export interface AvatarConfigUpdatedEvent {
  enabled: boolean;
  emotion_classifier: "classifier" | "off";
  show_emotion_in_chat: boolean;
  emotion_confidence_threshold: number;
  expression_fade_delay: number;
  subtitles_enabled: boolean; // NANO-100
  subtitle_fade_delay: number; // NANO-100
  stream_deck_enabled: boolean; // NANO-110
  avatar_always_on_top: boolean;
  subtitle_always_on_top: boolean;
  persisted: boolean;
}

export interface SetAvatarConfigPayload {
  enabled?: boolean;
  emotion_classifier?: "classifier" | "off";
  show_emotion_in_chat?: boolean;
  emotion_confidence_threshold?: number;
  expression_fade_delay?: number;
  subtitles_enabled?: boolean; // NANO-100
  subtitle_fade_delay?: number; // NANO-100
  stream_deck_enabled?: boolean; // NANO-110
  avatar_always_on_top?: boolean;
  subtitle_always_on_top?: boolean;
}

export interface RequestVTSHotkeysPayload {
  refresh?: boolean;
}

export interface RequestVTSExpressionsPayload {
  refresh?: boolean;
}

export interface SendVTSHotkeyPayload {
  name: string;
}

export interface SendVTSExpressionPayload {
  file: string;
  active?: boolean;
}

export interface SendVTSMovePayload {
  preset: string;
}

// NANO-045c-1: Block config API
export interface BlockInfo {
  id: string;
  label: string;
  order: number;
  enabled: boolean;
  is_static: boolean;
  section_header: string | null;
  has_override: boolean;
  content_wrapper: string | null;
}

export interface BlockConfigLoadedEvent {
  order: string[];
  disabled: string[];
  overrides: Record<string, string | null>;
  blocks: BlockInfo[];
}

export interface BlockConfigUpdatedEvent {
  success: boolean;
  persisted: boolean;
  order: string[];
  disabled: string[];
  overrides: Record<string, string | null>;
}

// ============================================
// NANO-043 Phase 6: Memory Curation GUI Types
// ============================================

export type MemoryCollectionType = "global" | "general" | "flashcards" | "summaries";

export interface MemoryDocument {
  id: string;
  content: string;
  metadata: {
    timestamp?: string;
    type?: string;
    source?: string;
    session_id?: string;
    edited_at?: string;
    promoted_at?: string;
    [key: string]: unknown;
  };
}

export interface MemorySearchResult extends MemoryDocument {
  collection: MemoryCollectionType;
  distance: number;
  score?: number; // NANO-107: composite retrieval score
}

// Server -> Client
export interface MemoryCountsEvent {
  global: number;
  general: number;
  flashcards: number;
  summaries: number;
  enabled: boolean;
  error?: string;
}

export interface MemoryListEvent {
  collection: MemoryCollectionType;
  memories: MemoryDocument[];
  error?: string;
}

export interface MemoryAddedEvent {
  success: boolean;
  collection: "global" | "general";
  memory?: MemoryDocument;
  error?: string;
}

export interface MemoryEditedEvent {
  success: boolean;
  old_id?: string;
  memory?: MemoryDocument;
  error?: string;
}

export interface MemoryDeletedEvent {
  success: boolean;
  collection?: MemoryCollectionType;
  id?: string;
  error?: string;
}

export interface MemoryPromotedEvent {
  success: boolean;
  source_collection?: string;
  source_id?: string;
  new_id?: string;
  deleted_source?: boolean;
  error?: string;
}

export interface MemorySearchResultsEvent {
  results: MemorySearchResult[];
  query: string;
  error?: string;
}

export interface FlashcardsClearedEvent {
  success: boolean;
  error?: string;
}

// NANO-025 Phase 7: Tool Call Visibility types
export interface ToolInvokedEvent {
  tool_name: string;
  arguments: Record<string, unknown>;
  iteration: number;
  tool_call_id: string;
  timestamp: string;
}

export interface ToolResultEvent {
  tool_name: string;
  success: boolean;
  result_summary: string;
  duration_ms: number;
  iteration: number;
  tool_call_id: string;
}

// NANO-027 Phase 3: Service Launch Events
export interface LaunchProgressEvent {
  status: "starting" | "loading_config" | "config_loaded" | "starting" | "started" | "skipped" | "complete";
  service: string | null;
  message: string;
}

export interface LaunchErrorEvent {
  error: string;
  service: string | null;
}

export interface LaunchCompleteEvent {
  services: string[];
}

export interface LaunchStatusEvent {
  in_progress: boolean;
  launched_services: string[];
  has_orchestrator: boolean;
}

// NANO-027 Phase 4: Orchestrator Ready Events
export interface OrchestratorReadyEvent {
  persona: string;
  has_orchestrator: boolean;
}

export interface OrchestratorErrorEvent {
  error: string;
}

// NANO-028: Graceful Shutdown Events
export interface ShutdownProgressEvent {
  message: string;
}

export interface ShutdownCompleteEvent {
  timestamp: string;
}

export interface ShutdownErrorEvent {
  error: string;
}

// NANO-034 Phase 4: Character Management Events
export interface CharacterInfo {
  id: string;
  name: string;
  description: string;
  voice: string | null;
  has_avatar: boolean;
  tags: string[];
}

export interface CharacterListEvent {
  characters: CharacterInfo[];
  active: string | null;
}

export interface CharacterDetailEvent {
  character_id: string;
  card: CharacterCardData;
  has_avatar: boolean;
}

export interface CharacterCreatedEvent {
  character_id: string;
  success: boolean;
}

export interface CharacterUpdatedEvent {
  character_id: string;
  success: boolean;
}

export interface CharacterDeletedEvent {
  character_id: string;
  success: boolean;
}

export interface CharacterErrorEvent {
  error: string;
}

export interface AvatarUploadedEvent {
  character_id: string;
  success: boolean;
}

export interface AvatarDataEvent {
  character_id: string;
  image_data: string | null;
}

// NANO-041a: Crop settings for avatar crop modal
export interface CropSettings {
  scale: number;
  offsetX: number;
  offsetY: number;
}

export interface AvatarDataEventExtended extends AvatarDataEvent {
  original_data?: string | null;
  crop_settings?: CropSettings | null;
}

// NANO-034 Phase 5: Codex Management Events
export interface GlobalCodexEvent {
  entries: CharacterBookEntry[];
  name: string;
  error?: string;
}

export interface CharacterCodexEvent {
  character_id: string;
  entries: CharacterBookEntry[];
  error?: string;
}

export interface CodexEntryCreatedEvent {
  character_id: string | null;
  entry_id: number;
  success: boolean;
}

export interface CodexEntryUpdatedEvent {
  character_id: string | null;
  entry_id: number;
  success: boolean;
}

export interface CodexEntryDeletedEvent {
  character_id: string | null;
  entry_id: number;
  success: boolean;
}

export interface CodexErrorEvent {
  error: string;
}

// Character Card V2 types (matching backend models)
export interface GenerationConfig {
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  top_k?: number;
  frequency_penalty?: number;
  presence_penalty?: number;
  repeat_penalty?: number;
  repeat_last_n?: number;
}

export interface SpindlExtensions {
  id?: string;
  voice?: string; // Legacy (Kokoro) — kept for backward compat
  language?: string; // Legacy (Kokoro) — kept for backward compat
  tts_voice_config?: Record<string, string>; // Provider-agnostic TTS params (NANO-054a)
  appearance?: string;
  rules?: string[];
  summarization_prompt?: string;
  generation?: GenerationConfig;
  summarization_generation?: GenerationConfig;
  avatar_vrm?: string; // VRM filename within character directory (NANO-097)
  avatar_expressions?: Record<string, Record<string, number>>; // Per-character expression composites (NANO-098)
  avatar_animations?: {
    default?: string;
    emotions?: Record<string, { threshold: number; clip: string }>;
  }; // Emotion-to-animation threshold map (NANO-098)
}

export interface CharacterBookEntry {
  keys: string[];
  content: string;
  extensions: Record<string, unknown>;
  enabled: boolean;
  insertion_order: number;
  case_sensitive?: boolean;
  name?: string;
  priority?: number;
  id?: number;
  comment?: string;
  selective?: boolean;
  secondary_keys?: string[];
  constant?: boolean;
  position?: "before_char" | "after_char";
  sticky?: number;
  cooldown?: number;
  delay?: number;
}

export interface CharacterBook {
  entries: CharacterBookEntry[];
  name?: string;
  description?: string;
  scan_depth?: number;
  token_budget?: number;
  recursive_scanning?: boolean;
  extensions: Record<string, unknown>;
}

export interface CharacterCardDataInner {
  name: string;
  description: string;
  personality: string;
  scenario: string;
  first_mes: string;
  mes_example: string;
  creator_notes: string;
  system_prompt: string;
  post_history_instructions: string;
  alternate_greetings: string[];
  tags: string[];
  creator: string;
  character_version: string;
  extensions: Record<string, unknown>;
  character_book?: CharacterBook;
}

export interface CharacterCardData {
  spec: "chara_card_v2";
  spec_version: string;
  data: CharacterCardDataInner;
}

// ============================================
// Client -> Server Events (GUI emits)
// ============================================

export interface RequestStatePayload {
  // Empty - just requesting current state
}

export interface RequestHealthPayload {
  // Empty - trigger health check
}

export interface RequestConfigPayload {
  // Empty - request full config
}

export interface RequestSessionsPayload {
  persona?: string;
}

export interface RequestSessionDetailPayload {
  filepath: string;
}

export interface SetPersonaPayload {
  persona_id: string;
}

export interface SetVADConfigPayload {
  threshold?: number;
  min_speech_ms?: number;
  min_silence_ms?: number;
}

export interface SetPipelineConfigPayload {
  summarization_threshold?: number;
}

export interface SetMemoryConfigPayload {
  top_k?: number;
  relevance_threshold?: number | null;
  dedup_threshold?: number | null;
  reflection_interval?: number;
  reflection_prompt?: string | null;
  reflection_system_message?: string | null;
  reflection_delimiter?: string;
}

// NANO-045d + NANO-052 follow-up: Prompt injection wrapper config
export interface SetPromptConfigPayload {
  rag_prefix?: string;
  rag_suffix?: string;
  codex_prefix?: string;
  codex_suffix?: string;
  example_dialogue_prefix?: string;
  example_dialogue_suffix?: string;
}

// NANO-045c-1: Block config
export interface SetBlockConfigPayload {
  order?: string[];
  disabled?: string[];
  overrides?: Record<string, string | null>;
}

// NANO-042: Reasoning config
export interface SetReasoningConfigPayload {
  reasoning_budget: number; // -1 = unlimited, 0 = disabled
}

export interface ReasoningConfigUpdatedEvent {
  reasoning_budget: number;
  persisted: boolean;
}

export interface ResumeSessionPayload {
  filename: string;
}

export interface DeleteSessionPayload {
  filepath: string;
}

// NANO-043 Phase 4: Session Summary
export interface GenerateSessionSummaryPayload {
  filepath: string;
}

// NANO-031: Text Input Mode
export interface SendMessagePayload {
  text: string;
  skip_tts?: boolean;
}

// NANO-027 Phase 3: Service Launch Payloads
export interface StartServicesPayload {
  services?: string[];  // Optional: specific services to start
  skip_orchestrator?: boolean;  // Optional: don't start orchestrator
}

export interface RequestLaunchStatusPayload {
  // Empty - just requesting current status
}

// NANO-034 Phase 4: Character Management Payloads
export interface RequestCharactersPayload {
  // Empty - request list of characters
}

export interface RequestCharacterPayload {
  character_id: string;
}

export interface CreateCharacterPayload {
  card: CharacterCardData;
  character_id?: string;
}

export interface UpdateCharacterPayload {
  character_id: string;
  card: CharacterCardData;
}

export interface DeleteCharacterPayload {
  character_id: string;
}

export interface UploadAvatarPayload {
  character_id: string;
  image_data: string; // Base64 encoded with data URL prefix
}

export interface RequestAvatarPayload {
  character_id: string;
}

// NANO-034 Phase 5: Codex Management Payloads
export interface RequestGlobalCodexPayload {
  // Empty - request global codex entries
}

export interface RequestCharacterCodexPayload {
  character_id: string;
}

export interface CreateCodexEntryPayload {
  entry: CharacterBookEntry;
  character_id?: string; // If provided, add to character; otherwise add to global
}

export interface UpdateCodexEntryPayload {
  entry: CharacterBookEntry;
  entry_id: number;
  character_id?: string;
}

export interface DeleteCodexEntryPayload {
  entry_id: number;
  character_id?: string;
}

// NANO-043 Phase 6: Memory Curation GUI Payloads
export interface RequestMemoryCountsPayload {
  // Empty - request collection counts
}

export interface RequestMemoriesPayload {
  collection: MemoryCollectionType;
}

export interface AddGeneralMemoryPayload {
  content: string;
}

export interface EditGeneralMemoryPayload {
  id: string;
  content: string;
}

// NANO-105: Global memory payloads
export interface AddGlobalMemoryPayload {
  content: string;
}

export interface EditGlobalMemoryPayload {
  id: string;
  content: string;
}

export interface DeleteMemoryPayload {
  collection: MemoryCollectionType;
  id: string;
}

export interface PromoteMemoryPayload {
  source_collection: "flashcards" | "summaries";
  id: string;
  delete_source?: boolean;
}

export interface SearchMemoriesPayload {
  query: string;
  top_k?: number;
}

export interface ClearFlashcardsPayload {
  // Empty - clear all flash cards
}

// NANO-036: Character Hot-Reload Payloads
export interface ReloadCharacterPayload {
  // Empty - reload currently active character from disk
}

export interface ReloadCharacterResponse {
  success: boolean;
  error?: string;
  current_state?: string;
  character_id?: string;
}

// ============================================
// Event Maps for Type-Safe Socket.IO
// ============================================

export interface ServerToClientEvents {
  state_changed: (event: StateChangedEvent) => void;
  transcription: (event: TranscriptionEvent) => void;
  response: (event: ResponseEvent) => void;
  tts_status: (event: TTSStatusEvent) => void;
  token_usage: (event: TokenUsageEvent) => void;
  health_status: (event: HealthStatusEvent) => void;
  context_updated: (event: ContextUpdatedEvent) => void;
  pipeline_error: (event: PipelineErrorEvent) => void;
  config_loaded: (event: ConfigLoadedEvent) => void;
  session_list: (event: SessionListEvent) => void;
  session_detail: (event: SessionDetailEvent) => void;
  session_resumed: (event: SessionResumedEvent) => void;
  session_deleted: (event: SessionDeletedEvent) => void;
  // NANO-071: Create New Session
  session_created: (event: SessionCreatedEvent) => void;
  // NANO-073a: Chat History
  chat_history: (event: ChatHistoryEvent) => void;
  // NANO-043 Phase 4: Session Summary
  session_summary_generated: (event: SessionSummaryGeneratedEvent) => void;
  prompt_snapshot: (event: PromptSnapshotEvent) => void;
  persona_changed: (event: PersonaChangedEvent) => void;
  persona_change_failed: (event: PersonaChangeFailedEvent) => void;
  vad_config_updated: (event: VADConfigUpdatedEvent) => void;
  pipeline_config_updated: (event: PipelineConfigUpdatedEvent) => void;
  memory_config_updated: (event: MemoryConfigUpdatedEvent) => void;
  // NANO-102: Memory curation config
  curation_config_updated: (event: Record<string, unknown>) => void;
  // NANO-045d: Prompt injection wrappers
  prompt_config_updated: (event: PromptConfigUpdatedEvent) => void;
  // NANO-045c-1: Block config
  block_config_loaded: (event: BlockConfigLoadedEvent) => void;
  block_config_updated: (event: BlockConfigUpdatedEvent) => void;
  // NANO-053: Generation parameters
  generation_params_updated: (event: GenerationParamsUpdatedEvent) => void;
  // NANO-065a: Runtime tools toggle
  tools_config_updated: (event: ToolsConfigUpdatedEvent) => void;
  // NANO-065b: Runtime LLM provider/model swap
  llm_config_updated: (event: LLMConfigUpdatedEvent) => void;
  vlm_config_updated: (event: VLMConfigUpdatedEvent) => void;
  openrouter_models: (event: OpenRouterModelsEvent) => void;
  // NANO-065b Enhancement: Dashboard Local LLM Launch
  llm_server_launched: (event: LLMServerLaunchedEvent) => void;
  local_llm_config: (event: LocalLLMConfigEvent) => void;
  // NANO-079: Dashboard VLM Launch
  vlm_server_launched: (event: VLMServerLaunchedEvent) => void;
  local_vlm_config: (event: LocalVLMConfigEvent) => void;
  // NANO-042: Reasoning config
  reasoning_config_updated: (event: ReasoningConfigUpdatedEvent) => void;
  tool_invoked: (event: ToolInvokedEvent) => void;
  tool_result: (event: ToolResultEvent) => void;
  // NANO-027 Phase 3: Service Launch Events
  launch_progress: (event: LaunchProgressEvent) => void;
  launch_error: (event: LaunchErrorEvent) => void;
  launch_complete: (event: LaunchCompleteEvent) => void;
  launch_status: (event: LaunchStatusEvent) => void;
  // NANO-027 Phase 4: Orchestrator Ready Events
  orchestrator_ready: (event: OrchestratorReadyEvent) => void;
  orchestrator_error: (event: OrchestratorErrorEvent) => void;
  // NANO-028: Graceful Shutdown Events
  shutdown_progress: (event: ShutdownProgressEvent) => void;
  shutdown_complete: (event: ShutdownCompleteEvent) => void;
  shutdown_error: (event: ShutdownErrorEvent) => void;
  // NANO-034 Phase 4: Character Management Events
  character_list: (event: CharacterListEvent) => void;
  character_detail: (event: CharacterDetailEvent) => void;
  character_created: (event: CharacterCreatedEvent) => void;
  character_updated: (event: CharacterUpdatedEvent) => void;
  character_deleted: (event: CharacterDeletedEvent) => void;
  character_error: (event: CharacterErrorEvent) => void;
  avatar_uploaded: (event: AvatarUploadedEvent) => void;
  avatar_data: (event: AvatarDataEvent) => void;
  // NANO-034 Phase 5: Codex Management Events
  global_codex: (event: GlobalCodexEvent) => void;
  character_codex: (event: CharacterCodexEvent) => void;
  codex_entry_created: (event: CodexEntryCreatedEvent) => void;
  codex_entry_updated: (event: CodexEntryUpdatedEvent) => void;
  codex_entry_deleted: (event: CodexEntryDeletedEvent) => void;
  codex_error: (event: CodexErrorEvent) => void;
  // NANO-056: Stimuli system events
  stimuli_config_updated: (event: StimuliConfigUpdatedEvent) => void;
  patience_progress: (event: PatienceProgressEvent) => void;
  twitch_status: (event: TwitchStatusEvent) => void;
  twitch_credentials_result: (event: { success: boolean; error: string | null }) => void;
  stimulus_fired: (event: StimulusFiredEvent) => void;
  // NANO-110: Addressing-others state
  addressing_others_state: (event: AddressingOthersStateEvent) => void;
  // NANO-110: Tauri build/install status
  tauri_build_status: (event: TauriBuildStatusEvent) => void;
  tauri_install_status: (event: TauriInstallStatusEvent) => void;
  // NANO-060b: VTubeStudio events
  vts_config_updated: (event: VTSConfigUpdatedEvent) => void;
  vts_status: (event: VTSStatusEvent) => void;
  // NANO-093: Avatar bridge events
  avatar_config_updated: (event: AvatarConfigUpdatedEvent) => void;
  // NANO-097: Avatar model swap on character switch
  avatar_load_model: (event: { path: string }) => void;
  // NANO-097: Avatar renderer connection status
  avatar_connection_status: (event: { connected: boolean }) => void;
  vts_hotkeys: (event: VTSHotkeysEvent) => void;
  vts_expressions: (event: VTSExpressionsEvent) => void;
  vts_hotkey_triggered: (event: VTSHotkeyTriggeredEvent) => void;
  vts_expression_triggered: (event: VTSExpressionTriggeredEvent) => void;
  vts_move_triggered: (event: VTSMoveTriggeredEvent) => void;
  // NANO-043 Phase 6: Memory Curation GUI Events
  memory_counts: (event: MemoryCountsEvent) => void;
  memory_list: (event: MemoryListEvent) => void;
  memory_added: (event: MemoryAddedEvent) => void;
  memory_edited: (event: MemoryEditedEvent) => void;
  memory_deleted: (event: MemoryDeletedEvent) => void;
  memory_promoted: (event: MemoryPromotedEvent) => void;
  memory_search_results: (event: MemorySearchResultsEvent) => void;
  flashcards_cleared: (event: FlashcardsClearedEvent) => void;
  // NANO-069: Audio output level for portrait
  audio_level: (event: AudioLevelEvent) => void;
  // NANO-073b: Mic input level for voice overlay
  mic_level: (event: MicLevelEvent) => void;
  // NANO-111: Streaming LLM sentence chunks
  llm_chunk: (event: { text: string; is_final: boolean }) => void;
  // NANO-111: Token-level LLM text for real-time display
  llm_token: (event: { token: string; is_final: boolean }) => void;
}

export interface ClientToServerEvents {
  request_state: (payload: RequestStatePayload) => void;
  request_health: (payload: RequestHealthPayload) => void;
  request_config: (payload: RequestConfigPayload) => void;
  request_sessions: (payload: RequestSessionsPayload) => void;
  request_session_detail: (payload: RequestSessionDetailPayload) => void;
  set_persona: (payload: SetPersonaPayload) => void;
  set_vad_config: (payload: SetVADConfigPayload) => void;
  set_pipeline_config: (payload: SetPipelineConfigPayload) => void;
  set_memory_config: (payload: SetMemoryConfigPayload) => void;
  // NANO-102: Memory curation config
  set_curation_config: (payload: Record<string, unknown>) => void;
  // NANO-045d: Prompt injection wrappers
  set_prompt_config: (payload: SetPromptConfigPayload) => void;
  // NANO-045c-1: Block config
  request_block_config: (payload: Record<string, never>) => void;
  set_block_config: (payload: SetBlockConfigPayload) => void;
  reset_block_config: (payload: Record<string, never>) => void;
  // NANO-053: Generation parameters
  set_generation_params: (payload: SetGenerationParamsPayload) => void;
  // NANO-065a: Runtime tools toggle
  request_tools_config: (payload: Record<string, never>) => void;
  set_tools_config: (payload: SetToolsConfigPayload) => void;
  // NANO-065b: Runtime LLM provider/model swap
  request_llm_config: (payload: Record<string, never>) => void;
  set_llm_provider: (payload: SetLLMProviderPayload) => void;
  request_openrouter_models: (payload: Record<string, never>) => void;
  // NANO-065b Enhancement: Dashboard Local LLM Launch
  launch_llm_server: (payload: LaunchLLMServerPayload) => void;
  request_local_llm_config: (payload: Record<string, never>) => void;
  // NANO-065c: Runtime VLM provider swap (extended NANO-079)
  request_vlm_config: (payload: Record<string, never>) => void;
  set_vlm_provider: (payload: SetVLMProviderPayload) => void;
  // NANO-079: Dashboard VLM Launch
  launch_vlm_server: (payload: LaunchVLMServerPayload) => void;
  request_local_vlm_config: (payload: Record<string, never>) => void;
  // NANO-042: Reasoning config
  set_reasoning_config: (payload: SetReasoningConfigPayload) => void;
  // NANO-056: Stimuli system
  set_stimuli_config: (payload: SetStimuliConfigPayload) => void;
  request_patience_progress: (payload: Record<string, never>) => void;
  request_twitch_status: (payload: Record<string, never>) => void;
  test_twitch_credentials: (payload: { app_id: string; app_secret: string; channel: string }) => void;
  typing_active: (payload: { active: boolean }) => void;
  // NANO-110: Tauri install
  check_tauri_install: (payload: Record<string, never>) => void;
  install_tauri_apps: (payload: Record<string, never>) => void;
  // NANO-110: Addressing-others
  addressing_others_start: (payload: { context_id: string }) => void;
  addressing_others_stop: (payload: Record<string, never>) => void;
  // NANO-060b: VTubeStudio
  set_vts_config: (payload: SetVTSConfigPayload) => void;
  request_vts_status: (payload: RequestVTSStatusPayload) => void;
  // NANO-093: Avatar bridge
  set_avatar_config: (payload: SetAvatarConfigPayload) => void;
  // NANO-097: Request avatar model reload after VRM change in editor
  reload_avatar_model: (payload: { character_id: string }) => void;
  // NANO-099: Request avatar renderer to rescan animation files
  avatar_rescan_animations: () => void;
  // NANO-098: Live preview of expression composites from character editor
  preview_avatar_expressions: (payload: { expressions: Record<string, Record<string, number>>; previewMood?: string }) => void;
  // NANO-098 Session 3: Live preview of animation clip from character editor
  preview_avatar_animation: (payload: { clip: string | null }) => void;
  // NANO-098 Session 3: Push animation config to renderer on save
  update_avatar_animation_config: (payload: { animations: SpindlExtensions["avatar_animations"] | null }) => void;
  request_vts_hotkeys: (payload: RequestVTSHotkeysPayload) => void;
  request_vts_expressions: (payload: RequestVTSExpressionsPayload) => void;
  send_vts_hotkey: (payload: SendVTSHotkeyPayload) => void;
  send_vts_expression: (payload: SendVTSExpressionPayload) => void;
  send_vts_move: (payload: SendVTSMovePayload) => void;
  resume_session: (payload: ResumeSessionPayload) => void;
  delete_session: (payload: DeleteSessionPayload) => void;
  // NANO-071: Create New Session
  create_session: () => void;
  // NANO-073a: Chat History
  request_chat_history: (payload: Record<string, never>) => void;
  // NANO-076: Prompt Snapshot Hydration
  request_prompt_snapshot: (payload: Record<string, never>) => void;
  pause_listening: () => void;
  resume_listening: () => void;
  // NANO-031: Text Input Mode
  send_message: (payload: SendMessagePayload) => void;
  // NANO-027 Phase 3: Service Launch Events
  start_services: (payload: StartServicesPayload) => void;
  request_launch_status: (payload: RequestLaunchStatusPayload) => void;
  // NANO-043 Phase 4: Session Summary
  generate_session_summary: (payload: GenerateSessionSummaryPayload) => void;
  // NANO-028: Graceful Shutdown
  shutdown_backend: () => void;
  // NANO-034 Phase 4: Character Management Events
  request_characters: (payload: RequestCharactersPayload) => void;
  request_character: (payload: RequestCharacterPayload) => void;
  create_character: (payload: CreateCharacterPayload) => void;
  update_character: (payload: UpdateCharacterPayload) => void;
  delete_character: (payload: DeleteCharacterPayload) => void;
  upload_avatar: (payload: UploadAvatarPayload) => void;
  request_avatar: (payload: RequestAvatarPayload) => void;
  // NANO-034 Phase 5: Codex Management Events
  request_global_codex: (payload: RequestGlobalCodexPayload) => void;
  request_character_codex: (payload: RequestCharacterCodexPayload) => void;
  create_codex_entry: (payload: CreateCodexEntryPayload) => void;
  update_codex_entry: (payload: UpdateCodexEntryPayload) => void;
  delete_codex_entry: (payload: DeleteCodexEntryPayload) => void;
  // NANO-036: Character Hot-Reload
  reload_character: (
    payload: ReloadCharacterPayload,
    callback: (response: ReloadCharacterResponse) => void
  ) => void;
  // NANO-043 Phase 6: Memory Curation GUI Events
  request_memory_counts: (payload: RequestMemoryCountsPayload) => void;
  request_memories: (payload: RequestMemoriesPayload) => void;
  add_general_memory: (payload: AddGeneralMemoryPayload) => void;
  edit_general_memory: (payload: EditGeneralMemoryPayload) => void;
  add_global_memory: (payload: AddGlobalMemoryPayload) => void;
  edit_global_memory: (payload: EditGlobalMemoryPayload) => void;
  delete_memory: (payload: DeleteMemoryPayload) => void;
  promote_memory: (payload: PromoteMemoryPayload) => void;
  search_memories: (payload: SearchMemoriesPayload) => void;
  clear_flashcards: (payload: ClearFlashcardsPayload) => void;
}
