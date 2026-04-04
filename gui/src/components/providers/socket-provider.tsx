"use client";

import { useEffect } from "react";
import { getSocket, connectSocket, disconnectSocket } from "@/lib/socket";
import { useConnectionStore, useAgentStore, usePromptStore, useSessionStore, useSettingsStore, useLauncherStore, useCharacterStore, useCodexStore, useMemoryStore, useBlockEditorStore, useVTSStore } from "@/lib/stores";
import { useChatStore } from "@/lib/stores/chat-store";
import type { MemoryCollectionType, ChatHistoryTurn } from "@/types/events";
import {
  safeParse,
  LLMConfigUpdatedSchema,
  VLMConfigUpdatedSchema,
  LocalLLMConfigEventSchema,
  LocalVLMConfigEventSchema,
  ToolsConfigUpdatedSchema,
  StimuliConfigUpdatedSchema,
  VADConfigUpdatedSchema,
  MemoryConfigUpdatedSchema,
  CurationConfigSchema,
} from "@/lib/schemas/runtime-schemas";

interface SocketProviderProps {
  children: React.ReactNode;
}

export function SocketProvider({ children }: SocketProviderProps) {
  const { setConnected, setConnecting, setError } = useConnectionStore();
  const {
    addTransition,
    setTranscription,
    setResponse,
    setTokenUsage,
    setHealth,
    setConfig,
    clearTranscription,
    clearResponse,
    setListeningPaused,
    addToolInvoked,
    updateToolResult,
    clearToolActivities,
    setShutdownProgress,
    setShutdownComplete,
    setShutdownError,
    setAudioLevel,
    setMicLevel,
    setStimulusActive,
  } = useAgentStore();
  const { setSnapshot, clearSnapshot } = usePromptStore();
  const {
    setSessions,
    setSessionDetail,
    setActionResult,
    removeSession,
    setPersonaFilter,
  } = useSessionStore();
  const {
    loadConfig,
    setVadConfig,
    setPipelineConfig,
    setMemoryConfig,
    setGenerationConfig,
    setStimuliConfig,
    setStimuliProgress,
    setToolsConfig,
    setSavingTools,
    setLLMConfig,
    setSavingLLM,
    setVLMConfig,
    setSavingVLM,
    setLocalLLMConfig,
    setLaunchingLLM,
    setLLMServerRunning,
    setLLMLaunchError,
    setLocalVLMConfig,
    setLaunchingVLM,
    setVLMServerRunning,
    setVLMLaunchError,
    setUnifiedVLM,
    setRestartRequired,
    setAvatarConfig,
  } = useSettingsStore();
  const {
    setLaunchProgress,
    setLaunchError,
    setLaunchComplete,
    setOrchestratorInitializing,
    setOrchestratorReady,
    setOrchestratorError,
    resetLaunch,
  } = useLauncherStore();
  const {
    setCharacters,
    setCharacterDetail,
    setActionResult: setCharacterActionResult,
    setActionError: setCharacterActionError,
    removeCharacter,
    setAvatar,
    setSwitchingCharacter,
  } = useCharacterStore();
  const {
    setGlobalCodex,
    setCharacterCodex,
    setActionResult: setCodexActionResult,
    setActionError: setCodexActionError,
    removeEntry: removeCodexEntry,
  } = useCodexStore();
  const {
    setCounts: setMemoryCounts,
    setMemories,
    setLoadingList: setMemoryLoadingList,
    addMemory,
    editMemory,
    removeMemory: removeMemoryFromStore,
    clearAllFlashcards,
    setActionResult: setMemoryActionResult,
    setSearchResults,
    setSearching: setMemorySearching,
  } = useMemoryStore();
  const {
    setBlockConfig,
    setBlockConfigUpdated,
  } = useBlockEditorStore();
  const {
    setStatus: setVTSStatus,
    setEnabled: setVTSEnabled,
    setHotkeys: setVTSHotkeys,
    setExpressions: setVTSExpressions,
  } = useVTSStore();
  const {
    addUserMessage,
    addAssistantMessage,
    updateAssistantMessage,
    hydrateHistory,
    clearHistory,
  } = useChatStore();

  useEffect(() => {
    const socket = getSocket();

    // Connection events
    socket.on("connect", () => {
      setConnected(true);
      // Request initial state
      socket.emit("request_config", {});
      socket.emit("request_health", {});
      socket.emit("request_state", {});
      socket.emit("request_vts_status", {});
      // NANO-073a: Hydrate chat history from backend JSONL
      socket.emit("request_chat_history", {});
      // NANO-076: Hydrate prompt snapshot from sidecar
      socket.emit("request_prompt_snapshot", {});
    });

    socket.on("disconnect", () => {
      setConnected(false);
      // Clear stale orchestrator data — a fresh config_loaded from a live
      // orchestrator is required to prove services are actually running.
      setHealth(null);
      setConfig(null);
      // NANO-068: Clear stale launcher state so the Launcher page doesn't
      // show "Systems Ready" from a previous session and trap the user in
      // a redirect loop with the Dashboard overlay.
      resetLaunch();
    });

    socket.on("connect_error", (error) => {
      setError(error.message);
    });

    // NANO-073a: Track current assistant message ID for streaming updates
    let currentAssistantMsgId: string | null = null;

    // Agent events
    socket.on("state_changed", (event) => {
      addTransition(event);
      // Track listening paused state (idle = paused when agent is running)
      setListeningPaused(event.to === "idle");
      // Clear transcription/response on certain state changes
      if (event.to === "listening") {
        clearTranscription();
      }
      if (event.to === "processing") {
        clearResponse();
        clearToolActivities(); // Clear previous tool activities when new processing starts
        // NANO-073a: Reset assistant message tracker for new exchange
        currentAssistantMsgId = null;
      }
    });

    socket.on("transcription", (event) => {
      setTranscription(event.text, event.is_final);
      // NANO-073d: Only add user message for voice origins.
      // Text input is already added optimistically by ChatInput.
      // NANO-074a: Stimulus messages are internal machinery — hide from chat.
      if (event.is_final && event.text.trim() && event.input_modality === "voice") {
        addUserMessage(event.text);
      }
    });

    // NANO-111: Token-level LLM text for real-time chat display (like ChatGPT)
    socket.on("llm_token", (event) => {
      if (!currentAssistantMsgId) {
        // First token — create the chat bubble
        currentAssistantMsgId = addAssistantMessage({
          text: event.token,
          isFinal: false,
        });
      } else {
        // Subsequent tokens — append to existing bubble
        const msg = useChatStore.getState().messages.find((m) => m.id === currentAssistantMsgId);
        if (msg) {
          updateAssistantMessage(currentAssistantMsgId, {
            text: msg.text + event.token,
          });
        }
      }
    });

    // NANO-111 Session 606: Sequential sentence rendering synced with TTS.
    // Each llm_chunk fires when playback reaches that sentence — [text][tts][text][tts].
    // Builds the bubble incrementally. Response event finalizes metadata after.
    socket.on("llm_chunk", (event) => {
      const chunk = {
        text: event.text,
        emotion: event.emotion,
        emotionConfidence: event.emotion_confidence,
      };
      if (!currentAssistantMsgId) {
        // First sentence — create the parent bubble
        currentAssistantMsgId = addAssistantMessage({
          text: event.text,
          isFinal: false,
          chunks: [chunk],
        });
      } else {
        // Subsequent sentences — append chunk to existing bubble
        const msg = useChatStore.getState().messages.find((m) => m.id === currentAssistantMsgId);
        if (msg) {
          updateAssistantMessage(currentAssistantMsgId, {
            text: (msg.text ? msg.text + " " : "") + event.text,
            chunks: [...(msg.chunks || []), chunk],
          });
        }
      }
    });

    // NANO-037: codex entries, NANO-042: reasoning, NANO-044: memories, NANO-056: stimulus source
    // NANO-111 Session 606: response finalizes the bubble with metadata.
    // llm_chunk builds chunks incrementally during playback.
    // response sets authoritative text + metadata but preserves existing chunks.
    socket.on("response", (event) => {
      setResponse(event.text, event.is_final, event.activated_codex_entries, event.reasoning, event.retrieved_memories, event.stimulus_source);

      if (!currentAssistantMsgId) {
        // No llm_chunk arrived (blocking path / text input / stimulus).
        // Build chunks from response event if available, otherwise plain text.
        const chunks = event.chunks?.map((c: { text: string; emotion?: string; emotion_confidence?: number }) => ({
          text: c.text,
          emotion: c.emotion,
          emotionConfidence: c.emotion_confidence,
        }));
        currentAssistantMsgId = addAssistantMessage({
          text: event.text,
          isFinal: event.is_final,
          reasoning: event.reasoning,
          activatedCodexEntries: event.activated_codex_entries,
          retrievedMemories: event.retrieved_memories,
          stimulusSource: event.stimulus_source,
          emotion: event.emotion,
          emotionConfidence: event.emotion_confidence,
          chunks,
        });
      } else {
        // llm_chunk already built chunks incrementally during playback.
        // Finalize with metadata — do NOT overwrite chunks (would re-render all at once).
        updateAssistantMessage(currentAssistantMsgId, {
          text: event.text,
          isFinal: event.is_final,
          reasoning: event.reasoning,
          activatedCodexEntries: event.activated_codex_entries,
          retrievedMemories: event.retrieved_memories,
          stimulusSource: event.stimulus_source,
          emotion: event.emotion,
          emotionConfidence: event.emotion_confidence,
        });
      }
      // Clear the tracked ID and stimulus flag when response is final
      if (event.is_final) {
        currentAssistantMsgId = null;
        setStimulusActive(false);
      }
    });

    // NANO-073a + NANO-075: Chat history hydration with metadata survival
    socket.on("chat_history", (event: { turns: ChatHistoryTurn[] }) => {
      if (event.turns && event.turns.length > 0) {
        const hydrated = event.turns
          // NANO-075: Suppress stimulus user turns on hydration (matches live path behavior)
          .filter((t) => !(t.role === "user" && t.input_modality === "stimulus"))
          .map((t, i) => ({
            id: `hydrated-${i}`,
            role: t.role as "user" | "assistant",
            text: t.text,
            timestamp: t.timestamp,
            isFinal: true,
            // NANO-075: Hydrate metadata fields (undefined when absent = backward compat)
            reasoning: t.reasoning,
            stimulusSource: t.stimulus_source,
            activatedCodexEntries: t.activated_codex_entries,
            retrievedMemories: t.retrieved_memories,
            // NANO-094: Emotion classifier metadata
            emotion: t.emotion,
            emotionConfidence: t.emotion_confidence,
          }));
        hydrateHistory(hydrated);
      }
    });

    socket.on("token_usage", (event) => {
      setTokenUsage(event);
    });

    socket.on("health_status", (event) => {
      setHealth(event);
    });

    socket.on("config_loaded", (event) => {
      setConfig(event);
      loadConfig(event);
      // NANO-077: Seed activeCharacterId from config so Characters page
      // Active badge works on initial load (REST endpoint returns active: null)
      if (event.persona?.id) {
        useCharacterStore.setState((state) => ({
          activeCharacterId: state.activeCharacterId ?? event.persona.id,
        }));
      }
    });

    socket.on("pipeline_error", (event) => {
      console.error("[Pipeline Error]", event.stage, event.message);
    });

    socket.on("prompt_snapshot", (event) => {
      setSnapshot(event);
    });

    // Session events
    socket.on("session_list", (event) => {
      setSessions(event.sessions, event.active_session);
    });

    socket.on("session_detail", (event) => {
      setSessionDetail(event);
    });

    socket.on("session_resumed", (event) => {
      setActionResult("resume", event.filepath, event.success, event.error);
      if (event.success) {
        clearResponse();
        // NANO-073a: Re-hydrate chat history for resumed session
        clearHistory();
        socket.emit("request_chat_history", {});
        // NANO-076: Re-hydrate prompt snapshot for resumed session
        clearSnapshot();
        socket.emit("request_prompt_snapshot", {});
      }
    });

    socket.on("session_deleted", (event) => {
      setActionResult("delete", event.filepath, event.success, event.error);
      if (event.success) {
        removeSession(event.filepath);
      }
    });

    // NANO-071: Create New Session
    socket.on("session_created", (event) => {
      setActionResult("create", event.filepath ?? "", event.success, event.error);
      if (event.success) {
        clearResponse();
        // NANO-073a: Clear chat history for fresh session
        clearHistory();
        // NANO-074b: Refresh session list so the new session appears
        socket.emit("request_sessions", {});
        // NANO-076: Clear stale snapshot and request fresh one
        clearSnapshot();
        socket.emit("request_prompt_snapshot", {});
      }
    });

    socket.on("session_summary_generated", (event) => {
      setActionResult("summarize", event.filepath, event.success, event.error);
    });

    // NANO-077: Runtime character switching — full hot-swap handler
    socket.on("persona_changed", (event) => {
      if (event.restart_required) {
        setRestartRequired(true);
      }
      // Hot-swap landed — update all character-dependent state
      setSwitchingCharacter(false);
      // Update active character without clearing the list (request_characters will refresh it)
      useCharacterStore.setState({ activeCharacterId: event.persona_id });
      setPersonaFilter(event.persona_id);
      // Clear selected session so auto-select picks the new character's active session
      // when session_list arrives (pushed by backend after persona_changed)
      useSessionStore.setState({
        selectedSession: null,
        selectedSessionTurns: [],
        isLoadingDetail: false,
      });
      // Clear chat history and re-hydrate for new character's session
      clearHistory();
      clearResponse();
      clearSnapshot();
      socket.emit("request_config", {});
      socket.emit("request_chat_history", {});
      socket.emit("request_prompt_snapshot", {});
      socket.emit("request_characters", {});
    });

    socket.on("persona_change_failed", (event: { error: string }) => {
      setSwitchingCharacter(false);
      setCharacterActionError(event.error);
    });

    socket.on("vad_config_updated", (event) => {
      const validated = safeParse(VADConfigUpdatedSchema, event, "vad_config_updated");
      if (validated) {
        setVadConfig(validated);
      }
    });

    socket.on("pipeline_config_updated", (event) => {
      setPipelineConfig(event);
    });

    socket.on("memory_config_updated", (event) => {
      const validated = safeParse(MemoryConfigUpdatedSchema, event, "memory_config_updated");
      if (validated) {
        // Merge partial update over existing config (preserves curation fields
        // that aren't included in the basic memory_config_updated event)
        const current = useSettingsStore.getState().memoryConfig;
        setMemoryConfig({ ...current, ...validated });
      }
    });

    // NANO-102: Curation config updates
    socket.on("curation_config_updated", (event) => {
      const validated = safeParse(CurationConfigSchema, event, "curation_config_updated");
      if (validated) {
        const current = useSettingsStore.getState().memoryConfig;
        setMemoryConfig({ ...current, curation: { ...current.curation, ...validated } });
      }
    });

    // NANO-053: Generation parameters
    socket.on("generation_params_updated", (event) => {
      setGenerationConfig(event);
    });

    // NANO-065a: Runtime tools toggle
    // NANO-089 Phase 4: Zod validation before store update
    socket.on("tools_config_updated", (event) => {
      const validated = safeParse(ToolsConfigUpdatedSchema, event, "tools_config_updated");
      if (validated && !validated.error) {
        setToolsConfig({
          master_enabled: validated.master_enabled,
          tools: validated.tools,
        });
      }
      setSavingTools(false);
    });

    // NANO-065b: Runtime LLM provider/model swap
    // NANO-089 Phase 2: Zod validation before store update
    socket.on("llm_config_updated", (event) => {
      const validated = safeParse(LLMConfigUpdatedSchema, event, "llm_config_updated");
      if (validated && !validated.error && validated.provider) {
        setLLMConfig({
          provider: validated.provider,
          model: validated.model ?? "",
          context_size: validated.context_size,
          available_providers: validated.available_providers,
        });
      }
      setSavingLLM(false);
    });

    // NANO-065c: Runtime VLM provider swap (extended NANO-079: unified toggle)
    // NANO-089 Phase 2: Zod validation before store update
    socket.on("vlm_config_updated", (event) => {
      const validated = safeParse(VLMConfigUpdatedSchema, event, "vlm_config_updated");
      if (validated && !validated.error && validated.provider) {
        setVLMConfig({
          provider: validated.provider,
          available_providers: validated.available_providers,
          healthy: validated.healthy,
          // Derive unified state from provider — backend doesn't track this flag
          unified_vlm: validated.provider === "llm",
        });
      }
      setSavingVLM(false);
    });

    // NANO-065b Enhancement: Dashboard Local LLM Launch
    socket.on("llm_server_launched", (event) => {
      if (event.success === null) {
        // Acknowledgment — launch in progress
        return;
      }
      setLaunchingLLM(false);
      if (event.success) {
        setLLMServerRunning(true);
        setLLMLaunchError(null);
        // Auto-swap to llama provider after successful launch
        socket.emit("set_llm_provider", { provider: "llama", config: {} });
      } else {
        setLLMLaunchError(event.error || "Launch failed");
      }
    });

    // NANO-089 Phase 2: Zod validation before store update
    socket.on("local_llm_config", (event) => {
      const validated = safeParse(LocalLLMConfigEventSchema, event, "local_llm_config");
      if (validated) {
        setLocalLLMConfig(validated.config);
        setLLMServerRunning(validated.server_running);
      }
    });

    // Request local LLM config on connect for hydration
    socket.emit("request_local_llm_config", {});

    // NANO-079: Dashboard VLM Launch
    socket.on("vlm_server_launched", (event) => {
      if (event.success === null) {
        // Acknowledgment — launch in progress
        return;
      }
      setLaunchingVLM(false);
      if (event.success) {
        setVLMServerRunning(true);
        setVLMLaunchError(null);
        // Auto-swap to llama provider after successful VLM launch
        socket.emit("set_vlm_provider", { provider: "llama", config: {} });
      } else {
        setVLMLaunchError(event.error || "VLM launch failed");
      }
    });

    // NANO-089 Phase 2: Zod validation before store update
    socket.on("local_vlm_config", (event) => {
      const validated = safeParse(LocalVLMConfigEventSchema, event, "local_vlm_config");
      if (validated) {
        // Merge with existing config to preserve frontend defaults (e.g. model_type)
        const current = useSettingsStore.getState().localVLMConfig;
        setLocalVLMConfig({ ...current, ...validated.config });
        setVLMServerRunning(validated.server_running);
      }
    });

    // Request local VLM config on connect for hydration
    socket.emit("request_local_vlm_config", {});

    // NANO-056: Stimuli system
    // NANO-056b: Zod validation before store update
    socket.on("stimuli_config_updated", (event) => {
      const validated = safeParse(StimuliConfigUpdatedSchema, event, "stimuli_config_updated");
      if (validated) {
        setStimuliConfig(validated);
      }
    });

    socket.on("patience_progress", (event) => {
      setStimuliProgress(event);
    });

    socket.on("stimulus_fired", (event) => {
      console.log("[Stimuli] Stimulus fired:", event.source, event.prompt_text);
      setStimulusActive(true);
    });

    // NANO-069: Audio output level for portrait visualization
    socket.on("audio_level", (event) => {
      setAudioLevel(event.level);
    });

    // NANO-073b: Mic input level for voice overlay visualization
    socket.on("mic_level", (event) => {
      setMicLevel(event.level);
    });

    // NANO-093: Avatar config confirmation (preserve avatar_connected from NANO-097)
    socket.on("avatar_config_updated", (event) => {
      setAvatarConfig({
        ...useSettingsStore.getState().avatarConfig,
        ...event,
      });
    });

    // NANO-097: Avatar renderer connection status
    socket.on("avatar_connection_status", (event: { connected: boolean }) => {
      setAvatarConfig({
        ...useSettingsStore.getState().avatarConfig,
        avatar_connected: event.connected,
      });
    });

    // NANO-110: Tauri app install status check result
    socket.on("tauri_install_status", (event: { avatar: boolean; subtitle: boolean; stream_deck: boolean }) => {
      const current = useSettingsStore.getState().avatarConfig;
      const allInstalled = event.avatar && event.subtitle && event.stream_deck;
      setAvatarConfig({ ...current, tauri_installed: allInstalled });
    });

    // NANO-110: Tauri app build progress (first-time install)
    socket.on("tauri_build_status", (event: { app: string; status: string; message?: string }) => {
      const current = useSettingsStore.getState().avatarConfig;
      const installing = event.status === "building";
      const msg = event.message || "";
      setAvatarConfig({
        ...current,
        tauri_installing: installing,
        tauri_install_message: msg,
        // Mark as installed when ready
        tauri_installed: event.status === "ready" ? true : current.tauri_installed,
      });
    });

    // NANO-060b: VTubeStudio events
    socket.on("vts_config_updated", (event) => {
      setVTSEnabled(event.enabled);
    });

    socket.on("vts_status", (event) => {
      setVTSStatus(event);
    });

    socket.on("vts_hotkeys", (event) => {
      setVTSHotkeys(event.hotkeys);
    });

    socket.on("vts_expressions", (event) => {
      setVTSExpressions(event.expressions);
    });

    // NANO-045c-1: Block config events
    socket.on("block_config_loaded", (event) => {
      setBlockConfig(event);
    });

    socket.on("block_config_updated", (event) => {
      setBlockConfigUpdated(event);
    });

    // Tool events (NANO-025 Phase 7)
    socket.on("tool_invoked", (event) => {
      addToolInvoked(event);
    });

    socket.on("tool_result", (event) => {
      updateToolResult(event);
    });

    // Launch events (NANO-027 Phase 3)
    socket.on("launch_progress", (event) => {
      setLaunchProgress({
        status: event.status === "started" ? "launching" : event.status as "starting" | "loading_config" | "config_loaded" | "complete",
        currentService: event.service,
        message: event.message,
        launchedServices: event.service && event.status === "started"
          ? [] // Will be populated by launch_complete
          : [],
      });
    });

    socket.on("launch_error", (event) => {
      setLaunchError(event.error, event.service);
    });

    socket.on("launch_complete", (event) => {
      setLaunchComplete(event.services);
      // After services complete, orchestrator init starts
      setOrchestratorInitializing();
    });

    // Orchestrator events (NANO-027 Phase 4)
    socket.on("orchestrator_ready", (event) => {
      setOrchestratorReady(event.persona);

      // Backend persists active session via .last_session marker —
      // just request the current state.
      socket.emit("request_chat_history", {});
      socket.emit("request_prompt_snapshot", {});
    });

    socket.on("orchestrator_error", (event) => {
      setOrchestratorError(event.error);
    });

    // Shutdown events (NANO-028)
    socket.on("shutdown_progress", (event) => {
      setShutdownProgress(event.message);
    });

    socket.on("shutdown_complete", () => {
      setShutdownComplete();
      // Reset launcher store so user can re-launch from /launcher
      resetLaunch();
    });

    socket.on("shutdown_error", (event) => {
      setShutdownError(event.error);
    });

    // Character events (NANO-034 Phase 4)
    socket.on("character_list", (event) => {
      setCharacters(event.characters, event.active);
    });

    socket.on("character_detail", (event) => {
      setCharacterDetail(event);
    });

    socket.on("character_created", (event) => {
      setCharacterActionResult("create", event.character_id, event.success);
    });

    socket.on("character_updated", (event) => {
      setCharacterActionResult("update", event.character_id, event.success);
    });

    socket.on("character_deleted", (event) => {
      setCharacterActionResult("delete", event.character_id, event.success);
      if (event.success) {
        removeCharacter(event.character_id);
      }
    });

    socket.on("character_error", (event) => {
      setCharacterActionError(event.error);
    });

    socket.on("avatar_uploaded", (event) => {
      setCharacterActionResult("avatar", event.character_id, event.success);
    });

    socket.on("avatar_data", (event) => {
      setAvatar(event.character_id, event.image_data);
    });

    // Codex events (NANO-034 Phase 5)
    socket.on("global_codex", (event) => {
      setGlobalCodex(event.entries, event.name);
    });

    socket.on("character_codex", (event) => {
      setCharacterCodex(event.character_id, event.entries);
    });

    socket.on("codex_entry_created", (event) => {
      setCodexActionResult("create", event.entry_id, event.character_id, event.success);
    });

    socket.on("codex_entry_updated", (event) => {
      setCodexActionResult("update", event.entry_id, event.character_id, event.success);
    });

    socket.on("codex_entry_deleted", (event) => {
      setCodexActionResult("delete", event.entry_id, event.character_id, event.success);
      if (event.success) {
        removeCodexEntry(event.entry_id, event.character_id);
      }
    });

    socket.on("codex_error", (event) => {
      setCodexActionError(event.error);
    });

    // Memory events (NANO-043 Phase 6)
    socket.on("memory_counts", (event) => {
      setMemoryCounts(
        { global: event.global ?? 0, general: event.general, flashcards: event.flashcards, summaries: event.summaries },
        event.enabled,
      );
    });

    socket.on("memory_list", (event) => {
      setMemories(event.collection, event.memories);
    });

    socket.on("memory_added", (event) => {
      if (event.success && event.memory) {
        addMemory(event.memory, event.collection as MemoryCollectionType);
      }
      setMemoryActionResult("add", event.success, event.error);
    });

    socket.on("memory_edited", (event) => {
      if (event.success && event.old_id && event.memory) {
        editMemory(event.old_id, event.memory);
      }
      setMemoryActionResult("edit", event.success, event.error);
    });

    socket.on("memory_deleted", (event) => {
      if (event.success && event.collection && event.id) {
        removeMemoryFromStore(event.collection as MemoryCollectionType, event.id);
      }
      setMemoryActionResult("delete", event.success, event.error);
    });

    socket.on("memory_promoted", (event) => {
      setMemoryActionResult("promote", event.success, event.error);
      // Refresh counts and lists after promotion
      if (event.success) {
        socket.emit("request_memory_counts", {});
        socket.emit("request_memories", { collection: "general" });
        if (event.deleted_source && event.source_collection) {
          socket.emit("request_memories", { collection: event.source_collection as MemoryCollectionType });
        }
      }
    });

    socket.on("memory_search_results", (event) => {
      setSearchResults(event.results, event.query);
      setMemorySearching(false);
    });

    socket.on("flashcards_cleared", (event) => {
      if (event.success) {
        clearAllFlashcards();
      }
      setMemoryActionResult("clear", event.success, event.error);
    });

    // Connect
    setConnecting(true);
    connectSocket();

    // Cleanup
    return () => {
      socket.off("connect");
      socket.off("disconnect");
      socket.off("connect_error");
      socket.off("state_changed");
      socket.off("transcription");
      socket.off("response");
      socket.off("llm_chunk");
      socket.off("llm_token");
      socket.off("token_usage");
      socket.off("health_status");
      socket.off("config_loaded");
      socket.off("pipeline_error");
      socket.off("prompt_snapshot");
      socket.off("session_list");
      socket.off("session_detail");
      socket.off("session_resumed");
      socket.off("session_deleted");
      socket.off("session_created");
      socket.off("chat_history");
      socket.off("session_summary_generated");
      socket.off("persona_changed");
      socket.off("persona_change_failed");
      socket.off("vad_config_updated");
      socket.off("pipeline_config_updated");
      socket.off("memory_config_updated");
      socket.off("generation_params_updated");
      socket.off("tools_config_updated");
      socket.off("llm_config_updated");
      socket.off("vlm_config_updated");
      socket.off("llm_server_launched");
      socket.off("local_llm_config");
      socket.off("vlm_server_launched");
      socket.off("local_vlm_config");
      socket.off("stimuli_config_updated");
      socket.off("patience_progress");
      socket.off("stimulus_fired");
      socket.off("audio_level");
      socket.off("mic_level");
      socket.off("block_config_loaded");
      socket.off("block_config_updated");
      // NANO-093: Avatar
      socket.off("avatar_config_updated");
      // NANO-097: Avatar connection status
      socket.off("avatar_connection_status");
      // NANO-110: Tauri install
      socket.off("tauri_install_status");
      socket.off("tauri_build_status");
      // NANO-060b: VTubeStudio
      socket.off("vts_config_updated");
      socket.off("vts_status");
      socket.off("vts_hotkeys");
      socket.off("vts_expressions");
      socket.off("tool_invoked");
      socket.off("tool_result");
      socket.off("launch_progress");
      socket.off("launch_error");
      socket.off("launch_complete");
      socket.off("orchestrator_ready");
      socket.off("orchestrator_error");
      socket.off("shutdown_progress");
      socket.off("shutdown_complete");
      socket.off("shutdown_error");
      socket.off("character_list");
      socket.off("character_detail");
      socket.off("character_created");
      socket.off("character_updated");
      socket.off("character_deleted");
      socket.off("character_error");
      socket.off("avatar_uploaded");
      socket.off("avatar_data");
      socket.off("global_codex");
      socket.off("character_codex");
      socket.off("codex_entry_created");
      socket.off("codex_entry_updated");
      socket.off("codex_entry_deleted");
      socket.off("codex_error");
      // NANO-043 Phase 6: Memory Curation GUI
      socket.off("memory_counts");
      socket.off("memory_list");
      socket.off("memory_added");
      socket.off("memory_edited");
      socket.off("memory_deleted");
      socket.off("memory_promoted");
      socket.off("memory_search_results");
      socket.off("flashcards_cleared");
      disconnectSocket();
    };
  }, [
    setConnected,
    setConnecting,
    setError,
    addTransition,
    setTranscription,
    setResponse,
    setTokenUsage,
    setHealth,
    setConfig,
    clearTranscription,
    clearResponse,
    setListeningPaused,
    setSnapshot,
    setSessions,
    setSessionDetail,
    setActionResult,
    removeSession,
    setPersonaFilter,
    loadConfig,
    setVadConfig,
    setPipelineConfig,
    setMemoryConfig,
    setRestartRequired,
    addToolInvoked,
    updateToolResult,
    clearToolActivities,
    setLaunchProgress,
    setLaunchError,
    setLaunchComplete,
    setOrchestratorInitializing,
    setOrchestratorReady,
    setOrchestratorError,
    setShutdownProgress,
    setShutdownComplete,
    setShutdownError,
    resetLaunch,
    setCharacters,
    setCharacterDetail,
    setCharacterActionResult,
    setCharacterActionError,
    removeCharacter,
    setAvatar,
    setSwitchingCharacter,
    setGlobalCodex,
    setCharacterCodex,
    setCodexActionResult,
    setCodexActionError,
    removeCodexEntry,
    setMemoryCounts,
    setMemories,
    setMemoryLoadingList,
    addMemory,
    editMemory,
    removeMemoryFromStore,
    clearAllFlashcards,
    setMemoryActionResult,
    setSearchResults,
    setMemorySearching,
    setBlockConfig,
    setBlockConfigUpdated,
    setAudioLevel,
    setMicLevel,
    setStimulusActive,
    setStimuliConfig,
    setStimuliProgress,
    setToolsConfig,
    setSavingTools,
    setLLMConfig,
    setSavingLLM,
    setVLMConfig,
    setSavingVLM,
    setVTSStatus,
    setVTSEnabled,
    setVTSHotkeys,
    setVTSExpressions,
    addUserMessage,
    addAssistantMessage,
    updateAssistantMessage,
    hydrateHistory,
    clearHistory,
  ]);

  return <>{children}</>;
}
