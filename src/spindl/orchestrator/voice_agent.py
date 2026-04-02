"""
Voice Agent Orchestrator - Central integration layer.

Connects all spindl components into a functioning voice conversation loop:
    Mic -> VAD -> StateMachine -> STT -> LLMPipeline -> TTS -> Speaker

Exposes EventBus and ContextManager for external integration.
"""

import os
import threading
import time
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from ..audio import AudioCapture, AudioPlayback
from ..core import AudioStateMachine, AgentCallbacks, AgentState
from ..core.event_bus import EventBus
from ..core.context_manager import ContextManager
from ..core.events import (
    AudioLevelEvent,
    MicLevelEvent,
    StateChangedEvent,
    TTSStartedEvent,
    TTSCompletedEvent,
    TTSInterruptedEvent,
)
from ..llm import LLMPipeline, PromptBuilder, LLMProviderRegistry, LLMProvider
from ..llm.provider_holder import ProviderHolder
from ..llm.registry import ProviderNotFoundError as LLMProviderNotFoundError
from ..llm.plugins import (
    ConversationHistoryManager,
    HistoryInjector,
    HistoryRecorder,
    SummarizationTrigger,
    BudgetEnforcer,
    TTSCleanupPlugin,
    create_codex_plugins,
)
from ..llm.plugins.reasoning_extractor import ReasoningExtractor
from ..llm.providers.registry import create_default_registry as create_prompt_provider_registry
from ..characters import CharacterLoader
from ..stt import STTProvider, STTProviderRegistry, STTProviderNotFoundError
from ..tools import ToolRegistry, ToolExecutor, create_tool_executor
from ..tts import TTSProvider, TTSProviderRegistry, ProviderNotFoundError

from ..memory.curation_client import CurationClient
from ..memory.embedding_client import EmbeddingClient
from ..memory.memory_store import MemoryStore
from ..memory.rag_injector import RAGInjector
from ..memory.reflection import ReflectionSystem
from ..memory.reflection_monitor import ReflectionMonitor
from ..memory.session_summary import SessionSummaryGenerator
from ..stimuli import StimuliEngine, PatienceModule, TwitchModule
from ..avatar import AvatarToolMoodSubscriber, ONNXEmotionClassifier
from ..vts import VTSDriver
from .callbacks import OrchestratorCallbacks
from .config import OrchestratorConfig

import logging

logger = logging.getLogger(__name__)


class VoiceAgentOrchestrator:
    """
    Main orchestrator for the spindl voice pipeline.

    Connects all components into a functioning voice conversation loop:
    - Audio capture -> VAD -> State machine
    - State machine callbacks -> STT -> LLM -> TTS
    - TTS -> Audio playback
    - Barge-in detection -> Playback interrupt

    Usage:
        config = OrchestratorConfig.from_yaml("config/spindl.yaml")
        agent = VoiceAgentOrchestrator(config)

        agent.start()
        # ... conversation happens ...
        agent.stop()
    """

    def __init__(self, config: OrchestratorConfig):
        """
        Initialize the voice agent orchestrator.

        Args:
            config: OrchestratorConfig with all settings.
        """
        self._config = config
        self._running = False
        self._setup_complete = False

        # Service clients (initialized in _setup)
        self._stt: Optional[STTProvider] = None  # NANO-061a: provider-based
        self._tts_provider: Optional[TTSProvider] = None  # NANO-015: provider-based
        self._llm_provider: Optional[LLMProvider] = None  # NANO-018/019: provider-based
        self._pipeline: Optional[LLMPipeline] = None

        # Tool system (NANO-024)
        self._tool_registry: Optional[ToolRegistry] = None
        self._tool_executor: Optional[ToolExecutor] = None

        # Codex system (NANO-037) - stored for hot-reload (NANO-036)
        self._codex_manager = None

        # Runtime generation parameter overrides (NANO-053)
        self._runtime_generation_overrides: Optional[dict] = None

        # Memory system (NANO-043)
        self._memory_store: Optional[MemoryStore] = None
        self._embedding_client: Optional[EmbeddingClient] = None

        # Audio components
        self._capture: Optional[AudioCapture] = None
        self._playback: Optional[AudioPlayback] = None
        self._state_machine: Optional[AudioStateMachine] = None

        # Orchestrator components
        self._callbacks: Optional[OrchestratorCallbacks] = None
        self._persona: Optional[dict] = None
        self._history_manager: Optional[ConversationHistoryManager] = None

        # Event system
        self._event_bus: Optional[EventBus] = None
        self._context_manager: Optional[ContextManager] = None

        # Stimuli system (NANO-056)
        self._stimuli_engine: Optional[StimuliEngine] = None

        # Addressing-others state (NANO-110)
        self._addressing_others: bool = False
        self._addressing_others_context_id: Optional[str] = None
        self._addressing_others_prompt: Optional[str] = None  # one-shot for next pipeline call

        # VTubeStudio driver (NANO-060)
        self._vts_driver: Optional[VTSDriver] = None

        # Avatar tool-mood subscriber (NANO-093)
        self._avatar_tool_mood: Optional[AvatarToolMoodSubscriber] = None

        # NANO-073b: Mic input level monitoring
        self._current_mic_rms: float = 0.0
        self._mic_level_thread: Optional[threading.Thread] = None

        # NANO-108: Mic health change callback (set by GUI server)
        self._on_health_change_callback: Optional[Callable] = None

    def _setup(self) -> None:
        """Initialize all components."""
        if self._setup_complete:
            return

        # Ensure conversations directory exists
        conversations_path = Path(self._config.conversations_dir)
        conversations_path.mkdir(parents=True, exist_ok=True)

        # Load character (NANO-034: ST V2 Character Cards)
        loader = CharacterLoader(self._config.characters_dir)
        self._persona = loader.load_as_dict(self._config.character_id)

        # STT: provider-based architecture (NANO-061a)
        stt_config = self._config.stt_config
        stt_registry = STTProviderRegistry(plugin_paths=stt_config.plugin_paths)

        try:
            stt_provider_class = stt_registry.get_provider_class(stt_config.provider)
        except STTProviderNotFoundError as e:
            raise RuntimeError(f"STT initialization failed: {e}")

        # Validate provider config
        config_errors = stt_provider_class.validate_config(stt_config.provider_config)
        if config_errors:
            raise RuntimeError(
                f"STT provider config invalid: {'; '.join(config_errors)}"
            )

        # Instantiate and initialize provider
        self._stt = stt_provider_class()
        self._stt.initialize(stt_config.provider_config)

        # TTS: provider-based architecture (NANO-015)
        tts_config = self._config.tts_config
        tts_registry = TTSProviderRegistry(plugin_paths=tts_config.plugin_paths)

        try:
            tts_provider_class = tts_registry.get_provider_class(tts_config.provider)
        except ProviderNotFoundError as e:
            raise RuntimeError(f"TTS initialization failed: {e}")

        # Validate provider config
        config_errors = tts_provider_class.validate_config(tts_config.provider_config)
        if config_errors:
            raise RuntimeError(
                f"TTS provider config invalid: {'; '.join(config_errors)}"
            )

        # Instantiate and initialize provider
        self._tts_provider = tts_provider_class()
        self._tts_provider.initialize(tts_config.provider_config)

        # LLM: provider-based architecture (NANO-018)
        llm_config = self._config.llm_config
        llm_registry = LLMProviderRegistry(plugin_paths=llm_config.plugin_paths)
        self._llm_registry = llm_registry  # Stored for runtime swap (NANO-065b)

        try:
            llm_provider_class = llm_registry.get_provider_class(llm_config.provider)
        except LLMProviderNotFoundError as e:
            raise RuntimeError(f"LLM initialization failed: {e}")

        # Validate provider config
        llm_config_errors = llm_provider_class.validate_config(llm_config.provider_config)
        if llm_config_errors:
            raise RuntimeError(
                f"LLM provider config invalid: {'; '.join(llm_config_errors)}"
            )

        # NANO-087: flag unified vision mode so LLM provider can pin to slot 0
        if self._config.vlm_config and self._config.vlm_config.provider == "llm":
            llm_config.provider_config["unified_vision"] = True

        # Instantiate and initialize provider, wrapped in ProviderHolder (NANO-065b)
        raw_llm_provider = llm_provider_class()
        raw_llm_provider.initialize(llm_config.provider_config)
        self._provider_holder = ProviderHolder(raw_llm_provider, llm_registry)
        self._llm_provider = self._provider_holder

        # Create history manager (shared across plugins)
        self._history_manager = ConversationHistoryManager(
            conversations_dir=self._config.conversations_dir,
            resume_session=self._config.resume_session,
            debug=self._config.debug,
        )

        # Create plugins using LLMProvider (NANO-019 Phase 3)
        # All plugins now use the provider interface for tokenization and generation
        summarizer = SummarizationTrigger(
            llm_provider=self._llm_provider,
            manager=self._history_manager,
            threshold=self._config.summarization_threshold,
            reserve_tokens=self._config.summarization_reserve_tokens,
            live_mode=self._config.memory_config.live_mode,
        )

        budget_enforcer = BudgetEnforcer(
            llm_provider=self._llm_provider,
            manager=self._history_manager,
            strategy=self._config.budget_strategy,
            response_reserve=self._config.response_reserve,
        )

        history_injector = HistoryInjector(self._history_manager)
        history_recorder = HistoryRecorder(self._history_manager)
        tts_cleanup = TTSCleanupPlugin()

        # Codex: Create activator and cooldown plugins (NANO-037)
        # Must be registered BEFORE budget_enforcer so codex tokens are counted
        # Store manager for hot-reload (NANO-036)
        self._codex_manager, codex_activator, codex_cooldown = create_codex_plugins(
            characters_dir=self._config.characters_dir,
            character_id=self._config.character_id,
        )

        # Memory: Initialize embedding client, memory store, RAG injector,
        # reflection system, and session summary generator (NANO-043 Phase 1-4)
        rag_injector = None
        self._rag_injector = None
        reflection_monitor = None
        self._reflection_system = None
        self._session_summary_generator = None
        if self._config.memory_config.enabled:
            self._embedding_client = EmbeddingClient(
                base_url=self._config.memory_config.embedding_base_url,
                timeout=self._config.memory_config.embedding_timeout,
            )

            # Memory dir lives alongside the character's data
            character_memory_dir = str(
                Path(self._config.characters_dir)
                / self._config.character_id
                / "memory"
            )
            resolved_memory_dir = str(Path(character_memory_dir).resolve())

            # NANO-102: Initialize curation client if configured
            curation_client = None
            curation_cfg = self._config.memory_config.curation
            if curation_cfg.enabled and curation_cfg.api_key:
                try:
                    curation_client = CurationClient(
                        api_key=curation_cfg.api_key,
                        model=curation_cfg.model,
                        system_prompt=curation_cfg.prompt,
                        timeout=curation_cfg.timeout,
                    )
                    logger.info("Memory curation client initialized (model=%s)", curation_cfg.model)
                except Exception as e:
                    logger.warning("Failed to initialize curation client: %s", e)

            # Global memory dir: sibling to character dirs (NANO-105)
            global_memory_dir = str(
                Path(self._config.characters_dir) / "global" / "memory"
            )

            # NANO-107: pass scoring weights from config
            mem_cfg = self._config.memory_config
            scoring_config = {
                "w_relevance": mem_cfg.scoring_w_relevance,
                "w_recency": mem_cfg.scoring_w_recency,
                "w_importance": mem_cfg.scoring_w_importance,
                "w_frequency": mem_cfg.scoring_w_frequency,
                "decay_base": mem_cfg.scoring_decay_base,
            }

            self._memory_store = MemoryStore(
                character_id=self._config.character_id,
                memory_dir=character_memory_dir,
                embedding_client=self._embedding_client,
                dedup_threshold=self._config.memory_config.dedup_threshold,
                curation_client=curation_client,
                global_memory_dir=global_memory_dir,
                scoring_config=scoring_config,
            )
            rag_injector = RAGInjector(
                memory_store=self._memory_store,
                top_k=self._config.memory_config.rag_top_k,
                relevance_threshold=self._config.memory_config.relevance_threshold,
                rag_prefix=self._config.prompt_config.rag_prefix,
                rag_suffix=self._config.prompt_config.rag_suffix,
            )
            self._rag_injector = rag_injector

            # Reflection system: async memory entry generation (Phase 3, NANO-104)
            self._reflection_system = ReflectionSystem(
                llm_provider=self._llm_provider,
                memory_store=self._memory_store,
                history_manager=self._history_manager,
                reflection_interval=self._config.memory_config.reflection_interval,
                max_tokens=self._config.memory_config.reflection_max_tokens,
                reflection_prompt=self._config.memory_config.reflection_prompt,
                reflection_system_message=self._config.memory_config.reflection_system_message,
                reflection_delimiter=self._config.memory_config.reflection_delimiter,
            )
            reflection_monitor = ReflectionMonitor(self._reflection_system)

            # Session summary generator: on-demand via GUI (Phase 4)
            self._session_summary_generator = SessionSummaryGenerator(
                llm_provider=self._llm_provider,
                memory_store=self._memory_store,
                max_tokens=self._config.memory_config.session_summary_max_tokens,
            )

            logger.info(
                "Memory system initialized for character '%s' "
                "(top_k=%d, reflection_interval=%d, live_mode=%s)",
                self._config.character_id,
                self._config.memory_config.rag_top_k,
                self._config.memory_config.reflection_interval,
                self._config.memory_config.live_mode,
            )

        # Assemble pipeline with provider-based prompt builder (NANO-014 Session 5)
        # Pipeline now uses LLMProvider directly (NANO-019)
        # Note: Vision is no longer injected into prompts—use tools instead (NANO-024)
        prompt_provider_registry = create_prompt_provider_registry()
        self._pipeline = LLMPipeline(self._llm_provider, PromptBuilder(prompt_provider_registry))

        # NANO-045a: Block-based prompt assembly (default on).
        # Explicit `prompt_blocks: false` in config disables it (legacy template mode).
        if self._config.prompt_blocks is False:
            pass  # Legacy mode — no block config
        elif self._config.prompt_blocks:
            self._pipeline.set_block_config(self._config.prompt_blocks)
        else:
            self._pipeline.set_block_config({})

        # NANO-045d: Codex injection wrappers from config
        self._pipeline.set_codex_wrappers(
            codex_prefix=self._config.prompt_config.codex_prefix,
            codex_suffix=self._config.prompt_config.codex_suffix,
        )

        # NANO-052 follow-up: Example dialogue wrappers from config
        self._pipeline.set_example_dialogue_wrappers(
            prefix=self._config.prompt_config.example_dialogue_prefix,
            suffix=self._config.prompt_config.example_dialogue_suffix,
        )

        # Create event bus for publish/subscribe communication
        # Note: Moved before _setup_tools() so tool executor can emit events (NANO-025 Phase 7)
        self._event_bus = EventBus()

        # Tools: Initialize tool registry and executor (NANO-024)
        self._setup_tools()

        # PreProcessors (order matters!)
        # 1. SummarizationTrigger - checks if summarization needed
        # 2. CodexActivator - activate codex entries based on user input (NANO-037)
        # 3. RAGInjector - query memories, stage for injection (NANO-043)
        # 4. BudgetEnforcer - enforces hard limits (includes codex + RAG tokens)
        # 5. HistoryInjector - injects remaining history into messages
        self._pipeline.register_pre_processor(summarizer)
        self._pipeline.register_pre_processor(codex_activator)
        if rag_injector:
            self._pipeline.register_pre_processor(rag_injector)
        self._pipeline.register_pre_processor(budget_enforcer)
        self._pipeline.register_pre_processor(history_injector)

        # PostProcessors
        # 0. ReasoningExtractor - strip inline <think> blocks (NANO-042)
        # 1. HistoryRecorder - store turn to JSONL
        # 2. ReflectionMonitor - signal reflection system (NANO-043 Phase 3)
        # 3. CodexCooldown - advance codex state after turn (NANO-037)
        # 4. TTSCleanupPlugin - strip *actions*, (parens), etc. for TTS
        self._pipeline.register_post_processor(ReasoningExtractor())
        self._pipeline.register_post_processor(history_recorder)
        if reflection_monitor:
            self._pipeline.register_post_processor(reflection_monitor)
        self._pipeline.register_post_processor(codex_cooldown)
        self._pipeline.register_post_processor(tts_cleanup)

        # Create audio components
        self._capture = AudioCapture(
            chunk_size=self._config.chunk_samples,
            on_chunk=self._audio_callback,
            on_health_change=self._on_mic_health_change,
        )

        self._playback = AudioPlayback(
            on_complete=self._on_playback_complete,
            on_interrupt=self._on_playback_interrupt,
            on_audio_level=self._on_audio_level,
        )

        # Configure playback sample rate from TTS provider properties (NANO-015)
        # Provider declares its output format, playback adapts
        tts_props = self._tts_provider.get_properties()
        self._playback.configure(
            sample_rate=tts_props.sample_rate,
            channels=tts_props.channels,
        )

        # Create state machine with VAD
        self._state_machine = AudioStateMachine(
            vad_threshold=self._config.vad_threshold,
            min_speech_ms=self._config.min_speech_ms,
            min_silence_ms=self._config.min_silence_ms,
            speech_pad_ms=self._config.speech_pad_ms,
        )

        # Create context manager for multimodal context aggregation
        self._context_manager = ContextManager(event_bus=self._event_bus)

        # Create orchestrator callbacks with event system (NANO-015: provider-based TTS)
        self._callbacks = OrchestratorCallbacks(
            stt_client=self._stt,
            tts_provider=self._tts_provider,
            llm_pipeline=self._pipeline,
            persona=self._persona,
            on_response_ready=self._on_response_ready,
            on_barge_in_triggered=self._handle_barge_in,
            on_empty_transcription=self._on_empty_transcription,
            on_error=self._on_processing_error,
            on_pause_listening=lambda: self.pause_listening(),
            on_resume_listening=lambda: self.resume_listening(),
            event_bus=self._event_bus,
            context_manager=self._context_manager,
            context_limit_getter=lambda: (
                self._provider_holder.provider.get_properties().context_length or 8192
            ) if self._provider_holder else 8192,
        )

        # NANO-076: Wire session file getter for snapshot sidecar persistence
        self._callbacks.set_session_file_getter(lambda: self.session_file)

        # NANO-110: Wire addressing-others state getters
        self._callbacks._is_addressing_others = lambda: self._addressing_others
        self._callbacks._consume_addressing_others_prompt = self._consume_addressing_others_prompt

        # Wrapper to emit state change events
        def on_state_change_with_event(transition):
            # Original logging
            self._callbacks.on_state_change(transition)
            # Emit event for external subscribers
            self._event_bus.emit(
                StateChangedEvent(
                    from_state=transition.from_state.value,
                    to_state=transition.to_state.value,
                    trigger=transition.trigger,
                )
            )

        # Wire callbacks to state machine
        self._state_machine._callbacks = AgentCallbacks(
            on_state_change=on_state_change_with_event,
            on_user_speech_start=None,  # Not needed for orchestration
            on_user_speech_end=self._callbacks.on_user_speech_end,
            on_barge_in=self._callbacks.on_barge_in,
            on_processing_complete=None,  # Not needed
            on_system_speech_end=None,  # Handled via playback callbacks
        )

        # Stimuli engine: autonomous stimulus system (NANO-056)
        self._stimuli_engine = None
        stimuli_cfg = self._config.stimuli_config
        self._stimuli_engine = StimuliEngine(
            state_machine=self._state_machine,
            callbacks=self._callbacks,
            event_bus=self._event_bus,
            enabled=stimuli_cfg.enabled,
            is_speaking=lambda: self._playback.is_playing,
        )
        # Register PATIENCE module
        patience = PatienceModule(
            timeout_seconds=stimuli_cfg.patience_seconds,
            prompt=stimuli_cfg.patience_prompt,
            enabled=stimuli_cfg.patience_enabled,
        )
        self._stimuli_engine.register_module(patience)

        # Register Twitch module if configured (NANO-056b)
        # Auth can come from config or TWITCH_APP_ID/TWITCH_APP_SECRET env vars
        if stimuli_cfg.twitch_channel:
            twitch = TwitchModule(
                channel=stimuli_cfg.twitch_channel,
                app_id=stimuli_cfg.twitch_app_id,
                app_secret=stimuli_cfg.twitch_app_secret,
                buffer_size=stimuli_cfg.twitch_buffer_size,
                max_message_length=stimuli_cfg.twitch_max_message_length,
                prompt_template=stimuli_cfg.twitch_prompt_template,
                enabled=stimuli_cfg.twitch_enabled,
            )
            self._stimuli_engine.register_module(twitch)
            logger.info(
                "Twitch module registered (channel=%s, enabled=%s)",
                stimuli_cfg.twitch_channel,
                stimuli_cfg.twitch_enabled,
            )

        logger.info(
            "Stimuli engine created (enabled=%s, patience=%.1fs)",
            stimuli_cfg.enabled,
            stimuli_cfg.patience_seconds,
        )

        # VTubeStudio driver (NANO-060)
        vts_cfg = self._config.vtubestudio_config
        if vts_cfg.enabled:
            self._vts_driver = VTSDriver(
                config=vts_cfg,
                event_bus=self._event_bus,
            )
            logger.info("[VTS] Driver created (host=%s:%d)", vts_cfg.host, vts_cfg.port)

        # Avatar tool-mood subscriber (NANO-093)
        if self._config.avatar_config.enabled:
            self._avatar_tool_mood = AvatarToolMoodSubscriber(
                event_bus=self._event_bus,
            )
            logger.info("Avatar bridge enabled")

        # Emotion classifier (NANO-094)
        avatar_cfg = self._config.avatar_config
        if avatar_cfg.emotion_classifier == "classifier":
            # Resolve model_dir relative to project root (src/spindl/orchestrator → ../../..)
            model_dir = avatar_cfg.emotion_model_path
            if not os.path.isabs(model_dir):
                project_root = Path(__file__).parent.parent.parent.parent
                model_dir = str(project_root / model_dir)
            classifier = ONNXEmotionClassifier(
                model_dir=model_dir,
                confidence_threshold=avatar_cfg.emotion_confidence_threshold,
            )
            # Eager init — download model + load ONNX session now, not on first classify
            if classifier._init():
                self._callbacks.set_emotion_classifier(classifier)
                logger.info("Emotion classifier ready (ONNX, model_dir=%s, threshold=%.2f)", model_dir, avatar_cfg.emotion_confidence_threshold)
            else:
                logger.warning("Emotion classifier failed to initialize — disabled")

        self._setup_complete = True

    def _setup_tools(self) -> None:
        """
        Initialize the tool system if enabled in config.

        Creates ToolRegistry, initializes configured tools, and wires
        the ToolExecutor into the LLM pipeline.
        """
        tools_config = self._config.tools_config

        if not tools_config.enabled:
            logger.debug("Tools disabled in config")
            return

        self._init_tool_system()

    def _init_tool_system(self) -> None:
        """
        Core tool system initialization (registry, tools, executor, pipeline wiring).

        Called from _setup_tools() at startup and from update_tools_config()
        for lazy on-demand initialization when tools were disabled at launch.
        """
        tools_config = self._config.tools_config

        # Create tool registry if not already present
        if not self._tool_registry:
            self._tool_registry = ToolRegistry(plugin_paths=tools_config.plugin_paths)

        # Build tools config with VLM config injected for screen_vision
        # The screen_vision tool needs access to the VLM provider config
        tools_raw = tools_config.to_raw_dict()

        # Inject VLM config into screen_vision if it exists
        # Skip injection entirely when VLM is disabled (provider: "none") —
        # stale vlm_provider in tools config must not override the VLM section
        vlm_config_section = self._config.vlm_config
        vlm_disabled = (
            not vlm_config_section
            or vlm_config_section.provider == "none"
        )

        if "screen_vision" in tools_raw.get("tools", {}) and vlm_disabled:
            # VLM is disabled — force screen_vision off regardless of stale
            # vlm_provider in tools config. Otherwise it inits with the LLM
            # endpoint and falsely reports VLM healthy. (Bug #12)
            tools_raw["tools"]["screen_vision"]["enabled"] = False
            logger.debug("VLM disabled (provider='none') — disabling screen_vision tool")

        if "screen_vision" in tools_raw.get("tools", {}) and not vlm_disabled:
            screen_vision_config = tools_raw["tools"]["screen_vision"]
            vlm_provider = screen_vision_config.get("vlm_provider", "llama")

            if vlm_provider == "llm":
                # Unified mode (NANO-030): build config from LLM provider + VLM overrides
                llm_cfg = self._config.llm_config.provider_config

                # Construct URL from LLM config (url preferred, fall back to host:port)
                llm_url = llm_cfg.get("url")
                if not llm_url:
                    llm_host = llm_cfg.get("host", "127.0.0.1")
                    llm_port = llm_cfg.get("port", 5557)
                    llm_url = f"http://{llm_host}:{llm_port}"

                # Base config from LLM provider
                vlm_provider_config = {
                    "url": llm_url,
                    "api_key": llm_cfg.get("api_key"),
                    "model": llm_cfg.get("model", "local-llm"),
                }

                # Apply vision-specific overrides from vlm.providers.llm
                vlm_overrides = vlm_config_section.providers.get("llm", {})
                vlm_provider_config.update(vlm_overrides)

                logger.debug(f"Unified vision mode: routing through LLM at {llm_url}")
            else:
                vlm_provider_config = vlm_config_section.providers.get(vlm_provider, {})

            # Also inject plugin paths from vlm config
            screen_vision_config["vlm_config"] = vlm_provider_config
            screen_vision_config["vlm_plugin_paths"] = vlm_config_section.plugin_paths

            logger.debug(f"Injected VLM config for screen_vision tool: provider={vlm_provider}")

        # Initialize tools from config
        self._tool_registry.initialize_tools(tools_raw)

        # Check if any tools are enabled
        enabled_tools = self._tool_registry.get_enabled_tools()
        if not enabled_tools:
            logger.info("No tools enabled after initialization")
            return

        # Create tool executor (NANO-025 Phase 7: pass event_bus for tool visibility)
        self._tool_executor = create_tool_executor(
            registry=self._tool_registry,
            max_iterations=tools_config.max_iterations,
            event_bus=self._event_bus,
        )

        # Wire executor into pipeline
        if self._tool_executor and self._pipeline:
            self._pipeline.set_tool_executor(self._tool_executor)
            tool_names = [t.name for t in enabled_tools]
            logger.info(f"Tool system initialized with {len(enabled_tools)} tool(s): {tool_names}")
            logger.info("Tools enabled: %s", ", ".join(tool_names))

    def _audio_callback(self, chunk: np.ndarray) -> None:
        """
        Called for each audio chunk from capture.

        Feeds audio to state machine which handles:
        - VAD processing
        - Audio buffering during speech
        - State transitions
        """
        if not self._running:
            return

        # NANO-073b: Compute mic RMS (volatile write, read by monitor thread)
        self._current_mic_rms = float(np.sqrt(np.mean(chunk ** 2)))

        # Feed to state machine (handles VAD internally)
        self._state_machine.process_audio(chunk)

    def _on_response_ready(self, audio: np.ndarray) -> None:
        """
        Called when TTS audio is ready to play.

        Args:
            audio: TTS audio at provider's native sample rate.
                   AudioPlayback is pre-configured to match this rate.
        """
        if not self._running:
            return

        # No resampling needed - AudioPlayback.configure() was called in _setup()
        # with the provider's native sample rate (e.g., 24kHz for Kokoro)

        # Emit TTS started event
        duration = len(audio) / self._playback.sample_rate
        if self._event_bus:
            self._event_bus.emit(TTSStartedEvent(duration=duration))

        # Transition state machine
        self._state_machine.start_system_speaking()

        # Start playback
        self._playback.play(audio)

    def _on_playback_complete(self) -> None:
        """Called when TTS playback finishes naturally."""
        if not self._running:
            return

        # Emit TTS completed event
        if self._event_bus:
            self._event_bus.emit(TTSCompletedEvent())

        self._state_machine.finish_system_speaking()

    def _on_playback_interrupt(self) -> None:
        """Called when playback is interrupted (barge-in stop)."""
        # Emit TTS interrupted event
        if self._event_bus:
            self._event_bus.emit(TTSInterruptedEvent())

        # State machine already transitioned via barge-in callback

    def _on_audio_level(self, level: float) -> None:
        """Called with RMS audio level during playback (NANO-069)."""
        if self._event_bus:
            self._event_bus.emit(AudioLevelEvent(level=level))

    def _on_mic_health_change(self, health: str) -> None:
        """Called when AudioCapture stream health changes (NANO-108)."""
        logger.info("Mic stream health changed: %s", health)
        if self._on_health_change_callback:
            self._on_health_change_callback()

    def _mic_level_monitor(self) -> None:
        """Background thread emitting mic input RMS at ~50ms intervals (NANO-073b)."""
        MIC_LEVEL_INTERVAL = 0.05  # 50ms, matches TTS output level rate
        last_emit_time = 0.0
        was_speaking = False

        while self._running:
            now = time.monotonic()
            if (now - last_emit_time) >= MIC_LEVEL_INTERVAL:
                last_emit_time = now

                # NANO-108 Layer 2: Check for audio input timeout
                if self._state_machine is not None:
                    self._state_machine.check_audio_timeout()

                is_speaking = (
                    self._state_machine is not None
                    and self._state_machine.state == AgentState.USER_SPEAKING
                )

                if is_speaking:
                    level = min(1.0, self._current_mic_rms)
                    if self._event_bus:
                        self._event_bus.emit(MicLevelEvent(level=level))
                    was_speaking = True
                elif was_speaking:
                    # Emit zero when transitioning out of speaking
                    if self._event_bus:
                        self._event_bus.emit(MicLevelEvent(level=0.0))
                    was_speaking = False

            time.sleep(0.01)

    def _handle_barge_in(self) -> None:
        """Handle barge-in by stopping playback."""
        self._playback.stop()

    def set_addressing_others(self, context_id: str) -> None:
        """
        Activate addressing-others mode (NANO-110).

        Suppresses voice pipeline calls and stops TTS if playing.
        Called when Stream Deck button is held or global hotkey pressed.

        Args:
            context_id: ID of the addressing context (e.g., "ctx_0").
        """
        self._addressing_others = True
        self._addressing_others_context_id = context_id

        # Stop TTS if persona is speaking — same as barge-in but no trigger
        if (
            self._playback
            and self._state_machine
            and self._state_machine.state == AgentState.SYSTEM_SPEAKING
        ):
            self._playback.stop()
            # Transition to LISTENING, not USER_SPEAKING — we're suppressing input
            self._state_machine.finish_system_speaking()

        # Pause stimuli engine — reuse "user is busy" semantic
        if self._stimuli_engine:
            self._stimuli_engine.user_typing = True

        logger.info("[NANO-110] Addressing others activated: context=%s", context_id)

    def clear_addressing_others(self) -> None:
        """
        Deactivate addressing-others mode (NANO-110).

        Resolves the prompt from the active context and stores it as a one-shot
        for the next pipeline call. Resumes stimuli engine.
        Called when Stream Deck button is released or global hotkey released.
        """
        from .config import AddressingContext
        from ..llm.prompt_library import DEFAULT_ADDRESSING_OTHERS_PROMPT

        # Resolve prompt from active context
        context_id = self._addressing_others_context_id
        resolved_prompt = DEFAULT_ADDRESSING_OTHERS_PROMPT
        if context_id:
            for ctx in self._config.stimuli_config.addressing_others_contexts:
                if ctx.id == context_id:
                    if ctx.prompt.strip():
                        resolved_prompt = ctx.prompt.strip()
                    break

        # Store as one-shot — consumed by next pipeline.run() call
        self._addressing_others_prompt = resolved_prompt

        self._addressing_others = False
        self._addressing_others_context_id = None

        # Resume stimuli engine
        if self._stimuli_engine:
            self._stimuli_engine.user_typing = False

        logger.info(
            "[NANO-110] Addressing others deactivated: context=%s, prompt=%s",
            context_id,
            resolved_prompt[:60] + "..." if len(resolved_prompt) > 60 else resolved_prompt,
        )

    @property
    def addressing_others(self) -> bool:
        """Whether addressing-others mode is active (NANO-110)."""
        return self._addressing_others

    @property
    def addressing_others_context_id(self) -> Optional[str]:
        """Active addressing-others context ID, or None (NANO-110)."""
        return self._addressing_others_context_id

    def _consume_addressing_others_prompt(self) -> Optional[str]:
        """
        Consume and return the one-shot addressing-others prompt (NANO-110).

        Returns the resolved prompt string if set, then clears it.
        Called by callbacks before pipeline.run().
        """
        prompt = self._addressing_others_prompt
        self._addressing_others_prompt = None
        return prompt

    def _on_empty_transcription(self) -> None:
        """Handle empty transcription (noise/silence detected as speech)."""
        # Return to listening state
        if self._state_machine.state == AgentState.PROCESSING:
            # Manually transition back to LISTENING since we have no response
            self._state_machine._transition(AgentState.LISTENING, "empty_transcription")

    def _on_processing_error(self, stage: str, error: Exception) -> None:
        """
        Handle processing errors.

        Args:
            stage: Which stage failed ("stt", "llm", "tts").
            error: The exception that occurred.
        """
        logger.error("%s error: %s", stage.upper(), error)

        # Return to listening state on error
        if self._state_machine.state == AgentState.PROCESSING:
            self._state_machine._transition(AgentState.LISTENING, f"{stage}_error")

    def start(self) -> None:
        """Start the voice agent."""
        if self._running:
            return

        self._setup()
        self._running = True

        # Start capture
        self._capture.start()

        # Activate state machine
        self._state_machine.activate()

        # Eagerly bootstrap the history session so session_file is available
        # immediately (for reflection, GUI active-session badge, etc.)
        # Without this, session_file is None until the first pipeline message.
        persona_id = self._persona.get("id") if self._persona else None
        if persona_id and self._history_manager:
            self._history_manager.ensure_session(persona_id)

        # Start reflection system background thread (NANO-043 Phase 3)
        if self._reflection_system:
            session_id = None
            if self._history_manager.session_file:
                session_id = self._history_manager.session_file.stem
            self._reflection_system.start(session_id=session_id)

        # NANO-107: set session_id for session-scoped memory retrieval
        if self._memory_store and self._history_manager and self._history_manager.session_file:
            self._memory_store.set_session_id(self._history_manager.session_file.stem)

        # Start stimuli engine (NANO-056)
        if self._stimuli_engine:
            self._stimuli_engine.start()

        # Start mic level monitor (NANO-073b)
        self._mic_level_thread = threading.Thread(
            target=self._mic_level_monitor, daemon=True
        )
        self._mic_level_thread.start()

        # Start VTubeStudio driver (NANO-060)
        if self._vts_driver:
            self._vts_driver.start()

        # Start avatar tool-mood subscriber (NANO-093)
        if self._avatar_tool_mood:
            self._avatar_tool_mood.start()

        persona_name = self._persona.get("name", "Unknown")
        logger.info("Started. Persona: %s", persona_name)

        if self._history_manager.session_file:
            logger.info("Session: %s", self._history_manager.session_file)

    def stop(self) -> None:
        """Stop the voice agent."""
        if not self._running:
            return

        self._running = False

        # Stop avatar tool-mood subscriber (NANO-093)
        if self._avatar_tool_mood:
            self._avatar_tool_mood.stop()

        # Stop VTubeStudio driver (NANO-060) — before stimuli/state machine
        if self._vts_driver:
            self._vts_driver.stop()

        # Stop stimuli engine (NANO-056) — before state machine deactivation
        if self._stimuli_engine:
            self._stimuli_engine.stop()

        # Stop reflection system background thread (NANO-043 Phase 3)
        if self._reflection_system:
            self._reflection_system.stop()

        # Deactivate state machine
        if self._state_machine:
            self._state_machine.deactivate()

        # Stop mic level monitor (NANO-073b)
        if self._mic_level_thread and self._mic_level_thread.is_alive():
            self._mic_level_thread.join(timeout=0.2)
            self._mic_level_thread = None

        # Stop audio
        if self._capture:
            self._capture.stop()
        if self._playback:
            self._playback.stop()

        # Shutdown tools (NANO-024)
        if self._tool_registry:
            self._tool_registry.shutdown()

        logger.info("Stopped.")

    def pause_listening(self) -> bool:
        """
        Pause audio input without stopping the agent.

        Transitions to IDLE state, stopping VAD processing but preserving
        session state. The agent can be resumed with resume_listening().

        Returns:
            True if successfully paused, False if already paused or not running.
        """
        if not self._running or not self._state_machine:
            return False

        current_state = self._state_machine.state
        if current_state == AgentState.IDLE:
            return False  # Already paused

        # Only pause if not in the middle of processing
        if current_state == AgentState.PROCESSING:
            logger.info("Cannot pause while processing - waiting for completion")
            return False

        self._state_machine.deactivate()
        logger.info("Listening paused")
        return True

    def resume_listening(self) -> bool:
        """
        Resume audio input after pausing.

        Transitions from IDLE to LISTENING state, reactivating VAD processing.

        Returns:
            True if successfully resumed, False if already listening or not running.
        """
        if not self._running or not self._state_machine:
            return False

        if self._state_machine.state != AgentState.IDLE:
            return False  # Already listening

        self._state_machine.activate()
        logger.info("Listening resumed")
        return True

    @property
    def is_listening_paused(self) -> bool:
        """Whether listening is currently paused (agent is in IDLE state while running)."""
        if not self._running or not self._state_machine:
            return False
        return self._state_machine.state == AgentState.IDLE

    @property
    def memory_store(self) -> Optional[MemoryStore]:
        """Get the MemoryStore for memory CRUD operations (NANO-043 Phase 6)."""
        return self._memory_store

    def health_check(self) -> dict[str, bool]:
        """
        Check if all services are available.

        Returns:
            Dict mapping service name to availability status.
        """
        self._setup()

        return {
            "stt": self._stt.health_check() if self._stt else False,
            "tts": self._tts_provider.health_check() if self._tts_provider else False,
            "llm": self._check_llm_health(),
            "vlm": self._check_vlm_health(),
            "embedding": self._embedding_client.health_check() if self._embedding_client else False,
            "mic": self._capture.stream_health if self._capture else "down",
        }

    def update_memory_config(
        self,
        top_k: Optional[int] = None,
        relevance_threshold: Optional[float] = ...,
        dedup_threshold: Optional[float] = ...,
        reflection_interval: Optional[int] = None,
        reflection_prompt: Optional[str] = ...,
        reflection_system_message: Optional[str] = ...,
        reflection_delimiter: Optional[str] = None,
    ) -> None:
        """
        Update memory/RAG config at runtime (no pipeline rebuild needed).

        Args:
            top_k: New max results. None = keep current.
            relevance_threshold: New distance threshold. None = disable filtering.
                                 Ellipsis (...) = keep current.
            dedup_threshold: New dedup L2 distance threshold. None = disable dedup.
                             Ellipsis (...) = keep current.
            reflection_interval: New reflection interval (turns). None = keep current.
            reflection_prompt: Custom prompt template. None = use built-in default.
                              Ellipsis (...) = keep current.
            reflection_system_message: Custom system message. None = use built-in default.
                                       Ellipsis (...) = keep current.
            reflection_delimiter: Custom delimiter. None = keep current.
        """
        if top_k is not None:
            self._config.memory_config.rag_top_k = top_k
        if relevance_threshold is not ...:
            self._config.memory_config.relevance_threshold = relevance_threshold
        if dedup_threshold is not ...:
            self._config.memory_config.dedup_threshold = dedup_threshold
            if self._memory_store:
                self._memory_store._dedup_threshold = dedup_threshold
        if reflection_interval is not None:
            self._config.memory_config.reflection_interval = reflection_interval
            if self._reflection_system:
                self._reflection_system._reflection_interval = reflection_interval
        if reflection_prompt is not ...:
            self._config.memory_config.reflection_prompt = reflection_prompt
            if self._reflection_system:
                self._reflection_system._reflection_prompt = reflection_prompt
        if reflection_system_message is not ...:
            self._config.memory_config.reflection_system_message = reflection_system_message
            if self._reflection_system:
                self._reflection_system._reflection_system_message = reflection_system_message
        if reflection_delimiter is not None:
            self._config.memory_config.reflection_delimiter = reflection_delimiter
            if self._reflection_system:
                self._reflection_system._reflection_delimiter = reflection_delimiter

        if self._rag_injector:
            self._rag_injector.update_config(
                top_k=top_k,
                relevance_threshold=relevance_threshold,
            )

    def update_prompt_config(
        self,
        rag_prefix: Optional[str] = ...,
        rag_suffix: Optional[str] = ...,
        codex_prefix: Optional[str] = ...,
        codex_suffix: Optional[str] = ...,
        example_dialogue_prefix: Optional[str] = ...,
        example_dialogue_suffix: Optional[str] = ...,
    ) -> None:
        """
        Update prompt injection wrappers at runtime (NANO-045d + NANO-052 follow-up).

        Args:
            rag_prefix: New RAG prefix. Ellipsis (...) = keep current.
            rag_suffix: New RAG suffix. Ellipsis (...) = keep current.
            codex_prefix: New codex prefix. Ellipsis (...) = keep current.
            codex_suffix: New codex suffix. Ellipsis (...) = keep current.
            example_dialogue_prefix: New example dialogue prefix. Ellipsis = keep current.
            example_dialogue_suffix: New example dialogue suffix. Ellipsis = keep current.
        """
        pc = self._config.prompt_config
        if rag_prefix is not ...:
            pc.rag_prefix = rag_prefix or ""
        if rag_suffix is not ...:
            pc.rag_suffix = rag_suffix or ""
        if codex_prefix is not ...:
            pc.codex_prefix = codex_prefix or ""
        if codex_suffix is not ...:
            pc.codex_suffix = codex_suffix or ""
        if example_dialogue_prefix is not ...:
            pc.example_dialogue_prefix = example_dialogue_prefix or ""
        if example_dialogue_suffix is not ...:
            pc.example_dialogue_suffix = example_dialogue_suffix or ""

        if self._rag_injector:
            self._rag_injector.update_config(
                rag_prefix=rag_prefix if rag_prefix is not ... else ...,
                rag_suffix=rag_suffix if rag_suffix is not ... else ...,
            )

        if self._pipeline:
            self._pipeline.set_codex_wrappers(
                codex_prefix=codex_prefix if codex_prefix is not ... else ...,
                codex_suffix=codex_suffix if codex_suffix is not ... else ...,
            )
            self._pipeline.set_example_dialogue_wrappers(
                prefix=example_dialogue_prefix if example_dialogue_prefix is not ... else ...,
                suffix=example_dialogue_suffix if example_dialogue_suffix is not ... else ...,
            )

    def update_generation_params(
        self,
        temperature: Optional[float] = ...,
        max_tokens: Optional[int] = ...,
        top_p: Optional[float] = ...,
        repeat_penalty: Optional[float] = ...,
        repeat_last_n: Optional[int] = ...,
        frequency_penalty: Optional[float] = ...,
        presence_penalty: Optional[float] = ...,
    ) -> None:
        """
        Update generation parameters at runtime (NANO-053, NANO-108).

        Applied as highest-priority overrides on the next LLM call.
        Also updates the provider_config in llm_config so save_to_yaml
        persists the values correctly.

        Args:
            temperature: New temperature. Ellipsis (...) = keep current.
            max_tokens: New max tokens. Ellipsis (...) = keep current.
            top_p: New top_p. Ellipsis (...) = keep current.
            repeat_penalty: New repeat_penalty. Ellipsis (...) = keep current.
            repeat_last_n: New repeat_last_n. Ellipsis (...) = keep current.
            frequency_penalty: New frequency_penalty. Ellipsis (...) = keep current.
            presence_penalty: New presence_penalty. Ellipsis (...) = keep current.
        """
        if self._runtime_generation_overrides is None:
            self._runtime_generation_overrides = {}

        if temperature is not ...:
            self._runtime_generation_overrides["temperature"] = temperature
            self._config.llm_config.provider_config["temperature"] = temperature
        if max_tokens is not ...:
            self._runtime_generation_overrides["max_tokens"] = max_tokens
            self._config.llm_config.provider_config["max_tokens"] = max_tokens
        if top_p is not ...:
            self._runtime_generation_overrides["top_p"] = top_p
            self._config.llm_config.provider_config["top_p"] = top_p
        if repeat_penalty is not ...:
            self._runtime_generation_overrides["repeat_penalty"] = repeat_penalty
            self._config.llm_config.provider_config["repeat_penalty"] = repeat_penalty
        if repeat_last_n is not ...:
            self._runtime_generation_overrides["repeat_last_n"] = repeat_last_n
            self._config.llm_config.provider_config["repeat_last_n"] = repeat_last_n
        if frequency_penalty is not ...:
            self._runtime_generation_overrides["frequency_penalty"] = frequency_penalty
            self._config.llm_config.provider_config["frequency_penalty"] = frequency_penalty
        if presence_penalty is not ...:
            self._runtime_generation_overrides["presence_penalty"] = presence_penalty
            self._config.llm_config.provider_config["presence_penalty"] = presence_penalty

        # Propagate to callbacks so next pipeline.run() uses them
        if self._callbacks:
            self._callbacks.update_generation_params(
                self._runtime_generation_overrides
            )

    def get_tools_state(self) -> dict:
        """
        Return current tools state for dashboard hydration (NANO-065a).

        Returns:
            Dict with master_enabled flag and per-tool enabled states.
        """
        from spindl.gui.response_models import ToolsConfigResponse

        master = (
            self._tool_executor is not None
            and self._pipeline is not None
            and self._pipeline._tool_executor is not None
        )
        tools = {}
        if self._tool_registry:
            for name in self._tool_registry._tools:
                label = name.replace("_", " ").title()
                tools[name] = {
                    "enabled": self._tool_registry.is_enabled(name),
                    "label": label,
                }
        state = {"master_enabled": master, "tools": tools}
        # NANO-089 Phase 4: validate response shape before returning
        try:
            ToolsConfigResponse.model_validate(state)
        except Exception as e:
            logger.warning("Tools state response validation warning: %s", e)
        return state

    def update_tools_config(
        self,
        master_enabled=...,
        tools: dict | None = None,
    ) -> dict:
        """
        Update tool enable/disable state at runtime (NANO-065a).

        Args:
            master_enabled: True/False to toggle master tool executor.
                Ellipsis (...) = keep current.
            tools: Dict of {tool_name: {"enabled": bool}} for per-tool toggles.
                None = no per-tool changes.

        Returns:
            Dict with success status and optional error message.
        """
        if master_enabled is not ...:
            if master_enabled and self._pipeline:
                if not self._tool_executor:
                    # NANO-089 Phase 4: precondition — VLM must be configured
                    vlm_provider = getattr(
                        self._config.vlm_config, "provider", ""
                    )
                    if not vlm_provider or vlm_provider == "none":
                        return {
                            "success": False,
                            "error": (
                                "No VLM provider configured. "
                                "Set up a VLM provider before enabling tools."
                            ),
                        }
                    # Lazy-init: tools were disabled at startup, create now.
                    # Force-enable all configured tools — per-tool states were
                    # stored as disabled because the master was off.  The master
                    # toggle ON means "activate everything".
                    for tool_cfg in self._config.tools_config.tools.values():
                        tool_cfg["enabled"] = True
                    logger.info("Tools master toggle: ON (lazy init)")
                    self._init_tool_system()
                if self._tool_executor:
                    self._pipeline.set_tool_executor(self._tool_executor)
                    self._config.tools_config.enabled = True
                    logger.info("Tools master toggle: ON")
                else:
                    logger.warning("Tools master toggle: ON failed — no tools initialized")
                    return {
                        "success": False,
                        "error": "Tools initialization failed — no tools available.",
                    }
            elif not master_enabled and self._pipeline:
                self._pipeline.set_tool_executor(None)
                self._config.tools_config.enabled = False
                logger.info("Tools master toggle: OFF")

        if tools:
            for name, cfg in tools.items():
                enabled = cfg.get("enabled")
                if enabled is not None and self._tool_registry:
                    self._tool_registry.set_enabled(name, enabled)
                    # Update in-memory config for YAML persistence
                    if name in self._config.tools_config.tools:
                        self._config.tools_config.tools[name]["enabled"] = enabled

        return {"success": True}

    # ── Runtime LLM provider/model swap (NANO-065b) ────────────────

    def get_llm_state(self) -> dict:
        """Return current LLM provider state for dashboard hydration."""
        from spindl.gui.response_models import LLMConfigResponse

        props = self._llm_provider.get_properties()
        available = []
        if self._llm_registry:
            available = self._llm_registry.list_available()
        state = {
            "provider": self._config.llm_config.provider,
            "model": props.model_name,
            "context_size": props.context_length,
            "available_providers": available,
        }
        # NANO-089: validate response shape before returning
        try:
            LLMConfigResponse.model_validate(state)
        except Exception as e:
            logger.warning("LLM state response validation warning: %s", e)
        return state

    def swap_llm_provider(
        self, provider_name: str, provider_config: dict
    ) -> dict:
        """Swap the active LLM provider at runtime.

        Args:
            provider_name: Registry name (e.g., "llama", "openrouter").
            provider_config: Provider config overrides. If empty, resolves
                from the stored YAML providers section.

        Returns:
            Dict with success status and new state, or error info.
        """
        from ..core import AgentState

        # Gate: reject if actively generating
        current_state = self._state_machine.state
        if current_state not in (AgentState.IDLE, AgentState.LISTENING):
            return {
                "success": False,
                "error": f"Cannot swap provider while {current_state.value}",
            }

        # Resolve provider class
        try:
            provider_class = self._llm_registry.get_provider_class(provider_name)
        except LLMProviderNotFoundError as e:
            return {"success": False, "error": str(e)}

        # Resolve config: merge stored YAML config with any overrides
        stored_config = self._config.llm_config.providers.get(provider_name, {})
        if provider_config:
            resolved_config = {**stored_config, **provider_config}
        else:
            resolved_config = stored_config

        if not resolved_config:
            return {
                "success": False,
                "error": f"No config found for provider '{provider_name}'",
            }

        # NANO-087: flag unified vision mode for slot pinning
        if self._config.vlm_config and self._config.vlm_config.provider == "llm":
            resolved_config["unified_vision"] = True

        # Validate config
        errors = provider_class.validate_config(resolved_config)
        if errors:
            return {"success": False, "error": "; ".join(errors)}

        # Instantiate and initialize
        try:
            new_provider = provider_class()
            new_provider.initialize(resolved_config)
        except Exception as e:
            return {"success": False, "error": f"Provider init failed: {e}"}

        # Swap via holder — all consumers auto-update
        old_provider = self._provider_holder.swap(new_provider)

        # Shutdown old provider
        try:
            old_provider.shutdown()
        except Exception as e:
            logger.warning("Old provider shutdown error: %s", e)

        # Update in-memory config for persistence
        self._config.llm_config.provider = provider_name
        self._config.llm_config.provider_config = resolved_config

        # NANO-089: validate config state after in-memory mutation
        try:
            self._config.llm_config.model_validate(
                self._config.llm_config.model_dump()
            )
        except Exception as e:
            logger.warning("LLM config validation warning after swap: %s", e)

        logger.info(
            "LLM provider swapped to %s (model: %s)",
            provider_name,
            new_provider.get_properties().model_name,
        )

        # Cascade to unified VLM if active (NANO-065c)
        if self._config.vlm_config.provider == "llm":
            try:
                self.swap_vlm_provider("llm", {})
                logger.info("VLM unified cascade: re-derived from new LLM provider")
            except Exception as e:
                logger.warning("VLM unified cascade failed: %s", e)

        return {"success": True, **self.get_llm_state()}

    # ── Runtime VLM provider swap (NANO-065c) ─────────────────────────

    def get_vlm_state(self) -> dict:
        """Return current VLM provider state for dashboard hydration."""
        from ..vision.registry import VLMProviderRegistry
        from spindl.gui.response_models import VLMConfigResponse

        vlm_cfg = self._config.vlm_config
        available: list[str] = []
        try:
            registry = VLMProviderRegistry(plugin_paths=vlm_cfg.plugin_paths)
            available = registry.list_providers()
        except Exception:
            available = ["llama", "openai", "llm"]

        healthy = self._check_vlm_health()

        # Include cloud provider config for dashboard hydration (bug #13).
        # Mask API key — only send enough for the user to recognize it.
        cloud_config = {}
        if vlm_cfg.provider == "openai":
            stored = vlm_cfg.providers.get("openai", {})
            api_key = stored.get("api_key", "")
            if api_key and not api_key.startswith("${"):
                masked = api_key[:4] + "..." + api_key[-4:] if len(api_key) > 8 else "••••"
            else:
                masked = api_key  # ${ENV_VAR} shown as-is
            cloud_config = {
                "api_key": masked,
                "model": stored.get("model", ""),
                "base_url": stored.get("base_url", ""),
            }

        state = {
            "provider": vlm_cfg.provider,
            "available_providers": available,
            "healthy": healthy,
        }
        if cloud_config:
            state["cloud_config"] = cloud_config
        # NANO-089: validate response shape before returning
        try:
            VLMConfigResponse.model_validate(state)
        except Exception as e:
            logger.warning("VLM state response validation warning: %s", e)
        return state

    def swap_vlm_provider(
        self, provider_name: str, provider_config: dict
    ) -> dict:
        """Swap the active VLM provider at runtime.

        Args:
            provider_name: Registry name (e.g., "llama", "openai", "llm").
            provider_config: Provider config overrides. If empty, resolves
                from the stored YAML providers section.

        Returns:
            Dict with success status and new state, or error info.
        """
        from ..core import AgentState

        # Gate: reject if actively generating
        current_state = self._state_machine.state
        if current_state not in (AgentState.IDLE, AgentState.LISTENING):
            return {
                "success": False,
                "error": f"Cannot swap VLM provider while {current_state.value}",
            }

        # Get screen_vision tool — lazy-init if tools were disabled at startup
        if not self._tool_registry:
            logger.info("Tool registry not initialized — attempting lazy init for VLM swap")
            self._config.vlm_config.provider = provider_name
            if provider_config:
                stored = self._config.vlm_config.providers.get(provider_name, {})
                stored.update(provider_config)
                self._config.vlm_config.providers[provider_name] = stored
            # Enable tools master + screen_vision
            self._config.tools_config.enabled = True
            for tool_cfg in self._config.tools_config.tools.values():
                tool_cfg["enabled"] = True
            try:
                self._init_tool_system()
            except Exception as e:
                logger.error("Lazy tool init failed during VLM swap: %s", e)
                return {"success": False, "error": f"Tool initialization failed: {e}"}
            if getattr(self, "_tool_executor", None) and self._pipeline:
                self._pipeline.set_tool_executor(self._tool_executor)
            if not self._tool_registry:
                return {"success": False, "error": "Tool registry initialization failed"}

        screen_vision = self._tool_registry.get_tool("screen_vision")
        if not screen_vision:
            # Tool failed to init at startup (e.g. no VLM config at launch).
            # Update VLM config in-memory so _init_tool_system injects the
            # correct provider, then re-init the tool system.
            logger.info("screen_vision not available — attempting lazy re-init")
            self._config.vlm_config.provider = provider_name
            if provider_config:
                stored = self._config.vlm_config.providers.get(provider_name, {})
                stored.update(provider_config)
                self._config.vlm_config.providers[provider_name] = stored
            # Ensure screen_vision is marked enabled in tools config
            tools_cfg = self._config.tools_config.tools
            if "screen_vision" in tools_cfg:
                tools_cfg["screen_vision"]["enabled"] = True
                tools_cfg["screen_vision"]["vlm_provider"] = provider_name
            try:
                self._init_tool_system()
            except Exception as e:
                logger.error("Lazy tool re-init failed: %s", e)
                return {
                    "success": False,
                    "error": f"Screen vision tool re-initialization failed: {e}",
                }
            screen_vision = self._tool_registry.get_tool("screen_vision")
            if not screen_vision:
                return {
                    "success": False,
                    "error": "Screen vision tool not initialized. Enable tools first.",
                }

        vlm_cfg = self._config.vlm_config

        # Resolve config
        if provider_name == "llm":
            # Unified mode: derive VLM config from current LLM provider
            llm_cfg = self._config.llm_config.provider_config

            llm_url = llm_cfg.get("url")
            if not llm_url:
                llm_host = llm_cfg.get("host", "127.0.0.1")
                llm_port = llm_cfg.get("port", 5557)
                llm_url = f"http://{llm_host}:{llm_port}"

            resolved_config = {
                "url": llm_url,
                "api_key": llm_cfg.get("api_key"),
                "model": llm_cfg.get("model", "local-llm"),
            }

            # Apply vision-specific overrides from vlm.providers.llm
            vlm_overrides = vlm_cfg.providers.get("llm", {})
            resolved_config.update(vlm_overrides)

            # Apply any caller overrides
            if provider_config:
                resolved_config.update(provider_config)
        else:
            # Standard provider: merge stored config with overrides
            stored_config = vlm_cfg.providers.get(provider_name, {})
            if provider_config:
                resolved_config = {**stored_config, **provider_config}
            else:
                resolved_config = stored_config

            if not resolved_config:
                return {
                    "success": False,
                    "error": f"No config found for VLM provider '{provider_name}'",
                }

        # Swap on the tool
        try:
            screen_vision.swap_vlm_provider(
                provider_name, resolved_config, vlm_cfg.plugin_paths
            )
        except Exception as e:
            return {"success": False, "error": f"VLM provider init failed: {e}"}

        # Update in-memory config for persistence
        self._config.vlm_config.provider = provider_name

        # NANO-089: validate config state after in-memory mutation
        try:
            self._config.vlm_config.model_validate(
                self._config.vlm_config.model_dump()
            )
        except Exception as e:
            logger.warning("VLM config validation warning after swap: %s", e)

        logger.info("VLM provider swapped to %s", provider_name)

        return {"success": True, **self.get_vlm_state()}

    def get_block_config(self) -> dict:
        """
        Return current block configuration as a serializable dict (NANO-045c-1).

        Reads from the live pipeline block config. If pipeline has no block
        config, returns defaults.

        Returns:
            Dict with keys: order, disabled, overrides, blocks (full metadata).
        """
        from spindl.llm.prompt_block import create_default_blocks

        # Get live blocks from pipeline, or defaults
        if self._pipeline and self._pipeline._block_config:
            blocks = self._pipeline._block_config
        else:
            blocks = create_default_blocks()

        # Build metadata list for GUI
        block_infos = []
        order_list = []
        disabled_list = []
        overrides_dict: dict[str, str | None] = {}

        for block in sorted(blocks, key=lambda b: b.order):
            order_list.append(block.id)
            if not block.enabled:
                disabled_list.append(block.id)
            if block.user_override is not None:
                overrides_dict[block.id] = block.user_override

            block_infos.append({
                "id": block.id,
                "label": block.label,
                "order": block.order,
                "enabled": block.enabled,
                "is_static": block.is_static,
                "section_header": block.section_header,
                "has_override": block.user_override is not None,
                "content_wrapper": block.content_wrapper,
            })

        return {
            "order": order_list,
            "disabled": disabled_list,
            "overrides": overrides_dict,
            "blocks": block_infos,
        }

    def update_block_config(self, config: dict) -> None:
        """
        Update block configuration at runtime (NANO-045c-1).

        Stores the raw config dict and hot-reloads the pipeline's block list.
        No restart required.

        Args:
            config: Dict with optional keys: order, disabled, overrides, wrappers.
        """
        self._config.prompt_blocks = config
        if self._pipeline:
            self._pipeline.set_block_config(config)

    def reset_block_config(self) -> None:
        """
        Reset block configuration to defaults (NANO-045c-1).

        Clears custom config and reloads pipeline with default block ordering.
        """
        self._config.prompt_blocks = {}
        if self._pipeline:
            self._pipeline.set_block_config({})

    def update_stimuli_config(
        self,
        enabled: Optional[bool] = None,
        patience_enabled: Optional[bool] = None,
        patience_seconds: Optional[float] = None,
        patience_prompt: Optional[str] = None,
        twitch_enabled: Optional[bool] = None,
        twitch_channel: Optional[str] = None,
        twitch_app_id: Optional[str] = None,
        twitch_app_secret: Optional[str] = None,
        twitch_buffer_size: Optional[int] = None,
        twitch_max_message_length: Optional[int] = None,
        twitch_prompt_template: Optional[str] = None,
        addressing_others_contexts: Optional[list] = None,
    ) -> None:
        """
        Update stimuli config at runtime (NANO-056).

        Args:
            enabled: Master enable/disable for stimuli engine.
            patience_enabled: Enable/disable PATIENCE timer.
            patience_seconds: PATIENCE timeout in seconds.
            patience_prompt: Custom PATIENCE prompt text.
            twitch_enabled: Enable/disable Twitch module (NANO-056b).
            twitch_channel: Twitch channel name.
            twitch_app_id: Twitch app ID.
            twitch_app_secret: Twitch app secret.
            twitch_buffer_size: Max buffered messages.
            twitch_max_message_length: Max message length filter.
            twitch_prompt_template: Prompt template for Twitch stimulus.
            addressing_others_contexts: List of AddressingContext dicts (NANO-110).
        """
        cfg = self._config.stimuli_config

        if enabled is not None:
            cfg.enabled = enabled
            if self._stimuli_engine:
                self._stimuli_engine.enabled = enabled
                # Reset activity timers when enabling so patience doesn't
                # fire immediately from stale elapsed time since engine start
                if enabled:
                    self._stimuli_engine.reset_activity()

        if self._stimuli_engine:
            # Find the PATIENCE module and update it
            for module in self._stimuli_engine.modules:
                if module.name == "patience":
                    if patience_enabled is not None:
                        module.enabled = patience_enabled
                        cfg.patience_enabled = patience_enabled
                    if patience_seconds is not None:
                        module.timeout_seconds = patience_seconds
                        cfg.patience_seconds = patience_seconds
                    if patience_prompt is not None:
                        module.prompt = patience_prompt
                        cfg.patience_prompt = patience_prompt
                    break

            # Find the Twitch module and update it (NANO-056b)
            for module in self._stimuli_engine.modules:
                if module.name == "twitch":
                    if twitch_enabled is not None:
                        module.enabled = twitch_enabled
                        cfg.twitch_enabled = twitch_enabled
                    if twitch_channel is not None:
                        module.channel = twitch_channel
                        cfg.twitch_channel = twitch_channel
                    if twitch_app_id is not None:
                        module.app_id = twitch_app_id
                        cfg.twitch_app_id = twitch_app_id
                    if twitch_app_secret is not None:
                        module.app_secret = twitch_app_secret
                        cfg.twitch_app_secret = twitch_app_secret
                    if twitch_buffer_size is not None:
                        module.buffer_size = twitch_buffer_size
                        cfg.twitch_buffer_size = twitch_buffer_size
                    if twitch_max_message_length is not None:
                        module.max_message_length = twitch_max_message_length
                        cfg.twitch_max_message_length = twitch_max_message_length
                    if twitch_prompt_template is not None:
                        module.prompt_template = twitch_prompt_template
                        cfg.twitch_prompt_template = twitch_prompt_template
                    break

            # Update config even if module isn't registered yet
            if twitch_enabled is not None:
                cfg.twitch_enabled = twitch_enabled
            if twitch_channel is not None:
                cfg.twitch_channel = twitch_channel
            if twitch_app_id is not None:
                cfg.twitch_app_id = twitch_app_id
            if twitch_app_secret is not None:
                cfg.twitch_app_secret = twitch_app_secret
            if twitch_buffer_size is not None:
                cfg.twitch_buffer_size = twitch_buffer_size
            if twitch_max_message_length is not None:
                cfg.twitch_max_message_length = twitch_max_message_length
            if twitch_prompt_template is not None:
                cfg.twitch_prompt_template = twitch_prompt_template

        # Addressing-others contexts (NANO-110) — config-only, no live module
        if addressing_others_contexts is not None:
            from .config import AddressingContext
            cfg.addressing_others_contexts = [
                AddressingContext(
                    id=ctx.get("id", f"ctx_{i}"),
                    label=ctx.get("label", "Others"),
                    prompt=ctx.get("prompt", ""),
                )
                for i, ctx in enumerate(addressing_others_contexts)
            ]

    def update_vts_config(
        self,
        enabled: Optional[bool] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
    ) -> None:
        """
        Update VTubeStudio config at runtime (NANO-060b).

        Args:
            enabled: Master enable/disable for VTS driver.
            host: WebSocket host for VTS.
            port: WebSocket port for VTS.
        """
        cfg = self._config.vtubestudio_config
        needs_restart = False

        if host is not None and host != cfg.host:
            cfg.host = host
            needs_restart = True

        if port is not None and port != cfg.port:
            cfg.port = port
            needs_restart = True

        if enabled is not None:
            cfg.enabled = enabled
            if enabled:
                if self._vts_driver and needs_restart:
                    # Host/port changed — restart
                    self._vts_driver.stop()
                    self._vts_driver.start()
                elif self._vts_driver:
                    # Driver exists, just start it
                    self._vts_driver.start()
                else:
                    # Driver was never created (disabled at boot) — create + start
                    self._vts_driver = VTSDriver(
                        config=cfg,
                        event_bus=self._event_bus,
                    )
                    if self._running:
                        self._vts_driver.start()
                    logger.info("[VTS] Driver created at runtime (host=%s:%d)", cfg.host, cfg.port)
            else:
                if self._vts_driver:
                    self._vts_driver.stop()
        elif needs_restart and self._vts_driver and cfg.enabled:
            # Host/port changed but enabled wasn't toggled — restart if running
            self._vts_driver.stop()
            self._vts_driver.start()

    @property
    def stimuli_engine(self) -> Optional[StimuliEngine]:
        """The stimuli engine instance, if created."""
        return self._stimuli_engine

    def _check_llm_health(self) -> bool:
        """Check if LLM provider is available."""
        if self._llm_provider:
            return self._llm_provider.health_check()

        # Fallback to legacy client check
        if not self._llm_client:
            return False

        try:
            # Try to get context length as a health check
            self._llm_client.get_context_length()
            return True
        except Exception:
            return False

    def _check_vlm_health(self) -> bool:
        """
        Check if VLM is available via the screen_vision tool.

        Returns:
            True if screen_vision tool is enabled and its VLM provider is healthy,
            False if tools are disabled or VLM is not available.
        """
        if not self._tool_registry:
            return False

        screen_vision = self._tool_registry.get_tool("screen_vision")
        if not screen_vision:
            return False

        return screen_vision.health_check()

    @property
    def is_running(self) -> bool:
        """Whether the voice agent is currently running."""
        return self._running

    @property
    def state(self) -> Optional[AgentState]:
        """Current state machine state."""
        if self._state_machine:
            return self._state_machine.state
        return None

    @property
    def session_file(self) -> Optional[Path]:
        """Current session file path."""
        if self._history_manager:
            return self._history_manager.session_file
        return None

    def get_prompt_snapshot(self) -> Optional[dict]:
        """
        Get the current prompt snapshot for the active session (NANO-076).

        Hot path: read latest from sidecar JSONL (fast, real token counts).
        Cold path: reconstruct via pipeline.build_snapshot() (estimated tokens).

        Returns:
            Snapshot dict, or None if no session/pipeline is available.
        """
        from ..history.snapshot_store import read_latest_snapshot

        # Hot path: try sidecar
        if self.session_file:
            snapshot = read_latest_snapshot(self.session_file)
            if snapshot is not None:
                return snapshot

        # Cold path: reconstruct
        if self._callbacks:
            return self._callbacks.build_prompt_snapshot()

        return None

    @property
    def turn_count(self) -> int:
        """Number of turns in current session."""
        if self._history_manager:
            return self._history_manager.turn_count
        return 0

    @property
    def config(self) -> OrchestratorConfig:
        """Get the orchestrator configuration."""
        return self._config

    @property
    def persona(self) -> Optional[dict]:
        """Get the loaded persona."""
        return self._persona

    @property
    def callbacks(self) -> Optional[OrchestratorCallbacks]:
        """Get the callbacks instance (for stats access)."""
        return self._callbacks

    @property
    def context_usage(self) -> dict:
        """
        Current context window usage.

        Returns:
            Dict with:
                - tokens: Current tokens in context
                - max_context: Context window size from active provider (NANO-096)
                - usage_percent: Percentage of budget used

        Example:
            >>> agent.context_usage
            {"tokens": 1120, "max_context": 8192, "usage_percent": 13.7}
        """
        if self._callbacks:
            return self._callbacks.context_usage
        # Fallback before callbacks are initialized — query provider if available
        limit = 8192
        if self._provider_holder:
            limit = self._provider_holder.provider.get_properties().context_length or 8192
        return {"tokens": 0, "max_context": limit, "usage_percent": 0.0}

    @property
    def event_bus(self) -> Optional[EventBus]:
        """
        Get the EventBus for subscribing to agent events.

        Available event types:
        - TRANSCRIPTION_READY: STT produced text
        - RESPONSE_READY: LLM produced response
        - TTS_STARTED: Audio playback began
        - TTS_COMPLETED: Audio playback finished
        - TTS_INTERRUPTED: Audio playback interrupted (barge-in)
        - STATE_CHANGED: Agent state transition
        - CONTEXT_UPDATED: ContextManager assembled new context
        - PIPELINE_ERROR: Processing error occurred

        Returns:
            EventBus instance, or None if not yet initialized.
        """
        return self._event_bus

    @property
    def context_manager(self) -> Optional[ContextManager]:
        """
        Get the ContextManager for registering context sources.

        Use this to add lorebook, vision, or other context sources:
            agent.context_manager.register_source(
                name="lorebook",
                provider=lambda: lorebook.search(agent.context_manager.pending_input),
                priority=10,
            )

        Returns:
            ContextManager instance, or None if not yet initialized.
        """
        return self._context_manager

    @property
    def vts_driver(self) -> Optional[VTSDriver]:
        """Get the VTubeStudio driver (NANO-060), or None if disabled."""
        return self._vts_driver

    def load_session(self, filepath: str) -> bool:
        """
        Load a specific session file.

        Args:
            filepath: Path to the JSONL session file

        Returns:
            True if session was loaded successfully, False otherwise
        """
        if not self._history_manager:
            return False
        from pathlib import Path
        result = self._history_manager.load_session(Path(filepath))
        # NANO-107: update session-scoped retrieval filter
        if result and self._memory_store and self._history_manager.session_file:
            self._memory_store.set_session_id(self._history_manager.session_file.stem)
        return result

    def create_new_session(self) -> bool:
        """
        Clear current session and start a fresh one for the same persona.

        The old session file is preserved (not deleted).

        Returns:
            True if new session was created, False otherwise
        """
        if not self._history_manager:
            return False
        # ensure_session() is lazy (first pipeline run). If no message has been
        # processed yet, _persona_id is None and clear_session() silently no-ops.
        # Bootstrap from the orchestrator's loaded persona so the manager knows
        # which persona to generate a filename for.
        persona_id = self._persona.get("id") if self._persona else None
        if persona_id:
            self._history_manager.ensure_session(persona_id)
        self._history_manager.clear_session()
        # NANO-107: update session-scoped retrieval filter
        if self._memory_store and self._history_manager.session_file:
            self._memory_store.set_session_id(self._history_manager.session_file.stem)
        return self._history_manager.session_file is not None

    def generate_session_summary(self, session_filepath: str) -> Optional[str]:
        """
        Generate a session summary from a specific session file.

        Reads turns from the JSONL file, generates summary via LLM,
        stores in ChromaDB summaries collection.

        Called by GUI server when user clicks "Summarize" button.

        Args:
            session_filepath: Path to the session JSONL file.

        Returns:
            Summary text, or None if generation failed or not configured.
        """
        if not self._session_summary_generator:
            logger.warning("Session summary requested but memory system is not enabled")
            return None

        from ..history.jsonl_store import read_visible_turns

        filepath = Path(session_filepath)
        if not filepath.exists():
            logger.warning("Session file not found: %s", session_filepath)
            return None

        turns = read_visible_turns(filepath)
        session_id = filepath.stem  # e.g., "spindle_20260207_143022"

        # Collect flash cards from this session if available
        flash_cards = None
        if self._memory_store:
            try:
                all_cards = self._memory_store.get_all("flashcards")
                flash_cards = [
                    c for c in all_cards
                    if c.get("metadata", {}).get("session_id") == session_id
                ]
            except Exception as e:
                logger.debug("Could not retrieve flash cards for summary: %s", e)

        return self._session_summary_generator.generate_and_store(
            session_turns=turns,
            session_id=session_id,
            flash_cards=flash_cards,
        )

    def switch_persona(self, persona_id: str) -> bool:
        """
        Switch to a different character at runtime (NANO-077).

        Full hot-swap pipeline: closes the current session, swaps all
        persona-dependent state (persona dict, codex, memory collections,
        conversation history), opens a fresh session for the new character,
        and updates the config.

        Only permitted when agent state is IDLE or LISTENING.

        Args:
            persona_id: The character folder name to switch to (e.g., "mryummers")

        Returns:
            True if switch succeeded, False if state prevents it.

        Raises:
            RuntimeError: If character loading fails.
        """
        # 1. State gate
        if self._state_machine:
            current_state = self._state_machine.state
            if current_state not in (AgentState.IDLE, AgentState.LISTENING):
                logger.info(
                    "switch_persona blocked: state is %s (need IDLE or LISTENING)",
                    current_state.value,
                )
                return False

        # 2. Same character? Delegate to reload
        current_id = self._persona.get("id") if self._persona else None
        if persona_id == current_id:
            logger.info("switch_persona: same character, delegating to reload_persona()")
            return self.reload_persona()

        logger.info("Switching persona: %s -> %s", current_id, persona_id)

        # 3. Close current session (flush pending turns, persist marker)
        if self._history_manager:
            # Any pending user input that hasn't been stored gets discarded
            # (no assistant response to pair it with)
            self._history_manager._pending_user_input = None

        # 4. Stop reflection system (writes through memory store)
        if self._reflection_system:
            self._reflection_system.stop()

        # 5. Load new character from disk
        loader = CharacterLoader(self._config.characters_dir)
        new_persona = loader.load_as_dict(persona_id)

        # 6. Update orchestrator persona reference
        self._persona = new_persona

        # 7. Update callbacks persona reference
        if self._callbacks:
            self._callbacks.update_persona(new_persona)

        # 8. Swap codex (clears old entries, loads new, resets state)
        if self._codex_manager:
            self._codex_manager.load_character(persona_id)

        # 9. Swap memory collections (if ChromaDB active)
        if self._memory_store:
            self._memory_store.switch_character(persona_id)

        # 10. Open new session
        if self._history_manager:
            self._history_manager.switch_to_persona(persona_id)

        # 11. Restart reflection system with new session_id
        if self._reflection_system:
            session_id = None
            if self._history_manager and self._history_manager.session_file:
                session_id = self._history_manager.session_file.stem
            self._reflection_system.start(session_id=session_id)

        # NANO-107: update session-scoped retrieval filter
        if self._memory_store and self._history_manager and self._history_manager.session_file:
            self._memory_store.set_session_id(self._history_manager.session_file.stem)

        # 12. Update config (in-memory only; YAML persistence handled by server)
        self._config.character_id = persona_id

        logger.info(
            "Persona switched to: %s (%s)",
            new_persona.get("name", persona_id),
            persona_id,
        )
        return True

    def reload_persona(self) -> bool:
        """
        Reload character card from disk and update all references.

        Hot-reloads the persona without restarting the agent. Only permitted
        when agent state is IDLE or LISTENING (not mid-conversation).

        Updates:
        - self._persona (orchestrator's reference)
        - self._callbacks._persona (pipeline's reference)
        - self._codex_manager (reloads character_book entries)

        Returns:
            True if reload succeeded, False if state prevents reload.

        Raises:
            RuntimeError: If character loading fails.

        Usage:
            # From GUI after saving character edits
            if agent.reload_persona():
                print("Character reloaded successfully")
            else:
                print(f"Cannot reload while {agent.state}")
        """
        # State gating: only allow reload when not processing
        if self._state_machine:
            current_state = self._state_machine.state
            if current_state not in (AgentState.IDLE, AgentState.LISTENING):
                logger.info(
                    "reload_persona blocked: state is %s (need IDLE or LISTENING)",
                    current_state.value,
                )
                return False

        # Get character ID from current persona
        character_id = self._persona.get("id") if self._persona else None
        if not character_id:
            character_id = self._config.character_id

        # Reload from disk
        loader = CharacterLoader(self._config.characters_dir)
        new_persona = loader.load_as_dict(character_id)

        # Update orchestrator reference
        self._persona = new_persona

        # Update callbacks reference
        if self._callbacks:
            self._callbacks.update_persona(new_persona)

        # Reload codex (character_book entries)
        if self._codex_manager:
            self._codex_manager.load_character(character_id)

        logger.info("Persona reloaded: %s", new_persona.get("name", character_id))
        return True

    def __enter__(self) -> "VoiceAgentOrchestrator":
        """Context manager entry - start agent."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - stop agent."""
        self.stop()
