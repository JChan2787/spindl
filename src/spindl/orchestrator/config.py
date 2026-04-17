"""Configuration models for VoiceAgentOrchestrator (NANO-089: Pydantic validation layer)."""

from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from ruamel.yaml import YAML as RuamelYAML

from spindl.utils.paths import resolve_relative_path


def _make_ruamel_yaml() -> RuamelYAML:
    """Create a configured ruamel.yaml instance for round-trip editing.

    Preserves comments, key ordering, quoting style, block scalar styles,
    and ${ENV_VAR} patterns. Used by save_to_yaml() and related methods.

    NANO-106: replaces all regex-based YAML surgery.
    """
    ry = RuamelYAML()
    ry.preserve_quotes = True
    ry.width = 4096  # prevent line wrapping
    ry.indent(mapping=2, sequence=4, offset=2)  # match spindl.yaml style

    # Represent None as 'null' (ruamel defaults to empty value)
    def _represent_none(self, data):
        return self.represent_scalar("tag:yaml.org,2002:null", "null")

    ry.representer.add_representer(type(None), _represent_none)
    return ry


class STTConfig(BaseModel):
    """
    STT provider configuration (NANO-061a).

    Encapsulates the provider-based STT configuration:
    - Whether the service is enabled
    - Which provider to use
    - Plugin paths for external providers
    - Provider-specific configuration
    """

    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    provider: str = "parakeet"
    plugin_paths: list[str] = Field(default_factory=list)
    provider_config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider")
    @classmethod
    def provider_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("STT provider cannot be empty")
        return v

    @classmethod
    def from_dict(cls, data: dict) -> "STTConfig":
        """Parse STT config from YAML dict.

        Supports both new provider-based format and legacy flat format:
        - New: stt.provider, stt.providers.<name>.*
        - Legacy: stt.host, stt.port, stt.timeout (no provider field)
        """
        provider = data.get("provider", "parakeet")
        plugin_paths = data.get("plugin_paths", [])

        # Get the provider-specific config from providers.<name>
        providers = data.get("providers", {})
        provider_config = providers.get(provider, {})

        # Backward compat: if no providers section, build from flat fields
        if not provider_config and "host" in data:
            provider_config = {
                "host": data.get("host", "127.0.0.1"),
                "port": data.get("port", 5555),
                "timeout": data.get("timeout", 30.0),
            }

        return cls(
            provider=provider,
            plugin_paths=plugin_paths,
            provider_config=provider_config,
        )


class TTSConfig(BaseModel):
    """
    TTS provider configuration (NANO-015).

    Encapsulates the provider-based TTS configuration:
    - Whether the service is enabled
    - Which provider to use
    - Plugin paths for external providers
    - Provider-specific configuration
    """

    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    provider: str = "kokoro"
    plugin_paths: list[str] = Field(default_factory=list)
    provider_config: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider")
    @classmethod
    def provider_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("TTS provider cannot be empty")
        return v

    @classmethod
    def from_dict(cls, data: dict) -> "TTSConfig":
        """Parse TTS config from YAML dict."""
        provider = data.get("provider", "kokoro")
        plugin_paths = data.get("plugin_paths", [])

        # Get the provider-specific config from providers.<name>
        providers = data.get("providers", {})
        provider_config = providers.get(provider, {})

        return cls(
            provider=provider,
            plugin_paths=plugin_paths,
            provider_config=provider_config,
        )


class LLMConfig(BaseModel):
    """
    LLM provider configuration (NANO-018).

    Encapsulates the provider-based LLM configuration:
    - Which provider to use
    - Plugin paths for external providers
    - Provider-specific configuration
    """

    model_config = ConfigDict(extra="ignore")

    provider: str = "llama"
    plugin_paths: list[str] = Field(default_factory=list)
    provider_config: dict[str, Any] = Field(default_factory=dict)
    providers: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @field_validator("provider")
    @classmethod
    def provider_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("LLM provider cannot be empty")
        return v

    @classmethod
    def from_dict(cls, data: dict) -> "LLMConfig":
        """Parse LLM config from YAML dict."""
        provider = data.get("provider", "llama")
        plugin_paths = data.get("plugin_paths", [])

        # Get the provider-specific config from providers.<name>
        providers = data.get("providers", {})
        provider_config = providers.get(provider, {})

        return cls(
            provider=provider,
            plugin_paths=plugin_paths,
            provider_config=provider_config,
            providers=providers,
        )


class VLMConfig(BaseModel):
    """
    VLM (Vision Language Model) backend configuration (NANO-023/024).

    Pure backend config for any tool or feature that needs vision capabilities.
    Vision is invoked via tools (e.g., screen_vision), not injected into prompts.

    Fields:
    - Which provider to use (llama, openai, llm, none, or plugin name)
    - Plugin paths for external providers
    - Screen capture settings
    - Provider-specific configuration
    """

    model_config = ConfigDict(extra="ignore")

    provider: str = "llama"
    plugin_paths: list[str] = Field(default_factory=list)
    capture_config: dict[str, Any] = Field(default_factory=dict)
    providers: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def provider_required_when_providers_exist(self) -> "VLMConfig":
        """Provider must be set when providers block is non-empty (NANO-089 Phase 3)."""
        if self.providers and not self.provider:
            raise ValueError(
                "VLM provider cannot be empty when providers are configured — "
                "no routing key to select a provider"
            )
        return self

    @classmethod
    def from_dict(cls, data: dict) -> "VLMConfig":
        """Parse VLM config from YAML dict."""
        return cls(
            provider=data.get("provider", "llama"),
            plugin_paths=data.get("plugin_paths", []),
            capture_config=data.get("capture", {}),
            providers=data.get("providers", {}),
        )

    def to_raw_dict(self) -> dict:
        """Convert back to raw dict format for tool initialization."""
        return {
            "provider": self.provider,
            "plugin_paths": self.plugin_paths,
            "capture": self.capture_config,
            "providers": self.providers,
        }


class ToolsConfig(BaseModel):
    """
    Tools configuration (NANO-024).

    Encapsulates the tool system configuration:
    - Whether tools are enabled globally
    - Plugin paths for external tools
    - Per-tool configuration and enable/disable
    - Max iterations for tool call loop
    """

    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    plugin_paths: list[str] = Field(default_factory=list)
    max_iterations: int = Field(default=5, ge=1)
    tools: dict[str, dict[str, Any]] = Field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "ToolsConfig":
        """Parse Tools config from YAML dict."""
        return cls(
            enabled=data.get("enabled", False),
            plugin_paths=data.get("plugin_paths", []),
            max_iterations=data.get("max_iterations", 5),
            tools=data.get("tools", {}),
        )

    def to_raw_dict(self) -> dict:
        """Convert back to raw dict format for tool registry."""
        return {
            "enabled": self.enabled,
            "plugin_paths": self.plugin_paths,
            "max_iterations": self.max_iterations,
            "tools": self.tools,
        }


class CurationConfig(BaseModel):
    """
    LLM-assisted memory curation via OpenRouter frontier model (NANO-102).

    When enabled, ambiguous near-duplicate memories are routed to a frontier
    model for ADD/SKIP/UPDATE/DELETE classification via function calling.
    """

    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    api_key: Optional[str] = None
    model: str = "anthropic/claude-haiku-4-5"
    prompt: Optional[str] = None
    timeout: float = Field(default=30.0, gt=0)

    @classmethod
    def from_dict(cls, data: dict) -> "CurationConfig":
        """Parse curation config from YAML dict."""
        return cls(
            enabled=data.get("enabled", False),
            api_key=data.get("api_key"),
            model=data.get("model", "anthropic/claude-haiku-4-5"),
            prompt=data.get("prompt"),
            timeout=data.get("timeout", 30.0),
        )


class MemoryConfig(BaseModel):
    """
    Memory system configuration (NANO-043).

    Encapsulates the long-term memory / RAG configuration:
    - Whether memory is enabled
    - Embedding server connection details
    - RAG query parameters
    - Reflection system parameters (Phase 3)
    - Session summary parameters (Phase 4)
    - Deduplication threshold (NANO-102)
    - LLM curation config (NANO-102)
    - Editable reflection prompt/delimiter (NANO-104)
    - Live mode flag
    """

    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    embedding_base_url: str = "http://127.0.0.1:5559"
    embedding_timeout: float = Field(default=10.0, gt=0)
    rag_top_k: int = Field(default=5, ge=1)
    relevance_threshold: Optional[float] = Field(default=0.25, ge=0.0, le=1.0)
    reflection_interval: int = Field(default=20, ge=1)
    reflection_max_tokens: int = Field(default=300, ge=1)
    reflection_prompt: Optional[str] = Field(
        default=None,
        description=(
            "Prompt template for reflection extraction. "
            "Must contain {transcript} placeholder. "
            "None = use built-in default."
        ),
    )
    reflection_system_message: Optional[str] = Field(
        default=None,
        description=(
            "System message for the reflection LLM call. "
            "None = use built-in default."
        ),
    )
    reflection_delimiter: str = Field(
        default="{qa}",
        description="Delimiter used to split LLM response into individual memory entries.",
    )
    session_summary_max_tokens: int = Field(default=500, ge=1)
    dedup_threshold: Optional[float] = Field(default=0.30, ge=0.0, le=2.0)
    curation: CurationConfig = Field(default_factory=CurationConfig)
    live_mode: bool = True

    # Retrieval scoring weights (NANO-107)
    scoring_w_relevance: float = Field(default=0.5, ge=0.0, le=1.0)
    scoring_w_recency: float = Field(default=0.2, ge=0.0, le=1.0)
    scoring_w_importance: float = Field(default=0.2, ge=0.0, le=1.0)
    scoring_w_frequency: float = Field(default=0.1, ge=0.0, le=1.0)
    scoring_decay_base: float = Field(default=0.9975, gt=0.0, lt=1.0)

    @field_validator("reflection_prompt")
    @classmethod
    def prompt_must_contain_transcript(cls, v: Optional[str]) -> Optional[str]:
        """Ensure custom reflection prompt contains {transcript} placeholder."""
        if v is not None and "{transcript}" not in v:
            raise ValueError("reflection_prompt must contain {transcript} placeholder")
        return v

    @field_validator("reflection_delimiter")
    @classmethod
    def delimiter_not_empty(cls, v: str) -> str:
        """Ensure reflection delimiter is not empty."""
        if not v:
            raise ValueError("reflection_delimiter cannot be empty")
        return v

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryConfig":
        """Parse memory config from YAML dict."""
        embedding = data.get("embedding", {})
        curation_data = data.get("curation", {})
        return cls(
            enabled=data.get("enabled", False),
            embedding_base_url=embedding.get("base_url", "http://127.0.0.1:5559"),
            embedding_timeout=embedding.get("timeout", 10.0),
            rag_top_k=data.get("top_k", 5),
            relevance_threshold=data.get("relevance_threshold"),
            reflection_interval=data.get("reflection_interval", 20),
            reflection_max_tokens=data.get("reflection_max_tokens", 300),
            reflection_prompt=data.get("reflection_prompt"),
            reflection_system_message=data.get("reflection_system_message"),
            reflection_delimiter=data.get("reflection_delimiter", "{qa}"),
            session_summary_max_tokens=data.get("session_summary_max_tokens", 500),
            dedup_threshold=data.get("dedup_threshold", 0.30),
            curation=CurationConfig.from_dict(curation_data),
            live_mode=data.get("live_mode", True),
            scoring_w_relevance=data.get("scoring_w_relevance", 0.5),
            scoring_w_recency=data.get("scoring_w_recency", 0.2),
            scoring_w_importance=data.get("scoring_w_importance", 0.2),
            scoring_w_frequency=data.get("scoring_w_frequency", 0.1),
            scoring_decay_base=data.get("scoring_decay_base", 0.9975),
        )


class PromptConfig(BaseModel):
    """
    Injection wrapper configuration (NANO-045d).

    Controls the prefix/suffix strings wrapped around RAG memories
    and codex entries when injected into the system prompt.
    """

    model_config = ConfigDict(extra="ignore")

    rag_prefix: str = (
        "The following are relevant memories about the user and past "
        "conversations. Use them to inform your response:"
    )
    rag_suffix: str = "End of memories."
    codex_prefix: str = "The following facts are always true in this context:"
    codex_suffix: str = ""
    example_dialogue_prefix: str = (
        "The following are example dialogues demonstrating this character's "
        "voice, tone, and response style. Use them as style reference only — "
        "do not repeat or quote them directly:"
    )
    example_dialogue_suffix: str = "End of style examples."

    @classmethod
    def from_dict(cls, data: dict) -> "PromptConfig":
        """Parse prompt config from YAML dict."""
        defaults = cls()
        return cls(
            rag_prefix=data.get("rag_prefix", defaults.rag_prefix),
            rag_suffix=data.get("rag_suffix", defaults.rag_suffix),
            codex_prefix=data.get("codex_prefix", defaults.codex_prefix),
            codex_suffix=data.get("codex_suffix", defaults.codex_suffix),
            example_dialogue_prefix=data.get("example_dialogue_prefix", defaults.example_dialogue_prefix),
            example_dialogue_suffix=data.get("example_dialogue_suffix", defaults.example_dialogue_suffix),
        )


class AddressingContext(BaseModel):
    """
    A single addressing-others context (NANO-110).

    Each context maps to one button in the Stream Deck and carries a prompt
    that tells the persona who the User was addressing when they return.
    """

    model_config = ConfigDict(extra="ignore")

    id: str = Field(description="Unique identifier, e.g. 'ctx_0', 'ctx_1'")
    label: str = Field(default="Others", description="Short label for Stream Deck button")
    prompt: str = Field(
        default="",
        description="Custom prompt. Empty = default fallback.",
    )


def _default_addressing_contexts() -> list[AddressingContext]:
    """Create the default addressing-others context list (one permanent entry)."""
    return [AddressingContext(id="ctx_0", label="Others", prompt="")]


class StimuliConfig(BaseModel):
    """
    Stimuli system configuration (NANO-056).

    Controls the autonomous stimulus engine:
    - Master enable/disable
    - PATIENCE idle timer settings
    - Twitch chat integration settings (NANO-056b)
    - Addressing-others contexts (NANO-110)
    """

    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    patience_enabled: bool = False
    patience_seconds: float = Field(default=60.0, ge=1.0)
    patience_prompt: str = (
        "Continue the conversation naturally. "
        "You have been idle. Think of something interesting to say or ask."
    )

    # Twitch integration (NANO-056b)
    twitch_enabled: bool = False
    twitch_channel: str = ""
    twitch_app_id: str = ""
    twitch_app_secret: str = ""
    twitch_buffer_size: int = Field(default=10, ge=1, le=50)
    twitch_max_message_length: int = Field(default=300, ge=50, le=1000)
    twitch_prompt_template: str = (
        "**You just received new messages in Twitch chat.** "
        "Reply as co-host \u2014 natural, in character, one unified response. "
        "Ignore anything off-topic or spammy.\n"
        "\n"
        "```chat\n"
        "{messages}\n"
        "```"
    )

    # NANO-115: Audience transcript injection controls
    twitch_audience_window: int = Field(default=25, ge=25, le=300)
    twitch_audience_char_cap: int = Field(default=150, ge=50, le=500)

    # Addressing-others contexts (NANO-110)
    addressing_others_contexts: list[AddressingContext] = Field(
        default_factory=_default_addressing_contexts,
    )

    @classmethod
    def from_dict(cls, data: dict) -> "StimuliConfig":
        """Parse stimuli config from YAML dict."""
        defaults = cls()
        patience = data.get("patience", {})
        twitch = data.get("twitch", {})

        # Parse addressing-others contexts (NANO-110)
        addressing = data.get("addressing_others", {})
        raw_contexts = addressing.get("contexts", [])
        if raw_contexts:
            contexts = [
                AddressingContext(
                    id=ctx.get("id", f"ctx_{i}"),
                    label=ctx.get("label", "Others"),
                    prompt=ctx.get("prompt", ""),
                )
                for i, ctx in enumerate(raw_contexts)
            ]
        else:
            contexts = _default_addressing_contexts()

        return cls(
            enabled=data.get("enabled", defaults.enabled),
            patience_enabled=patience.get("enabled", defaults.patience_enabled),
            patience_seconds=patience.get("seconds", defaults.patience_seconds),
            patience_prompt=patience.get("prompt", defaults.patience_prompt),
            twitch_enabled=twitch.get("enabled", defaults.twitch_enabled),
            twitch_channel=twitch.get("channel", defaults.twitch_channel),
            twitch_app_id=twitch.get("app_id", defaults.twitch_app_id),
            twitch_app_secret=twitch.get("app_secret", defaults.twitch_app_secret),
            twitch_buffer_size=twitch.get(
                "buffer_size", defaults.twitch_buffer_size
            ),
            twitch_max_message_length=twitch.get(
                "max_message_length", defaults.twitch_max_message_length
            ),
            twitch_prompt_template=twitch.get(
                "prompt_template", defaults.twitch_prompt_template
            ),
            twitch_audience_window=twitch.get(
                "audience_window", defaults.twitch_audience_window
            ),
            twitch_audience_char_cap=twitch.get(
                "audience_char_cap", defaults.twitch_audience_char_cap
            ),
            addressing_others_contexts=contexts,
        )


class VTubeStudioConfig(BaseModel):
    """
    VTubeStudio driver configuration (NANO-060).

    Controls the VTS WebSocket driver:
    - Connection settings (host, port, auth token)
    - Plugin identity
    - Expression and position preset mappings
    """

    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    host: str = "localhost"
    port: int = Field(default=8001, ge=1, le=65535)
    token_path: str = "./vtubeStudio_token.txt"
    plugin_name: str = "spindl"
    developer: str = "spindl"
    expressions: dict = Field(default_factory=dict)
    positions: dict = Field(default_factory=dict)
    thinking_hotkey: str = ""
    idle_hotkey: str = ""

    @classmethod
    def from_dict(cls, data: dict) -> "VTubeStudioConfig":
        """Parse VTubeStudio config from YAML dict."""
        defaults = cls()
        return cls(
            enabled=data.get("enabled", defaults.enabled),
            host=data.get("host", defaults.host),
            port=data.get("port", defaults.port),
            token_path=data.get("token_path", defaults.token_path),
            plugin_name=data.get("plugin_name", defaults.plugin_name),
            developer=data.get("developer", defaults.developer),
            expressions=data.get("expressions", defaults.expressions),
            positions=data.get("positions", defaults.positions),
            thinking_hotkey=data.get("thinking_hotkey", defaults.thinking_hotkey),
            idle_hotkey=data.get("idle_hotkey", defaults.idle_hotkey),
        )


class AvatarConfig(BaseModel):
    """
    Avatar renderer configuration (NANO-093).

    Controls the avatar bridge:
    - Master enable/disable
    - Emotion classifier mode (NANO-094)
    - Chat display of classified emotions
    """

    model_config = ConfigDict(extra="ignore")

    enabled: bool = False
    emotion_classifier: Literal["classifier", "off"] = "off"
    emotion_model_path: str = "models/emotion"
    emotion_confidence_threshold: float = 0.3
    expression_fade_delay: float = 1.0  # Seconds after TTS ends before face+body expressions fade out
    show_emotion_in_chat: bool = True
    subtitles_enabled: bool = False  # NANO-100: Show subtitle overlay window in avatar app
    subtitle_fade_delay: float = 1.5  # NANO-100: Seconds to hold subtitle text after TTS ends
    stream_deck_enabled: bool = False  # NANO-110: Show stream deck overlay window
    avatar_always_on_top: bool = True  # Avatar window always-on-top z-order
    subtitle_always_on_top: bool = True  # Subtitle window always-on-top z-order
    global_animations_dir: str = "spindl-avatar/public/animations"  # Shared animation FBX pool path (NANO-098)
    base_animations: dict[str, Optional[str]] = Field(
        default_factory=lambda: {"idle": None, "happy": None, "sad": None, "angry": None, "curious": None}
    )  # NANO-099: global default FBX clip names per emotion slot

    @classmethod
    def from_dict(cls, data: dict) -> "AvatarConfig":
        """Parse avatar config from YAML dict."""
        defaults = cls()
        # YAML parses unquoted `off` as boolean False — coerce back to string
        raw_classifier = data.get("emotion_classifier", defaults.emotion_classifier)
        if isinstance(raw_classifier, bool):
            raw_classifier = "off"
        # Legacy: coerce "rule" → "off" (rule-based classifier removed in NANO-094 rev 3)
        if raw_classifier == "rule":
            raw_classifier = "off"
        return cls(
            enabled=data.get("enabled", defaults.enabled),
            emotion_classifier=raw_classifier,
            emotion_model_path=data.get("emotion_model_path", defaults.emotion_model_path),
            emotion_confidence_threshold=data.get(
                "emotion_confidence_threshold", defaults.emotion_confidence_threshold
            ),
            expression_fade_delay=data.get(
                "expression_fade_delay", defaults.expression_fade_delay
            ),
            show_emotion_in_chat=data.get(
                "show_emotion_in_chat", defaults.show_emotion_in_chat
            ),
            subtitles_enabled=data.get(
                "subtitles_enabled", defaults.subtitles_enabled
            ),
            subtitle_fade_delay=data.get(
                "subtitle_fade_delay", defaults.subtitle_fade_delay
            ),
            stream_deck_enabled=data.get(
                "stream_deck_enabled", defaults.stream_deck_enabled
            ),
            avatar_always_on_top=data.get(
                "avatar_always_on_top", defaults.avatar_always_on_top
            ),
            subtitle_always_on_top=data.get(
                "subtitle_always_on_top", defaults.subtitle_always_on_top
            ),
            global_animations_dir=data.get(
                "global_animations_dir", defaults.global_animations_dir
            ),
            base_animations={
                "idle": data.get("base_animations", {}).get("idle"),
                "happy": data.get("base_animations", {}).get("happy"),
                "sad": data.get("base_animations", {}).get("sad"),
                "angry": data.get("base_animations", {}).get("angry"),
                "curious": data.get("base_animations", {}).get("curious"),
            },
        )


class OrchestratorConfig(BaseModel):
    """
    Configuration for VoiceAgentOrchestrator (NANO-089: Pydantic validation layer).

    Centralizes all settings for the voice agent pipeline:
    - Service endpoints (STT, TTS, LLM)
    - Audio capture/playback parameters
    - VAD tuning
    - Pipeline behavior (history, summarization, budget)
    - Persona selection
    - VLM backend settings (NANO-023/024)
    - Tools settings (NANO-024)
    - Memory / RAG settings (NANO-043)
    - VTubeStudio driver (NANO-060)
    """

    model_config = ConfigDict(extra="ignore")

    # STT configuration (NANO-061a: provider-based)
    stt_config: STTConfig = Field(default_factory=STTConfig)

    # TTS configuration (NANO-015: provider-based)
    tts_config: TTSConfig = Field(default_factory=TTSConfig)

    # LLM configuration (NANO-018/019: provider-based, clean break)
    llm_config: LLMConfig = Field(default_factory=LLMConfig)

    # VLM configuration (NANO-023/024: backend for vision tools)
    vlm_config: VLMConfig = Field(default_factory=VLMConfig)

    # Memory configuration (NANO-043: long-term memory / RAG)
    memory_config: MemoryConfig = Field(default_factory=MemoryConfig)

    # Prompt injection wrappers (NANO-045d: configurable RAG/codex framing)
    prompt_config: PromptConfig = Field(default_factory=PromptConfig)

    # Stimuli system (NANO-056: autonomous stimulus engine)
    stimuli_config: StimuliConfig = Field(default_factory=StimuliConfig)

    # VTubeStudio driver (NANO-060: VTS output driver)
    vtubestudio_config: VTubeStudioConfig = Field(default_factory=VTubeStudioConfig)

    # Tools configuration (NANO-024: on-demand tool calling)
    tools_config: ToolsConfig = Field(default_factory=ToolsConfig)

    # Avatar renderer bridge (NANO-093: orchestrator-avatar connection)
    avatar_config: AvatarConfig = Field(default_factory=AvatarConfig)

    # Audio settings
    # Note: playback_sample_rate removed - now derived from TTS provider properties
    capture_sample_rate: int = 16000
    chunk_samples: int = 512

    # VAD settings
    vad_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    min_speech_ms: int = Field(default=250, gt=0)
    min_silence_ms: int = Field(default=500, gt=0)
    speech_pad_ms: int = Field(default=300, ge=0)

    # Pipeline settings
    conversations_dir: str = "./conversations"
    resume_session: bool = True
    summarization_threshold: float = Field(default=0.6, gt=0.0, le=1.0)
    summarization_reserve_tokens: int = Field(default=512, ge=0)
    budget_strategy: Literal["truncate", "drop", "reject"] = "truncate"
    response_reserve: int = Field(default=300, ge=0)

    # NANO-115: History splice/flatten override
    # "auto" defers to provider capability, "splice" forces role-array,
    # "flatten" forces bracket-formatted text in system prompt.
    force_role_history: Literal["auto", "splice", "flatten"] = "auto"

    # Character settings (NANO-034: ST V2 Character Cards)
    character_id: str = "spindle"
    characters_dir: str = "./characters"

    # Legacy persona settings (deprecated, use character_id/characters_dir)
    persona_id: str = "spindle"  # @deprecated: use character_id
    personas_dir: str = "./personas"  # @deprecated: use characters_dir

    # Prompt composition (NANO-045a: block-based prompt assembly)
    # Raw dict from spindl.yaml `prompt_blocks` section. None = legacy template mode.
    prompt_blocks: Optional[dict] = None

    # Debug settings
    debug: bool = False

    @model_validator(mode="after")
    def vlm_none_must_not_have_stale_mmproj(self) -> "OrchestratorConfig":
        """When VLM is disabled (provider='none'), mmproj_path in the LLM
        provider config is a stale zombie that will cause architecture mismatches
        on the next launch. Warn loudly and strip it. (Session 606 bug)."""
        if self.vlm_config.provider == "none":
            llama_cfg = self.llm_config.providers.get("llama", {})
            if llama_cfg.get("mmproj_path"):
                import warnings
                warnings.warn(
                    f"VLM is disabled (provider='none') but llm.providers.llama "
                    f"contains stale mmproj_path: {llama_cfg['mmproj_path']} — "
                    f"stripping to prevent architecture mismatch on launch",
                    UserWarning,
                    stacklevel=2,
                )
                llama_cfg.pop("mmproj_path", None)
        return self

    @classmethod
    def from_yaml(cls, path: str) -> "OrchestratorConfig":
        """
        Load configuration from YAML file.

        Args:
            path: Path to YAML configuration file.

        Returns:
            OrchestratorConfig instance with values from file.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If config file is malformed.
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return cls._from_dict(data)

    @classmethod
    def _from_dict(cls, data: dict) -> "OrchestratorConfig":
        """Build config from dictionary (parsed YAML)."""
        config = cls()

        # STT settings (NANO-061a: provider-based)
        if "stt" in data:
            config.stt_config = STTConfig.from_dict(data["stt"])

        # TTS settings (NANO-015: provider-based)
        if "tts" in data:
            config.tts_config = TTSConfig.from_dict(data["tts"])

        # NANO-112: Propagate enabled flags from launcher services
        launcher_services = data.get("launcher", {}).get("services", {})
        if "stt" in launcher_services:
            stt_enabled = launcher_services["stt"].get("enabled", True)
            config.stt_config.enabled = stt_enabled
        if "tts" in launcher_services:
            tts_enabled = launcher_services["tts"].get("enabled", True)
            config.tts_config.enabled = tts_enabled

        # LLM settings (NANO-018/019: provider-based, clean break)
        if "llm" in data:
            config.llm_config = LLMConfig.from_dict(data["llm"])
            # NANO-115: History splice/flatten override
            config.force_role_history = data["llm"].get(
                "force_role_history", config.force_role_history
            )

        # VLM settings (NANO-023/024: backend for vision tools)
        if "vlm" in data:
            config.vlm_config = VLMConfig.from_dict(data["vlm"])

        # Tools settings (NANO-024: on-demand tool calling)
        if "tools" in data:
            config.tools_config = ToolsConfig.from_dict(data["tools"])

        # Memory settings (NANO-043: long-term memory / RAG)
        if "memory" in data:
            config.memory_config = MemoryConfig.from_dict(data["memory"])

        # Prompt injection wrappers (NANO-045d)
        if "prompt" in data:
            config.prompt_config = PromptConfig.from_dict(data["prompt"])

        # Stimuli system (NANO-056)
        if "stimuli" in data:
            config.stimuli_config = StimuliConfig.from_dict(data["stimuli"])

        # VTubeStudio driver (NANO-060)
        if "vtubestudio" in data:
            config.vtubestudio_config = VTubeStudioConfig.from_dict(data["vtubestudio"])

        # Avatar renderer bridge (NANO-093)
        if "avatar" in data:
            config.avatar_config = AvatarConfig.from_dict(data["avatar"])

        # Audio settings
        if "audio" in data:
            audio = data["audio"]
            config.capture_sample_rate = audio.get("capture_rate", config.capture_sample_rate)
            # playback_rate removed - derived from TTS provider properties
            config.chunk_samples = audio.get("chunk_size", config.chunk_samples)

        # VAD settings
        if "vad" in data:
            vad = data["vad"]
            config.vad_threshold = vad.get("threshold", config.vad_threshold)
            config.min_speech_ms = vad.get("min_speech_ms", config.min_speech_ms)
            config.min_silence_ms = vad.get("min_silence_ms", config.min_silence_ms)
            config.speech_pad_ms = vad.get("speech_pad_ms", config.speech_pad_ms)

        # Pipeline settings
        if "pipeline" in data:
            pipeline = data["pipeline"]
            config.conversations_dir = resolve_relative_path(
                pipeline.get("conversations_dir", config.conversations_dir)
            )
            config.resume_session = pipeline.get("resume_session", config.resume_session)
            config.summarization_threshold = pipeline.get("summarization_threshold", config.summarization_threshold)
            config.summarization_reserve_tokens = pipeline.get("summarization_reserve_tokens", config.summarization_reserve_tokens)
            config.budget_strategy = pipeline.get("budget_strategy", config.budget_strategy)
            config.response_reserve = pipeline.get("response_reserve", config.response_reserve)

        # Prompt composition (NANO-045a: block-based prompt assembly)
        if "prompt_blocks" in data:
            config.prompt_blocks = data["prompt_blocks"]

        # Character settings (NANO-034: ST V2 Character Cards)
        if "character" in data:
            character = data["character"]
            config.character_id = character.get("default", config.character_id)
            config.characters_dir = resolve_relative_path(
                character.get("directory", config.characters_dir)
            )

        # Legacy persona settings (deprecated, fallback if character not specified)
        if "persona" in data and "character" not in data:
            persona = data["persona"]
            config.persona_id = persona.get("default", config.persona_id)
            config.personas_dir = resolve_relative_path(
                persona.get("directory", config.personas_dir)
            )
            # Map legacy to new for compatibility
            config.character_id = config.persona_id
            config.characters_dir = config.personas_dir

        return config

    def to_dict(self) -> dict:
        """Export config as dictionary (for serialization)."""
        return {
            "stt": {
                "provider": self.stt_config.provider,
                "plugin_paths": self.stt_config.plugin_paths,
                "providers": {
                    self.stt_config.provider: self.stt_config.provider_config,
                },
            },
            "tts": {
                "provider": self.tts_config.provider,
                "plugin_paths": self.tts_config.plugin_paths,
                "providers": {
                    self.tts_config.provider: self.tts_config.provider_config,
                },
            },
            "llm": {
                "provider": self.llm_config.provider,
                "plugin_paths": self.llm_config.plugin_paths,
                "providers": {
                    self.llm_config.provider: self.llm_config.provider_config,
                },
            },
            "vlm": self.vlm_config.to_raw_dict(),
            "tools": self.tools_config.to_raw_dict(),
            "audio": {
                "capture_rate": self.capture_sample_rate,
                "chunk_size": self.chunk_samples,
            },
            "vad": {
                "threshold": self.vad_threshold,
                "min_speech_ms": self.min_speech_ms,
                "min_silence_ms": self.min_silence_ms,
                "speech_pad_ms": self.speech_pad_ms,
            },
            "pipeline": {
                "conversations_dir": self.conversations_dir,
                "resume_session": self.resume_session,
                "summarization_threshold": self.summarization_threshold,
                "summarization_reserve_tokens": self.summarization_reserve_tokens,
                "budget_strategy": self.budget_strategy,
                "response_reserve": self.response_reserve,
            },
            "persona": {
                "default": self.persona_id,
                "directory": self.personas_dir,
            },
        }

    def _validate_before_write(self) -> None:
        """Validate current config state before persisting to YAML (NANO-089).

        Re-validates the full config through Pydantic. Catches any in-memory
        mutations that violated type/value constraints before they hit disk.

        Raises:
            ValueError: If config state is invalid.
        """
        from pydantic import ValidationError

        try:
            self.model_validate(self.model_dump())
        except ValidationError as e:
            raise ValueError(
                f"Config validation failed — refusing to write invalid state: {e}"
            ) from e

    def save_to_yaml(self, path: str) -> None:
        """
        Persist GUI-configurable settings to YAML file.

        Uses ruamel.yaml round-trip parsing to preserve comments, key ordering,
        quoting style, block scalar styles, and ${ENV_VAR} patterns.

        NANO-106: replaces regex-based line surgery with dict key-path assignment.

        Args:
            path: Path to YAML configuration file.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            IOError: If file cannot be written.
            ValueError: If config state is invalid (NANO-089).
        """
        # NANO-089: validate before writing
        self._validate_before_write()

        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        # Read original for rollback
        original_content = config_path.read_text(encoding="utf-8")

        ry = _make_ruamel_yaml()
        data = ry.load(original_content)

        # --- VAD ---
        if "vad" not in data:
            data["vad"] = {}
        data["vad"]["threshold"] = self.vad_threshold
        data["vad"]["min_speech_ms"] = self.min_speech_ms
        data["vad"]["min_silence_ms"] = self.min_silence_ms
        data["vad"]["speech_pad_ms"] = self.speech_pad_ms

        # --- Pipeline ---
        if "pipeline" not in data:
            data["pipeline"] = {}
        data["pipeline"]["summarization_threshold"] = self.summarization_threshold

        # --- Memory ---
        if "memory" not in data:
            data["memory"] = {}
        mem = data["memory"]
        mem["top_k"] = self.memory_config.rag_top_k
        mem["relevance_threshold"] = self.memory_config.relevance_threshold
        mem["dedup_threshold"] = self.memory_config.dedup_threshold
        mem["reflection_interval"] = self.memory_config.reflection_interval
        mem["reflection_prompt"] = self.memory_config.reflection_prompt
        mem["reflection_system_message"] = self.memory_config.reflection_system_message
        mem["reflection_delimiter"] = self.memory_config.reflection_delimiter

        # Retrieval scoring (NANO-107)
        mem["scoring_w_relevance"] = self.memory_config.scoring_w_relevance
        mem["scoring_w_recency"] = self.memory_config.scoring_w_recency
        mem["scoring_w_importance"] = self.memory_config.scoring_w_importance
        mem["scoring_w_frequency"] = self.memory_config.scoring_w_frequency
        mem["scoring_decay_base"] = self.memory_config.scoring_decay_base

        # Curation (nested under memory)
        if "curation" not in mem:
            mem["curation"] = {}
        cur = mem["curation"]
        cur["enabled"] = self.memory_config.curation.enabled
        cur["api_key"] = self.memory_config.curation.api_key
        cur["model"] = self.memory_config.curation.model
        cur["timeout"] = self.memory_config.curation.timeout

        # --- Prompt wrappers ---
        if "prompt" not in data:
            data["prompt"] = {}
        data["prompt"]["rag_prefix"] = self.prompt_config.rag_prefix
        data["prompt"]["rag_suffix"] = self.prompt_config.rag_suffix
        data["prompt"]["codex_prefix"] = self.prompt_config.codex_prefix
        data["prompt"]["codex_suffix"] = self.prompt_config.codex_suffix
        data["prompt"]["example_dialogue_prefix"] = self.prompt_config.example_dialogue_prefix
        data["prompt"]["example_dialogue_suffix"] = self.prompt_config.example_dialogue_suffix

        # --- LLM ---
        if "llm" not in data:
            data["llm"] = {}
        data["llm"]["provider"] = self.llm_config.provider

        # LLM generation parameters — write to the active provider's section (NANO-108)
        provider_name = self.llm_config.provider
        if "providers" in data["llm"] and provider_name in data["llm"]["providers"]:
            prov = data["llm"]["providers"][provider_name]
            prov["temperature"] = self.llm_config.provider_config.get("temperature", 0.7)
            prov["max_tokens"] = self.llm_config.provider_config.get("max_tokens", 256)
            prov["top_p"] = self.llm_config.provider_config.get("top_p", 0.95)
            prov["top_k"] = self.llm_config.provider_config.get("top_k", 40)
            prov["min_p"] = self.llm_config.provider_config.get("min_p", 0.05)
            prov["repeat_penalty"] = self.llm_config.provider_config.get("repeat_penalty", 1.1)
            prov["repeat_last_n"] = self.llm_config.provider_config.get("repeat_last_n", 64)
            prov["frequency_penalty"] = self.llm_config.provider_config.get("frequency_penalty", 0.0)
            prov["presence_penalty"] = self.llm_config.provider_config.get("presence_penalty", 0.0)

        # NANO-115: History splice/flatten override
        data["llm"]["force_role_history"] = self.force_role_history

        # --- VLM ---
        if "vlm" not in data:
            data["vlm"] = {}
        data["vlm"]["provider"] = self.vlm_config.provider

        # --- Stimuli ---
        if "stimuli" not in data:
            data["stimuli"] = {}
        stim = data["stimuli"]
        stim["enabled"] = self.stimuli_config.enabled
        if "patience" not in stim:
            stim["patience"] = {}
        stim["patience"]["enabled"] = self.stimuli_config.patience_enabled
        stim["patience"]["seconds"] = self.stimuli_config.patience_seconds
        stim["patience"]["prompt"] = self.stimuli_config.patience_prompt

        # Twitch (nested under stimuli)
        if "twitch" not in stim:
            stim["twitch"] = {}
        tw = stim["twitch"]
        tw["enabled"] = self.stimuli_config.twitch_enabled
        tw["channel"] = self.stimuli_config.twitch_channel
        tw["app_id"] = self.stimuli_config.twitch_app_id
        tw["app_secret"] = self.stimuli_config.twitch_app_secret
        tw["buffer_size"] = self.stimuli_config.twitch_buffer_size
        tw["max_message_length"] = self.stimuli_config.twitch_max_message_length
        tw["prompt_template"] = self.stimuli_config.twitch_prompt_template
        tw["audience_window"] = self.stimuli_config.twitch_audience_window
        tw["audience_char_cap"] = self.stimuli_config.twitch_audience_char_cap

        # Addressing-others contexts (NANO-110, nested under stimuli)
        if "addressing_others" not in stim:
            stim["addressing_others"] = {}
        ao = stim["addressing_others"]
        ao["contexts"] = [
            {"id": ctx.id, "label": ctx.label, "prompt": ctx.prompt}
            for ctx in self.stimuli_config.addressing_others_contexts
        ]

        # --- Tools ---
        if "tools" not in data:
            data["tools"] = {}
        data["tools"]["enabled"] = self.tools_config.enabled

        # --- VTubeStudio ---
        if "vtubestudio" in data:
            data["vtubestudio"]["enabled"] = self.vtubestudio_config.enabled

        # --- Character ---
        if "character" not in data:
            data["character"] = {}
        data["character"]["default"] = self.character_id

        # --- Avatar ---
        if "avatar" not in data:
            data["avatar"] = {}
        av = data["avatar"]
        av["enabled"] = self.avatar_config.enabled
        av["emotion_classifier"] = self.avatar_config.emotion_classifier
        av["emotion_model_path"] = self.avatar_config.emotion_model_path
        av["emotion_confidence_threshold"] = self.avatar_config.emotion_confidence_threshold
        av["expression_fade_delay"] = self.avatar_config.expression_fade_delay
        av["show_emotion_in_chat"] = self.avatar_config.show_emotion_in_chat
        av["subtitles_enabled"] = self.avatar_config.subtitles_enabled
        av["subtitle_fade_delay"] = self.avatar_config.subtitle_fade_delay
        av["avatar_always_on_top"] = self.avatar_config.avatar_always_on_top
        av["subtitle_always_on_top"] = self.avatar_config.subtitle_always_on_top
        av["stream_deck_enabled"] = self.avatar_config.stream_deck_enabled
        # NANO-099: base_animations is managed by the Next.js API route
        # (/api/avatar/base-animations). Do NOT write it here.

        # Write with parse-back verification (NANO-105 hardening)
        from io import StringIO

        out = StringIO()
        ry.dump(data, out)
        new_content = out.getvalue()

        config_path.write_text(new_content, encoding="utf-8")

        # Verify the file we just wrote is valid YAML
        try:
            yaml.safe_load(new_content)
        except yaml.YAMLError as e:
            # Roll back
            config_path.write_text(original_content, encoding="utf-8")
            print(
                f"[Config] YAML round-trip produced invalid file - rolled back. Error: {e}",
                flush=True,
            )

    def save_block_config_to_yaml(self, path: str) -> None:
        """
        Persist prompt_blocks config to YAML file (NANO-045c-1).

        Uses ruamel.yaml round-trip to set or remove the ``prompt_blocks``
        key while preserving all other formatting.

        NANO-106: replaces section-boundary scanning with direct key access.

        Args:
            path: Path to YAML configuration file.

        Raises:
            FileNotFoundError: If config file doesn't exist.
        """
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {path}")

        pb = self.prompt_blocks
        has_content = (
            pb
            and isinstance(pb, dict)
            and (pb.get("order") or pb.get("disabled") or pb.get("overrides") or pb.get("wrappers"))
        )

        ry = _make_ruamel_yaml()
        with open(config_path, "r", encoding="utf-8") as f:
            data = ry.load(f)

        if has_content:
            data["prompt_blocks"] = pb
        else:
            # Remove existing section (reset to defaults)
            if "prompt_blocks" in data:
                del data["prompt_blocks"]

        with open(config_path, "w", encoding="utf-8") as f:
            ry.dump(data, f)

    # NOTE: validate() removed in NANO-089. All checks are now Pydantic
    # field constraints that fire at construction time:
    # - Empty provider → field_validator on STTConfig, TTSConfig, LLMConfig
    # - VAD threshold 0-1 → Field(ge=0.0, le=1.0)
    # - min_speech_ms > 0 → Field(gt=0)
    # - budget_strategy enum → Literal["truncate", "drop", "reject"]
    # - summarization_threshold 0-1 → Field(gt=0.0, le=1.0)
