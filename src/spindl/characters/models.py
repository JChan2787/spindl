"""
Character Card models following SillyTavern V2 specification.

Implements the de facto standard for AI character definitions, enabling
ecosystem interoperability with thousands of existing ST character cards
while providing spindl specific extensions.

Spec: https://github.com/malfoyslastname/character-card-spec-v2
"""

from typing import Any, Literal

from pydantic import BaseModel, Field


class GenerationConfig(BaseModel):
    """LLM generation parameters."""

    temperature: float | None = None
    max_tokens: int | None = None
    top_p: float | None = None
    top_k: int | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None
    repeat_penalty: float | None = None
    repeat_last_n: int | None = None


class SpindlExtensions(BaseModel):
    """
    spindl specific extensions stored in Character Card V2 extensions block.

    These fields are not part of the ST V2 spec but are preserved for
    spindl functionality (TTS, summarization, generation params).
    """

    id: str | None = None
    voice: str | None = None  # Legacy (Kokoro) — kept for backward compat
    language: str | None = None  # Legacy (Kokoro) — kept for backward compat
    tts_voice_config: dict | None = None  # Provider-agnostic TTS params (NANO-054a)
    appearance: str | None = None
    rules: list[str] | None = None
    summarization_prompt: str | None = None
    generation: GenerationConfig | None = None
    summarization_generation: GenerationConfig | None = None
    avatar_vrm: str | None = None  # VRM filename within character directory (NANO-097)
    avatar_expressions: dict[str, dict[str, float]] | None = None  # Per-character expression composites (NANO-098)
    avatar_animations: dict | None = None  # Emotion-to-animation threshold map (NANO-098)


class CharacterBookEntry(BaseModel):
    """
    A single entry in a character book (lorebook/codex).

    Follows ST V2 CharacterBookEntry specification with full support
    for keyword matching, secondary keys, and timed effects.
    """

    keys: list[str] = Field(default_factory=list)
    content: str = ""
    extensions: dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    insertion_order: int = 0
    case_sensitive: bool | None = None
    name: str | None = None
    priority: int | None = None
    id: int | None = None
    comment: str | None = None
    selective: bool | None = None
    secondary_keys: list[str] | None = None
    constant: bool | None = None
    position: Literal["before_char", "after_char"] | None = None

    # ST extensions for timed effects (not in base spec, but widely used)
    sticky: int | None = None  # Stay active for N messages after trigger
    cooldown: int | None = None  # Cannot re-trigger for N messages
    delay: int | None = None  # Activate N messages after trigger


class CharacterBook(BaseModel):
    """
    A collection of lorebook entries (codex).

    Can be embedded in a character card or stored separately as a global codex.
    """

    entries: list[CharacterBookEntry] = Field(default_factory=list)
    name: str | None = None
    description: str | None = None
    scan_depth: int | None = None
    token_budget: int | None = None
    recursive_scanning: bool | None = None
    extensions: dict[str, Any] = Field(default_factory=dict)


class CharacterCardData(BaseModel):
    """
    The data block of a Character Card V2.

    Contains all character definition fields, both V1 legacy and V2 additions.
    """

    # V1 Legacy fields (all required for compatibility)
    name: str
    description: str = ""
    personality: str = ""
    scenario: str = ""
    first_mes: str = ""
    mes_example: str = ""

    # V2 New fields
    creator_notes: str = ""
    system_prompt: str = ""
    post_history_instructions: str = ""
    alternate_greetings: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    creator: str = ""
    character_version: str = ""

    # Extensions block (MUST default to empty dict per spec)
    extensions: dict[str, Any] = Field(default_factory=dict)

    # Embedded character book (optional)
    character_book: CharacterBook | None = None

    @property
    def spindl(self) -> SpindlExtensions | None:
        """Access spindl extensions if present."""
        if "spindl" not in self.extensions:
            return None
        return SpindlExtensions.model_validate(self.extensions["spindl"])

    def get_spindl(self) -> SpindlExtensions:
        """Get spindl extensions, creating empty if not present."""
        if "spindl" not in self.extensions:
            self.extensions["spindl"] = {}
        return SpindlExtensions.model_validate(self.extensions["spindl"])


class CharacterCard(BaseModel):
    """
    SillyTavern Character Card V2.

    The de facto standard for AI character definitions. This model enables
    bidirectional portability with the ST ecosystem while supporting
    spindl specific extensions.

    Usage:
        card = CharacterCard.model_validate(json_data)
        card.data.name  # "Spindle"
        card.data.spindl.voice  # "af_bella"

    Compatibility:
        The `to_persona_dict()` method provides backward compatibility
        with the existing PersonaLoader interface, allowing gradual migration.
    """

    spec: Literal["chara_card_v2"] = "chara_card_v2"
    spec_version: str = "2.0"
    data: CharacterCardData

    def to_persona_dict(self) -> dict[str, Any]:
        """
        Convert to legacy persona dict format for backward compatibility.

        Returns a dict matching the PersonaLoader output format, allowing
        existing code (VoiceAgentOrchestrator, persona providers) to work
        without modification.

        Returns:
            Dict with keys: id, name, voice, language, description, personality,
            rules, system_prompt, summarization_prompt, generation, etc.
        """
        nano = self.data.spindl
        result: dict[str, Any] = {
            "id": nano.id if nano else self.data.name.lower().replace(" ", "_"),
            "name": self.data.name,
        }

        # Map ST V2 native fields
        if self.data.personality:
            result["personality"] = self.data.personality
        if self.data.system_prompt:
            result["system_prompt"] = self.data.system_prompt

        # Description migration: data.description is canonical source,
        # fall back to nano.appearance for un-migrated cards
        if self.data.description:
            result["description"] = self.data.description
        elif nano and nano.appearance:
            result["description"] = nano.appearance

        # Scenario (ST V2 native field)
        if self.data.scenario:
            result["scenario"] = self.data.scenario

        # Example dialogue fields (style reference material)
        if self.data.first_mes:
            result["first_mes"] = self.data.first_mes
        if self.data.mes_example:
            result["mes_example"] = self.data.mes_example
        if self.data.alternate_greetings:
            result["alternate_greetings"] = self.data.alternate_greetings

        # Map TTS voice configuration (NANO-054a: provider-agnostic)
        if nano:
            if nano.tts_voice_config:
                result["tts_voice_config"] = nano.tts_voice_config
            elif nano.voice or nano.language:
                # Legacy migration: construct tts_voice_config from bare Kokoro fields
                legacy_config: dict[str, str] = {}
                if nano.voice:
                    legacy_config["voice"] = nano.voice
                if nano.language:
                    legacy_config["language"] = nano.language
                result["tts_voice_config"] = legacy_config

        # Map remaining spindl extensions
        if nano:
            if nano.rules:
                result["rules"] = nano.rules
            if nano.summarization_prompt:
                result["summarization_prompt"] = nano.summarization_prompt
            if nano.generation:
                result["generation"] = nano.generation.model_dump(exclude_none=True)
            if nano.summarization_generation:
                result["summarization_generation"] = nano.summarization_generation.model_dump(
                    exclude_none=True
                )
            if nano.avatar_vrm:
                result["avatar_vrm"] = nano.avatar_vrm
            if nano.avatar_expressions:
                result["avatar_expressions"] = nano.avatar_expressions
            if nano.avatar_animations:
                result["avatar_animations"] = nano.avatar_animations

        return result

    @classmethod
    def from_persona_dict(cls, persona: dict[str, Any]) -> "CharacterCard":
        """
        Create CharacterCard from legacy persona dict (migration helper).

        Args:
            persona: Dict from PersonaLoader or YAML persona file

        Returns:
            CharacterCard with fields mapped to appropriate locations
        """
        # Build spindl extensions
        nano_ext: dict[str, Any] = {}
        if "id" in persona:
            nano_ext["id"] = persona["id"]
        if "tts_voice_config" in persona:
            nano_ext["tts_voice_config"] = persona["tts_voice_config"]
        elif "voice" in persona or "language" in persona:
            # Legacy path: preserve bare fields for backward compat
            if "voice" in persona:
                nano_ext["voice"] = persona["voice"]
            if "language" in persona:
                nano_ext["language"] = persona["language"]
        if "appearance" in persona:
            nano_ext["appearance"] = persona["appearance"]
        if "rules" in persona:
            nano_ext["rules"] = persona["rules"]
        if "summarization_prompt" in persona:
            nano_ext["summarization_prompt"] = persona["summarization_prompt"]
        if "generation" in persona:
            nano_ext["generation"] = persona["generation"]
        if "summarization_generation" in persona:
            nano_ext["summarization_generation"] = persona["summarization_generation"]
        if "avatar_vrm" in persona:
            nano_ext["avatar_vrm"] = persona["avatar_vrm"]
        if "avatar_expressions" in persona:
            nano_ext["avatar_expressions"] = persona["avatar_expressions"]
        if "avatar_animations" in persona:
            nano_ext["avatar_animations"] = persona["avatar_animations"]

        # Build extensions block
        extensions: dict[str, Any] = {}
        if nano_ext:
            extensions["spindl"] = nano_ext

        # Build data block
        data = CharacterCardData(
            name=persona.get("name", "Unknown"),
            description=persona.get("description", ""),
            personality=persona.get("personality", ""),
            system_prompt=persona.get("system_prompt", ""),
            extensions=extensions,
        )

        return cls(data=data)
