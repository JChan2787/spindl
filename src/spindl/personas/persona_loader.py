"""
Persona loader for voice assistant configurations.

.. deprecated:: 0.5.0
    PersonaLoader is deprecated and will be removed in a future version.
    Use :class:`spindl.characters.CharacterLoader` instead, which provides
    ST V2 Character Card support and ecosystem interoperability.

    Migration:
        1. Run: python scripts/migrate_persona.py --all
        2. Update config: change 'persona' section to 'character' section
        3. Replace PersonaLoader with CharacterLoader in your code

    See NANO-034 for details.

Loads persona definitions from YAML files in the personas directory.

Supports two persona formats:
1. Legacy format: Uses monolithic `system_prompt` field
2. Structured format: Uses `appearance`, `personality`, `rules` fields

Both formats are valid. The structured format is preferred for new personas
as it enables template-based prompt building via ContextProviders.
"""

from pathlib import Path

import yaml


class PersonaLoader:
    """
    Loads persona configurations from YAML files.

    Each persona file defines a voice assistant's identity, including:
    - Voice settings (TTS voice ID, language)
    - System prompt OR structured fields (appearance, personality, rules)
    - Generation parameters (temperature, max_tokens)

    Example legacy persona:
        id: spindle
        name: "Spindle"
        system_prompt: |
            You are Spindle, a helpful voice assistant...

    Example structured persona (NANO-014):
        id: spindle
        name: "Spindle"
        appearance: |
            You have the body of a robot spider...
        personality: |
            Helpful, concise, slightly playful.
        rules:
            - NO asterisks or action markers
            - Keep responses concise

    Both formats are supported. A persona can have both for backward compatibility.
    """

    # Core identity fields always required
    REQUIRED_FIELDS = {"id", "name"}

    # Structured fields for template-based prompts
    STRUCTURED_FIELDS = {"description", "appearance", "personality", "rules"}

    def __init__(self, personas_dir: str = "./personas"):
        """
        Initialize with personas directory path.

        Args:
            personas_dir: Path to directory containing persona YAML files.
                          Defaults to ./personas relative to working directory.
        """
        self.personas_dir = Path(personas_dir)

    def load(self, persona_id: str) -> dict:
        """
        Load persona by ID.

        Args:
            persona_id: Persona identifier (filename without .yaml extension)

        Returns:
            Persona configuration dict with keys:
                - id: str
                - name: str
                - voice: str (optional, for TTS)
                - language: str (optional, for TTS)
                - system_prompt: str (optional if structured fields present)
                - appearance: str (optional, for structured format)
                - personality: str (optional, for structured format)
                - rules: list[str] | str (optional, for structured format)
                - generation: dict (optional, LLM params)

        Raises:
            FileNotFoundError: Persona file not found
            ValueError: Invalid persona format (missing required fields or parse error)
        """
        persona_path = self.personas_dir / f"{persona_id}.yaml"

        if not persona_path.exists():
            raise FileNotFoundError(
                f"Persona '{persona_id}' not found at {persona_path}"
            )

        try:
            with open(persona_path, "r", encoding="utf-8") as f:
                persona = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Failed to parse persona '{persona_id}': {e}")

        if not isinstance(persona, dict):
            raise ValueError(
                f"Persona '{persona_id}' must be a YAML mapping, got {type(persona).__name__}"
            )

        # Validate required fields
        missing = self.REQUIRED_FIELDS - set(persona.keys())
        if missing:
            raise ValueError(
                f"Persona '{persona_id}' missing required fields: {missing}"
            )

        # Validate that persona has EITHER system_prompt OR at least one structured field
        has_system_prompt = bool(persona.get("system_prompt"))
        has_structured = any(persona.get(field) for field in self.STRUCTURED_FIELDS)

        if not has_system_prompt and not has_structured:
            raise ValueError(
                f"Persona '{persona_id}' must have either 'system_prompt' or "
                f"at least one structured field ({', '.join(sorted(self.STRUCTURED_FIELDS))})"
            )

        return persona

    def has_structured_fields(self, persona: dict) -> bool:
        """
        Check if persona uses structured format.

        Args:
            persona: Loaded persona dict

        Returns:
            True if persona has any structured fields (appearance, personality, rules)
        """
        return any(persona.get(field) for field in self.STRUCTURED_FIELDS)

    def has_legacy_prompt(self, persona: dict) -> bool:
        """
        Check if persona has legacy system_prompt.

        Args:
            persona: Loaded persona dict

        Returns:
            True if persona has system_prompt field
        """
        return bool(persona.get("system_prompt"))

    def list_personas(self) -> list[str]:
        """
        List available persona IDs.

        Returns:
            List of persona IDs (filenames without .yaml extension)
        """
        if not self.personas_dir.exists():
            return []

        return [
            p.stem
            for p in self.personas_dir.glob("*.yaml")
            if p.is_file()
        ]
