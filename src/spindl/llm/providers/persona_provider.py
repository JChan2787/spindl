"""
Persona providers - Extract persona information from BuildContext.

These providers fill persona-related placeholders in the prompt template:
- [PERSONA_NAME] - The persona's name (required)
- [PERSONA_APPEARANCE] - Physical description (optional, collapses if empty)
- [PERSONA_PERSONALITY] - Personality traits (optional, collapses if empty)
- [SCENARIO] - Scene/setting context (optional, collapses if empty)
- [EXAMPLE_DIALOGUE] - Style reference from first_mes, alternate_greetings, mes_example (optional)
- [PERSONA_RULES] - Persona-specific rules (optional, collapses if empty)
"""

from typing import Optional

from ..build_context import BuildContext
from ..context_provider import ContextProvider


class PersonaNameProvider(ContextProvider):
    """
    Provides the persona name for [PERSONA_NAME] placeholder.

    This is a required field - returns "Assistant" as fallback if not specified.
    Unlike other persona providers, this does NOT collapse if missing.
    """

    @property
    def placeholder(self) -> str:
        return "[PERSONA_NAME]"

    def provide(self, context: BuildContext) -> Optional[str]:
        name = context.persona.get("name")
        if name:
            return str(name)
        # Fallback - persona name is required for template to make sense
        return "Assistant"


class PersonaAppearanceProvider(ContextProvider):
    """
    Provides persona appearance for [PERSONA_APPEARANCE] placeholder.

    Returns None if not specified, causing section to collapse.
    """

    @property
    def placeholder(self) -> str:
        return "[PERSONA_APPEARANCE]"

    def provide(self, context: BuildContext) -> Optional[str]:
        description = context.persona.get("description")
        if description and str(description).strip():
            return str(description).strip()
        return None


class PersonaPersonalityProvider(ContextProvider):
    """
    Provides persona personality for [PERSONA_PERSONALITY] placeholder.

    Returns None if not specified, causing section to collapse.
    """

    @property
    def placeholder(self) -> str:
        return "[PERSONA_PERSONALITY]"

    def provide(self, context: BuildContext) -> Optional[str]:
        personality = context.persona.get("personality")
        if personality and str(personality).strip():
            return str(personality).strip()
        return None


class ScenarioProvider(ContextProvider):
    """
    Provides scenario context for [SCENARIO] placeholder.

    Returns None if not specified, causing section to collapse.
    """

    @property
    def placeholder(self) -> str:
        return "[SCENARIO]"

    def provide(self, context: BuildContext) -> Optional[str]:
        scenario = context.persona.get("scenario")
        if scenario and str(scenario).strip():
            return str(scenario).strip()
        return None


class PersonaRulesProvider(ContextProvider):
    """
    Provides persona-specific rules for [PERSONA_RULES] placeholder.

    Handles rules as either:
    - A list of strings (formatted as bullet points)
    - A single string (returned as-is)

    Returns None if not specified, causing section to collapse.
    """

    @property
    def placeholder(self) -> str:
        return "[PERSONA_RULES]"

    def provide(self, context: BuildContext) -> Optional[str]:
        rules = context.persona.get("rules")

        if not rules:
            return None

        # Handle list of rules - format as bullet points
        if isinstance(rules, list):
            formatted_rules = []
            for rule in rules:
                rule_text = str(rule).strip()
                if rule_text:
                    # Add bullet if not already present
                    if not rule_text.startswith("-"):
                        rule_text = f"- {rule_text}"
                    formatted_rules.append(rule_text)
            if formatted_rules:
                return "\n".join(formatted_rules)
            return None

        # Handle string rules - return as-is
        rules_str = str(rules).strip()
        if rules_str:
            return rules_str
        return None


class ExampleDialogueProvider(ContextProvider):
    """
    Provides example dialogue for [EXAMPLE_DIALOGUE] placeholder.

    Combines first_mes, alternate_greetings, and mes_example into a single
    style reference block. The LLM uses these examples to understand the
    character's voice, tone, and response patterns.

    Supports configurable prefix/suffix wrappers (NANO-052 follow-up) to
    frame the content for the LLM. Wrappers are set at runtime via the
    pipeline's set_example_dialogue_wrappers() method.

    Returns None if all fields are empty, causing section to collapse.
    """

    def __init__(self) -> None:
        self._prefix: str = ""
        self._suffix: str = ""

    def set_wrappers(self, prefix: str = "", suffix: str = "") -> None:
        """Update prefix/suffix wrappers at runtime."""
        self._prefix = prefix
        self._suffix = suffix

    @property
    def placeholder(self) -> str:
        return "[EXAMPLE_DIALOGUE]"

    def provide(self, context: BuildContext) -> Optional[str]:
        char_name = context.persona.get("name", "Assistant")
        parts: list[str] = []

        # First message — strongest style signal
        first_mes = context.persona.get("first_mes", "")
        if first_mes and first_mes.strip():
            parts.append(self._substitute(first_mes.strip(), char_name))

        # Alternate greetings — additional style samples
        alt_greetings = context.persona.get("alternate_greetings", [])
        for greeting in alt_greetings:
            if greeting and greeting.strip():
                parts.append(self._substitute(greeting.strip(), char_name))

        # Message examples — few-shot dialogue examples
        mes_example = context.persona.get("mes_example", "")
        if mes_example and mes_example.strip():
            cleaned = self._clean_mes_example(mes_example.strip(), char_name)
            if cleaned:
                parts.append(cleaned)

        if not parts:
            return None

        content = "\n\n".join(parts)

        # Wrap with configurable prefix/suffix (NANO-052 follow-up)
        if self._prefix:
            content = f"{self._prefix}\n{content}"
        if self._suffix:
            content = f"{content}\n{self._suffix}"

        return content

    def _substitute(self, text: str, char_name: str) -> str:
        """Replace {{char}} and {{user}} placeholders with actual names."""
        text = text.replace("{{char}}", char_name)
        text = text.replace("{{user}}", "User")
        return text

    def _clean_mes_example(self, text: str, char_name: str) -> str:
        """Clean ST-format mes_example: strip <START> tags, substitute names."""
        # Replace <START> delimiter with clean separator
        text = text.replace("<START>", "---")
        # Strip leading separator if present
        text = text.lstrip("-").lstrip()
        # If only separator remains, return empty
        if not text.strip():
            return ""
        return self._substitute(text, char_name)
