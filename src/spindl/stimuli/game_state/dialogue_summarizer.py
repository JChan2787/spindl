"""
Rolling compression summarizer for game dialogue (NANO-116 Phase B.2).

Cloud LLM via OpenRouter. Persona-infused summaries — the summary reads
in SpindL's voice (or whichever character persona is active). Fires only
when accumulated raw dialogue lines exceed the prompt token budget.

Part of the atomic response cycle: drain → respond → summarize if needed
→ TTS completes → gate reopens. No new bridge stimuli responded to until
the full cycle completes.

Follows the CurationClient pattern (NANO-102) for lightweight OpenRouter
calls — separate from the conversation LLM.
"""

import logging
import os
import re
from typing import Any, Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
CHAT_ENDPOINT = "/chat/completions"

# Environment variable pattern: ${VAR_NAME}
_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")

DEFAULT_SUMMARIZER_SYSTEM_PROMPT = (
    "You are maintaining a rolling character knowledge summary for a livestream "
    "AI co-host. The co-host commentates on a video game while the User plays.\n"
    "\n"
    "You will receive:\n"
    "1. The previous summary (if any) — your last compression pass.\n"
    "2. New dialogue lines from in-game characters since the last summary.\n"
    "\n"
    "Produce an updated summary that:\n"
    "- Identifies each named character with a brief description.\n"
    "- Tracks what each character has said, done, or revealed.\n"
    "- Notes relationships between characters.\n"
    "- Preserves narrative progression (earlier events stay unless contradicted).\n"
    "- Reads naturally in the co-host's voice — not clinical, not robotic.\n"
    "- Is concise: aim for 1-3 short paragraphs total, not per-character.\n"
    "\n"
    "This summary will be injected into the co-host's system prompt as context "
    "for future commentary. Write it as notes the co-host would reference, "
    "not as a script to read aloud."
)


def _resolve_env_vars(value: str) -> str:
    """Replace ${VAR_NAME} patterns with environment variable values."""
    def _replace(match: re.Match) -> str:
        var_name = match.group(1)
        return os.environ.get(var_name, match.group(0))
    return _ENV_VAR_PATTERN.sub(_replace, value)


class DialogueSummarizer:
    """
    Rolling compression summarizer for game character dialogue.

    Makes a single OpenRouter chat completion call:
    - System prompt: persona-infused summarization instructions (configurable).
    - User message: previous summary + new raw dialogue lines.
    - Response: updated summary blob.

    Config sourced from YAML stimuli.game_state.dialogue.summarizer section.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "anthropic/claude-sonnet-4-20250514",
        base_url: str = DEFAULT_BASE_URL,
        persona_prompt: str = "",
        character_name: str = "",
        character_description: str = "",
        character_personality: str = "",
        character_scenario: str = "",
        max_tokens: int = 512,
        timeout: float = 30.0,
    ):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._character_name = character_name
        self._max_tokens = max_tokens
        if persona_prompt:
            self._persona_prompt = persona_prompt
        else:
            self._persona_prompt = self._build_default_prompt(
                character_name, character_description,
                character_personality, character_scenario,
            )
        self._timeout = timeout

    @staticmethod
    def _build_default_prompt(
        name: str, description: str, personality: str, scenario: str,
    ) -> str:
        prompt = DEFAULT_SUMMARIZER_SYSTEM_PROMPT
        if name:
            prompt = prompt.replace(
                "a livestream AI co-host",
                f"a livestream AI co-host named {name}",
            )
        persona_block = ""
        if description:
            persona_block += f"\n\nCo-host appearance: {description}"
        if personality:
            persona_block += f"\n\nCo-host personality: {personality}"
        if scenario:
            persona_block += f"\n\nScenario context: {scenario}"
        if persona_block:
            prompt += persona_block
        return prompt

    @property
    def api_key(self) -> str:
        return self._api_key

    @api_key.setter
    def api_key(self, value: str) -> None:
        self._api_key = value

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, value: str) -> None:
        self._model = value

    @property
    def persona_prompt(self) -> str:
        return self._persona_prompt

    @persona_prompt.setter
    def persona_prompt(self, value: str) -> None:
        self._persona_prompt = value

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @max_tokens.setter
    def max_tokens(self, value: int) -> None:
        self._max_tokens = max(64, value)

    def is_configured(self) -> bool:
        """Check if the summarizer has the minimum config to make a call."""
        resolved_key = _resolve_env_vars(self._api_key) if self._api_key else ""
        return bool(resolved_key and self._model)

    def summarize(
        self,
        previous_summary: str,
        dialogue_lines: list[dict],
        codex_context: str = "",
    ) -> Optional[str]:
        """Run a summarization pass. Returns the updated summary blob or None on failure.

        Args:
            previous_summary: The previous summary blob (empty string if first pass).
            dialogue_lines: List of dialogue line dicts from DialogueStore.get_unsummarized_lines().
            codex_context: Pre-formatted Codex entries activated against the dialogue
                lines. Gives the summarizer character context from the user's lorebook.
        """
        if not dialogue_lines:
            logger.debug("Summarizer called with no dialogue lines, skipping.")
            return None

        if not self.is_configured():
            logger.warning(
                "Dialogue summarizer not configured (missing API key or model). "
                "Skipping summarization."
            )
            return None

        resolved_key = _resolve_env_vars(self._api_key)

        # Build user message
        user_parts: list[str] = []
        if previous_summary:
            user_parts.append("## Previous Summary")
            user_parts.append(previous_summary)
            user_parts.append("")

        if codex_context:
            user_parts.append("## Codex Context")
            user_parts.append(codex_context)
            user_parts.append("")

        user_parts.append("## New Dialogue Lines")
        for line_dict in dialogue_lines:
            speaker = line_dict.get("speaker", "???")
            text = line_dict.get("text", "")
            source = line_dict.get("source", "")
            repeat = line_dict.get("repeat_count", 1)

            parts = [f"{speaker}: {text}"]
            if repeat and repeat > 1:
                parts.append(f"(x{repeat})")
            if source == "cinematic":
                parts.append("[cinematic]")

            # Staple gameplay context if present
            ctx = line_dict.get("gameplay_context", {})
            if ctx.get("combat_active"):
                ctx_note = f"  [combat: {ctx.get('enemy_count', '?')} enemies"
                if ctx.get("hp_ratio") is not None:
                    ctx_note += f", HP {ctx['hp_ratio']:.0%}"
                ctx_note += "]"
                parts.append(ctx_note)

            user_parts.append(" ".join(parts))

        user_message = "\n".join(user_parts)

        # Make the API call
        try:
            response = requests.post(
                f"{self._base_url}{CHAT_ENDPOINT}",
                headers={
                    "Authorization": f"Bearer {resolved_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": self._persona_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    "temperature": 0.3,
                    "max_tokens": self._max_tokens,
                },
                timeout=self._timeout,
            )

            if response.status_code != 200:
                logger.error(
                    "Dialogue summarizer API error %d: %s",
                    response.status_code,
                    response.text[:200],
                )
                return None

            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                logger.error("Dialogue summarizer: no choices in response")
                return None

            content = choices[0].get("message", {}).get("content", "")
            if not content:
                logger.error("Dialogue summarizer: empty content in response")
                return None

            logger.info(
                "Dialogue summarizer produced %d-char summary (model=%s, "
                "input_lines=%d, previous_summary=%s)",
                len(content),
                self._model,
                len(dialogue_lines),
                "yes" if previous_summary else "no",
            )
            return content.strip()

        except requests.Timeout:
            logger.error(
                "Dialogue summarizer timed out after %.1fs", self._timeout
            )
            return None
        except requests.RequestException as e:
            logger.error("Dialogue summarizer request failed: %s", e)
            return None
        except (KeyError, ValueError) as e:
            logger.error("Dialogue summarizer response parse error: %s", e)
            return None
