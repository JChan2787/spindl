"""
CurationClient — LLM-assisted memory deduplication via OpenRouter (NANO-102 Phase 2).

Lightweight OpenRouter client that classifies ambiguous near-duplicate memories
using frontier model function calling. Independent from the conversation LLM —
separate API key, model, and timeout. Runs on the reflection background thread.

Decision types:
  ADD    — genuinely new information, store it
  SKIP   — already captured by existing memory
  UPDATE — merge new info into existing memory (returns merged text)
  DELETE — new memory contradicts existing (remove old, store new)

All validation failures resolve to ADD (permissive) — never lose a memory.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)

# Environment variable pattern: ${VAR_NAME}
ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
CHAT_ENDPOINT = "/chat/completions"

DEFAULT_SYSTEM_PROMPT = (
    "You are a memory deduplication judge for an AI character's long-term memory store.\n"
    "Given a NEW memory and EXISTING similar memories, call the memory_decision tool\n"
    "with the appropriate action. Prefer SKIP over UPDATE when the new memory adds no\n"
    "meaningful information. Prefer UPDATE over DELETE when facts can be merged.\n"
    "Only use DELETE when the new memory directly contradicts an existing one."
)

MEMORY_DECISION_TOOL = {
    "type": "function",
    "function": {
        "name": "memory_decision",
        "description": "Decide how to handle a candidate memory relative to existing memories.",
        "parameters": {
            "type": "object",
            "required": ["action"],
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["ADD", "SKIP", "UPDATE", "DELETE"],
                    "description": (
                        "ADD: genuinely new info. SKIP: already captured. "
                        "UPDATE: improves/corrects existing. DELETE: contradicts existing."
                    ),
                },
                "target_id": {
                    "type": "string",
                    "description": (
                        "ID of the existing memory to UPDATE or DELETE. "
                        "Required for UPDATE/DELETE, omit for ADD/SKIP."
                    ),
                },
                "merged_text": {
                    "type": "string",
                    "description": (
                        "Merged content combining old and new information. "
                        "Required for UPDATE, omit otherwise."
                    ),
                },
            },
        },
    },
}


def _resolve_env_vars(value: str) -> str:
    """Replace ${VAR_NAME} patterns with environment variable values."""
    def replace_match(match: re.Match) -> str:
        var_name = match.group(1)
        env_value = os.environ.get(var_name)
        if env_value is None:
            raise ValueError(
                f"Environment variable '{var_name}' not set. "
                f"Required for curation API key."
            )
        return env_value

    return ENV_VAR_PATTERN.sub(replace_match, value)


@dataclass
class CurationDecision:
    """Result of a curation classification."""

    action: str  # ADD, SKIP, UPDATE, DELETE
    target_id: Optional[str] = None
    merged_text: Optional[str] = None
    raw_response: Optional[str] = None


class CurationClient:
    """
    OpenRouter-based LLM judge for memory deduplication.

    Uses function calling to force structured ADD/SKIP/UPDATE/DELETE
    classification with 7 server-side validation guards.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "anthropic/claude-haiku-4-5",
        system_prompt: Optional[str] = None,
        timeout: float = 30.0,
        base_url: str = DEFAULT_BASE_URL,
    ):
        self._api_key = _resolve_env_vars(api_key) if "${" in api_key else api_key
        self._model = model
        self._system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._timeout = timeout
        self._base_url = base_url.rstrip("/")

    def classify(
        self,
        candidate: str,
        existing: list[dict],
    ) -> CurationDecision:
        """
        Classify a candidate memory against existing similar memories.

        Args:
            candidate: The new memory text to evaluate.
            existing: List of dicts with keys: id, content, distance.

        Returns:
            CurationDecision with validated action.
        """
        valid_ids = {e["id"] for e in existing}

        # Build user message
        existing_lines = []
        for e in existing:
            existing_lines.append(
                f'[id: {e["id"]}] "{e["content"]}" (distance: {e.get("distance", "?")})'
            )
        user_content = f'NEW: "{candidate}"\n\nEXISTING:\n' + "\n".join(existing_lines)

        messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "user", "content": user_content},
        ]

        payload = {
            "model": self._model,
            "messages": messages,
            "tools": [MEMORY_DECISION_TOOL],
            "tool_choice": {"type": "function", "function": {"name": "memory_decision"}},
            "temperature": 0,
            "max_tokens": 200,
            "stream": False,
        }

        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "HTTP-Referer": "https://github.com/spindl-ai/spindl",
            "X-Title": "SpindL Memory Curation",
            "Content-Type": "application/json",
        }

        # --- Make API call ---
        try:
            response = requests.post(
                f"{self._base_url}{CHAT_ENDPOINT}",
                headers=headers,
                json=payload,
                timeout=self._timeout,
            )
        except requests.exceptions.Timeout:
            logger.warning("Curation API timed out after %.1fs — fallback to ADD", self._timeout)
            return CurationDecision(action="ADD", raw_response="TIMEOUT")
        except requests.exceptions.ConnectionError:
            logger.warning("Curation API unreachable — fallback to ADD")
            return CurationDecision(action="ADD", raw_response="CONNECTION_ERROR")
        except Exception as e:
            logger.warning("Curation API request failed: %s — fallback to ADD", e)
            return CurationDecision(action="ADD", raw_response=str(e))

        if response.status_code != 200:
            logger.warning(
                "Curation API returned %d: %s — fallback to ADD",
                response.status_code,
                response.text[:200],
            )
            return CurationDecision(action="ADD", raw_response=response.text[:200])

        # --- Parse response ---
        try:
            data = response.json()
        except Exception:
            logger.warning("Curation API returned non-JSON — fallback to ADD")
            return CurationDecision(action="ADD", raw_response=response.text[:200])

        raw_str = json.dumps(data, indent=None)[:500]

        # Guard 1: Response must be a tool call
        choice = data.get("choices", [{}])[0]
        message = choice.get("message", {})
        tool_calls = message.get("tool_calls", [])

        if not tool_calls:
            logger.warning("Curation: model returned text instead of tool call — fallback to ADD")
            return CurationDecision(action="ADD", raw_response=raw_str)

        # Parse the first tool call
        tc = tool_calls[0]
        func = tc.get("function", {})
        func_name = func.get("name", "")
        args_str = func.get("arguments", "{}")

        if func_name != "memory_decision":
            logger.warning("Curation: unexpected function '%s' — fallback to ADD", func_name)
            return CurationDecision(action="ADD", raw_response=raw_str)

        try:
            args = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            logger.warning("Curation: failed to parse tool arguments — fallback to ADD")
            return CurationDecision(action="ADD", raw_response=raw_str)

        action = args.get("action", "").upper()
        target_id = args.get("target_id")
        merged_text = args.get("merged_text")

        # Guard 2: Action must be in enum
        if action not in {"ADD", "SKIP", "UPDATE", "DELETE"}:
            logger.warning("Curation: unknown action '%s' — fallback to ADD", action)
            return CurationDecision(action="ADD", raw_response=raw_str)

        # Guard 3: Target ID validation (UPDATE/DELETE must reference an existing ID)
        if action in ("UPDATE", "DELETE"):
            if not target_id or target_id not in valid_ids:
                logger.warning(
                    "Curation: %s with invalid target_id '%s' (valid: %s) — fallback to ADD",
                    action,
                    target_id,
                    valid_ids,
                )
                return CurationDecision(action="ADD", raw_response=raw_str)

        # Guard 4: Merged text validation (UPDATE only)
        if action == "UPDATE":
            if not merged_text or len(merged_text.strip()) < 10:
                logger.warning("Curation: UPDATE with empty/short merged_text — fallback to SKIP")
                return CurationDecision(action="SKIP", raw_response=raw_str)

            # Merged text must differ from both candidate and existing
            existing_content = next(
                (e["content"] for e in existing if e["id"] == target_id), ""
            )
            if merged_text.strip() == candidate.strip():
                logger.warning("Curation: UPDATE merged_text identical to candidate — fallback to SKIP")
                return CurationDecision(action="SKIP", raw_response=raw_str)
            if merged_text.strip() == existing_content.strip():
                logger.warning("Curation: UPDATE merged_text identical to existing — fallback to SKIP")
                return CurationDecision(action="SKIP", raw_response=raw_str)

        # --- All guards passed ---
        decision = CurationDecision(
            action=action,
            target_id=target_id,
            merged_text=merged_text,
            raw_response=raw_str,
        )

        logger.info(
            "Curation decision: %s (target=%s, merged=%s): %.60s...",
            action,
            target_id or "N/A",
            bool(merged_text),
            candidate,
        )

        return decision
