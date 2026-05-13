"""
Twitch message selection pass (NANO-130 Phase 1).

Evaluates buffered Twitch messages and selects the single best one
to respond to — or rejects all if none pass the quality/safety gate.

Two modes:
  - LLM: structured-output call to OpenRouter, returns one index or null.
  - Heuristic: score by length, question detection, metadata flags.

LLM failure falls back to heuristic. Never batches. One winner or silence.
"""

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Optional

import requests

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
CHAT_ENDPOINT = "/chat/completions"

_ENV_VAR_PATTERN = re.compile(r"\$\{([^}]+)\}")

_SELECTION_SYSTEM_PROMPT = (
    "You are a message selector for a livestream AI co-host's Twitch chat. "
    "You will receive a numbered list of chat messages. "
    "Select the single best message to respond to, or return null if none "
    "are appropriate.\n"
    "\n"
    "SELECT: substantive content, on-topic, polite, first-time chatters, "
    "questions, compliments, genuine reactions.\n"
    "REJECT ALL: toxic/hateful content, sexual content, doxxing, slurs, "
    "harassment, spam, bot messages, one-word reactions, emote-only, "
    "copypasta, off-topic noise, single emojis.\n"
    "\n"
    "Respond with ONLY a JSON object: {\"selected\": <1-based index>} "
    "or {\"selected\": null} if every message should be rejected.\n"
    "Do not include any other text."
)

_HEURISTIC_MIN_SCORE = 2.0


def _resolve_env_vars(value: str) -> str:
    def _replace(match: re.Match) -> str:
        return os.environ.get(match.group(1), match.group(0))
    return _ENV_VAR_PATTERN.sub(_replace, value)


@dataclass
class SelectionResult:
    selected_index: Optional[int]
    mode: str
    reason: str = ""


class TwitchSelector:
    def __init__(
        self,
        api_key: str = "",
        model: str = "",
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 10.0,
    ):
        self._api_key = api_key
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

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

    def is_configured(self) -> bool:
        resolved_key = _resolve_env_vars(self._api_key) if self._api_key else ""
        return bool(resolved_key and self._model)

    def select(
        self,
        messages: list[dict],
        mode: str = "llm",
    ) -> SelectionResult:
        if not messages:
            return SelectionResult(selected_index=None, mode=mode, reason="empty")

        if mode == "llm" and self.is_configured():
            result = self._select_llm(messages)
            if result is not None:
                return result
            print("[NANO-130] LLM selection failed, falling back to heuristic", flush=True)

        return self._select_heuristic(messages)

    def _select_llm(self, messages: list[dict]) -> Optional[SelectionResult]:
        resolved_key = _resolve_env_vars(self._api_key)

        numbered = []
        for i, m in enumerate(messages):
            numbered.append(f"{i + 1}. {m['username']}: {m['text']}")
        user_message = "\n".join(numbered)

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
                        {"role": "system", "content": _SELECTION_SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ],
                    "temperature": 0.0,
                    "max_tokens": 30,
                    "response_format": {"type": "json_object"},
                },
                timeout=self._timeout,
            )

            if response.status_code != 200:
                print(
                    f"[NANO-130] Selection pass API error {response.status_code}: {response.text[:200]}",
                    flush=True,
                )
                return None

            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                print("[NANO-130] Selection pass: no choices in response", flush=True)
                return None

            content = choices[0].get("message", {}).get("content", "")
            if not content:
                print("[NANO-130] Selection pass: empty content from LLM", flush=True)
                return None

            content = content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[-1].rsplit("```", 1)[0].strip()

            print(f"[NANO-130] Selection pass raw content: {content!r}", flush=True)
            parsed = json.loads(content)
            selected = parsed.get("selected")

            if selected is None:
                return SelectionResult(
                    selected_index=None, mode="llm", reason="all_rejected"
                )

            idx = int(selected) - 1
            if idx < 0 or idx >= len(messages):
                print(
                    f"[NANO-130] Selection pass returned out-of-range index {idx + 1} (count={len(messages)})",
                    flush=True,
                )
                return None

            return SelectionResult(
                selected_index=idx, mode="llm", reason="selected"
            )

        except requests.Timeout:
            print(f"[NANO-130] Selection pass timed out after {self._timeout:.1f}s", flush=True)
            return None
        except requests.RequestException as e:
            print(f"[NANO-130] Selection pass request failed: {e}", flush=True)
            return None
        except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
            print(f"[NANO-130] Selection pass response parse error: {e}", flush=True)
            return None

    def _select_heuristic(self, messages: list[dict]) -> SelectionResult:
        best_idx = -1
        best_score = -1.0

        for i, m in enumerate(messages):
            score = self._score_message(m["text"])
            if score > best_score:
                best_score = score
                best_idx = i

        if best_score < _HEURISTIC_MIN_SCORE:
            return SelectionResult(
                selected_index=None,
                mode="heuristic",
                reason=f"all_below_threshold ({best_score:.1f} < {_HEURISTIC_MIN_SCORE})",
            )

        return SelectionResult(
            selected_index=best_idx,
            mode="heuristic",
            reason=f"score={best_score:.1f}",
        )

    @staticmethod
    def _score_message(text: str) -> float:
        score = 0.0
        stripped = text.strip()
        length = len(stripped)

        if length < 3:
            return 0.0

        if length >= 10:
            score += 1.0
        if length >= 30:
            score += 1.0
        if length >= 60:
            score += 0.5

        if "?" in stripped:
            score += 2.0

        words = stripped.split()
        if len(words) >= 4:
            score += 1.0

        alpha_ratio = sum(1 for c in stripped if c.isalpha()) / max(1, length)
        if alpha_ratio < 0.3:
            score -= 2.0

        return score
