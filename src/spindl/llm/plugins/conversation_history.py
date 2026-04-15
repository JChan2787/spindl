"""Conversation history plugin for persistent multi-turn conversations."""

from datetime import datetime, timezone
from pathlib import Path

from .base import PipelineContext, PreProcessor, PostProcessor
from ...history.jsonl_store import (
    generate_session_filename,
    generate_uuid,
    append_turn,
    read_visible_turns,
    get_latest_session,
    get_next_turn_id,
    save_last_session,
)


class ConversationHistoryManager:
    """
    Manages conversation history state and JSONL persistence.

    Shared between the pre-processor and post-processor plugins.
    Handles session file management, turn storage, and history retrieval.

    Storage format: One JSONL file per session.
    File location: {conversations_dir}/{persona_id}_{timestamp}.jsonl

    Turn format:
        {"turn_id": 1,
         "uuid": "...",
         "role": "user"|"assistant"|"summary",
         "content": "...",
         "timestamp": "ISO8601",
         "hidden": false}

    The `hidden` field supports future summarization:
    - hidden=False: Included in context injection
    - hidden=True: Stored but excluded from injection
    """

    def __init__(
        self,
        conversations_dir: str = "./conversations",
        resume_session: bool = False,
        debug: bool = False,
    ):
        """
        Initialize the conversation history manager.

        Args:
            conversations_dir: Directory for JSONL session files
            resume_session: If True, attempts to load the most recent session
                           for the persona. If False, always starts a new session.
            debug: If True, enables verbose debug logging.
        """
        self._conversations_dir = Path(conversations_dir)
        self._resume_session = resume_session
        self._debug = debug

        self._session_file: Path | None = None
        self._persona_id: str | None = None
        self._history: list[dict] = []
        self._pending_user_input: str | None = None
        self._next_turn_id: int = 1

    def _persist_active_session(self) -> None:
        """Write the active session filename to the .last_session marker."""
        if self._session_file and self._persona_id:
            save_last_session(
                self._conversations_dir, self._persona_id, self._session_file.name
            )

    def ensure_session(self, persona_id: str) -> None:
        """
        Ensure a session file exists for the given persona.

        Called on first pipeline run. Creates new session or resumes existing.

        Args:
            persona_id: Persona identifier (e.g., "spindle")
        """
        if self._session_file is not None and self._persona_id == persona_id:
            if self._debug:
                print(f"[DEBUG:HistoryManager] ensure_session() - SKIPPED (already initialized, _history={len(self._history)})")
            return

        if self._debug:
            print(f"[DEBUG:HistoryManager] ensure_session() - INITIALIZING for persona '{persona_id}'")
            print(f"[DEBUG:HistoryManager]   resume_session={self._resume_session}")

        self._persona_id = persona_id

        if self._resume_session:
            existing = get_latest_session(self._conversations_dir, persona_id)
            if existing:
                self._session_file = existing
                self._history = read_visible_turns(existing)
                self._next_turn_id = get_next_turn_id(existing)
                if self._debug:
                    print(f"[DEBUG:HistoryManager]   RESUMED existing session: {existing}")
                    print(f"[DEBUG:HistoryManager]   Loaded {len(self._history)} turns, next_turn_id={self._next_turn_id}")
                self._persist_active_session()
                return

        filename = generate_session_filename(persona_id)
        self._session_file = self._conversations_dir / filename
        self._history = []
        self._next_turn_id = 1
        self._persist_active_session()
        if self._debug:
            print(f"[DEBUG:HistoryManager]   CREATED new session: {self._session_file}")
            print(f"[DEBUG:HistoryManager]   Starting with empty history")

    def get_visible_history(self) -> list[dict]:
        """
        Get visible history as LLM-compatible messages.

        Returns:
            List of message dicts with role and content fields.
            Summary turns are mapped to system role with prefix.
        """
        messages = []
        for turn in self._history:
            role = turn["role"]
            if role == "summary":
                messages.append({
                    "role": "system",
                    "content": f"[Previous conversation summary]: {turn['content']}"
                })
            else:
                messages.append({
                    "role": role,
                    "content": turn["content"]
                })
        return messages

    def stash_user_input(self, user_input: str) -> None:
        """
        Stash user input for later storage by post-processor.

        Args:
            user_input: The current user utterance
        """
        self._pending_user_input = user_input

    def store_turn(
        self,
        response: str,
        reasoning: str = None,
        input_modality: str = None,
        stimulus_source: str = None,
        activated_codex_entries: list = None,
        retrieved_memories: list = None,
        tts_text: str = None,
    ) -> None:
        """
        Store the pending user input and assistant response to JSONL.

        Appends two records:
            1. User turn (from stashed input, with input_modality)
            2. Assistant turn (from response, with reasoning and metadata)

        Also updates in-memory history for subsequent turns in same session.
        Reasoning is stored in JSONL for inspection but NOT replayed to the LLM
        (HistoryInjector only reads 'content', not 'reasoning').

        NANO-109: When tts_text is provided, it becomes the 'content' field
        (what HistoryInjector replays to the LLM). The raw response is preserved
        as 'display_content' in JSONL for inspection. This steers the LLM away
        from generating action markers and narrative prose by showing it clean
        dialogue in its own conversation history.

        Args:
            response: The assistant's raw response text
            reasoning: Optional thinking/reasoning content from the LLM (NANO-042)
            input_modality: Input source — "VOICE", "TEXT", or "stimulus" (NANO-075)
            stimulus_source: Stimulus identifier — "patience", "custom", etc. (NANO-075)
            activated_codex_entries: Codex entries activated for display (NANO-075)
            retrieved_memories: RAG memories retrieved for display (NANO-075)
            tts_text: TTS-cleaned text for history replay (NANO-109)
        """
        if self._session_file is None or self._pending_user_input is None:
            return

        timestamp = datetime.now(timezone.utc).isoformat()

        user_turn = {
            "turn_id": self._next_turn_id,
            "uuid": generate_uuid(),
            "role": "user",
            "content": self._pending_user_input,
            "timestamp": timestamp,
            "hidden": False,
        }

        # NANO-075: Persist input modality on user turn
        if input_modality:
            user_turn["input_modality"] = input_modality

        # NANO-109: Use cleaned text for history content (steers LLM output).
        # Preserve raw response as display_content for JSONL inspection.
        history_content = tts_text if tts_text else response

        assistant_turn = {
            "turn_id": self._next_turn_id + 1,
            "uuid": generate_uuid(),
            "role": "assistant",
            "content": history_content,
            "timestamp": timestamp,
            "hidden": False,
        }

        # NANO-109: Preserve raw response when tts_text differs
        if tts_text and tts_text != response:
            assistant_turn["display_content"] = response

        # NANO-042: Store reasoning alongside assistant turn for inspection
        if reasoning:
            assistant_turn["reasoning"] = reasoning

        # NANO-075: Persist metadata for hydration survival
        if stimulus_source:
            assistant_turn["stimulus_source"] = stimulus_source
        if activated_codex_entries:
            assistant_turn["activated_codex_entries"] = activated_codex_entries
        if retrieved_memories:
            assistant_turn["retrieved_memories"] = retrieved_memories

        append_turn(self._session_file, user_turn)
        append_turn(self._session_file, assistant_turn)

        self._history.append(user_turn)
        self._history.append(assistant_turn)

        self._next_turn_id += 2
        self._pending_user_input = None

    def amend_last_assistant_content(self, truncated_content: str) -> None:
        """
        Amend the last assistant turn's content to reflect what was actually
        delivered before barge-in (NANO-111 Phase 2.5).

        Updates both in-memory history and the JSONL file on disk.
        Called when barge-in truncates the response to only delivered sentences.

        Args:
            truncated_content: The text that was actually spoken/delivered.
        """
        if not self._history:
            return

        # Find the last assistant turn in memory
        for i in range(len(self._history) - 1, -1, -1):
            if self._history[i].get("role") == "assistant":
                original = self._history[i].get("content", "")
                self._history[i]["content"] = truncated_content
                # Preserve the full generation as display_content for inspection
                if "display_content" not in self._history[i]:
                    self._history[i]["display_content"] = original
                break
        else:
            return  # No assistant turn found

        # Amend the JSONL file on disk
        if self._session_file and self._session_file.exists():
            from ...history.jsonl_store import patch_last_turn
            patch_last_turn(self._session_file, {
                "content": truncated_content,
                "display_content": original,
                "barge_in_truncated": True,
            })

    @property
    def session_file(self) -> Path | None:
        """Get the current session file path."""
        return self._session_file

    @property
    def turn_count(self) -> int:
        """Get the number of visible turns in current session."""
        return len(self._history)

    def get_history(self) -> list[dict]:
        """Get a copy of the current visible history."""
        return list(self._history)

    def clear_session(self) -> None:
        """
        Clear in-memory history and start a new session file.

        Does not delete the old session file.
        """
        if self._persona_id:
            filename = generate_session_filename(self._persona_id)
            self._session_file = self._conversations_dir / filename
        self._history = []
        self._pending_user_input = None
        self._next_turn_id = 1
        self._persist_active_session()

    def switch_to_persona(self, persona_id: str) -> None:
        """
        Switch to a persona, resuming their latest session if one exists.

        Used by runtime character switching (NANO-077). Checks for an existing
        session via get_latest_session (marker file first, then most recent
        non-empty file). Only creates a new session if no prior sessions exist.

        Args:
            persona_id: The new persona identifier (e.g., "mryummers")
        """
        self._persona_id = persona_id

        # Try to resume the latest session for this persona
        latest = get_latest_session(self._conversations_dir, persona_id)
        if latest:
            self._session_file = latest
            self._history = read_visible_turns(latest)
            self._next_turn_id = get_next_turn_id(latest)
        else:
            # No prior sessions — create a fresh one
            filename = generate_session_filename(persona_id)
            self._session_file = self._conversations_dir / filename
            self._history = []
            self._next_turn_id = 1

        self._pending_user_input = None
        self._persist_active_session()

    def load_session(self, filepath: Path) -> bool:
        """
        Load a specific session file.

        Switches the active session to the specified file and loads
        its visible history into memory.

        Args:
            filepath: Path to the JSONL session file

        Returns:
            True if session was loaded successfully, False otherwise
        """
        if not filepath.exists():
            return False

        try:
            self._session_file = filepath
            self._history = read_visible_turns(filepath)
            self._next_turn_id = get_next_turn_id(filepath)

            # Extract persona_id from filename (format: {persona}_{timestamp}.jsonl)
            parts = filepath.stem.rsplit("_", 2)
            if len(parts) >= 2:
                self._persona_id = parts[0]

            self._pending_user_input = None
            self._persist_active_session()

            if self._debug:
                print(f"[DEBUG:HistoryManager] load_session() - Loaded {filepath}")
                print(f"[DEBUG:HistoryManager]   {len(self._history)} visible turns, next_turn_id={self._next_turn_id}")

            return True
        except Exception as e:
            if self._debug:
                print(f"[DEBUG:HistoryManager] load_session() - FAILED: {e}")
            return False


class HistoryInjector(PreProcessor):
    """
    PreProcessor that injects conversation history into the prompt.

    Two paths, selected by the active provider's `supports_role_history`
    capability (stashed on context.metadata by the pipeline):

    - **Flattened (default, False):** formats visible history turns as inline
      text and substitutes into the [RECENT_HISTORY] placeholder in the
      system message. Preserves positional emphasis of
      "Respond as [PERSONA_NAME]." at the end. Messages stay [system, user].

    - **Spliced (NANO-114, True):** collapses [RECENT_HISTORY] placeholder to
      empty string and splices real role-tagged messages into context.messages
      between the system prompt and the current user turn. Required for
      strict chat-template models (Gemma-3, Gemma-4) whose jinja templates
      refuse to treat flattened bracket headers as turn boundaries.
    """

    PLACEHOLDER = "[RECENT_HISTORY]"

    def __init__(self, manager: ConversationHistoryManager):
        """
        Initialize the history injector.

        Args:
            manager: Shared ConversationHistoryManager instance
        """
        self._manager = manager

    @property
    def name(self) -> str:
        return "history_injector"

    def process(self, context: PipelineContext) -> PipelineContext:
        """
        Inject conversation history via flatten or splice path based on
        provider capability.

        Args:
            context: Current pipeline context

        Returns:
            Modified context. Flatten path: system message updated.
            Splice path: context.messages extended with role-array history.
        """
        persona_id = context.persona.get("id", "unknown")
        persona_name = context.persona.get("name", "Assistant")
        self._manager.ensure_session(persona_id)
        self._manager.stash_user_input(context.user_input)

        history_messages = self._manager.get_visible_history()

        use_splice = bool(context.metadata.get("provider_supports_role_history", False))

        if self._manager._debug:
            mode = "SPLICE" if use_splice else "FLATTEN"
            print(f"[DEBUG:HistoryInjector] ====== HISTORY INJECTION START [{mode}] ======")
            print(f"[DEBUG:HistoryInjector] In-memory _history count: {len(self._manager._history)}")
            print(f"[DEBUG:HistoryInjector] Visible history messages: {len(history_messages)}")
            print(f"[DEBUG:HistoryInjector] Current user input: {context.user_input[:50]}...")
            if history_messages:
                for i, msg in enumerate(history_messages):
                    print(f"[DEBUG:HistoryInjector]   [{i}] {msg['role']}: {msg['content'][:60]}...")
            else:
                print(f"[DEBUG:HistoryInjector]   (no history to inject)")
            print(f"[DEBUG:HistoryInjector] ====== HISTORY INJECTION END ======")

        # Always compute flattened text (used for flatten path AND for token
        # counting on splice path — Workshop block counts stay honest).
        if history_messages:
            formatted = self._format_history(history_messages, persona_name)
        else:
            formatted = ""

        # NANO-045b: Stash formatted history text for per-block token counting
        context.metadata["history_formatted"] = formatted

        if use_splice:
            self._splice_role_history(context, history_messages)
        else:
            self._inject_flattened(context, formatted)

        return context

    def _inject_flattened(self, context: PipelineContext, formatted: str) -> None:
        """
        Legacy path: substitute [RECENT_HISTORY] in the system message with
        bracket-formatted history text.
        """
        if context.messages and context.messages[0].get("role") == "system":
            system_content = context.messages[0]["content"]
            if self.PLACEHOLDER in system_content:
                system_content = system_content.replace(self.PLACEHOLDER, formatted)
                # Collapse 3+ consecutive newlines to 2
                import re
                system_content = re.sub(r"\n{3,}", "\n\n", system_content)
                context.messages[0]["content"] = system_content

    def _splice_role_history(
        self, context: PipelineContext, history_messages: list[dict]
    ) -> None:
        """
        NANO-114: Collapse [RECENT_HISTORY] placeholder to empty string and
        splice real role-tagged messages between the system prompt and the
        current user turn.

        Also strips the orphan "### Conversation" section header when both
        [CONVERSATION_SUMMARY] and [RECENT_HISTORY] collapsed to empty (no
        summary in history and no turns to inject).
        """
        if not context.messages or context.messages[0].get("role") != "system":
            return

        import re

        system_content = context.messages[0]["content"]

        # Collapse the history placeholder to empty
        system_content = system_content.replace(self.PLACEHOLDER, "")

        # Strip orphan "### Conversation" header when its content blocks are
        # both empty. Matches the header followed by any amount of whitespace
        # before the next "###" section or end of string.
        system_content = re.sub(
            r"\n### Conversation\s*(?=\n###|\Z)",
            "",
            system_content,
        )

        # Collapse 3+ consecutive newlines to 2
        system_content = re.sub(r"\n{3,}", "\n\n", system_content)
        context.messages[0]["content"] = system_content

        if not history_messages:
            return

        # Splice history into context.messages between system (idx 0) and
        # the current user turn (last message). Summary turns (role=system)
        # are preserved as intermediate system messages — llama.cpp + jinja
        # handles multi-system arrays; if a specific model regresses, fall
        # back to coercing summaries onto the main system prompt.
        current_user = context.messages[-1]
        context.messages[:] = (
            [context.messages[0]]
            + list(history_messages)
            + [current_user]
        )

    def _format_history(
        self, messages: list[dict], persona_name: str
    ) -> str:
        """
        Format history messages as inline text for system prompt.

        Uses bracket format to match NANO-037 hydrated example:
            [user]: What should I make for dinner?
            [Spindle]: Oh, hmm—have you tried shakshuka?

        Args:
            messages: List of message dicts with role and content
            persona_name: Persona name for assistant labels

        Returns:
            Formatted history string
        """
        lines = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "user":
                lines.append(f"[user]: {content}")
            elif role == "assistant":
                lines.append(f"[{persona_name}]: {content}")
            elif role == "system":
                # Summary turns mapped to system role by get_visible_history()
                lines.append(content)
        return "\n".join(lines)


class HistoryRecorder(PostProcessor):
    """
    PostProcessor that records turns to JSONL.

    Appends the current exchange (user input + assistant response)
    to the session JSONL file after each LLM call.
    """

    def __init__(self, manager: ConversationHistoryManager):
        """
        Initialize the history recorder.

        Args:
            manager: Shared ConversationHistoryManager instance
        """
        self._manager = manager

    @property
    def name(self) -> str:
        return "history_recorder"

    def process(self, context: PipelineContext, response: str) -> str:
        """
        Store the current turn to JSONL.

        Args:
            context: Pipeline context (reads metadata["reasoning"] for NANO-042)
            response: The assistant's response text

        Returns:
            Response unchanged (storage only, no transformation)
        """
        if self._manager._debug:
            print(f"[DEBUG:HistoryRecorder] ====== HISTORY RECORDING START ======")
            print(f"[DEBUG:HistoryRecorder] Pending user input: {self._manager._pending_user_input[:50] if self._manager._pending_user_input else 'NONE'}...")
            print(f"[DEBUG:HistoryRecorder] Response to store: {response[:50]}...")
            print(f"[DEBUG:HistoryRecorder] In-memory _history BEFORE store: {len(self._manager._history)}")

        # NANO-042: Pass reasoning from context metadata for JSONL persistence
        reasoning = context.metadata.get("reasoning")

        # NANO-109: Use TTS-cleaned text for history content (steers LLM away
        # from generating action markers and narrative prose). Raw response
        # preserved as display_content in JSONL for inspection.
        tts_text = context.metadata.get("tts_text")

        # NANO-075: Extract metadata for JSONL persistence (hydration survival)
        input_modality = context.metadata.get("input_modality")
        stimulus_source = context.metadata.get("stimulus_source")

        # Codex/memory display data — extract same way pipeline._extract_codex_display_data() does
        # These are ActivationResult dataclass objects from CodexActivatorPlugin
        codex_results = context.metadata.get("codex_results", [])
        activated_codex = [
            {"name": r.entry_name or "", "keys": [r.matched_keyword] if r.matched_keyword else [],
             "activation_method": r.reason}
            for r in codex_results
            if r.activated
        ] if codex_results else None

        rag_results = context.metadata.get("rag_results", [])
        retrieved_memories = [
            {
                "content_preview": r.get("content", "")[:100],
                "collection": r.get("collection", "general"),
                "distance": r.get("distance", 0.0),
                **({"score": r["score"]} if "score" in r else {}),
            }
            for r in rag_results
        ] if rag_results else None

        self._manager.store_turn(
            response,
            reasoning=reasoning,
            input_modality=input_modality,
            stimulus_source=stimulus_source,
            activated_codex_entries=activated_codex,
            retrieved_memories=retrieved_memories,
            tts_text=tts_text,
        )

        if self._manager._debug:
            print(f"[DEBUG:HistoryRecorder] In-memory _history AFTER store: {len(self._manager._history)}")
            print(f"[DEBUG:HistoryRecorder] ====== HISTORY RECORDING END ======")
        return response


def create_history_plugins(
    conversations_dir: str = "./conversations",
    resume_session: bool = False,
    debug: bool = False,
) -> tuple[HistoryInjector, HistoryRecorder]:
    """
    Factory function to create paired history plugins.

    Creates a shared ConversationHistoryManager and returns both
    the pre-processor (injector) and post-processor (recorder)
    that share it.

    Args:
        conversations_dir: Directory for JSONL session files
        resume_session: If True, resume most recent session for persona
        debug: If True, enables verbose debug logging.

    Returns:
        Tuple of (pre_processor, post_processor) sharing the same manager

    Usage:
        injector, recorder = create_history_plugins()
        pipeline.register_pre_processor(injector)
        pipeline.register_post_processor(recorder)
    """
    manager = ConversationHistoryManager(conversations_dir, resume_session, debug)
    return HistoryInjector(manager), HistoryRecorder(manager)
