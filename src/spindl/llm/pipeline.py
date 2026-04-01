"""
LLMPipeline - Plugin-aware orchestrator for LLM calls.

Coordinates pre-processors, LLM provider, and post-processors
into a single pipeline.

Session 5 (NANO-014): Integrated with BuildContext and provider-based prompt building.
Session 1 (NANO-017): Added token usage tracking via PipelineResult.
Session 1 (NANO-019): Migrated from LlamaClient to LLMProvider interface.
Session X (NANO-024): Added tool calling support via ToolExecutor.
"""

import asyncio
from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from .build_context import BuildContext, InputModality
from .base import LLMProvider, LLMResponse
from .prompt_block import PromptBlock, create_default_blocks, load_block_config
from .prompt_builder import PromptBuilder
from .plugins.base import PipelineContext, PreProcessor, PostProcessor

if TYPE_CHECKING:
    from ..tools import ToolExecutor


@dataclass
class TokenUsage:
    """
    Token usage statistics from an LLM response.

    Provider-agnostic representation of token counts.
    """

    input_tokens: int
    """Tokens in the prompt."""

    output_tokens: int
    """Tokens in the response."""

    reasoning_tokens: Optional[int] = None
    """Tokens used for reasoning/thinking content (NANO-042)."""

    @property
    def total_tokens(self) -> int:
        """Total tokens (input + output)."""
        return self.input_tokens + self.output_tokens

    @property
    def prompt_tokens(self) -> int:
        """Alias for input_tokens (OpenAI naming convention)."""
        return self.input_tokens

    @property
    def completion_tokens(self) -> int:
        """Alias for output_tokens (OpenAI naming convention)."""
        return self.output_tokens

    @classmethod
    def from_llm_response(cls, response: LLMResponse) -> "TokenUsage":
        """Create TokenUsage from LLMResponse."""
        return cls(
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            reasoning_tokens=response.reasoning_tokens,
        )


@dataclass
class PipelineResult:
    """
    Result from an LLM pipeline execution.

    Contains the final response text, token usage statistics,
    and the messages sent to the LLM (for prompt inspection).
    """

    content: str
    """Final processed response text."""

    usage: TokenUsage
    """Token usage statistics from the LLM call."""

    messages: list[dict]
    """The final message list sent to the LLM ([{role, content}, ...])."""

    input_modality: str = "TEXT"
    """Input modality for this call ("VOICE" or "TEXT")."""

    state_trigger: Optional[str] = None
    """Optional state trigger (e.g., "barge_in", "empty_transcription")."""

    activated_codex_entries: list = None
    """List of codex entries activated for this response (NANO-037 Phase 2).
    Each dict contains: name, keys, activation_method."""

    retrieved_memories: list = None
    """List of retrieved memories for this response (NANO-044).
    Each dict contains: content_preview, collection, distance."""

    reasoning: Optional[str] = None
    """Thinking/reasoning content from the LLM, if present (NANO-042)."""

    tts_text: Optional[str] = None
    """TTS-safe version of the response with formatting stripped (NANO-109).
    None when TTSCleanupPlugin is not registered."""

    block_contents: Optional[list[dict]] = None
    """Per-block content data for token breakdown (NANO-045b).
    Each dict has: id, label, section, chars, deferred.
    None when legacy template mode is used (no block_config)."""

    def __post_init__(self):
        """Initialize mutable defaults."""
        if self.activated_codex_entries is None:
            self.activated_codex_entries = []
        if self.retrieved_memories is None:
            self.retrieved_memories = []


class LLMPipeline:
    """
    Plugin-aware LLM pipeline orchestrator.

    Executes the full flow:
    1. Build initial context from user input and persona
    2. Run pre-processors in registration order
    3. Call LLM with processed messages (with optional tool calling loop)
    4. Run post-processors in registration order
    5. Return final response
    """

    def __init__(
        self,
        provider: LLMProvider,
        prompt_builder: PromptBuilder,
        pre_processors: Optional[list[PreProcessor]] = None,
        post_processors: Optional[list[PostProcessor]] = None,
        tool_executor: Optional["ToolExecutor"] = None,
    ):
        """
        Initialize pipeline with provider and plugins.

        Args:
            provider: LLMProvider instance for LLM calls
            prompt_builder: PromptBuilder for message assembly
            pre_processors: Initial list of pre-processors (optional)
            post_processors: Initial list of post-processors (optional)
            tool_executor: Optional ToolExecutor for tool calling support (NANO-024)
        """
        self.provider = provider
        self.prompt_builder = prompt_builder
        self._pre_processors: list[PreProcessor] = pre_processors or []
        self._post_processors: list[PostProcessor] = post_processors or []
        self._tool_executor: Optional["ToolExecutor"] = tool_executor
        self._block_config: Optional[list[PromptBlock]] = None
        # Codex injection wrappers (NANO-045d)
        self._codex_prefix: str = "The following facts are always true in this context:"
        self._codex_suffix: str = ""

    def set_tool_executor(self, executor: Optional["ToolExecutor"]) -> None:
        """
        Set or clear the tool executor.

        Args:
            executor: ToolExecutor instance, or None to disable tool calling
        """
        self._tool_executor = executor

    def set_block_config(self, prompt_blocks_config: Optional[dict]) -> None:
        """
        Configure block-based prompt assembly (NANO-045a).

        When set, the pipeline uses configurable block ordering instead of
        the static CONVERSATION_TEMPLATE. When None, uses the legacy
        template path (byte-identical output).

        Args:
            prompt_blocks_config: The `prompt_blocks` section from spindl.yaml,
                                  or None to use legacy template mode.
        """
        if prompt_blocks_config is not None:
            self._block_config = load_block_config(prompt_blocks_config)
        else:
            self._block_config = None

    def set_codex_wrappers(
        self,
        codex_prefix: Optional[str] = ...,
        codex_suffix: Optional[str] = ...,
    ) -> None:
        """
        Update codex injection wrappers at runtime (NANO-045d).

        Args:
            codex_prefix: Prefix before codex content. Ellipsis = keep current.
            codex_suffix: Suffix after codex content. Ellipsis = keep current.
        """
        if codex_prefix is not ...:
            self._codex_prefix = codex_prefix or ""
        if codex_suffix is not ...:
            self._codex_suffix = codex_suffix or ""

    def set_example_dialogue_wrappers(
        self,
        prefix: Optional[str] = ...,
        suffix: Optional[str] = ...,
    ) -> None:
        """
        Update example dialogue wrappers at runtime (NANO-052 follow-up).

        Finds the ExampleDialogueProvider in the prompt builder's provider
        registry and sets its prefix/suffix wrappers.

        Args:
            prefix: Prefix before example dialogue. Ellipsis = keep current.
            suffix: Suffix after example dialogue. Ellipsis = keep current.
        """
        from .providers.persona_provider import ExampleDialogueProvider

        for provider in self.prompt_builder._providers:
            if isinstance(provider, ExampleDialogueProvider):
                current_prefix = provider._prefix
                current_suffix = provider._suffix
                provider.set_wrappers(
                    prefix=prefix if prefix is not ... else current_prefix,
                    suffix=suffix if suffix is not ... else current_suffix,
                )
                break

    def register_pre_processor(self, plugin: PreProcessor) -> None:
        """
        Add a pre-processor to the pipeline.

        Plugins execute in registration order.
        """
        self._pre_processors.append(plugin)

    def register_post_processor(self, plugin: PostProcessor) -> None:
        """
        Add a post-processor to the pipeline.

        Plugins execute in registration order.
        """
        self._post_processors.append(plugin)

    def run(
        self,
        user_input: str,
        persona: dict,
        generation_params: Optional[dict] = None,
        state_trigger: Optional[str] = None,
        input_modality: InputModality = InputModality.TEXT,
        last_assistant_message: Optional[str] = None,
        stimulus_source: Optional[str] = None,
        stimulus_metadata: Optional[dict] = None,
    ) -> PipelineResult:
        """
        Execute full pipeline.

        Args:
            user_input: User's utterance
            persona: Persona config dict (must have 'system_prompt' or structured fields)
            generation_params: Optional LLM params (temperature, max_tokens, etc.)
                              Defaults pulled from persona['generation'] if present
            state_trigger: Optional state machine trigger for voice state injection
                          (e.g., "barge_in", "empty_transcription")
            input_modality: Input modality (VOICE, TEXT). Defaults to TEXT.
            last_assistant_message: Optional previous assistant response for barge-in context.
                                   When user interrupts, this contains what was being said.
            stimulus_source: Optional stimulus identifier (NANO-075). Persisted to JSONL
                            for hydration. E.g., "patience", "custom".
            stimulus_metadata: Optional metadata from stimulus module (NANO-056b).
                              Contains pre-formatted content for prompt block injection
                              (e.g., "twitch_content" for Twitch Chat block).

        Returns:
            PipelineResult with final response text and token usage statistics

        Raises:
            ConnectionError: LLM server unreachable
            TimeoutError: LLM request timed out
            RuntimeError: LLM generation failed
        """
        # 1. Build initial context
        # Create BuildContext for provider-based prompt building
        build_context = BuildContext(
            input_content=user_input,
            input_modality=input_modality,
            persona=persona,
            state_trigger=state_trigger,
            last_assistant_message=last_assistant_message,
            block_config=self._block_config,
        )

        context = PipelineContext(
            user_input=user_input,
            persona=persona,
            messages=self.prompt_builder.build(
                persona, user_input, build_context=build_context
            ),
        )

        # NANO-045b: Transfer per-block content data for token counting
        if build_context.block_contents is not None:
            context.metadata["block_contents"] = build_context.block_contents

        # NANO-075: Stash metadata for HistoryRecorder JSONL persistence
        context.metadata["input_modality"] = input_modality.value
        if stimulus_source:
            context.metadata["stimulus_source"] = stimulus_source

        # NANO-056b: Populate twitch content from stimulus metadata
        if stimulus_metadata and "twitch_content" in stimulus_metadata:
            context.metadata["twitch_content"] = stimulus_metadata["twitch_content"]

        # 2. Run pre-processors
        for plugin in self._pre_processors:
            context = plugin.process(context)

        # 2b. Inject codex content into system prompt (if any)
        # CodexActivatorPlugin populates metadata["codex_content"] during preprocessing.
        # We replace the [CODEX_CONTEXT] placeholder here, after all preprocessors run.
        self._inject_codex_content(context)

        # 2c. Inject RAG memory content into system prompt (NANO-043 Phase 2)
        # RAGInjector populates metadata["rag_content"] during preprocessing.
        # We replace the [RAG_CONTEXT] placeholder here, same pattern as codex.
        self._inject_rag_content(context)

        # 2d. Inject Twitch chat content into system prompt (NANO-056b)
        # Twitch content arrives via stimulus_metadata, populated above.
        self._inject_twitch_content(context)

        # 2e. Patch deferred block char counts now that injections are done (NANO-045b)
        self._update_deferred_block_contents(context)

        # 3. Resolve generation parameters
        params = self._resolve_generation_params(persona, generation_params)

        # DEBUG: Dump full prompt to log for vision debugging
        # TODO: Remove or gate behind debug flag after NANO-023 testing
        # Unwrap ProviderHolder to show the actual inner provider
        from .provider_holder import ProviderHolder
        _inner = self.provider.provider if isinstance(self.provider, ProviderHolder) else self.provider
        provider_type = type(_inner).__name__
        is_cloud = _inner.__class__.is_cloud_provider()
        provider_label = f"{provider_type} ({'cloud' if is_cloud else 'local'})"
        print(f"[Prompt] --- BEGIN FULL PROMPT [{provider_label}] ---", flush=True)
        for msg in context.messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            # Truncate for readability but show enough to see vision injection
            if len(content) > 2000:
                display = content[:1000] + "\n...[truncated]...\n" + content[-500:]
            else:
                display = content
            print(f"[Prompt] [{role}]:\n{display}", flush=True)
        print("[Prompt] --- END FULL PROMPT ---", flush=True)

        # 4. Call LLM (with optional tool execution loop)
        if self._tool_executor is not None:
            # Use tool executor for tool-aware generation
            llm_response = self._run_with_tools(context.messages, params)
        else:
            # Direct LLM call (no tools)
            llm_response = self.provider.generate(
                messages=context.messages,
                **params,
            )

        # 5. Stash reasoning in context metadata for post-processors (NANO-042)
        # HistoryRecorder reads this to persist reasoning alongside the turn in JSONL
        if llm_response.reasoning:
            context.metadata["reasoning"] = llm_response.reasoning

        # 6. Run post-processors on content
        response_text = llm_response.content
        for plugin in self._post_processors:
            response_text = plugin.process(context, response_text)

        # 7. Extract activated codex entries for GUI display (NANO-037 Phase 2)
        # CodexActivatorPlugin populates metadata["codex_results"] during preprocessing
        activated_codex = self._extract_codex_display_data(context)

        # 8. Extract retrieved memories for GUI display (NANO-044)
        # RAGInjector populates metadata["rag_results"] during preprocessing
        retrieved_memories = self._extract_rag_display_data(context)

        return PipelineResult(
            content=response_text,
            usage=TokenUsage.from_llm_response(llm_response),
            messages=context.messages,
            input_modality=input_modality.value,
            state_trigger=state_trigger,
            activated_codex_entries=activated_codex,
            retrieved_memories=retrieved_memories,
            reasoning=context.metadata.get("reasoning"),
            tts_text=context.metadata.get("tts_text"),
            block_contents=context.metadata.get("block_contents"),
        )

    def build_snapshot(self, persona: dict) -> dict:
        """
        Build a prompt snapshot without calling the LLM (NANO-076).

        Duplicates the context-building steps of run() (steps 1-2d) with a
        sentinel user input, then estimates token counts via tiktoken.

        This is the cold reconstruction path — used when no sidecar exists
        (legacy sessions) or for new sessions before the first message.

        Args:
            persona: Persona config dict.

        Returns:
            Snapshot dict with messages, token_breakdown, block_contents,
            input_modality, estimated flag.
        """
        import tiktoken

        sentinel = "[snapshot_preview]"

        # 1. Build initial context (mirrors run() steps 1-2d)
        build_context = BuildContext(
            input_content=sentinel,
            input_modality=InputModality.TEXT,
            persona=persona,
            state_trigger=None,
            last_assistant_message=None,
            block_config=self._block_config,
        )

        context = PipelineContext(
            user_input=sentinel,
            persona=persona,
            messages=self.prompt_builder.build(
                persona, sentinel, build_context=build_context
            ),
        )

        if build_context.block_contents is not None:
            context.metadata["block_contents"] = build_context.block_contents

        # 2. Run pre-processors
        for plugin in self._pre_processors:
            context = plugin.process(context)

        # 2b-2e. Inject deferred content and patch block char counts
        self._inject_codex_content(context)
        self._inject_rag_content(context)
        self._inject_twitch_content(context)
        self._update_deferred_block_contents(context)

        # Estimate tokens via tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        prompt_tokens = 0
        system_content = ""
        user_content = ""
        for msg in context.messages:
            content = msg.get("content", "")
            prompt_tokens += len(enc.encode(content)) + 4  # overhead per message
            role = msg.get("role", "")
            if role == "system":
                system_content = content
            elif role == "user":
                user_content = content

        # Build token breakdown inline (mirrors callbacks._build_token_breakdown)
        block_contents = context.metadata.get("block_contents")
        total_chars = len(system_content) + len(user_content)
        system_tokens = 0
        user_tokens = 0
        if total_chars > 0 and prompt_tokens > 0:
            system_ratio = len(system_content) / total_chars
            system_tokens = int(prompt_tokens * system_ratio)
            user_tokens = prompt_tokens - system_tokens

        breakdown = {
            "total": prompt_tokens,
            "prompt": prompt_tokens,
            "completion": 0,
            "system": system_tokens,
            "user": user_tokens,
            "sections": {"agent": 0, "context": 0, "rules": 0, "conversation": 0},
        }

        if block_contents:
            total_block_chars = sum(b["chars"] for b in block_contents)
            section_key_map = {"Agent": "agent", "Context": "context", "Rules": "rules", "Conversation": "conversation"}
            blocks_data = []
            current_section_key = None
            for block in block_contents:
                if total_block_chars > 0 and system_tokens > 0:
                    tokens = int(system_tokens * (block["chars"] / total_block_chars))
                else:
                    tokens = 0
                blocks_data.append({
                    "id": block["id"],
                    "label": block["label"],
                    "section": block.get("section"),
                    "tokens": tokens,
                    "content": block.get("content", ""),
                })
                if block.get("section"):
                    current_section_key = section_key_map.get(block["section"])
                if current_section_key and current_section_key in breakdown["sections"]:
                    breakdown["sections"][current_section_key] += tokens
            breakdown["blocks"] = blocks_data

        return {
            "messages": context.messages,
            "token_breakdown": breakdown,
            "block_contents": block_contents,
            "input_modality": "TEXT",
            "state_trigger": None,
            "estimated": True,
        }

    def _run_with_tools(self, messages: list[dict], params: dict) -> LLMResponse:
        """
        Run LLM with tool execution loop.

        Uses asyncio to run the async tool executor in a sync context.

        Args:
            messages: Prepared message list
            params: Generation parameters

        Returns:
            Final LLMResponse after tool loop completes
        """
        async def _async_execute():
            result = await self._tool_executor.execute(
                provider=self.provider,
                messages=messages,
                temperature=params.get("temperature", 0.7),
                max_tokens=params.get("max_tokens", 256),
                top_p=params.get("top_p"),
                stop=params.get("stop"),
            )
            return result.response

        # Run async executor in sync context
        # Try to get existing loop, create new one if none exists
        try:
            loop = asyncio.get_running_loop()
            # We're already in an async context - shouldn't happen in voice pipeline
            # but handle it gracefully
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, _async_execute())
                return future.result()
        except RuntimeError:
            # No running loop - create one
            return asyncio.run(_async_execute())

    def _resolve_generation_params(
        self,
        persona: dict,
        override: Optional[dict],
    ) -> dict:
        """
        Resolve generation parameters from persona defaults and overrides.

        Priority: override > persona['generation'] > hardcoded defaults
        """
        # Hardcoded defaults
        params = {
            "temperature": 0.7,
            "max_tokens": 256,
            "top_p": 0.95,
            "repeat_penalty": 1.1,
            "repeat_last_n": 64,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }

        # Persona defaults
        if "generation" in persona:
            persona_gen = persona["generation"]
            if "temperature" in persona_gen:
                params["temperature"] = persona_gen["temperature"]
            if "max_tokens" in persona_gen:
                params["max_tokens"] = persona_gen["max_tokens"]
            if "top_p" in persona_gen:
                params["top_p"] = persona_gen["top_p"]
            if "stop" in persona_gen:
                params["stop"] = persona_gen["stop"]
            if "repeat_penalty" in persona_gen:
                params["repeat_penalty"] = persona_gen["repeat_penalty"]
            if "repeat_last_n" in persona_gen:
                params["repeat_last_n"] = persona_gen["repeat_last_n"]
            if "frequency_penalty" in persona_gen:
                params["frequency_penalty"] = persona_gen["frequency_penalty"]
            if "presence_penalty" in persona_gen:
                params["presence_penalty"] = persona_gen["presence_penalty"]

        # Explicit overrides
        if override:
            params.update(override)

        return params

    def _inject_codex_content(self, context: PipelineContext) -> None:
        """
        Inject codex content into the system prompt.

        Replaces [CODEX_CONTEXT] placeholder with activated codex content.
        Wraps content with configurable prefix/suffix (NANO-045d).
        If no content, placeholder collapses to empty string.

        Args:
            context: Pipeline context with metadata["codex_content"] populated
                    by CodexActivatorPlugin (may be empty string or absent)
        """
        codex_content = context.metadata.get("codex_content", "")

        # Wrap with configurable prefix/suffix (NANO-045d)
        if codex_content:
            if self._codex_prefix:
                codex_content = f"{self._codex_prefix}\n{codex_content}"
            if self._codex_suffix:
                codex_content = f"{codex_content}\n{self._codex_suffix}"

        if context.messages and context.messages[0].get("role") == "system":
            system_prompt = context.messages[0]["content"]

            # Replace placeholder with content (or empty string)
            system_prompt = system_prompt.replace("[CODEX_CONTEXT]", codex_content)

            # Clean up excess whitespace from empty placeholder
            # Collapse 3+ consecutive newlines to 2
            import re
            system_prompt = re.sub(r"\n{3,}", "\n\n", system_prompt)

            context.messages[0]["content"] = system_prompt

    def _inject_rag_content(self, context: PipelineContext) -> None:
        """
        Inject RAG memory content into the system prompt (NANO-043 Phase 2).

        Replaces [RAG_CONTEXT] placeholder with retrieved memories.
        If no content, placeholder collapses to empty string.
        Follows the same pattern as _inject_codex_content().

        Args:
            context: Pipeline context with metadata["rag_content"] populated
                    by RAGInjector (may be empty string or absent)
        """
        rag_content = context.metadata.get("rag_content", "")

        if context.messages and context.messages[0].get("role") == "system":
            system_prompt = context.messages[0]["content"]

            # Replace placeholder with content (or empty string)
            system_prompt = system_prompt.replace("[RAG_CONTEXT]", rag_content)

            # Clean up excess whitespace from empty placeholder
            import re
            system_prompt = re.sub(r"\n{3,}", "\n\n", system_prompt)

            context.messages[0]["content"] = system_prompt

    def _inject_twitch_content(self, context: PipelineContext) -> None:
        """
        Inject Twitch chat content into the system prompt (NANO-056b).

        Replaces [TWITCH_CONTEXT] placeholder with formatted Twitch messages.
        If no content, placeholder collapses to empty string.
        Follows the same pattern as _inject_codex_content() and _inject_rag_content().

        Args:
            context: Pipeline context with metadata["twitch_content"] populated
                    from stimulus metadata (may be empty string or absent)
        """
        twitch_content = context.metadata.get("twitch_content", "")

        if context.messages and context.messages[0].get("role") == "system":
            system_prompt = context.messages[0]["content"]

            # Replace placeholder with content (or empty string)
            system_prompt = system_prompt.replace("[TWITCH_CONTEXT]", twitch_content)

            # Clean up excess whitespace from empty placeholder
            import re
            system_prompt = re.sub(r"\n{3,}", "\n\n", system_prompt)

            context.messages[0]["content"] = system_prompt

    def _update_deferred_block_contents(self, context: PipelineContext) -> None:
        """
        Patch deferred block char counts after injection (NANO-045b).

        Injection blocks (codex, RAG, twitch, history) emit placeholder strings at
        build time. After preprocessors and inject methods run, the real
        content is available in context.metadata. This method updates the
        deferred entries with actual character counts.
        """
        block_contents = context.metadata.get("block_contents")
        if not block_contents:
            return

        for entry in block_contents:
            if not entry.get("deferred"):
                continue

            block_id = entry["id"]
            if block_id == "codex_context":
                real_content = context.metadata.get("codex_content", "")
                entry["chars"] = len(real_content)
                entry["content"] = real_content
                entry["deferred"] = False
            elif block_id == "rag_context":
                real_content = context.metadata.get("rag_content", "")
                entry["chars"] = len(real_content)
                entry["content"] = real_content
                entry["deferred"] = False
            elif block_id == "twitch_context":
                real_content = context.metadata.get("twitch_content", "")
                entry["chars"] = len(real_content)
                entry["content"] = real_content
                entry["deferred"] = False
            elif block_id == "recent_history":
                history_text = context.metadata.get("history_formatted", "")
                entry["chars"] = len(history_text)
                entry["content"] = history_text
                entry["deferred"] = False

    def _extract_codex_display_data(self, context: PipelineContext) -> list[dict]:
        """
        Extract codex activation data for GUI display (NANO-037 Phase 2).

        Transforms internal ActivationResult objects into serializable dicts
        for sending to the frontend.

        Args:
            context: Pipeline context with metadata["codex_results"] populated
                    by CodexActivatorPlugin

        Returns:
            List of dicts with: name, keys, activation_method
        """
        codex_results = context.metadata.get("codex_results", [])
        if not codex_results:
            return []

        display_data = []
        for result in codex_results:
            # result is an ActivationResult dataclass from codex/models.py
            # Only include activated entries
            if not result.activated:
                continue

            # Build keys list - include matched_keyword if available
            keys = [result.matched_keyword] if result.matched_keyword else []

            display_data.append({
                "name": result.entry_name or "Unnamed Entry",
                "keys": keys,
                "activation_method": result.reason,  # "keyword_match", "constant", "sticky_active"
            })

        return display_data

    def _extract_rag_display_data(self, context: PipelineContext) -> list[dict]:
        """
        Extract RAG memory data for GUI display (NANO-044).

        Transforms raw MemoryStore.query() results into serializable dicts
        for sending to the frontend MemoryIndicator component.

        Args:
            context: Pipeline context with metadata["rag_results"] populated
                    by RAGInjector

        Returns:
            List of dicts with: content_preview, collection, distance
        """
        rag_results = context.metadata.get("rag_results", [])
        if not rag_results:
            return []

        display_data = []
        for mem in rag_results:
            content = mem.get("content", "")
            entry = {
                "content_preview": content[:80] if len(content) > 80 else content,
                "collection": mem.get("collection", "unknown"),
                "distance": mem.get("distance", 0.0),
            }
            if "score" in mem:
                entry["score"] = mem["score"]
            display_data.append(entry)

        return display_data
