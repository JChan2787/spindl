"""Summarization trigger plugin for automatic context compression."""

from typing import Optional

from .base import PipelineContext, PreProcessor
from .conversation_history import ConversationHistoryManager
from ..base import LLMProvider
from ...history.jsonl_store import append_summary


# Default summarization prompt when persona doesn't define one
DEFAULT_SUMMARIZATION_PROMPT = """
Summarize this conversation between a user and {persona_name}.

Focus on:
1. Key topics discussed
2. User preferences or facts they shared
3. Important context for continuing the conversation
4. The general tone and rapport

Keep the summary under 150 words. Write in third person.

Conversation:
{conversation}

Summary:
"""


class SummarizationTrigger(PreProcessor):
    """
    PreProcessor that triggers summarization when context budget is exceeded.

    Monitors token usage and automatically generates summaries to compress
    conversation history when approaching the context window limit.

    Flow:
    1. Calculate tokens in visible history
    2. If (tokens / n_ctx) > threshold, trigger summarization
    3. Generate summary via LLM call
    4. Insert summary turn, mark old turns hidden
    5. Update manager's in-memory history

    Must be registered BEFORE HistoryInjector in the pipeline.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        manager: ConversationHistoryManager,
        threshold: float = 0.6,
        reserve_tokens: int = 512,
        live_mode: bool = False,
    ):
        """
        Initialize the summarization trigger.

        Args:
            llm_provider: LLMProvider instance for tokenization and summary generation
            manager: ConversationHistoryManager instance (shared with HistoryInjector)
            threshold: Trigger summarization when history exceeds this fraction
                       of available context (default: 0.6 = 60%)
            reserve_tokens: Tokens to reserve for system prompt + current turn
                           + response (default: 512)
            live_mode: If True, disables mid-session summarization entirely.
                      Use during live streams where the summarization LLM call
                      would cause latency spikes. Context management falls to
                      BudgetEnforcer FIFO truncation only. (NANO-043 Phase 3)
        """
        self._provider = llm_provider
        self._manager = manager
        self._threshold = threshold
        self._reserve_tokens = reserve_tokens
        self._live_mode = live_mode

        # Cached context length (queried once on first use)
        self._n_ctx: Optional[int] = None

    @property
    def name(self) -> str:
        return "summarization_trigger"

    def _get_context_length(self) -> int:
        """Get and cache the model's context window size."""
        if self._n_ctx is None:
            props = self._provider.get_properties()
            if props.context_length is None:
                # Fallback for providers that don't report context length
                self._n_ctx = 8192
            else:
                self._n_ctx = props.context_length
        return self._n_ctx

    def _calculate_available_budget(self, context: PipelineContext) -> int:
        """
        Calculate token budget available for history.

        Available = n_ctx - reserve - system_prompt_tokens - current_input_tokens
        """
        n_ctx = self._get_context_length()

        # Count system prompt tokens
        system_prompt = context.persona.get("system_prompt", "")
        system_tokens = self._provider.count_tokens(system_prompt)

        # Count current user input tokens
        input_tokens = self._provider.count_tokens(context.user_input)

        available = n_ctx - self._reserve_tokens - system_tokens - input_tokens
        return max(0, available)

    def _count_history_tokens(self) -> int:
        """Count total tokens in visible history."""
        history = self._manager.get_history()
        if not history:
            return 0

        # Concatenate all content for tokenization
        text = "\n".join(turn["content"] for turn in history)
        return self._provider.count_tokens(text)

    def _format_conversation_for_summary(self) -> str:
        """Format visible history as text for summarization prompt."""
        history = self._manager.get_history()
        lines = []
        for turn in history:
            role = turn["role"]
            content = turn["content"]
            if role == "user":
                lines.append(f"User: {content}")
            elif role == "assistant":
                lines.append(f"Assistant: {content}")
            elif role == "summary":
                lines.append(f"[Previous summary]: {content}")
        return "\n".join(lines)

    def _get_summarization_prompt(self, context: PipelineContext) -> str:
        """Get the summarization prompt template from persona or default."""
        persona = context.persona

        template = persona.get("summarization_prompt", DEFAULT_SUMMARIZATION_PROMPT)
        conversation = self._format_conversation_for_summary()
        persona_name = persona.get("name", "the assistant")

        return template.format(
            conversation=conversation,
            persona_name=persona_name,
        )

    def _get_summarization_params(self, context: PipelineContext) -> dict:
        """Get LLM generation params for summarization."""
        persona = context.persona

        # Try summarization-specific params first
        if "summarization_generation" in persona:
            params = dict(persona["summarization_generation"])
        elif "generation" in persona:
            params = dict(persona["generation"])
        else:
            params = {}

        # Defaults for summarization (lower temp, reasonable length)
        params.setdefault("temperature", 0.3)
        params.setdefault("max_tokens", 300)
        params.setdefault("top_p", 0.9)

        return params

    def _generate_summary(self, context: PipelineContext) -> str:
        """Generate summary via LLM call."""
        prompt = self._get_summarization_prompt(context)
        params = self._get_summarization_params(context)

        messages = [
            {"role": "system", "content": "You are a helpful assistant that creates concise conversation summaries."},
            {"role": "user", "content": prompt},
        ]

        llm_response = self._provider.generate(messages=messages, **params)
        return llm_response.content.strip()

    def _perform_summarization(self, context: PipelineContext) -> None:
        """
        Execute the full summarization flow.

        1. Generate summary via LLM
        2. Determine which turns to hide (all visible turns)
        3. Append summary turn to JSONL
        4. Mark old turns as hidden
        5. Reload manager's in-memory history
        """
        history = self._manager.get_history()
        if not history:
            return

        # Get the highest turn_id in current visible history
        max_turn_id = max(turn["turn_id"] for turn in history)

        # Generate summary
        summary_content = self._generate_summary(context)

        # Append to JSONL and mark old turns hidden
        session_file = self._manager.session_file
        if session_file is None:
            return

        summary_turn = append_summary(
            filepath=session_file,
            summary_content=summary_content,
            summarizes_up_to=max_turn_id,
        )

        # Update manager's in-memory state
        # Clear old history, add only the summary
        self._manager._history = [summary_turn]
        self._manager._next_turn_id = summary_turn["turn_id"] + 1

    def process(self, context: PipelineContext) -> PipelineContext:
        """
        Check token budget and trigger summarization if needed.

        Args:
            context: Current pipeline context

        Returns:
            Context unchanged (summarization modifies manager state)
        """
        # Ensure manager has a session
        persona_id = context.persona.get("id", "unknown")
        self._manager.ensure_session(persona_id)

        # Live mode: skip summarization entirely (NANO-043 Phase 3)
        # Context management handled by BudgetEnforcer FIFO truncation.
        if self._live_mode:
            return context

        # Check if summarization is needed
        history_tokens = self._count_history_tokens()

        if history_tokens == 0:
            return context

        available_budget = self._calculate_available_budget(context)
        threshold_budget = int(available_budget * self._threshold)

        if history_tokens > threshold_budget:
            # Trigger summarization
            self._perform_summarization(context)

        return context


def create_summarization_trigger(
    llm_provider: LLMProvider,
    manager: ConversationHistoryManager,
    threshold: float = 0.6,
    reserve_tokens: int = 512,
    live_mode: bool = False,
) -> SummarizationTrigger:
    """
    Factory function to create a SummarizationTrigger.

    Args:
        llm_provider: LLMProvider for tokenization and summary generation
        manager: ConversationHistoryManager (same instance used by HistoryInjector)
        threshold: Fraction of context to use before triggering (default: 0.6)
        reserve_tokens: Tokens to reserve for overhead (default: 512)
        live_mode: If True, disables mid-session summarization (NANO-043 Phase 3)

    Returns:
        Configured SummarizationTrigger instance

    Usage:
        manager = ConversationHistoryManager(...)
        injector = HistoryInjector(manager)
        recorder = HistoryRecorder(manager)

        summarizer = create_summarization_trigger(provider, manager, threshold=0.6)

        pipeline.register_pre_processor(summarizer)  # FIRST
        pipeline.register_pre_processor(injector)    # SECOND
        pipeline.register_post_processor(recorder)
    """
    return SummarizationTrigger(
        llm_provider=llm_provider,
        manager=manager,
        threshold=threshold,
        reserve_tokens=reserve_tokens,
        live_mode=live_mode,
    )
