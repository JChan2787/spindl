"""Token budget enforcement plugin for hard context limits."""

from enum import Enum
from typing import Optional

from .base import PipelineContext, PreProcessor
from .conversation_history import ConversationHistoryManager
from ..base import LLMProvider


class EnforcementStrategy(Enum):
    """Strategy for handling token budget overflow."""

    TRUNCATE = "truncate"  # Remove oldest turns until budget met
    DROP = "drop"  # Drop all history if truncation insufficient
    REJECT = "reject"  # Raise exception if budget exceeded


class TokenBudgetExceeded(Exception):
    """Raised when token budget cannot be satisfied."""

    def __init__(self, message: str, requested: int, available: int):
        super().__init__(message)
        self.requested = requested
        self.available = available


class BudgetEnforcer(PreProcessor):
    """
    PreProcessor that enforces hard token budget limits.

    Ensures the total context (system prompt + codex + history + user input + reserve)
    never exceeds the model's n_ctx. Applies configured strategy when limits
    are exceeded.

    Must be registered AFTER CodexActivatorPlugin but BEFORE HistoryInjector.
    This allows codex tokens to be included in budget calculations.

    Flow:
    1. Calculate token budget breakdown (including codex from context.metadata)
    2. If under budget, pass through unchanged
    3. If over budget:
       - TRUNCATE: Remove oldest non-summary turns until budget satisfied
       - DROP: Clear all history (fallback if truncation fails)
       - REJECT: Raise TokenBudgetExceeded

    Codex entries have priority over history—if budget is tight, history is
    truncated first while codex entries are preserved.

    If user input alone exceeds available space (after system prompt and reserve),
    always raises TokenBudgetExceeded regardless of strategy.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        manager: ConversationHistoryManager,
        strategy: str = "truncate",
        response_reserve: int = 300,
        strict: bool = False,
    ):
        """
        Initialize the budget enforcer.

        Args:
            llm_provider: LLMProvider for tokenization and context length
            manager: ConversationHistoryManager instance
            strategy: Enforcement strategy ("truncate", "drop", "reject")
            response_reserve: Tokens to reserve for LLM response generation
            strict: If True, raise exception when drop would be needed
                    (truncation failed to satisfy budget)
        """
        self._provider = llm_provider
        self._manager = manager
        self._strategy = EnforcementStrategy(strategy)
        self._response_reserve = response_reserve
        self._strict = strict

        self._n_ctx: Optional[int] = None

    @property
    def name(self) -> str:
        return "budget_enforcer"

    def _get_context_length(self) -> int:
        """Get and cache n_ctx from the provider."""
        if self._n_ctx is None:
            props = self._provider.get_properties()
            if props.context_length is None:
                # Fallback for providers that don't report context length
                # DeepSeek: 128K, typical llama.cpp: 4K-8K
                self._n_ctx = 8192
            else:
                self._n_ctx = props.context_length
        return self._n_ctx

    def _count_tokens(self, text: str) -> int:
        """Count tokens in text using the provider's tokenizer."""
        return self._provider.count_tokens(text)

    def _calculate_budget(self, context: PipelineContext) -> dict:
        """
        Calculate token budget breakdown.

        Includes codex tokens from context.metadata if CodexActivatorPlugin
        has run before this plugin (which it should). Also includes RAG
        memory tokens from RAGInjector (NANO-043 Phase 2).

        Returns:
            dict with keys:
                - n_ctx: Total context window
                - system_tokens: System prompt tokens
                - codex_tokens: Activated codex entry tokens
                - rag_tokens: Retrieved memory tokens (NANO-043)
                - user_tokens: Current user input tokens
                - reserve: Response reserve
                - available_for_history: Tokens available for history (after codex + RAG)
                - history_tokens: Current history token count
                - over_budget: Tokens over budget (0 if under)
        """
        n_ctx = self._get_context_length()

        system_prompt = context.persona.get("system_prompt", "")
        system_tokens = self._count_tokens(system_prompt)

        user_tokens = self._count_tokens(context.user_input)

        # Get codex tokens from metadata (set by CodexActivatorPlugin)
        # Use the estimate for quick budget check, or 0 if codex not active
        codex_tokens = context.metadata.get("codex_tokens_estimate", 0)

        # Get RAG memory tokens from metadata (set by RAGInjector, NANO-043)
        # Uses same estimate pattern as codex. 0 if memory disabled or no results.
        rag_tokens = context.metadata.get("rag_tokens_estimate", 0)

        # Codex and RAG have priority over history, so subtract both from available space
        available_for_history = (
            n_ctx - system_tokens - codex_tokens - rag_tokens - user_tokens - self._response_reserve
        )
        available_for_history = max(0, available_for_history)

        history = self._manager.get_history()
        if history:
            history_text = "\n".join(t["content"] for t in history)
            history_tokens = self._count_tokens(history_text)
        else:
            history_tokens = 0

        over_budget = max(0, history_tokens - available_for_history)

        return {
            "n_ctx": n_ctx,
            "system_tokens": system_tokens,
            "codex_tokens": codex_tokens,
            "rag_tokens": rag_tokens,
            "user_tokens": user_tokens,
            "reserve": self._response_reserve,
            "available_for_history": available_for_history,
            "history_tokens": history_tokens,
            "over_budget": over_budget,
        }

    def _truncate_history(self, available: int) -> bool:
        """
        Truncate oldest turns until budget is satisfied.

        Protects summary turns—removes them only as a last resort.

        Args:
            available: Tokens available for history

        Returns:
            True if truncation succeeded (history now fits),
            False if history exhausted but still over budget
        """
        history = self._manager._history

        while history:
            # Count current history tokens
            if history:
                history_text = "\n".join(t["content"] for t in history)
                current_tokens = self._count_tokens(history_text)
            else:
                current_tokens = 0

            # Check if we're under budget now
            if current_tokens <= available:
                return True

            # Find oldest non-summary turn to remove
            non_summary_idx = None
            for i, turn in enumerate(history):
                if turn["role"] != "summary":
                    non_summary_idx = i
                    break

            if non_summary_idx is not None:
                # Remove oldest non-summary turn
                history.pop(non_summary_idx)
            else:
                # All remaining turns are summaries—remove oldest summary
                # This is a last resort when even summaries don't fit
                history.pop(0)

        # History is now empty
        return True

    def _drop_history(self) -> None:
        """Drop all history."""
        self._manager._history = []

    def _check_minimum_viable(self, budget: dict) -> bool:
        """
        Check if request is viable even with zero history.

        Returns:
            True if system + codex + RAG + user + reserve fits in n_ctx
        """
        minimum_required = (
            budget["system_tokens"]
            + budget.get("codex_tokens", 0)
            + budget.get("rag_tokens", 0)
            + budget["user_tokens"]
            + budget["reserve"]
        )
        return minimum_required <= budget["n_ctx"]

    def process(self, context: PipelineContext) -> PipelineContext:
        """
        Enforce token budget.

        Args:
            context: Pipeline context

        Returns:
            Context unchanged (but manager history may be modified)

        Raises:
            TokenBudgetExceeded: If budget cannot be satisfied
                - Always raised if user input alone exceeds budget
                - Raised with REJECT strategy
                - Raised with strict=True when truncation fails
        """
        budget = self._calculate_budget(context)

        # FIRST: Check if request is viable at all (even with no history)
        # This must come before the early return, because a massive user input
        # will clamp available_for_history to 0, making over_budget appear as 0
        if not self._check_minimum_viable(budget):
            codex_tokens = budget.get("codex_tokens", 0)
            rag_tokens = budget.get("rag_tokens", 0)
            codex_msg = f" + codex ({codex_tokens} tokens)" if codex_tokens else ""
            rag_msg = f" + memories ({rag_tokens} tokens)" if rag_tokens else ""
            raise TokenBudgetExceeded(
                f"User input ({budget['user_tokens']} tokens) plus system prompt "
                f"({budget['system_tokens']} tokens){codex_msg}{rag_msg} exceeds context window "
                f"({budget['n_ctx']} tokens) even with no history.",
                requested=(
                    budget["user_tokens"]
                    + budget["system_tokens"]
                    + codex_tokens
                    + rag_tokens
                    + budget["reserve"]
                ),
                available=budget["n_ctx"],
            )

        # Under budget—nothing to do
        if budget["over_budget"] == 0:
            return context

        # Apply enforcement strategy
        if self._strategy == EnforcementStrategy.REJECT:
            raise TokenBudgetExceeded(
                f"Token budget exceeded by {budget['over_budget']} tokens.",
                requested=budget["history_tokens"],
                available=budget["available_for_history"],
            )

        if self._strategy == EnforcementStrategy.DROP:
            self._drop_history()
            return context

        # TRUNCATE strategy (default)
        success = self._truncate_history(budget["available_for_history"])

        if not success:
            # Truncation couldn't satisfy budget (shouldn't happen after emptying)
            if self._strict:
                raise TokenBudgetExceeded(
                    "Cannot satisfy token budget even after truncation.",
                    requested=budget["history_tokens"],
                    available=budget["available_for_history"],
                )
            else:
                # Fall back to drop
                self._drop_history()

        return context


def create_budget_enforcer(
    llm_provider: LLMProvider,
    manager: ConversationHistoryManager,
    strategy: str = "truncate",
    response_reserve: int = 300,
    strict: bool = False,
) -> BudgetEnforcer:
    """
    Factory function to create a BudgetEnforcer.

    Args:
        llm_provider: LLMProvider for tokenization
        manager: ConversationHistoryManager
        strategy: "truncate", "drop", or "reject"
        response_reserve: Tokens to reserve for response
        strict: Raise exception instead of dropping when truncation fails

    Returns:
        Configured BudgetEnforcer instance

    Usage:
        manager = ConversationHistoryManager(...)
        summarizer = create_summarization_trigger(provider, manager)
        codex_manager, codex_activator, codex_cooldown = create_codex_plugins(...)
        enforcer = create_budget_enforcer(provider, manager)
        injector = HistoryInjector(manager)
        recorder = HistoryRecorder(manager)

        # CRITICAL: Registration order matters!
        pipeline.register_pre_processor(summarizer)      # Soft limit (60%)
        pipeline.register_pre_processor(codex_activator) # Activate codex entries
        pipeline.register_pre_processor(enforcer)        # Hard limit (includes codex)
        pipeline.register_pre_processor(injector)        # Injects compliant history
        pipeline.register_post_processor(recorder)
        pipeline.register_post_processor(codex_cooldown) # Update codex state
    """
    return BudgetEnforcer(
        llm_provider=llm_provider,
        manager=manager,
        strategy=strategy,
        response_reserve=response_reserve,
        strict=strict,
    )
