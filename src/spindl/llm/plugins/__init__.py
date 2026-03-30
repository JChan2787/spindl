"""
LLM pipeline plugins.

Provides base classes for pre/post processing and concrete implementations.
"""

from .base import PipelineContext, PreProcessor, PostProcessor
from .tts_cleanup import TTSCleanupPlugin
from .conversation_history import (
    ConversationHistoryManager,
    HistoryInjector,
    HistoryRecorder,
    create_history_plugins,
)
from .summarization import (
    SummarizationTrigger,
    create_summarization_trigger,
    DEFAULT_SUMMARIZATION_PROMPT,
)
from .budget_enforcer import (
    BudgetEnforcer,
    create_budget_enforcer,
    EnforcementStrategy,
    TokenBudgetExceeded,
)
from .codex_activator import (
    CodexActivatorPlugin,
    create_codex_activator,
)
from .codex_cooldown import (
    CodexCooldownPlugin,
    create_codex_cooldown,
    create_codex_plugins,
)

__all__ = [
    # Base
    "PipelineContext",
    "PreProcessor",
    "PostProcessor",
    # TTS
    "TTSCleanupPlugin",
    # History (NANO-004)
    "ConversationHistoryManager",
    "HistoryInjector",
    "HistoryRecorder",
    "create_history_plugins",
    # Summarization (NANO-005)
    "SummarizationTrigger",
    "create_summarization_trigger",
    "DEFAULT_SUMMARIZATION_PROMPT",
    # Budget Enforcement (NANO-006)
    "BudgetEnforcer",
    "create_budget_enforcer",
    "EnforcementStrategy",
    "TokenBudgetExceeded",
    # Codex Activation (NANO-034)
    "CodexActivatorPlugin",
    "create_codex_activator",
    "CodexCooldownPlugin",
    "create_codex_cooldown",
    "create_codex_plugins",
]
