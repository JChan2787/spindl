"""
Base classes for LLM pipeline plugins.

Defines the interfaces for pre-processors and post-processors
that can modify prompts and responses in the pipeline.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class PipelineContext:
    """
    Shared context passed through the pipeline.

    Carries state from input through pre-processors, LLM call,
    and post-processors. Plugins can read/modify as needed.
    """

    user_input: str
    """Original user utterance."""

    persona: dict
    """Loaded persona configuration."""

    messages: list[dict] = field(default_factory=list)
    """OpenAI-style message list. Built by PromptBuilder, modifiable by plugins."""

    metadata: dict[str, Any] = field(default_factory=dict)
    """Scratch space for plugins to pass data to each other."""


class PreProcessor(ABC):
    """
    Base class for pre-processing plugins.

    Pre-processors run BEFORE the LLM call. They can modify:
    - context.messages (add/remove/edit messages)
    - context.metadata (stash data for post-processors)

    Examples:
    - RAGContextPlugin: Inject relevant memories into system prompt
    - HistoryPlugin: Add conversation history to messages
    - TokenBudgetPlugin: Trim context to fit token limits
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin identifier for logging/debugging."""

    @abstractmethod
    def process(self, context: PipelineContext) -> PipelineContext:
        """
        Modify context before LLM call.

        Args:
            context: Current pipeline context

        Returns:
            Modified context (can return same object mutated)
        """


class PostProcessor(ABC):
    """
    Base class for post-processing plugins.

    Post-processors run AFTER the LLM call. They receive the
    raw response and can transform it before it reaches the caller.

    Examples:
    - TTSCleanupPlugin: Strip asterisks, parentheticals for TTS
    - MemoryStorePlugin: Store interaction to RAG system
    - LoggingPlugin: Log request/response for debugging
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin identifier for logging/debugging."""

    @abstractmethod
    def process(self, context: PipelineContext, response: str) -> str:
        """
        Modify response after LLM call.

        Args:
            context: Pipeline context (read-only recommended)
            response: Raw LLM response text

        Returns:
            Modified response text
        """
