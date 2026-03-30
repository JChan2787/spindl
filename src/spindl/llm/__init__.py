"""
LLM integration module for spindl.

Provides:
- Provider abstraction for swappable LLM backends (NANO-018, NANO-019)
- Prompt building and plugin-aware pipeline orchestration

Note: LlamaClient is intentionally NOT exported here. It is an internal
implementation detail used by LlamaProvider. External code should use
the LLMProvider abstraction instead.
"""

# Provider system (NANO-018)
from .base import LLMProvider, LLMProperties, LLMResponse, StreamChunk
from .registry import LLMProviderRegistry, ProviderNotFoundError, create_default_registry

# Pipeline components
from .prompt_builder import PromptBuilder
from .pipeline import LLMPipeline, PipelineResult, TokenUsage
from .plugins import PipelineContext, PreProcessor, PostProcessor

__all__ = [
    # Provider system (NANO-018)
    "LLMProvider",
    "LLMProperties",
    "LLMResponse",
    "StreamChunk",
    "LLMProviderRegistry",
    "ProviderNotFoundError",
    "create_default_registry",
    # Pipeline
    "PromptBuilder",
    "LLMPipeline",
    "PipelineResult",
    "TokenUsage",
    "PipelineContext",
    "PreProcessor",
    "PostProcessor",
]
