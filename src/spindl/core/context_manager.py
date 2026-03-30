"""
Central aggregator for multimodal context before LLM calls.

Provides:
- Registration of context sources (lorebook, vision, memory, etc.)
- Context assembly from all registered sources
- Event emission when context is updated
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from .events import ContextUpdatedEvent, EventType
from .event_bus import EventBus

logger = logging.getLogger(__name__)


@dataclass
class ContextSource:
    """A registered source of context data."""

    name: str
    provider: Callable[[], Optional[dict[str, Any]]]  # Returns context or None
    priority: int = 0  # Higher = merged last (can override earlier sources)


@dataclass
class AggregatedContext:
    """
    Result of context aggregation from multiple sources.

    Contains:
    - primary_input: The main user input (e.g., transcription)
    - sources: Data from each contributing source
    - contributing_sources: Names of sources that provided data

    Usage:
        context = context_manager.assemble("Hello, how are you?")

        # Access primary input
        user_text = context.primary_input

        # Access source data
        lorebook_data = context.get("lorebook")
        visual_data = context.get("vision")
    """

    primary_input: str
    sources: dict[str, Any] = field(default_factory=dict)
    contributing_sources: list[str] = field(default_factory=list)

    def get(self, source_name: str, default: Any = None) -> Any:
        """
        Get context from a specific source.

        Args:
            source_name: Name of the registered source.
            default: Value to return if source not present.

        Returns:
            Source data or default.
        """
        return self.sources.get(source_name, default)

    def has_source(self, source_name: str) -> bool:
        """Check if a source contributed to this context."""
        return source_name in self.contributing_sources


class ContextManager:
    """
    Central aggregator for multimodal context before LLM calls.

    Registered sources are queried when assemble() is called.
    Results are merged by priority (higher priority sources can override).

    Current sources: voice transcription (passed through)
    Future sources: lorebook retrieval, image classification, memory search

    Usage:
        ctx_mgr = ContextManager(event_bus)

        # Register lorebook retrieval (future)
        ctx_mgr.register_source(
            name="lorebook",
            provider=lambda: lorebook.search(ctx_mgr.pending_input),
            priority=10
        )

        # When user speaks, assemble all context
        context = ctx_mgr.assemble("Hello, how are you?")
        # Returns AggregatedContext with transcription + any lorebook matches

    Thread Safety:
        - Source registration is not thread-safe (do it during setup)
        - assemble() can be called from any thread
    """

    def __init__(self, event_bus: Optional[EventBus] = None):
        """
        Initialize the context manager.

        Args:
            event_bus: Optional EventBus for emitting CONTEXT_UPDATED events.
        """
        self._event_bus = event_bus
        self._sources: list[ContextSource] = []
        self._pending_input: Optional[str] = None

    def register_source(
        self,
        name: str,
        provider: Callable[[], Optional[dict[str, Any]]],
        priority: int = 0,
    ) -> None:
        """
        Register a context source.

        If a source with the same name exists, it is replaced.

        Args:
            name: Unique identifier for this source.
            provider: Callable that returns context dict or None.
                     Can access self.pending_input to query based on user input.
            priority: Higher = merged last (can override earlier sources).
        """
        # Remove existing source with same name
        self._sources = [s for s in self._sources if s.name != name]

        self._sources.append(
            ContextSource(
                name=name,
                provider=provider,
                priority=priority,
            )
        )

        # Keep sorted by priority (lower first, so higher overrides)
        self._sources.sort(key=lambda s: s.priority)

        logger.debug(f"ContextManager: registered source '{name}' with priority {priority}")

    def unregister_source(self, name: str) -> bool:
        """
        Remove a context source by name.

        Args:
            name: Name of source to remove.

        Returns:
            True if source was found and removed.
        """
        original_len = len(self._sources)
        self._sources = [s for s in self._sources if s.name != name]
        removed = len(self._sources) < original_len

        if removed:
            logger.debug(f"ContextManager: unregistered source '{name}'")

        return removed

    def assemble(self, primary_input: str) -> AggregatedContext:
        """
        Assemble context from all registered sources.

        Sets pending_input before querying sources so they can access it.
        Emits CONTEXT_UPDATED event after assembly.

        Args:
            primary_input: The main user input (e.g., transcription).

        Returns:
            AggregatedContext with all source data merged.
        """
        self._pending_input = primary_input

        result = AggregatedContext(primary_input=primary_input)

        for source in self._sources:
            try:
                data = source.provider()
                if data is not None:
                    result.sources[source.name] = data
                    result.contributing_sources.append(source.name)
            except Exception as e:
                logger.warning(
                    f"ContextManager: source '{source.name}' failed: {e}"
                )

        self._pending_input = None

        # Emit event if we have an event bus
        if self._event_bus is not None:
            self._event_bus.emit(
                ContextUpdatedEvent(sources=result.contributing_sources)
            )

        return result

    @property
    def pending_input(self) -> Optional[str]:
        """
        Get the pending input during context assembly.

        Useful for sources that need to query based on user input.
        Only valid during assemble() execution.

        Returns:
            The primary_input being assembled, or None if not in assembly.
        """
        return self._pending_input

    @property
    def registered_sources(self) -> list[str]:
        """Get names of registered sources in priority order."""
        return [s.name for s in self._sources]

    @property
    def source_count(self) -> int:
        """Get number of registered sources."""
        return len(self._sources)
