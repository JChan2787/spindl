"""Tests for RAG pipeline integration — _inject_rag_content() and prompt_builder changes.

NANO-043 Phase 2. Tests the pipeline-level injection mechanism and
verifies that [RAG_CONTEXT] placeholder is handled correctly.
"""

import re
from unittest.mock import MagicMock

import pytest

from spindl.llm.plugins.base import PipelineContext
from spindl.llm.pipeline import LLMPipeline
from spindl.llm.prompt_builder import PromptBuilder
from spindl.llm.prompt_library import CONVERSATION_TEMPLATE


# ---------------------------------------------------------------------------
# Tests: [RAG_CONTEXT] placeholder in template
# ---------------------------------------------------------------------------

class TestRAGContextPlaceholder:
    """Verify [RAG_CONTEXT] exists in the prompt template."""

    def test_placeholder_exists_in_template(self) -> None:
        """CONVERSATION_TEMPLATE contains [RAG_CONTEXT] placeholder."""
        assert "[RAG_CONTEXT]" in CONVERSATION_TEMPLATE

    def test_placeholder_after_codex(self) -> None:
        """[RAG_CONTEXT] appears after [CODEX_CONTEXT] in template."""
        codex_pos = CONVERSATION_TEMPLATE.index("[CODEX_CONTEXT]")
        rag_pos = CONVERSATION_TEMPLATE.index("[RAG_CONTEXT]")
        assert rag_pos > codex_pos

    def test_placeholder_before_rules(self) -> None:
        """[RAG_CONTEXT] appears before ### Rules section."""
        rag_pos = CONVERSATION_TEMPLATE.index("[RAG_CONTEXT]")
        rules_pos = CONVERSATION_TEMPLATE.index("### Rules")
        assert rag_pos < rules_pos


# ---------------------------------------------------------------------------
# Tests: PromptBuilder preserves [RAG_CONTEXT]
# ---------------------------------------------------------------------------

class TestPromptBuilderPreservesRAG:
    """Verify prompt_builder no longer strips [RAG_CONTEXT]."""

    def test_rag_placeholder_survives_build(self) -> None:
        """[RAG_CONTEXT] is NOT stripped by PromptBuilder — survives for pipeline injection."""
        builder = PromptBuilder()  # Legacy mode (no providers)
        messages = builder.build(
            persona={"system_prompt": "Test prompt with [RAG_CONTEXT] in it."},
            user_input="Hello",
        )
        system_content = messages[0]["content"]
        assert "[RAG_CONTEXT]" in system_content

    def test_rag_placeholder_survives_provider_mode(self) -> None:
        """In provider mode, [RAG_CONTEXT] survives PromptBuilder into the system message."""
        from spindl.llm.providers.registry import create_default_registry

        registry = create_default_registry()
        builder = PromptBuilder(providers=registry)

        # Minimal persona for provider mode
        persona = {
            "name": "TestChar",
            "description": "Tall and dark.",
            "personality": "Friendly.",
            "rules": "Be nice.",
        }

        messages = builder.build(persona=persona, user_input="Hello")
        system_content = messages[0]["content"]

        # [RAG_CONTEXT] should still be in the system message — pipeline replaces it later
        assert "[RAG_CONTEXT]" in system_content


# ---------------------------------------------------------------------------
# Tests: Pipeline._inject_rag_content()
# ---------------------------------------------------------------------------

class TestPipelineInjectRAGContent:
    """Tests for LLMPipeline._inject_rag_content() method."""

    def _make_pipeline(self) -> LLMPipeline:
        """Create a minimal pipeline with mock provider."""
        mock_provider = MagicMock()
        mock_builder = MagicMock(spec=PromptBuilder)
        return LLMPipeline(provider=mock_provider, prompt_builder=mock_builder)

    def test_replaces_placeholder_with_content(self) -> None:
        """[RAG_CONTEXT] is replaced with memory content."""
        pipeline = self._make_pipeline()
        context = PipelineContext(
            user_input="test",
            persona={},
            messages=[{"role": "system", "content": "Prompt with [RAG_CONTEXT] here."}],
            metadata={"rag_content": "Relevant memories:\n- Alex likes mango.\nEnd of memories."},
        )

        pipeline._inject_rag_content(context)

        system = context.messages[0]["content"]
        assert "[RAG_CONTEXT]" not in system
        assert "Relevant memories:" in system
        assert "Alex likes mango." in system
        assert "End of memories." in system

    def test_empty_content_collapses_placeholder(self) -> None:
        """Empty rag_content collapses [RAG_CONTEXT] to nothing."""
        pipeline = self._make_pipeline()
        context = PipelineContext(
            user_input="test",
            persona={},
            messages=[{"role": "system", "content": "Before\n[RAG_CONTEXT]\nAfter"}],
            metadata={"rag_content": ""},
        )

        pipeline._inject_rag_content(context)

        system = context.messages[0]["content"]
        assert "[RAG_CONTEXT]" not in system
        # Should not have triple+ newlines
        assert "\n\n\n" not in system

    def test_missing_metadata_collapses_placeholder(self) -> None:
        """Missing rag_content key in metadata collapses placeholder."""
        pipeline = self._make_pipeline()
        context = PipelineContext(
            user_input="test",
            persona={},
            messages=[{"role": "system", "content": "Before\n[RAG_CONTEXT]\nAfter"}],
            metadata={},
        )

        pipeline._inject_rag_content(context)

        system = context.messages[0]["content"]
        assert "[RAG_CONTEXT]" not in system

    def test_no_system_message_is_safe(self) -> None:
        """No crash when messages list is empty."""
        pipeline = self._make_pipeline()
        context = PipelineContext(
            user_input="test",
            persona={},
            messages=[],
            metadata={"rag_content": "some content"},
        )

        # Should not raise
        pipeline._inject_rag_content(context)

    def test_cleans_up_excess_newlines(self) -> None:
        """Collapsed placeholder doesn't leave triple+ newlines."""
        pipeline = self._make_pipeline()
        context = PipelineContext(
            user_input="test",
            persona={},
            messages=[{"role": "system", "content": "Line1\n\n\n[RAG_CONTEXT]\n\n\nLine2"}],
            metadata={"rag_content": ""},
        )

        pipeline._inject_rag_content(context)

        system = context.messages[0]["content"]
        # No more than 2 consecutive newlines
        assert not re.search(r"\n{3,}", system)


# ---------------------------------------------------------------------------
# Tests: BudgetEnforcer RAG token accounting
# ---------------------------------------------------------------------------

class TestBudgetEnforcerRAGTokens:
    """Tests verifying BudgetEnforcer accounts for RAG tokens."""

    def test_rag_tokens_in_budget_calculation(self) -> None:
        """BudgetEnforcer includes rag_tokens_estimate in budget."""
        from spindl.llm.plugins.budget_enforcer import BudgetEnforcer

        mock_provider = MagicMock()
        mock_provider.get_properties.return_value = MagicMock(context_length=4096)
        mock_provider.count_tokens.side_effect = lambda text: len(text) // 4

        mock_manager = MagicMock()
        mock_manager.get_history.return_value = []

        enforcer = BudgetEnforcer(
            llm_provider=mock_provider,
            manager=mock_manager,
        )

        context = PipelineContext(
            user_input="Hello",
            persona={"system_prompt": "You are a test bot."},
            metadata={
                "codex_tokens_estimate": 100,
                "rag_tokens_estimate": 200,
            },
        )

        budget = enforcer._calculate_budget(context)

        assert budget["rag_tokens"] == 200
        # RAG tokens should reduce available_for_history
        # available = n_ctx - system - codex - rag - user - reserve
        # With RAG=200, available should be less than without RAG
        assert budget["available_for_history"] < (
            budget["n_ctx"] - budget["system_tokens"] - budget["codex_tokens"]
            - budget["user_tokens"] - budget["reserve"]
        )

    def test_zero_rag_tokens_no_impact(self) -> None:
        """Zero rag_tokens_estimate has no impact on budget."""
        from spindl.llm.plugins.budget_enforcer import BudgetEnforcer

        mock_provider = MagicMock()
        mock_provider.get_properties.return_value = MagicMock(context_length=4096)
        mock_provider.count_tokens.side_effect = lambda text: len(text) // 4

        mock_manager = MagicMock()
        mock_manager.get_history.return_value = []

        enforcer = BudgetEnforcer(
            llm_provider=mock_provider,
            manager=mock_manager,
        )

        # Without RAG
        context_no_rag = PipelineContext(
            user_input="Hello",
            persona={"system_prompt": "You are a test bot."},
            metadata={"codex_tokens_estimate": 100},
        )
        budget_no_rag = enforcer._calculate_budget(context_no_rag)

        # With RAG=0
        context_rag_zero = PipelineContext(
            user_input="Hello",
            persona={"system_prompt": "You are a test bot."},
            metadata={"codex_tokens_estimate": 100, "rag_tokens_estimate": 0},
        )
        budget_rag_zero = enforcer._calculate_budget(context_rag_zero)

        assert budget_no_rag["available_for_history"] == budget_rag_zero["available_for_history"]


# ---------------------------------------------------------------------------
# Tests: MemoryConfig parsing
# ---------------------------------------------------------------------------

class TestMemoryConfig:
    """Tests for MemoryConfig dataclass and YAML parsing."""

    def test_default_values(self) -> None:
        """MemoryConfig has correct defaults."""
        from spindl.orchestrator.config import MemoryConfig

        config = MemoryConfig()
        assert config.enabled is False
        assert config.embedding_base_url == "http://127.0.0.1:5559"
        assert config.embedding_timeout == 10.0
        assert config.rag_top_k == 5

    def test_from_dict_full(self) -> None:
        """MemoryConfig.from_dict parses all fields."""
        from spindl.orchestrator.config import MemoryConfig

        data = {
            "enabled": True,
            "embedding": {
                "base_url": "http://localhost:9999",
                "timeout": 30.0,
            },
            "top_k": 10,
        }
        config = MemoryConfig.from_dict(data)

        assert config.enabled is True
        assert config.embedding_base_url == "http://localhost:9999"
        assert config.embedding_timeout == 30.0
        assert config.rag_top_k == 10

    def test_from_dict_partial(self) -> None:
        """MemoryConfig.from_dict handles partial config with defaults."""
        from spindl.orchestrator.config import MemoryConfig

        data = {"enabled": True}
        config = MemoryConfig.from_dict(data)

        assert config.enabled is True
        assert config.embedding_base_url == "http://127.0.0.1:5559"
        assert config.embedding_timeout == 10.0
        assert config.rag_top_k == 5

    def test_from_dict_empty(self) -> None:
        """MemoryConfig.from_dict handles empty dict."""
        from spindl.orchestrator.config import MemoryConfig

        config = MemoryConfig.from_dict({})
        assert config.enabled is False

    def test_orchestrator_config_parses_memory(self) -> None:
        """OrchestratorConfig._from_dict parses memory section."""
        from spindl.orchestrator.config import OrchestratorConfig

        data = {
            "memory": {
                "enabled": True,
                "embedding": {
                    "base_url": "http://localhost:5559",
                    "timeout": 15.0,
                },
                "top_k": 3,
            },
        }
        config = OrchestratorConfig._from_dict(data)

        assert config.memory_config.enabled is True
        assert config.memory_config.embedding_base_url == "http://localhost:5559"
        assert config.memory_config.embedding_timeout == 15.0
        assert config.memory_config.rag_top_k == 3

    def test_orchestrator_config_default_memory(self) -> None:
        """OrchestratorConfig without memory section uses defaults."""
        from spindl.orchestrator.config import OrchestratorConfig

        config = OrchestratorConfig._from_dict({})
        assert config.memory_config.enabled is False


# ---------------------------------------------------------------------------
# Tests: _extract_rag_display_data (NANO-044)
# ---------------------------------------------------------------------------

class TestExtractRAGDisplayData:
    """Tests for LLMPipeline._extract_rag_display_data() method."""

    def _make_pipeline(self) -> LLMPipeline:
        """Create a minimal pipeline with mock provider."""
        mock_provider = MagicMock()
        mock_builder = MagicMock(spec=PromptBuilder)
        return LLMPipeline(provider=mock_provider, prompt_builder=mock_builder)

    def test_empty_results_returns_empty(self) -> None:
        """No rag_results in metadata → empty list."""
        pipeline = self._make_pipeline()
        context = PipelineContext(user_input="test", persona={}, messages=[], metadata={})
        assert pipeline._extract_rag_display_data(context) == []

    def test_empty_list_returns_empty(self) -> None:
        """Empty rag_results list → empty list."""
        pipeline = self._make_pipeline()
        context = PipelineContext(
            user_input="test", persona={}, messages=[],
            metadata={"rag_results": []},
        )
        assert pipeline._extract_rag_display_data(context) == []

    def test_extracts_content_preview(self) -> None:
        """content_preview is first 80 chars of content."""
        pipeline = self._make_pipeline()
        long_content = "A" * 120
        context = PipelineContext(
            user_input="test", persona={}, messages=[],
            metadata={"rag_results": [
                {"content": long_content, "collection": "general", "distance": 0.3, "metadata": {}},
            ]},
        )
        result = pipeline._extract_rag_display_data(context)
        assert len(result) == 1
        assert result[0]["content_preview"] == "A" * 80

    def test_short_content_not_truncated(self) -> None:
        """Short content is kept as-is."""
        pipeline = self._make_pipeline()
        context = PipelineContext(
            user_input="test", persona={}, messages=[],
            metadata={"rag_results": [
                {"content": "Short.", "collection": "general", "distance": 0.3, "metadata": {}},
            ]},
        )
        result = pipeline._extract_rag_display_data(context)
        assert result[0]["content_preview"] == "Short."

    def test_extracts_collection_and_distance(self) -> None:
        """collection and distance are passed through."""
        pipeline = self._make_pipeline()
        context = PipelineContext(
            user_input="test", persona={}, messages=[],
            metadata={"rag_results": [
                {"content": "Mem", "collection": "flashcards", "distance": 0.42, "metadata": {}},
            ]},
        )
        result = pipeline._extract_rag_display_data(context)
        assert result[0]["collection"] == "flashcards"
        assert result[0]["distance"] == 0.42

    def test_multiple_results(self) -> None:
        """Multiple results are all extracted."""
        pipeline = self._make_pipeline()
        context = PipelineContext(
            user_input="test", persona={}, messages=[],
            metadata={"rag_results": [
                {"content": "General mem", "collection": "general", "distance": 0.2, "metadata": {}},
                {"content": "Flash card", "collection": "flashcards", "distance": 0.5, "metadata": {}},
                {"content": "Summary", "collection": "summaries", "distance": 0.7, "metadata": {}},
            ]},
        )
        result = pipeline._extract_rag_display_data(context)
        assert len(result) == 3
        assert [r["collection"] for r in result] == ["general", "flashcards", "summaries"]
