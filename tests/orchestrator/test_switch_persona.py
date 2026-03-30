"""Tests for runtime character switching (NANO-077 Phase 1).

Tests cover:
- switch_persona: state gating (IDLE/LISTENING allowed, PROCESSING blocked)
- switch_persona: same-character delegates to reload_persona
- switch_persona: persona dict updated after swap
- switch_persona: codex swapped (load_character called with new id)
- switch_persona: memory store swapped (switch_character called)
- switch_persona: history manager gets new session (switch_to_persona called)
- switch_persona: reflection system stopped and restarted
- switch_persona: config.character_id updated in-memory
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spindl.core import AgentState
from spindl.orchestrator.voice_agent import VoiceAgentOrchestrator


def _make_orchestrator(
    state: str = "listening",
    character_id: str = "spindle",
    has_memory: bool = True,
    has_reflection: bool = True,
) -> VoiceAgentOrchestrator:
    """Create a VoiceAgentOrchestrator with mocked internals for swap testing."""
    orch = VoiceAgentOrchestrator.__new__(VoiceAgentOrchestrator)

    # State machine
    orch._state_machine = MagicMock()
    orch._state_machine.state = AgentState(state)

    # Persona
    orch._persona = {"id": character_id, "name": "Spindle"}

    # Config
    orch._config = MagicMock()
    orch._config.character_id = character_id
    orch._config.characters_dir = "./characters"

    # Callbacks
    orch._callbacks = MagicMock()

    # Codex manager
    orch._codex_manager = MagicMock()

    # Memory store
    if has_memory:
        orch._memory_store = MagicMock()
    else:
        orch._memory_store = None

    # History manager
    orch._history_manager = MagicMock()
    orch._history_manager.session_file = Path("/tmp/mryummers_20260306_0335.jsonl")
    orch._history_manager._pending_user_input = None

    # Reflection system
    if has_reflection:
        orch._reflection_system = MagicMock()
    else:
        orch._reflection_system = None

    return orch


class TestSwitchPersonaStateGating:
    """State gating: only IDLE and LISTENING are allowed."""

    def test_allowed_when_listening(self) -> None:
        orch = _make_orchestrator(state="listening")
        with patch.object(orch, "reload_persona", return_value=True):
            with patch(
                "spindl.orchestrator.voice_agent.CharacterLoader"
            ) as MockLoader:
                MockLoader.return_value.load_as_dict.return_value = {
                    "id": "mryummers",
                    "name": "Mister Yummers",
                }
                result = orch.switch_persona("mryummers")
        assert result is True

    def test_allowed_when_idle(self) -> None:
        orch = _make_orchestrator(state="idle")
        with patch(
            "spindl.orchestrator.voice_agent.CharacterLoader"
        ) as MockLoader:
            MockLoader.return_value.load_as_dict.return_value = {
                "id": "mryummers",
                "name": "Mister Yummers",
            }
            result = orch.switch_persona("mryummers")
        assert result is True

    def test_blocked_when_processing(self) -> None:
        orch = _make_orchestrator(state="processing")
        result = orch.switch_persona("mryummers")
        assert result is False

    def test_no_state_machine_allows_swap(self) -> None:
        """If no state machine (headless mode), swap should proceed."""
        orch = _make_orchestrator()
        orch._state_machine = None
        with patch(
            "spindl.orchestrator.voice_agent.CharacterLoader"
        ) as MockLoader:
            MockLoader.return_value.load_as_dict.return_value = {
                "id": "mryummers",
                "name": "Mister Yummers",
            }
            result = orch.switch_persona("mryummers")
        assert result is True


class TestSwitchPersonaSameCharacter:
    """Same character ID delegates to reload_persona."""

    def test_same_id_delegates_to_reload(self) -> None:
        orch = _make_orchestrator(character_id="spindle")
        with patch.object(orch, "reload_persona", return_value=True) as mock_reload:
            result = orch.switch_persona("spindle")
        assert result is True
        mock_reload.assert_called_once()


class TestSwitchPersonaSwapPipeline:
    """Full swap pipeline: persona, codex, memory, history, reflection, config."""

    @pytest.fixture
    def orch_and_loader(self):
        """Set up orchestrator with patched CharacterLoader."""
        orch = _make_orchestrator(character_id="spindle")
        new_persona = {"id": "mryummers", "name": "Mister Yummers"}
        with patch(
            "spindl.orchestrator.voice_agent.CharacterLoader"
        ) as MockLoader:
            MockLoader.return_value.load_as_dict.return_value = new_persona
            orch.switch_persona("mryummers")
        return orch, new_persona

    def test_persona_dict_updated(self, orch_and_loader) -> None:
        orch, new_persona = orch_and_loader
        assert orch._persona == new_persona

    def test_callbacks_updated(self, orch_and_loader) -> None:
        orch, new_persona = orch_and_loader
        orch._callbacks.update_persona.assert_called_once_with(new_persona)

    def test_codex_swapped(self, orch_and_loader) -> None:
        orch, _ = orch_and_loader
        orch._codex_manager.load_character.assert_called_once_with("mryummers")

    def test_memory_store_swapped(self, orch_and_loader) -> None:
        orch, _ = orch_and_loader
        orch._memory_store.switch_character.assert_called_once_with("mryummers")

    def test_history_manager_switched(self, orch_and_loader) -> None:
        orch, _ = orch_and_loader
        orch._history_manager.switch_to_persona.assert_called_once_with("mryummers")

    def test_reflection_stopped_and_restarted(self, orch_and_loader) -> None:
        orch, _ = orch_and_loader
        orch._reflection_system.stop.assert_called_once()
        orch._reflection_system.start.assert_called_once()

    def test_config_character_id_updated(self, orch_and_loader) -> None:
        orch, _ = orch_and_loader
        assert orch._config.character_id == "mryummers"

    def test_no_memory_store_skips_gracefully(self) -> None:
        orch = _make_orchestrator(has_memory=False)
        with patch(
            "spindl.orchestrator.voice_agent.CharacterLoader"
        ) as MockLoader:
            MockLoader.return_value.load_as_dict.return_value = {
                "id": "mryummers",
                "name": "Mister Yummers",
            }
            result = orch.switch_persona("mryummers")
        assert result is True

    def test_no_reflection_skips_gracefully(self) -> None:
        orch = _make_orchestrator(has_reflection=False)
        with patch(
            "spindl.orchestrator.voice_agent.CharacterLoader"
        ) as MockLoader:
            MockLoader.return_value.load_as_dict.return_value = {
                "id": "mryummers",
                "name": "Mister Yummers",
            }
            result = orch.switch_persona("mryummers")
        assert result is True

    def test_pending_user_input_discarded(self) -> None:
        orch = _make_orchestrator()
        orch._history_manager._pending_user_input = "half-typed message"
        with patch(
            "spindl.orchestrator.voice_agent.CharacterLoader"
        ) as MockLoader:
            MockLoader.return_value.load_as_dict.return_value = {
                "id": "mryummers",
                "name": "Mister Yummers",
            }
            orch.switch_persona("mryummers")
        assert orch._history_manager._pending_user_input is None
