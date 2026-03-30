"""Tests for stimuli data models."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spindl.stimuli.models import StimulusData, StimulusSource


class TestStimulusSource:
    """Tests for StimulusSource enum."""

    def test_patience_value(self):
        assert StimulusSource.PATIENCE.value == "patience"

    def test_custom_value(self):
        assert StimulusSource.CUSTOM.value == "custom"

    def test_module_value(self):
        assert StimulusSource.MODULE.value == "module"

    def test_twitch_value(self):
        assert StimulusSource.TWITCH.value == "twitch"

    def test_enum_members(self):
        assert len(StimulusSource) == 4


class TestStimulusData:
    """Tests for StimulusData dataclass."""

    def test_basic_construction(self):
        data = StimulusData(
            source=StimulusSource.PATIENCE,
            user_input="Hello world",
        )
        assert data.source == StimulusSource.PATIENCE
        assert data.user_input == "Hello world"
        assert data.metadata == {}

    def test_with_metadata(self):
        data = StimulusData(
            source=StimulusSource.CUSTOM,
            user_input="Test prompt",
            metadata={"elapsed_seconds": 42.5},
        )
        assert data.metadata["elapsed_seconds"] == 42.5

    def test_default_metadata_is_independent(self):
        """Ensure default_factory creates independent dicts."""
        data1 = StimulusData(source=StimulusSource.PATIENCE, user_input="a")
        data2 = StimulusData(source=StimulusSource.PATIENCE, user_input="b")
        data1.metadata["key"] = "value"
        assert "key" not in data2.metadata
