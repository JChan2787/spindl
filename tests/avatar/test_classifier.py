"""Tests for ONNXEmotionClassifier (NANO-094)."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from spindl.avatar.classifier import (
    ONNXEmotionClassifier,
    GOEMOTION_LABELS,
    GOEMOTION_TO_BUCKET,
    BUCKET_TO_MOOD,
    _softmax,
)

# Legacy alias used in some tests
GOEMOTION_TO_MOOD = GOEMOTION_TO_BUCKET


class TestGoEmotionBucketMapping:
    """Verify the GoEmotions → bucket → avatar mood mapping."""

    def test_all_28_labels_mapped(self):
        """Every GoEmotions label must have a bucket mapping entry."""
        for label in GOEMOTION_LABELS:
            assert label in GOEMOTION_TO_BUCKET, f"Missing mapping for label: {label}"

    def test_mapping_count(self):
        """28 GoEmotions labels → 28 mapping entries."""
        assert len(GOEMOTION_TO_BUCKET) == 28

    def test_neutral_maps_to_neutral_bucket(self):
        """Neutral label should map to 'neutral' bucket."""
        assert GOEMOTION_TO_BUCKET["neutral"] == "neutral"

    def test_neutral_bucket_maps_to_none_mood(self):
        """Neutral bucket should map to None (no avatar mood change)."""
        assert BUCKET_TO_MOOD["neutral"] is None

    def test_all_buckets_are_valid(self):
        """All bucket values must be one of the 5 broad categories."""
        valid_buckets = {"happy", "sad", "angry", "surprised", "neutral"}
        for label, bucket in GOEMOTION_TO_BUCKET.items():
            assert bucket in valid_buckets, (
                f"Label '{label}' maps to bucket '{bucket}' which is not valid"
            )

    def test_bucket_count(self):
        """5 broad emotion buckets (including neutral)."""
        buckets = {b for b in GOEMOTION_TO_BUCKET.values()}
        assert len(buckets) == 5

    def test_all_buckets_have_avatar_mood(self):
        """Every bucket must map to an avatar mood (or None for neutral)."""
        for bucket in {"happy", "sad", "angry", "surprised", "neutral"}:
            assert bucket in BUCKET_TO_MOOD

    def test_avatar_moods_are_valid_presets(self):
        """All non-None bucket → mood values must be valid NANO-092 avatar presets."""
        valid_presets = {
            "default", "error", "frustrated", "annoyed", "success", "pleased",
            "amused", "grateful", "warn", "curious", "intrigued", "melancholy",
            "concerned", "pensive", "focused", "skeptical", "contemplative",
            "smirk", "deadpan", "dramatic",
        }
        for bucket, mood in BUCKET_TO_MOOD.items():
            if mood is not None:
                assert mood in valid_presets, (
                    f"Bucket '{bucket}' maps to '{mood}' which is not a valid avatar preset"
                )

    def test_label_order_matches_model_output(self):
        """Labels list must have exactly 28 entries matching GoEmotions."""
        assert len(GOEMOTION_LABELS) == 28
        assert GOEMOTION_LABELS[0] == "admiration"
        assert GOEMOTION_LABELS[-1] == "neutral"


class TestSoftmax:
    """Verify softmax implementation."""

    def test_output_sums_to_one(self):
        """Softmax output should sum to ~1.0."""
        logits = np.array([1.0, 2.0, 3.0])
        result = _softmax(logits)
        assert abs(result.sum() - 1.0) < 1e-6

    def test_highest_logit_gets_highest_prob(self):
        """Highest logit should have highest probability."""
        logits = np.array([1.0, 5.0, 2.0])
        result = _softmax(logits)
        assert np.argmax(result) == 1

    def test_numerical_stability(self):
        """Should not overflow with large logits."""
        logits = np.array([1000.0, 1001.0, 999.0])
        result = _softmax(logits)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
        assert abs(result.sum() - 1.0) < 1e-6


class TestONNXEmotionClassifier:
    """Test classifier with mocked ONNX session and tokenizer."""

    def _make_logits(self, label_idx: int, confidence: float = 0.9) -> np.ndarray:
        """Create logits array where label_idx dominates with given confidence.

        Uses log-space to ensure softmax produces approximately the desired confidence.
        """
        logits = np.zeros((1, 28))
        # Set target logit high enough to dominate after softmax
        logits[0, label_idx] = np.log(confidence / (1 - confidence) * 27)
        return logits

    @patch("spindl.avatar.classifier.ONNXEmotionClassifier._ensure_model_files")
    def test_classify_high_confidence(self, mock_ensure):
        """High-confidence classification returns mood and score."""
        mock_ensure.return_value = True

        classifier = ONNXEmotionClassifier(confidence_threshold=0.3)

        # Mock ONNX session
        mock_session = MagicMock()
        joy_idx = GOEMOTION_LABELS.index("joy")
        mock_session.run.return_value = [self._make_logits(joy_idx, 0.85)]

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_encoding = MagicMock()
        mock_encoding.ids = [101, 2023, 2003, 2307, 102] + [0] * 507
        mock_encoding.attention_mask = [1, 1, 1, 1, 1] + [0] * 507
        mock_tokenizer.encode.return_value = mock_encoding

        classifier._session = mock_session
        classifier._tokenizer = mock_tokenizer
        classifier._initialized = True

        mood, confidence = classifier.classify("This is great!")
        assert mood == "amused"  # joy → happy bucket → amused
        assert confidence is not None
        assert confidence > 0.5  # Should be high (normalized 0-1)

    @patch("spindl.avatar.classifier.ONNXEmotionClassifier._ensure_model_files")
    def test_classify_below_threshold(self, mock_ensure):
        """Below-threshold classification returns (None, None)."""
        mock_ensure.return_value = True

        # With bucket summing, uniform logits give ~39% to happy (11 labels),
        # so threshold must be higher than that to test the below-threshold path
        classifier = ONNXEmotionClassifier(confidence_threshold=0.95)

        mock_session = MagicMock()
        # Uniform logits → probability spreads across buckets, none hits 95%
        mock_session.run.return_value = [np.zeros((1, 28))]

        mock_tokenizer = MagicMock()
        mock_encoding = MagicMock()
        mock_encoding.ids = [101, 102] + [0] * 510
        mock_encoding.attention_mask = [1, 1] + [0] * 510
        mock_tokenizer.encode.return_value = mock_encoding

        classifier._session = mock_session
        classifier._tokenizer = mock_tokenizer
        classifier._initialized = True

        mood, confidence = classifier.classify("Some text")
        assert mood is None
        assert confidence is None

    @patch("spindl.avatar.classifier.ONNXEmotionClassifier._ensure_model_files")
    def test_classify_neutral_returns_none(self, mock_ensure):
        """Neutral classification returns (None, None) even with high confidence."""
        mock_ensure.return_value = True

        classifier = ONNXEmotionClassifier(confidence_threshold=0.3)

        mock_session = MagicMock()
        neutral_idx = GOEMOTION_LABELS.index("neutral")
        mock_session.run.return_value = [self._make_logits(neutral_idx, 0.9)]

        mock_tokenizer = MagicMock()
        mock_encoding = MagicMock()
        mock_encoding.ids = [101, 102] + [0] * 510
        mock_encoding.attention_mask = [1, 1] + [0] * 510
        mock_tokenizer.encode.return_value = mock_encoding

        classifier._session = mock_session
        classifier._tokenizer = mock_tokenizer
        classifier._initialized = True

        mood, confidence = classifier.classify("Hello.")
        assert mood is None
        assert confidence is None

    def test_classify_empty_text(self):
        """Empty or whitespace text returns (None, None) without init."""
        classifier = ONNXEmotionClassifier()
        assert classifier.classify("") == (None, None)
        assert classifier.classify("   ") == (None, None)

    def test_classify_disabled_after_init_failure(self):
        """Classifier disables after init failure, doesn't retry."""
        classifier = ONNXEmotionClassifier(model_dir="/nonexistent/path")
        classifier._disabled = True

        mood, confidence = classifier.classify("Some text")
        assert mood is None
        assert confidence is None

    @patch("spindl.avatar.classifier.ONNXEmotionClassifier._ensure_model_files")
    def test_confidence_is_percentage(self, mock_ensure):
        """Confidence should be returned as a percentage (0-100), not a probability."""
        mock_ensure.return_value = True

        classifier = ONNXEmotionClassifier(confidence_threshold=0.3)

        mock_session = MagicMock()
        admiration_idx = GOEMOTION_LABELS.index("admiration")
        mock_session.run.return_value = [self._make_logits(admiration_idx, 0.75)]

        mock_tokenizer = MagicMock()
        mock_encoding = MagicMock()
        mock_encoding.ids = [101, 102] + [0] * 510
        mock_encoding.attention_mask = [1, 1] + [0] * 510
        mock_tokenizer.encode.return_value = mock_encoding

        classifier._session = mock_session
        classifier._tokenizer = mock_tokenizer
        classifier._initialized = True

        mood, confidence = classifier.classify("Amazing work!")
        assert mood == "amused"  # admiration → happy bucket → amused
        assert confidence is not None
        assert 0 < confidence <= 100


class TestONNXEmotionClassifierInit:
    """Test lazy initialization behavior."""

    def test_not_initialized_at_construction(self):
        """Classifier should not load model at construction time."""
        classifier = ONNXEmotionClassifier()
        assert classifier._initialized is False
        assert classifier._session is None
        assert classifier._tokenizer is None

    def test_disabled_flag_prevents_retry(self):
        """Once disabled, _init() returns False without retrying."""
        classifier = ONNXEmotionClassifier()
        classifier._disabled = True
        assert classifier._init() is False


class TestAvatarConfigUpdates:
    """Test AvatarConfig changes for NANO-094."""

    def test_drop_rule_from_literal(self):
        """AvatarConfig should only accept 'classifier' and 'off'."""
        from spindl.orchestrator.config import AvatarConfig

        config = AvatarConfig(emotion_classifier="classifier")
        assert config.emotion_classifier == "classifier"

        config = AvatarConfig(emotion_classifier="off")
        assert config.emotion_classifier == "off"

        with pytest.raises(Exception):
            AvatarConfig(emotion_classifier="rule")

    def test_from_dict_coerces_rule_to_off(self):
        """Legacy 'rule' value should be coerced to 'off'."""
        from spindl.orchestrator.config import AvatarConfig

        config = AvatarConfig.from_dict({"emotion_classifier": "rule"})
        assert config.emotion_classifier == "off"

    def test_from_dict_coerces_yaml_false_to_off(self):
        """YAML unquoted 'off' parsed as boolean False should coerce to 'off'."""
        from spindl.orchestrator.config import AvatarConfig

        config = AvatarConfig.from_dict({"emotion_classifier": False})
        assert config.emotion_classifier == "off"

    def test_new_fields_have_defaults(self):
        """New fields should have sensible defaults."""
        from spindl.orchestrator.config import AvatarConfig

        config = AvatarConfig()
        assert config.emotion_model_path == "models/emotion"
        assert config.emotion_confidence_threshold == 0.3
        assert config.expression_fade_delay == 1.0
        assert config.show_emotion_in_chat is True

    def test_from_dict_reads_new_fields(self):
        """from_dict should parse the new NANO-094 fields."""
        from spindl.orchestrator.config import AvatarConfig

        config = AvatarConfig.from_dict({
            "enabled": True,
            "emotion_classifier": "classifier",
            "emotion_model_path": "/custom/path",
            "emotion_confidence_threshold": 0.5,
            "expression_fade_delay": 2.0,
            "show_emotion_in_chat": False,
        })
        assert config.enabled is True
        assert config.emotion_classifier == "classifier"
        assert config.emotion_model_path == "/custom/path"
        assert config.emotion_confidence_threshold == 0.5
        assert config.expression_fade_delay == 2.0
        assert config.show_emotion_in_chat is False


class TestResponseReadyEventEmotion:
    """Test emotion fields on ResponseReadyEvent."""

    def test_default_emotion_fields(self):
        """Emotion fields should default to None."""
        from spindl.core.events import ResponseReadyEvent

        event = ResponseReadyEvent(text="Hello", user_input="Hi")
        assert event.emotion is None
        assert event.emotion_confidence is None

    def test_emotion_fields_populated(self):
        """Emotion fields should be settable."""
        from spindl.core.events import ResponseReadyEvent

        event = ResponseReadyEvent(
            text="Hello",
            user_input="Hi",
            emotion="amused",
            emotion_confidence=72.5,
        )
        assert event.emotion == "amused"
        assert event.emotion_confidence == 72.5
