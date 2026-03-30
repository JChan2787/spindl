"""Avatar integration for SpindL — bridges orchestrator events to the avatar renderer."""

from .tool_mood import AvatarToolMoodSubscriber, TOOL_MOOD_MAP
from .classifier import ONNXEmotionClassifier, GOEMOTION_TO_MOOD

__all__ = [
    "AvatarToolMoodSubscriber",
    "TOOL_MOOD_MAP",
    "ONNXEmotionClassifier",
    "GOEMOTION_TO_MOOD",
]
