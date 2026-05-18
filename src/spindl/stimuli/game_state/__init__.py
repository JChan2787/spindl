"""
Game-state bridge stimulus module (NANO-116).

Consumes events from the SPNDL-001 game-state bridge via TCP and
feeds gameplay + narrative awareness into the StimuliEngine.
"""

from .dialogue_buffer import DialogueBuffer, DialogueLine, GameplaySnapshot
from .dialogue_store import DialogueStore
from .dialogue_summarizer import DialogueSummarizer
from .module import GameStateModule

__all__ = [
    "DialogueBuffer",
    "DialogueLine",
    "DialogueStore",
    "DialogueSummarizer",
    "GameplaySnapshot",
    "GameStateModule",
]
