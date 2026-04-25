"""
Game-state bridge stimulus module (NANO-116).

Consumes events from the SPNDL-001 game-state bridge via TCP and
feeds gameplay + narrative awareness into the StimuliEngine.
"""

from .module import GameStateModule

__all__ = ["GameStateModule"]
