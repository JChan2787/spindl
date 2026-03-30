"""
GUI module for spindl.

Provides Socket.IO server for real-time communication with the web GUI.
"""

from .server import GUIServer
from .bridge import EventBridge

__all__ = ["GUIServer", "EventBridge"]
