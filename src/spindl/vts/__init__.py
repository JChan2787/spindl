"""
VTubeStudio driver for spindl (NANO-060).

Drives VTubeStudio as an output driver from agent pipeline events:
- WebSocket connection + token-based auth
- Hotkey triggering, expression activation, model positioning
- Queue-based async command dispatch from any thread
"""

from .driver import VTSDriver

__all__ = ["VTSDriver"]
