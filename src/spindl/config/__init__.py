"""
Configuration management for spindl.

Provides loading and access to service configuration.
"""

from .config_loader import get_config_path, load_config

__all__ = ["get_config_path", "load_config"]
