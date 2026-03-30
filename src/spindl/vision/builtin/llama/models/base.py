"""
VLM Model Configuration Base - Abstract base for model-specific launch configurations.

This module defines the protocol for model-specific llama-server launch commands.
Different VLM architectures (Gemma3, Qwen2-VL, LLaVA, etc.) require different
flags and configurations.

Two-Layer Architecture:
    - Provider Layer: How to talk to the VLM (HTTP client, API format)
    - Model Config Layer: How to LAUNCH the VLM (architecture-specific flags)

This is the Model Config Layer.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelLaunchConfig:
    """
    Generated launch configuration for llama-server.

    Contains everything needed to start the VLM server process.
    """

    command: str
    """Full llama-server command with all arguments."""

    health_url: str
    """Health check endpoint URL for the server."""

    port: int
    """Port the server will listen on."""


class VLMModelConfig(ABC):
    """
    Abstract base for VLM model configurations.

    Each VLM architecture (Gemma3, Qwen2-VL, LLaVA, MiniCPM-V, etc.) implements
    this to define how to construct the llama-server launch command.

    This abstraction allows users to specify `model_type: "gemma3"` in config
    and have the system automatically construct the correct command with
    appropriate flags (--mmproj for architectures that need it, etc.).
    """

    @property
    @abstractmethod
    def model_type(self) -> str:
        """
        Identifier for this model type.

        Used in config as `vision.providers.llama.model_type`.

        Examples: "gemma3", "qwen2_vl", "llava", "minicpm_v"
        """
        ...

    @property
    @abstractmethod
    def requires_mmproj(self) -> bool:
        """
        Whether this architecture requires a separate mmproj file.

        Gemma3, LLaVA, MiniCPM-V: True (base model + mmproj)
        Qwen2-VL: False (unified GGUF with vision built-in)
        """
        ...

    @abstractmethod
    def generate_launch_config(
        self,
        executable_path: str,
        model_path: str,
        mmproj_path: Optional[str],
        port: int = 5558,
        context_size: int = 8192,
        gpu_layers: int = 99,
        extra_args: Optional[list[str]] = None,
        device: Optional[str] = None,
        tensor_split: Optional[list[float]] = None,
    ) -> ModelLaunchConfig:
        """
        Generate the llama-server launch command for this model.

        Args:
            executable_path: Full path to the llama-server executable (REQUIRED)
            model_path: Path to the GGUF model file
            mmproj_path: Path to the mmproj file (if required by architecture)
            port: Port for the server to listen on
            context_size: Context window size in tokens
            gpu_layers: Number of layers to offload to GPU (-1 or 99 for all)
            extra_args: Additional llama-server arguments if needed
            device: GPU device to use (e.g., "CUDA0", "CUDA1")
            tensor_split: GPU split ratios for multi-GPU (e.g., [0.5, 0.5])

        Returns:
            ModelLaunchConfig with command, health_url, and port

        Raises:
            ValueError: If required config is missing (e.g., mmproj for Gemma3)
        """
        ...

    @abstractmethod
    def validate_config(
        self,
        model_path: str,
        mmproj_path: Optional[str],
    ) -> tuple[bool, Optional[str]]:
        """
        Validate configuration for this model type.

        Called before generate_launch_config() to catch errors early.

        Args:
            model_path: Path to the GGUF model file
            mmproj_path: Path to the mmproj file (may be None)

        Returns:
            Tuple of (is_valid, error_message).
            error_message is None if valid, descriptive string if invalid.
        """
        ...
