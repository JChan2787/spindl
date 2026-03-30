"""
Gemma3 Model Configuration - Launch config for Gemma 3 VLM architecture.

Gemma 3 models (including X-Ray_Alpha) use a two-file architecture:
- Base GGUF model file
- Separate mmproj (multimodal projector) file

Both are required for vision capabilities.
"""

from typing import Optional

from .base import ModelLaunchConfig, VLMModelConfig


class Gemma3ModelConfig(VLMModelConfig):
    """
    Gemma 3 VLM configuration.

    Supports models like:
    - X-Ray_Alpha (Gemma 3 4B, uncensored vision)
    - Gemma 3 4B/12B/27B with mmproj

    Architecture: Base model + separate mmproj file required.
    """

    @property
    def model_type(self) -> str:
        return "gemma3"

    @property
    def requires_mmproj(self) -> bool:
        return True

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
        Generate llama-server command for Gemma 3 models.

        Args:
            executable_path: Full path to the llama-server executable (REQUIRED)
            model_path: Path to the base GGUF model
            mmproj_path: Path to the mmproj file (REQUIRED)
            port: Server port (default 5558)
            context_size: Context window (default 8192)
            gpu_layers: GPU layers (default 99 = all)
            extra_args: Additional llama-server flags
            device: GPU device to use (e.g., "CUDA1")

        Returns:
            ModelLaunchConfig with complete launch command

        Raises:
            ValueError: If mmproj_path is not provided
        """
        if not mmproj_path:
            raise ValueError(
                f"Gemma 3 architecture requires mmproj_path. "
                f"Please provide the path to the multimodal projector file."
            )

        # Build command arguments
        # Note: Paths are NOT quoted here. The launcher passes the full command
        # string to `cmd /c` which handles path parsing. Embedded quotes break
        # Windows command execution.
        args = [
            executable_path,
            "-m",
            model_path,
            "--mmproj",
            mmproj_path,
            "-c",
            str(context_size),
            "--port",
            str(port),
            "-ngl",
            str(gpu_layers),
        ]

        # GPU selection (tensor_split takes precedence over device)
        if tensor_split:
            split_str = ",".join(str(x) for x in tensor_split)
            args.extend(["--tensor-split", split_str])
        elif device:
            args.extend(["--device", device])

        # Add any extra arguments
        if extra_args:
            args.extend(extra_args)

        return ModelLaunchConfig(
            command=" ".join(args),
            health_url=f"http://127.0.0.1:{port}/health",
            port=port,
        )

    def validate_config(
        self,
        model_path: str,
        mmproj_path: Optional[str],
    ) -> tuple[bool, Optional[str]]:
        """
        Validate Gemma 3 configuration.

        Checks that mmproj_path is provided (required for this architecture).

        Args:
            model_path: Path to the GGUF model file
            mmproj_path: Path to the mmproj file

        Returns:
            (True, None) if valid, (False, error_message) if invalid
        """
        if not model_path:
            return False, "model_path is required"

        if not mmproj_path:
            return False, (
                "Gemma 3 architecture requires mmproj_path. "
                "This model uses a separate multimodal projector file."
            )

        return True, None
