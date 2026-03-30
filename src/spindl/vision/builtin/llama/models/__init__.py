"""
VLM Model Configurations - Registry of model-specific launch configs.

Each model architecture has its own config class that knows how to
construct the correct llama-server launch command.
"""

from typing import Type

from .base import ModelLaunchConfig, VLMModelConfig
from .gemma3 import Gemma3ModelConfig

# Registry of available model configurations
# Add new model types here as they're implemented
MODEL_CONFIGS: dict[str, Type[VLMModelConfig]] = {
    "gemma3": Gemma3ModelConfig,
    # Future model types (documented as placeholders):
    # "qwen2_vl": Qwen2VLModelConfig,
    # "llava": LLaVAModelConfig,
    # "minicpm_v": MiniCPMVModelConfig,
}


def get_model_config(model_type: str) -> VLMModelConfig:
    """
    Get a model configuration instance by type name.

    Args:
        model_type: Model type identifier (e.g., "gemma3", "qwen2_vl")

    Returns:
        Instantiated VLMModelConfig for the specified type

    Raises:
        ValueError: If model_type is not recognized
    """
    if model_type not in MODEL_CONFIGS:
        available = ", ".join(sorted(MODEL_CONFIGS.keys()))
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Available types: {available}"
        )
    return MODEL_CONFIGS[model_type]()


def list_model_types() -> list[str]:
    """
    List all available model type identifiers.

    Returns:
        Sorted list of model type strings
    """
    return sorted(MODEL_CONFIGS.keys())


__all__ = [
    "ModelLaunchConfig",
    "VLMModelConfig",
    "Gemma3ModelConfig",
    "MODEL_CONFIGS",
    "get_model_config",
    "list_model_types",
]
