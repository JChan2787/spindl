"""Response schema models for GUI socket events (NANO-089).

These Pydantic models enforce the shape of config response payloads
emitted via socket.io. They ensure the frontend always receives
complete, well-typed data — preventing bugs like #1 (missing unified_vlm
field) and #13 (missing cloud_config).
"""

from typing import Optional

from pydantic import BaseModel


class LLMConfigResponse(BaseModel):
    """Response shape for llm_config_updated socket events."""

    provider: str
    model: Optional[str] = None
    context_size: Optional[int] = None
    available_providers: list[str] = []


class CloudVLMConfig(BaseModel):
    """Masked cloud VLM credentials for dashboard hydration."""

    api_key: str = ""
    model: str = ""
    base_url: str = ""


class VLMConfigResponse(BaseModel):
    """Response shape for vlm_config_updated socket events."""

    provider: str
    available_providers: list[str] = []
    healthy: bool = False
    cloud_config: Optional[CloudVLMConfig] = None


class ToolState(BaseModel):
    """Per-tool state in tools config response."""

    enabled: bool
    label: str


class ToolsConfigResponse(BaseModel):
    """Response shape for tools_config_updated socket events."""

    master_enabled: bool
    tools: dict[str, ToolState] = {}
    persisted: Optional[bool] = None
    error: Optional[str] = None
