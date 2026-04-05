"""
Launcher configuration dataclasses and parsing.

Handles loading and validation of the launcher: section from spindl.yaml.
Also reads TTS provider configuration for provider-driven services (NANO-015).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

import yaml


@dataclass
class HealthCheckConfig:
    """Health check configuration for a service."""

    type: Literal["tcp", "http", "provider", "none"]
    host: str | None = None
    port: int | None = None
    url: str | None = None
    timeout: int | None = None  # Override default timeout

    def __post_init__(self):
        if self.type == "tcp" and (self.host is None or self.port is None):
            raise ValueError("TCP health check requires 'host' and 'port'")
        if self.type == "http" and self.url is None:
            raise ValueError("HTTP health check requires 'url'")
        # "provider" type: health check is delegated to TTSProvider.health_check()
        # No additional fields required - provider config is read from tts section


@dataclass
class ServiceConfig:
    """Configuration for a single service."""

    name: str
    platform: Literal["native", "wsl"]
    command: str | None  # None for provider-driven services (e.g., TTS)
    health_check: HealthCheckConfig
    enabled: bool = True
    wsl_distro: str | None = None
    depends_on: list[str] = field(default_factory=list)
    pass_debug: bool = False

    def __post_init__(self):
        if self.platform == "wsl" and self.wsl_distro is None:
            raise ValueError(f"Service '{self.name}' uses WSL platform but 'wsl_distro' not specified")


@dataclass
class TTSProviderConfig:
    """
    TTS provider configuration for launcher (NANO-015).

    Extracted from tts: section to enable provider-driven service startup.
    """

    provider: str = "kokoro"
    plugin_paths: list[str] = field(default_factory=list)
    provider_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMProviderConfig:
    """
    LLM provider configuration for launcher (NANO-019).

    Extracted from llm: section to enable provider-driven service startup.
    Cloud providers (e.g., DeepSeek) return None from get_server_command(),
    causing the launcher to skip the LLM service entirely.
    """

    provider: str = "llama"
    plugin_paths: list[str] = field(default_factory=list)
    provider_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class VisionProviderConfig:
    """
    Vision (VLM) provider configuration for launcher (NANO-023).

    Extracted from vision: section to enable provider-driven VLM service startup.
    Cloud providers (e.g., OpenAI) return None from get_server_command(),
    causing the launcher to skip the VLM service entirely.
    """

    enabled: bool = False
    provider: str = "llama"
    plugin_paths: list[str] = field(default_factory=list)
    provider_config: dict[str, Any] = field(default_factory=dict)


@dataclass
class LauncherConfig:
    """Top-level launcher configuration."""

    services: dict[str, ServiceConfig]
    log_file: str = "./logs/launcher.log"
    log_level: Literal["debug", "info", "warning", "error"] = "info"
    health_check_timeout: int = 60

    # TTS provider config for provider-driven TTS service (NANO-015)
    tts_provider_config: Optional[TTSProviderConfig] = None

    # LLM provider config for provider-driven LLM service (NANO-019)
    llm_provider_config: Optional[LLMProviderConfig] = None

    # Vision (VLM) provider config for provider-driven VLM service (NANO-023)
    vision_provider_config: Optional[VisionProviderConfig] = None

    # LLM context size for -c flag injection (NANO-096: reads from provider config)
    llm_context_size: int = 8192

    # Per-service log level overrides (NANO-017)
    # e.g., {"stt": "error", "llm": "warning"}
    service_levels: dict[str, str] = field(default_factory=dict)

    # Regex patterns to suppress from console output (NANO-017)
    # e.g., ["Connection from", "-> success:", "slot \\d+:"]
    suppress_patterns: list[str] = field(default_factory=list)

    def get_startup_order(self) -> list[str]:
        """
        Return service names in dependency-respecting startup order.

        Uses Kahn's algorithm for topological sort.

        Returns:
            List of service names in startup order.

        Raises:
            ValueError: If circular dependency detected.
        """
        # Build adjacency list and in-degree counts
        in_degree: dict[str, int] = {name: 0 for name in self.services}
        dependents: dict[str, list[str]] = {name: [] for name in self.services}

        for name, svc in self.services.items():
            for dep in svc.depends_on:
                if dep not in self.services:
                    raise ValueError(f"Service '{name}' depends on unknown service '{dep}'")
                dependents[dep].append(name)
                in_degree[name] += 1

        # Start with services that have no dependencies
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result: list[str] = []

        while queue:
            # Sort for deterministic order
            queue.sort()
            current = queue.pop(0)
            result.append(current)

            for dependent in dependents[current]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(result) != len(self.services):
            # Find the cycle for better error message
            remaining = set(self.services.keys()) - set(result)
            raise ValueError(f"Circular dependency detected involving: {remaining}")

        return result


def _parse_health_check(data: dict) -> HealthCheckConfig:
    """Parse health check configuration from dict."""
    return HealthCheckConfig(
        type=data.get("type", "none"),
        host=data.get("host"),
        port=data.get("port"),
        url=data.get("url"),
        timeout=data.get("timeout"),
    )


def _parse_service(name: str, data: dict) -> ServiceConfig:
    """Parse service configuration from dict."""
    if "platform" not in data:
        raise ValueError(f"Service '{name}' missing required field 'platform'")
    if "health_check" not in data:
        raise ValueError(f"Service '{name}' missing required field 'health_check'")

    # Command is optional for provider-driven services (e.g., TTS)
    # The launcher will derive the command from the provider
    command = data.get("command")

    return ServiceConfig(
        name=name,
        platform=data["platform"],
        command=command,
        health_check=_parse_health_check(data["health_check"]),
        enabled=data.get("enabled", True),
        wsl_distro=data.get("wsl_distro"),
        depends_on=data.get("depends_on", []),
        pass_debug=data.get("pass_debug", False),
    )


def _parse_tts_provider_config(config: dict) -> Optional[TTSProviderConfig]:
    """
    Parse TTS provider configuration from the tts: section.

    Args:
        config: Full config dict (root level)

    Returns:
        TTSProviderConfig if tts section exists, None otherwise
    """
    tts_data = config.get("tts")
    if tts_data is None:
        return None

    provider = tts_data.get("provider", "kokoro")
    plugin_paths = tts_data.get("plugin_paths", [])

    # Get provider-specific config
    providers = tts_data.get("providers", {})
    provider_config = providers.get(provider, {})

    return TTSProviderConfig(
        provider=provider,
        plugin_paths=plugin_paths,
        provider_config=provider_config,
    )


def _parse_llm_provider_config(config: dict) -> Optional[LLMProviderConfig]:
    """
    Parse LLM provider configuration from the llm: section.

    Args:
        config: Full config dict (root level)

    Returns:
        LLMProviderConfig if llm section exists, None otherwise
    """
    llm_data = config.get("llm")
    if llm_data is None:
        return None

    provider = llm_data.get("provider", "llama")
    plugin_paths = llm_data.get("plugin_paths", [])

    # Get provider-specific config
    providers = llm_data.get("providers", {})
    provider_config = providers.get(provider, {})

    return LLMProviderConfig(
        provider=provider,
        plugin_paths=plugin_paths,
        provider_config=provider_config,
    )


def _parse_vision_provider_config(config: dict) -> Optional[VisionProviderConfig]:
    """
    Parse VLM provider configuration from the vlm: section.

    Args:
        config: Full config dict (root level)

    Returns:
        VisionProviderConfig if vlm section exists, None otherwise
    """
    vlm_data = config.get("vlm")
    if vlm_data is None:
        return None

    provider = vlm_data.get("provider", "llama")
    plugin_paths = vlm_data.get("plugin_paths", [])

    # provider: "none" means VLM is explicitly disabled — don't treat
    # "section exists" as "enabled". Session 606 bug: enabled=True with
    # provider="none" caused stale mmproj_path to survive and break launches.
    if provider == "none":
        return VisionProviderConfig(
            enabled=False,
            provider="none",
            plugin_paths=plugin_paths,
            provider_config={},
        )

    # Get provider-specific config
    providers = vlm_data.get("providers", {})
    provider_config = providers.get(provider, {})

    return VisionProviderConfig(
        enabled=True,
        provider=provider,
        plugin_paths=plugin_paths,
        provider_config=provider_config,
    )


def load_launcher_config(config_path: str | None = None) -> LauncherConfig:
    """
    Load launcher configuration from YAML file.

    Args:
        config_path: Path to configuration file.

    Returns:
        LauncherConfig with parsed services and TTS provider config.

    Raises:
        FileNotFoundError: Config file not found.
        ValueError: Invalid config format or missing required fields.
    """
    from spindl.config.config_loader import get_config_path
    path = Path(config_path if config_path is not None else get_config_path())

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Failed to parse YAML: {e}")

    if not isinstance(config, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(config).__name__}")

    launcher_data = config.get("launcher")
    if launcher_data is None:
        raise ValueError("Config missing 'launcher' section")

    services_data = launcher_data.get("services")
    if not services_data:
        raise ValueError("Launcher config missing 'services' section")

    services = {}
    for name, svc_data in services_data.items():
        services[name] = _parse_service(name, svc_data)

    # Parse TTS provider config (NANO-015)
    tts_provider_config = _parse_tts_provider_config(config)

    # Parse LLM provider config (NANO-019)
    llm_provider_config = _parse_llm_provider_config(config)

    # Parse Vision (VLM) provider config (NANO-023)
    vision_provider_config = _parse_vision_provider_config(config)

    # Parse logging config (NANO-017)
    logging_data = launcher_data.get("logging", {})
    service_levels = logging_data.get("services", {})
    suppress_patterns = logging_data.get("suppress", [])

    # Read context_size from LLM provider config (NANO-096: replaces pipeline.max_context)
    llm_context_size = 8192
    if llm_provider_config:
        llm_context_size = llm_provider_config.provider_config.get("context_size", 8192)

    return LauncherConfig(
        services=services,
        log_file=launcher_data.get("log_file", "./logs/launcher.log"),
        log_level=launcher_data.get("log_level", "info"),
        health_check_timeout=launcher_data.get("health_check_timeout", 60),
        tts_provider_config=tts_provider_config,
        llm_provider_config=llm_provider_config,
        vision_provider_config=vision_provider_config,
        service_levels=service_levels,
        suppress_patterns=suppress_patterns,
        llm_context_size=llm_context_size,
    )
