"""
Health check implementations for launcher services.

Supports TCP socket checks, HTTP endpoint checks, provider-based checks,
and no-op checks.
"""

import socket
import time
import urllib.request
import urllib.error
from dataclasses import dataclass
from typing import Callable, Literal, Optional, Protocol, Tuple


class HealthCheckProvider(Protocol):
    """Protocol for provider-based health checks."""

    def health_check(self) -> bool:
        """Return True if provider is healthy."""
        ...


class TTSProviderProtocol(Protocol):
    """Protocol for TTS providers used in health checks."""

    def initialize(self, config: dict) -> None:
        """Initialize the provider with config."""
        ...

    def health_check(self) -> bool:
        """Return True if provider is healthy."""
        ...


from .config import HealthCheckConfig


@dataclass
class HealthCheckResult:
    """Result of a health check attempt."""

    success: bool
    message: str
    attempts: int
    elapsed_seconds: float


def _check_tcp(host: str, port: int) -> bool:
    """
    Attempt TCP connection to host:port.

    Returns True if connection succeeds.
    """
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except (socket.timeout, socket.error, OSError):
        return False


def _check_http(url: str) -> bool:
    """
    Attempt HTTP GET request to URL.

    Returns True if response is 2xx.
    """
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return False


def check_health(
    config: HealthCheckConfig,
    default_timeout: int = 60,
    retry_interval: float = 2.0,
    on_retry: Callable[[int, int], None] | None = None,
    provider: Optional[HealthCheckProvider] = None,
    provider_factory: Optional[Tuple[type, dict]] = None,
    process_alive_check: Optional[Callable[[], bool]] = None,
) -> HealthCheckResult:
    """
    Perform health check with retries until success or timeout.

    Args:
        config: Health check configuration.
        default_timeout: Default timeout in seconds if not specified in config.
        retry_interval: Seconds between retry attempts.
        on_retry: Optional callback called on each retry with (attempt, max_attempts).
        provider: Optional pre-initialized provider instance for "provider" type health checks.
        provider_factory: Optional tuple of (provider_class, provider_config) for lazy
            initialization. Used when the provider needs to be initialized as part of
            the health check (server may not be ready yet).
        process_alive_check: Optional callback that returns True if the process is still
            running. If provided and returns False, health check fails immediately instead
            of retrying against a dead process.

    Returns:
        HealthCheckResult with success status and metadata.
    """
    if config.type == "none":
        return HealthCheckResult(
            success=True,
            message="No health check configured (fire-and-forget)",
            attempts=0,
            elapsed_seconds=0.0,
        )

    timeout = config.timeout or default_timeout
    max_attempts = max(1, int(timeout / retry_interval))

    start_time = time.monotonic()
    target = "unknown"

    # For provider health checks with factory, we'll initialize inside the loop
    provider_instance: Optional[HealthCheckProvider] = provider

    for attempt in range(1, max_attempts + 1):
        if config.type == "tcp":
            success = _check_tcp(config.host, config.port)
            target = f"{config.host}:{config.port}"
        elif config.type == "http":
            success = _check_http(config.url)
            target = config.url
        elif config.type == "provider":
            # Use factory to create provider if not already initialized
            if provider_instance is None and provider_factory is not None:
                provider_class, provider_config = provider_factory
                try:
                    provider_instance = provider_class()
                    provider_instance.initialize(provider_config)
                    target = f"provider:{provider_class.__name__}"
                except ConnectionError:
                    # Server not ready yet — this is expected during startup
                    success = False
                    target = f"provider:{provider_class.__name__} (connecting...)"
                    if on_retry and attempt < max_attempts:
                        on_retry(attempt, max_attempts)
                    if attempt < max_attempts:
                        time.sleep(retry_interval)
                    continue
                except Exception as e:
                    success = False
                    target = f"provider:{provider_class.__name__} (init error: {e})"
                    if on_retry and attempt < max_attempts:
                        on_retry(attempt, max_attempts)
                    if attempt < max_attempts:
                        time.sleep(retry_interval)
                    continue

            if provider_instance is None:
                return HealthCheckResult(
                    success=False,
                    message="Provider health check requires provider instance or factory",
                    attempts=attempt,
                    elapsed_seconds=time.monotonic() - start_time,
                )

            try:
                success = provider_instance.health_check()
                target = f"provider:{type(provider_instance).__name__}"
            except Exception as e:
                success = False
                target = f"provider:{type(provider_instance).__name__} (error: {e})"
        else:
            return HealthCheckResult(
                success=False,
                message=f"Unknown health check type: {config.type}",
                attempts=attempt,
                elapsed_seconds=time.monotonic() - start_time,
            )

        if success:
            return HealthCheckResult(
                success=True,
                message=f"Health check passed for {target}",
                attempts=attempt,
                elapsed_seconds=time.monotonic() - start_time,
            )

        # Early exit if process died — no point retrying against a dead port
        if process_alive_check is not None and not process_alive_check():
            elapsed = time.monotonic() - start_time
            return HealthCheckResult(
                success=False,
                message=f"Process exited during health check for {target} after {attempt} attempts ({elapsed:.1f}s)",
                attempts=attempt,
                elapsed_seconds=elapsed,
            )

        if on_retry and attempt < max_attempts:
            on_retry(attempt, max_attempts)

        if attempt < max_attempts:
            time.sleep(retry_interval)

    elapsed = time.monotonic() - start_time
    return HealthCheckResult(
        success=False,
        message=f"Health check failed for {target} after {max_attempts} attempts ({elapsed:.1f}s)",
        attempts=max_attempts,
        elapsed_seconds=elapsed,
    )
