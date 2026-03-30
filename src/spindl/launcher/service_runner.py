"""
Service process management for the launcher.

Handles spawning processes (native and WSL), output capture,
and graceful shutdown. Supports provider-driven services (NANO-015).
"""

import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import psutil

from .config import ServiceConfig, TTSProviderConfig, LLMProviderConfig, VisionProviderConfig
from .health_check import check_health, HealthCheckResult, HealthCheckProvider
from .log_aggregator import LogAggregator


@dataclass
class RunningService:
    """Tracks a running service process."""

    name: str
    process: subprocess.Popen
    stdout_thread: threading.Thread
    stderr_thread: threading.Thread
    started_at: float = field(default_factory=time.monotonic)


def kill_process_tree(pid: int, timeout: float = 5.0) -> tuple[list[int], list[int]]:
    """
    Kill a process and all its children. Cross-platform.

    Args:
        pid: The root process ID to kill.
        timeout: Seconds to wait for graceful termination before force-killing.

    Returns:
        Tuple of (terminated_pids, killed_pids) for logging.
    """
    terminated = []
    force_killed = []

    try:
        parent = psutil.Process(pid)
        children = parent.children(recursive=True)

        # Collect all processes to terminate (children + parent)
        all_procs = children + [parent]

        # First pass: graceful terminate
        for proc in all_procs:
            try:
                proc.terminate()
            except psutil.NoSuchProcess:
                pass

        # Wait for graceful shutdown
        gone, alive = psutil.wait_procs(all_procs, timeout=timeout)
        terminated = [p.pid for p in gone]

        # Second pass: force kill survivors
        for proc in alive:
            try:
                proc.kill()
                force_killed.append(proc.pid)
            except psutil.NoSuchProcess:
                pass

    except psutil.NoSuchProcess:
        # Parent already dead, nothing to do
        pass

    return terminated, force_killed


class ServiceRunner:
    """
    Manages lifecycle of service processes.

    Handles spawning, output capture, health checking, and shutdown.
    Supports provider-driven services for TTS (NANO-015).
    """

    def __init__(
        self,
        logger: LogAggregator,
        default_health_timeout: int = 60,
        debug_mode: bool = False,
        tts_provider_config: Optional[TTSProviderConfig] = None,
        llm_provider_config: Optional[LLMProviderConfig] = None,
        vision_provider_config: Optional[VisionProviderConfig] = None,
        llm_context_size: int = 8192,
    ):
        """
        Initialize the service runner.

        Args:
            logger: Log aggregator for output.
            default_health_timeout: Default health check timeout.
            debug_mode: If True, append --debug to services with pass_debug=True.
            tts_provider_config: TTS provider configuration for provider-driven services.
            llm_provider_config: LLM provider configuration for provider-driven services.
            vision_provider_config: Vision (VLM) provider configuration for provider-driven services.
            llm_context_size: Context size for LLM server -c flag (NANO-096: from provider config).
        """
        self._logger = logger
        self._default_health_timeout = default_health_timeout
        self._debug_mode = debug_mode
        self._tts_provider_config = tts_provider_config
        self._llm_provider_config = llm_provider_config
        self._vision_provider_config = vision_provider_config
        self._llm_context_size = llm_context_size
        self._running: dict[str, RunningService] = {}
        self._cloud_services: set[str] = set()  # Services using cloud providers (no process)
        self._shutdown_requested = False
        self._lock = threading.Lock()

        # Cached TTS provider class and instance for health checks
        self._tts_provider_class: Optional[type] = None
        self._tts_provider_instance: Optional[HealthCheckProvider] = None

        # Cached LLM provider class for health checks (NANO-019)
        self._llm_provider_class: Optional[type] = None

        # Cached VLM provider class for health checks (NANO-023)
        self._vlm_provider_class: Optional[type] = None

    def _get_tts_provider_class(self) -> type:
        """
        Get the TTS provider class from registry.

        Returns:
            TTSProvider subclass

        Raises:
            RuntimeError: If TTS provider config not available or provider not found
        """
        if self._tts_provider_class is not None:
            return self._tts_provider_class

        if self._tts_provider_config is None:
            raise RuntimeError("TTS provider config not available")

        # Import registry here to avoid circular imports
        from ..tts import TTSProviderRegistry, ProviderNotFoundError

        registry = TTSProviderRegistry(
            plugin_paths=self._tts_provider_config.plugin_paths
        )

        try:
            self._tts_provider_class = registry.get_provider_class(
                self._tts_provider_config.provider
            )
            return self._tts_provider_class
        except ProviderNotFoundError as e:
            raise RuntimeError(str(e))

    def _get_tts_provider_for_health_check(self) -> tuple[type, dict]:
        """
        Get TTS provider class and config for health checks.

        For provider-based health checks, we need the provider class and config
        but NOT an initialized instance — initialization would fail if the server
        isn't ready yet, which is exactly what the health check is testing.

        Returns:
            Tuple of (provider_class, provider_config)

        Raises:
            RuntimeError: If TTS provider config not available
        """
        provider_class = self._get_tts_provider_class()
        return provider_class, self._tts_provider_config.provider_config

    def _get_tts_server_command(self) -> str:
        """
        Get the TTS server command from provider.

        Returns:
            Shell command string to start TTS server

        Raises:
            RuntimeError: If provider doesn't provide a server command
        """
        provider_class = self._get_tts_provider_class()

        # Validate config first
        errors = provider_class.validate_config(
            self._tts_provider_config.provider_config
        )
        if errors:
            raise RuntimeError(
                f"TTS provider config validation failed: {'; '.join(errors)}"
            )

        # Get server command
        command = provider_class.get_server_command(
            self._tts_provider_config.provider_config
        )
        if command is None:
            raise RuntimeError(
                f"TTS provider '{self._tts_provider_config.provider}' "
                "doesn't require a server (in-process provider)"
            )

        return command

    def _get_llm_provider_class(self) -> type:
        """
        Get the LLM provider class from registry.

        Returns:
            LLMProvider subclass

        Raises:
            RuntimeError: If LLM provider config not available or provider not found
        """
        if self._llm_provider_class is not None:
            return self._llm_provider_class

        if self._llm_provider_config is None:
            raise RuntimeError("LLM provider config not available")

        # Import registry here to avoid circular imports
        from ..llm import LLMProviderRegistry, ProviderNotFoundError

        registry = LLMProviderRegistry(
            plugin_paths=self._llm_provider_config.plugin_paths
        )

        try:
            self._llm_provider_class = registry.get_provider_class(
                self._llm_provider_config.provider
            )
            return self._llm_provider_class
        except ProviderNotFoundError as e:
            raise RuntimeError(str(e))

    def _get_llm_provider_for_health_check(self) -> tuple[type, dict]:
        """
        Get LLM provider class and config for health checks.

        Returns:
            Tuple of (provider_class, provider_config)

        Raises:
            RuntimeError: If LLM provider config not available
        """
        provider_class = self._get_llm_provider_class()
        return provider_class, self._llm_provider_config.provider_config

    def _get_llm_server_command(self) -> Optional[str]:
        """
        Get the LLM server command from provider.

        Returns:
            Shell command string to start LLM server, or None if cloud provider

        Raises:
            RuntimeError: If config validation fails
        """
        provider_class = self._get_llm_provider_class()

        # Validate config first
        errors = provider_class.validate_config(
            self._llm_provider_config.provider_config
        )
        if errors:
            raise RuntimeError(
                f"LLM provider config validation failed: {'; '.join(errors)}"
            )

        # Get server command (may be None for cloud providers)
        return provider_class.get_server_command(
            self._llm_provider_config.provider_config
        )

    def _get_vlm_provider_class(self) -> type:
        """
        Get the VLM provider class from registry.

        Returns:
            VLMProvider subclass

        Raises:
            RuntimeError: If VLM provider config not available or provider not found
        """
        if self._vlm_provider_class is not None:
            return self._vlm_provider_class

        if self._vision_provider_config is None:
            raise RuntimeError("Vision provider config not available")

        # Import registry here to avoid circular imports
        from ..vision import VLMProviderRegistry

        registry = VLMProviderRegistry(
            plugin_paths=self._vision_provider_config.plugin_paths
        )

        try:
            self._vlm_provider_class = registry.get_provider_class(
                self._vision_provider_config.provider
            )
            return self._vlm_provider_class
        except ValueError as e:
            raise RuntimeError(str(e))

    def _get_vlm_provider_for_health_check(self) -> tuple[type, dict]:
        """
        Get VLM provider class and config for health checks.

        Returns:
            Tuple of (provider_class, provider_config)

        Raises:
            RuntimeError: If VLM provider config not available
        """
        provider_class = self._get_vlm_provider_class()
        return provider_class, self._vision_provider_config.provider_config

    def _get_vlm_server_command(self) -> Optional[str]:
        """
        Get the VLM server command from provider.

        Returns:
            Shell command string to start VLM server, or None if cloud provider

        Raises:
            RuntimeError: If config validation fails
        """
        provider_class = self._get_vlm_provider_class()

        # Cloud providers don't need a local server
        if provider_class.is_cloud_provider():
            return None

        # Validate config first
        errors = provider_class.validate_config(
            self._vision_provider_config.provider_config
        )
        if errors:
            raise RuntimeError(
                f"VLM provider config validation failed: {'; '.join(errors)}"
            )

        # Get server command
        return provider_class.get_server_command(
            self._vision_provider_config.provider_config
        )

    def _build_command(self, config: ServiceConfig) -> Optional[list[str] | str]:
        """
        Build the command for subprocess.

        Returns:
            Command string (native, run with shell=True) or list (WSL),
            or None if service should be skipped (e.g., cloud provider).
        """
        command = config.command

        # For TTS service without explicit command, derive from provider
        if command is None and config.name == "tts":
            command = self._get_tts_server_command()
            self._logger.log_launcher(
                f"Derived TTS server command from provider: {command[:80]}...",
                level="debug",
            )

        # For LLM service, check if provider needs a server
        if config.name == "llm" and self._llm_provider_config is not None:
            provider_class = self._get_llm_provider_class()

            # Cloud providers don't need a local server - skip entirely
            if provider_class.is_cloud_provider():
                self._logger.log_launcher(
                    f"LLM provider '{self._llm_provider_config.provider}' is cloud-based, "
                    "no local server needed",
                    level="info",
                )
                return None

            # Local provider - use config command (preferred) or provider command
            provider_command = self._get_llm_server_command()
            if command is None and provider_command is not None:
                command = provider_command
                self._logger.log_launcher(
                    f"Derived LLM server command from provider: {command[:80]}...",
                    level="debug",
                )

            # Inject -c flag from provider config context_size (NANO-096)
            # Replace existing -c value or append if not present
            import re
            if re.search(r'-c\s+\d+', command):
                command = re.sub(r'-c\s+\d+', f'-c {self._llm_context_size}', command)
            else:
                command = f"{command} -c {self._llm_context_size}"
            self._logger.log_launcher(
                f"LLM context size set to {self._llm_context_size} tokens (from provider config)",
                level="info",
            )

            # Inject -np 2 for unified vision mode (NANO-087)
            # When VLM routes through LLM (provider == "llm"), the same llama-server
            # handles both chat and vision describe requests. Two slots prevent KV cache
            # thrashing — chat pins to slot 0, describe pins to slot 1.
            if (
                self._vision_provider_config is not None
                and self._vision_provider_config.provider == "llm"
                and not re.search(r'(-np|--parallel)\s+\d+', command)
            ):
                command = f"{command} -np 2"
                self._logger.log_launcher(
                    "Unified vision mode: injected -np 2 (chat slot 0, describe slot 1)",
                    level="info",
                )

        # For VLM service, check if provider needs a server (NANO-023)
        if config.name == "vlm" and self._vision_provider_config is not None:
            provider_class = self._get_vlm_provider_class()

            # Cloud providers don't need a local server - skip entirely
            if provider_class.is_cloud_provider():
                self._logger.log_launcher(
                    f"VLM provider '{self._vision_provider_config.provider}' is cloud-based, "
                    "no local server needed",
                    level="info",
                )
                return None

            # Local provider - use config command (preferred) or provider command
            provider_command = self._get_vlm_server_command()
            if command is None and provider_command is not None:
                command = provider_command
                self._logger.log_launcher(
                    f"Derived VLM server command from provider: {command[:80]}...",
                    level="debug",
                )

        if command is None:
            raise RuntimeError(
                f"Service '{config.name}' has no command and is not a provider-driven service"
            )

        # Append --debug if requested
        if self._debug_mode and config.pass_debug:
            command = f"{command} --debug"

        if config.platform == "wsl":
            # WSL: wrap command in bash -ic (interactive, so ~/.bashrc loads conda init)
            return [
                "wsl", "-d", config.wsl_distro,
                "bash", "-ic", command
            ]
        else:
            # Native: return raw command string.
            # Popen receives this with shell=True (see start_service).
            # On Windows, shell=True invokes cmd /S /c "command" which
            # correctly preserves inner quotes — unlike the manual
            # cmd /c wrapping we used before, which broke on paths
            # with spaces (e.g. "VS CODE Projects").
            return command

    def _output_reader(self, service_name: str, stream, is_stderr: bool = False):
        """Thread function to read output from a process stream."""
        try:
            for line in iter(stream.readline, ''):
                if not line:
                    break
                line = line.rstrip('\r\n')
                if line:
                    self._logger.log(
                        service_name,
                        line,
                        level="info",
                        is_stderr=is_stderr,
                    )
        except Exception as e:
            if not self._shutdown_requested:
                self._logger.log(
                    service_name,
                    f"Output reader error: {e}",
                    level="error",
                    is_stderr=True,
                )

    def start_service(self, config: ServiceConfig) -> bool:
        """
        Start a service and perform health check.

        Args:
            config: Service configuration.

        Returns:
            True if service started and passed health check.
        """
        if not config.enabled:
            self._logger.log_launcher(f"Skipping disabled service: {config.name}")
            return True

        self._logger.log_launcher(f"Starting service: {config.name} ({config.platform})")

        cmd = self._build_command(config)

        # Cloud provider - no local server needed, consider it "started"
        if cmd is None:
            self._logger.log_launcher(
                f"Service {config.name} uses cloud provider, no local process needed"
            )
            with self._lock:
                self._cloud_services.add(config.name)
            return True

        try:
            # WSL commands are lists (no shell needed); native commands
            # are strings that need shell=True so cmd.exe /S /c handles
            # quoted paths with spaces correctly.
            use_shell = isinstance(cmd, str)

            # On Windows, wrap with 'start' to force a visible console window
            # per service with full unfiltered logs.
            if sys.platform == "win32" and use_shell:
                cmd = f'start "{config.name}" /wait {cmd}'

            process = subprocess.Popen(
                cmd,
                shell=use_shell,
            )

            self._logger.log_launcher(
                f"Process spawned for {config.name} (PID: {process.pid})",
                level="debug",
            )

            # Start output reader threads (only if pipes are available)
            stdout_thread = None
            stderr_thread = None
            if process.stdout:
                stdout_thread = threading.Thread(
                    target=self._output_reader,
                    args=(config.name, process.stdout, False),
                    daemon=True,
                )
                stdout_thread.start()
            if process.stderr:
                stderr_thread = threading.Thread(
                    target=self._output_reader,
                    args=(config.name, process.stderr, True),
                    daemon=True,
                )
                stderr_thread.start()

            # Track the running service
            with self._lock:
                self._running[config.name] = RunningService(
                    name=config.name,
                    process=process,
                    stdout_thread=stdout_thread,
                    stderr_thread=stderr_thread,
                )

            # Perform health check
            self._logger.log_launcher(f"Performing health check for {config.name}...")

            def on_retry(attempt: int, max_attempts: int):
                self._logger.log_launcher(
                    f"Health check retry {attempt}/{max_attempts} for {config.name}...",
                    level="info",
                )

            # Early-exit callback: stop retrying if the process died
            def process_alive() -> bool:
                return process.poll() is None

            # For provider-type health checks, get the provider factory (class + config)
            # We use a factory instead of a pre-initialized instance because
            # initialization may fail if the server isn't ready yet — that's
            # exactly what the health check is testing.
            provider_factory = None
            if config.health_check.type == "provider":
                try:
                    if config.name == "tts":
                        provider_factory = self._get_tts_provider_for_health_check()
                    elif config.name == "llm":
                        provider_factory = self._get_llm_provider_for_health_check()
                    elif config.name == "vlm":
                        provider_factory = self._get_vlm_provider_for_health_check()
                except Exception as e:
                    self._logger.log_launcher(
                        f"Failed to get provider for health check: {e}",
                        level="error",
                    )
                    return False

            result = check_health(
                config.health_check,
                default_timeout=self._default_health_timeout,
                retry_interval=2.0,
                on_retry=on_retry,
                provider_factory=provider_factory,
                process_alive_check=process_alive,
            )

            if result.success:
                self._logger.log_launcher(
                    f"Service {config.name} is healthy ({result.attempts} attempts, {result.elapsed_seconds:.1f}s)"
                )
                return True
            else:
                self._logger.log_launcher(
                    f"Health check FAILED for {config.name}: {result.message}",
                    level="error",
                )
                # Check if process died during health check
                if process.poll() is not None:
                    self._logger.log_launcher(
                        f"Process for {config.name} exited with code {process.returncode}",
                        level="error",
                    )
                return False

        except FileNotFoundError as e:
            self._logger.log_launcher(
                f"Failed to start {config.name}: command not found - {e}",
                level="error",
            )
            return False
        except Exception as e:
            self._logger.log_launcher(
                f"Failed to start {config.name}: {e}",
                level="error",
            )
            return False

    def stop_service(self, name: str, timeout: float = 5.0) -> bool:
        """
        Stop a single running service by name.

        Args:
            name: Service name (e.g. "llm", "tts", "stt").
            timeout: Max seconds to wait for graceful termination.

        Returns:
            True if service was stopped (or was already stopped).
        """
        with self._lock:
            svc = self._running.get(name)
            if svc is None:
                # Not tracked — might be cloud-only or never started
                self._cloud_services.discard(name)
                return True

        # Check if already exited
        if svc.process.poll() is not None:
            self._logger.log_launcher(
                f"{name} already exited (code: {svc.process.returncode})"
            )
            with self._lock:
                self._running.pop(name, None)
                self._cloud_services.discard(name)
            return True

        self._logger.log_launcher(f"Stopping {name} (PID: {svc.process.pid})...")

        try:
            terminated, force_killed = kill_process_tree(
                svc.process.pid, timeout=timeout
            )
            if force_killed:
                self._logger.log_launcher(
                    f"{name}: {len(terminated)} terminated, {len(force_killed)} force-killed",
                    level="warning",
                )
            else:
                self._logger.log_launcher(
                    f"{name}: {len(terminated)} processes terminated gracefully"
                )
        except Exception as e:
            self._logger.log_launcher(
                f"Error stopping {name}: {e}", level="error"
            )
            return False

        with self._lock:
            self._running.pop(name, None)
            self._cloud_services.discard(name)

        return True

    def shutdown_all(self, timeout_per_service: float = 5.0) -> None:
        """
        Gracefully shutdown all running services.

        Services are terminated in reverse startup order.

        Args:
            timeout_per_service: Max seconds to wait for each service to terminate.
        """
        self._shutdown_requested = True

        with self._lock:
            service_names = list(self._running.keys())

        # Reverse order for shutdown
        service_names.reverse()

        self._logger.log_launcher(f"Shutting down {len(service_names)} services...")

        for name in service_names:
            self.stop_service(name, timeout=timeout_per_service)

        self._logger.log_launcher("All services shut down")

    def get_running_services(self) -> list[str]:
        """Return names of currently running services."""
        with self._lock:
            return list(self._running.keys())

    def is_service_running(self, name: str) -> bool:
        """Check if a service is still running."""
        with self._lock:
            # Cloud services are always "running" (no process to check)
            if name in self._cloud_services:
                return True
            if name not in self._running:
                return False
            return self._running[name].process.poll() is None
