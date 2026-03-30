"""
Log aggregation for launcher services.

Captures output from multiple services, prefixes with service names,
and writes to both console and log file.

Supports per-service log levels and pattern-based suppression (NANO-017).
"""

import re
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import TextIO, Literal, Pattern

from spindl.utils.paths import resolve_relative_path


class LogAggregator:
    """
    Aggregates log output from multiple services.

    Thread-safe. Writes to console with prefixes and optionally to a log file
    with timestamps.
    """

    def __init__(
        self,
        log_file: str | None = None,
        log_level: Literal["debug", "info", "warning", "error"] = "info",
        service_levels: dict[str, str] | None = None,
        suppress_patterns: list[str] | None = None,
    ):
        """
        Initialize the log aggregator.

        Args:
            log_file: Optional path to log file. If None, only console output.
            log_level: Minimum log level to display (global default).
            service_levels: Per-service log level overrides (e.g., {"stt": "error"}).
            suppress_patterns: Regex patterns to suppress (matched messages are hidden).
        """
        self._lock = threading.Lock()
        self._log_file: TextIO | None = None
        self._log_level = log_level
        self._service_levels = service_levels or {}
        self._level_priority = {
            "debug": 0,
            "info": 1,
            "warning": 2,
            "error": 3,
        }

        # Compile suppress patterns once for performance
        self._suppress_patterns: list[Pattern] = []
        for pattern in (suppress_patterns or []):
            try:
                self._suppress_patterns.append(re.compile(pattern))
            except re.error as e:
                # Log invalid patterns but don't crash
                print(f"[LAUNCHER] Warning: Invalid suppress pattern '{pattern}': {e}")

        if log_file:
            log_path = Path(resolve_relative_path(log_file))
            log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_file = open(log_path, "a", encoding="utf-8")

    def _should_log(self, level: str) -> bool:
        """Check if message at given level should be logged (global level)."""
        return self._level_priority.get(level, 1) >= self._level_priority.get(self._log_level, 1)

    def _should_log_service(self, service: str, message: str, level: str) -> bool:
        """
        Check if a service message should be logged.

        Applies per-service level override and pattern suppression.

        Args:
            service: Service name (e.g., "stt", "llm").
            message: The log message content.
            level: Log level of the message.

        Returns:
            True if message should be displayed, False to suppress.
        """
        # Get effective log level for this service (service-specific or global default)
        effective_level = self._service_levels.get(service.lower(), self._log_level)

        # Check level threshold
        if self._level_priority.get(level, 1) < self._level_priority.get(effective_level, 1):
            return False

        # Check suppress patterns
        for pattern in self._suppress_patterns:
            if pattern.search(message):
                return False

        return True

    def log(
        self,
        service: str,
        message: str,
        level: Literal["debug", "info", "warning", "error"] = "info",
        is_stderr: bool = False,
    ) -> None:
        """
        Log a message from a service.

        Args:
            service: Service name (used in prefix).
            message: Log message.
            level: Log level.
            is_stderr: If True, message came from stderr (adds [ERR] marker).
        """
        if not self._should_log_service(service, message, level):
            return

        prefix = f"[NANO-{service.upper()}]"
        if is_stderr:
            prefix = f"{prefix} [ERR]"

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with self._lock:
            # Console output (no timestamp for cleaner display)
            console_line = f"{prefix} {message}"
            print(console_line, file=sys.stderr if is_stderr else sys.stdout, flush=True)

            # File output (with timestamp)
            if self._log_file:
                file_line = f"{timestamp} {prefix} {message}\n"
                self._log_file.write(file_line)
                self._log_file.flush()

    def log_launcher(
        self,
        message: str,
        level: Literal["debug", "info", "warning", "error"] = "info",
    ) -> None:
        """
        Log a message from the launcher itself (not a service).

        Args:
            message: Log message.
            level: Log level.
        """
        if not self._should_log(level):
            return

        prefix = "[LAUNCHER]"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with self._lock:
            print(f"{prefix} {message}", flush=True)

            if self._log_file:
                self._log_file.write(f"{timestamp} {prefix} {message}\n")
                self._log_file.flush()

    def close(self) -> None:
        """Close the log file if open."""
        with self._lock:
            if self._log_file:
                self._log_file.close()
                self._log_file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
