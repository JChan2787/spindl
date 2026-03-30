"""
E2E Test Harness - Lifecycle management for browser-driven tests.

NANO-029: Manages the full test lifecycle:
1. Start Next.js frontend as subprocess (SPINDL_CONFIG points to fixture config)
2. Wait for frontend health check
3. Start Python backend (GUI server) as subprocess with --config and SPINDL_TEST_MODE=1
4. Wait for backend health check
5. Launch Playwright browser
6. Connect Socket.IO test client
7. Clean up all processes on stop (including atexit handler for crashes)
"""

import asyncio
import atexit
import os
import signal
import socket
import subprocess
import sys
import threading
from pathlib import Path
from typing import Optional

import socketio
from playwright.async_api import async_playwright, Browser, Page


# Track active harness PIDs for cleanup on exit
_active_server_pids: set[int] = set()


def _cleanup_on_exit():
    """Kill any orphaned server processes on interpreter exit."""
    # Kill orphaned processes
    for pid in list(_active_server_pids):
        try:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True,
                    timeout=5,
                )
            else:
                os.kill(pid, signal.SIGKILL)
            print(f"[E2EHarness] Cleanup: killed orphaned PID {pid}")
        except Exception:
            pass
    _active_server_pids.clear()

    # Clean up Next.js dev lock file (safety net for orphan node processes)
    try:
        nextjs_lock = Path(__file__).parent.parent.parent / "gui" / ".next" / "dev" / "lock"
        if nextjs_lock.exists():
            nextjs_lock.unlink()
            print("[E2EHarness] Cleanup: removed Next.js dev lock file")
    except Exception:
        pass


atexit.register(_cleanup_on_exit)


def _output_reader(name: str, stream, is_stderr: bool = False) -> None:
    """
    Thread function to read and print output from a subprocess stream.

    Prevents pipe buffer deadlocks by continuously draining the output.
    This is critical on Windows where pipe buffers are small (~4KB).

    Args:
        name: Process name for log prefix (e.g., "frontend", "backend")
        stream: The stdout or stderr stream to read from
        is_stderr: Whether this is stderr (for log formatting)
    """
    prefix = f"[{name}]"
    if is_stderr:
        prefix = f"[{name}:err]"

    try:
        for line in iter(stream.readline, ''):
            if not line:
                break
            line = line.rstrip('\r\n')
            if line:
                print(f"{prefix} {line}")
    except Exception as e:
        print(f"{prefix} Output reader error: {e}")


def _find_free_port() -> int:
    """Find a free port by binding to port 0 and letting the OS assign one."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


class E2EHarness:
    """
    Manages E2E test lifecycle: server, browser, and Socket.IO client.

    Usage:
        async with E2EHarness(config_path="fixtures/config/spindl_e2e_local.yaml") as harness:
            await harness.send_message("Hello")
            event = await harness.wait_for_event("response", timeout=10.0)
            assert event["text"]  # Non-empty response
    """

    def __init__(
        self,
        config_path: str,
        host: str = "127.0.0.1",
        backend_port: int = 0,
        frontend_port: int = 0,
        headless: bool = True,
        auto_start_services: bool = True,
    ):
        """
        Initialize the E2E harness.

        Args:
            config_path: Path to spindl config YAML (relative to tests_e2e/).
            host: Server host.
            backend_port: Python Socket.IO backend port. Use 0 (default) for auto-assign.
            frontend_port: Next.js frontend port. Use 0 (default) for auto-assign.
            headless: Whether to run browser in headless mode.
            auto_start_services: If True, automatically emit start_services and wait for
                orchestrator_ready. Set to False for tests that want to click the Launch button.
        """
        self._config_path = config_path
        self._host = host
        # Auto-assign ports if 0 to prevent test collisions
        self._backend_port = backend_port if backend_port != 0 else _find_free_port()
        self._frontend_port = frontend_port  # Will be assigned in _start_frontend if 0
        self._headless = headless
        self._auto_start_services = auto_start_services

        # Subprocess and browser state
        self._server_process: Optional[subprocess.Popen] = None
        self._frontend_process: Optional[subprocess.Popen] = None
        self._playwright = None
        self._browser: Optional[Browser] = None
        self._page: Optional[Page] = None

        # Output reader threads (prevent pipe buffer deadlocks)
        self._output_threads: list[threading.Thread] = []

        # Socket.IO client for event verification
        self._sio: Optional[socketio.AsyncClient] = None
        self._event_queue: asyncio.Queue = asyncio.Queue()
        self._received_events: dict[str, list] = {}

        # Resolved config path (set during start())
        self._config_full_path: Optional[Path] = None

    @property
    def page(self) -> Page:
        """Get the Playwright page for browser interaction."""
        if not self._page:
            raise RuntimeError("Harness not started - call start() first")
        return self._page

    @property
    def backend_url(self) -> str:
        """Get the Python Socket.IO backend URL."""
        return f"http://{self._host}:{self._backend_port}"

    @property
    def frontend_url(self) -> str:
        """Get the Next.js frontend URL."""
        return f"http://{self._host}:{self._frontend_port}"

    async def start(self) -> None:
        """
        Start the test environment.

        1. Sets SPINDL_CONFIG so frontend reads fixture config directly
        2. Launches GUI server subprocess with --config pointing to fixture
        3. Waits for server to be healthy
        4. Starts Playwright browser
        5. Connects Socket.IO client for event interception
        """
        # Resolve config path relative to tests_e2e/
        tests_e2e_dir = Path(__file__).parent.parent
        self._config_full_path = tests_e2e_dir / self._config_path

        if not self._config_full_path.exists():
            raise FileNotFoundError(f"Config not found: {self._config_full_path}")

        # NANO-081: No config swap needed — frontend reads via SPINDL_CONFIG,
        # backend reads via --config flag. Real config/spindl.yaml is never touched.

        # Start frontend subprocess (Next.js)
        await self._start_frontend()

        # Wait for frontend to be ready
        await self._wait_for_frontend()

        # Start server subprocess
        await self._start_server(self._config_full_path)

        # Wait for server health
        await self._wait_for_health()

        # Start browser
        await self._start_browser()

        # Connect Socket.IO client
        await self._connect_socket()

        # Trigger service startup and wait for orchestrator (GUI-first mode)
        # Skip if auto_start_services=False (for tests that click the button manually)
        if self._auto_start_services:
            await self._trigger_service_startup()

    async def _trigger_service_startup(self, timeout: float = 60.0) -> None:
        """
        Trigger service startup and wait for orchestrator to be ready.

        NANO-031: In GUI-first mode, the orchestrator doesn't exist until
        start_services is emitted. This method triggers that flow and waits
        for the orchestrator_ready event before tests can proceed.

        Args:
            timeout: Maximum time to wait for orchestrator_ready.
        """
        if not self._sio:
            raise RuntimeError("Socket.IO not connected")

        print("[E2EHarness] Triggering service startup...")
        await self._sio.emit("start_services", {})

        # Wait for orchestrator_ready event
        try:
            event = await self.wait_for_event("orchestrator_ready", timeout=timeout)
            print(f"[E2EHarness] Orchestrator ready (persona: {event.get('persona', 'unknown')})")
        except TimeoutError:
            raise TimeoutError(f"Orchestrator not ready after {timeout}s - check service configuration")

    async def _start_frontend(self) -> None:
        """Start the Next.js frontend as a subprocess."""
        project_root = Path(__file__).parent.parent.parent
        gui_dir = project_root / "gui"

        if not gui_dir.exists():
            raise FileNotFoundError(f"GUI directory not found: {gui_dir}")

        # Use dynamic port for frontend too
        self._frontend_port = _find_free_port()

        env = os.environ.copy()
        env["PORT"] = str(self._frontend_port)
        # Tell the frontend where the backend Socket.IO server is
        env["NEXT_PUBLIC_SOCKET_URL"] = self.backend_url
        # Enable test mode so write-config respects enabled:false for services
        env["SPINDL_TEST_MODE"] = "1"
        # NANO-081: Point frontend at fixture config instead of swapping real config
        env["SPINDL_CONFIG"] = str(self._config_full_path)

        # Start Next.js dev server
        self._frontend_process = subprocess.Popen(
            ["npm", "run", "dev"],
            cwd=str(gui_dir),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            shell=True,  # Required for npm on Windows
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )

        # Start output reader threads to prevent pipe buffer deadlocks
        stdout_thread = threading.Thread(
            target=_output_reader,
            args=("frontend", self._frontend_process.stdout, False),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_output_reader,
            args=("frontend", self._frontend_process.stderr, True),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        self._output_threads.extend([stdout_thread, stderr_thread])

        # Track PID for atexit cleanup
        _active_server_pids.add(self._frontend_process.pid)
        print(f"[E2EHarness] Frontend started (PID: {self._frontend_process.pid}, port: {self._frontend_port})")

    async def _wait_for_frontend(self, timeout: float = 60.0, poll_interval: float = 1.0) -> None:
        """Wait for the Next.js frontend to be ready."""
        import aiohttp

        deadline = asyncio.get_event_loop().time() + timeout

        while asyncio.get_event_loop().time() < deadline:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(self.frontend_url, timeout=aiohttp.ClientTimeout(total=2.0)) as resp:
                        if resp.status == 200:
                            print("[E2EHarness] Frontend healthy (HTTP 200)")
                            return
            except Exception:
                pass

            # Check if process died
            if self._frontend_process and self._frontend_process.poll() is not None:
                # Output already captured by reader threads, just report exit code
                raise RuntimeError(
                    f"Frontend process died with exit code {self._frontend_process.returncode}. "
                    "Check console output above for details."
                )

            await asyncio.sleep(poll_interval)

        raise TimeoutError(f"Frontend not ready after {timeout}s")

    async def _start_server(self, config_path: Path) -> None:
        """Start the GUI server as a subprocess."""
        # Find the run_gui_standalone.py script
        project_root = Path(__file__).parent.parent.parent
        script_path = project_root / "scripts" / "run_gui_standalone.py"

        if not script_path.exists():
            raise FileNotFoundError(f"Server script not found: {script_path}")

        # Environment with test mode enabled
        env = os.environ.copy()
        env["SPINDL_TEST_MODE"] = "1"

        # Start server process
        self._server_process = subprocess.Popen(
            [
                sys.executable,
                str(script_path),
                "--config", str(config_path),
                "--host", self._host,
                "--port", str(self._backend_port),
            ],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            # Use process group for clean termination on Windows
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        )

        # Start output reader threads to prevent pipe buffer deadlocks
        stdout_thread = threading.Thread(
            target=_output_reader,
            args=("backend", self._server_process.stdout, False),
            daemon=True,
        )
        stderr_thread = threading.Thread(
            target=_output_reader,
            args=("backend", self._server_process.stderr, True),
            daemon=True,
        )
        stdout_thread.start()
        stderr_thread.start()
        self._output_threads.extend([stdout_thread, stderr_thread])

        # Track PID for atexit cleanup
        _active_server_pids.add(self._server_process.pid)
        print(f"[E2EHarness] Server started (PID: {self._server_process.pid})")

    async def _wait_for_health(self, timeout: float = 30.0, poll_interval: float = 0.5) -> None:
        """Wait for the backend server to be healthy via Socket.IO connection test."""
        deadline = asyncio.get_event_loop().time() + timeout

        while asyncio.get_event_loop().time() < deadline:
            try:
                # Try to connect via Socket.IO - if it succeeds, server is healthy
                test_client = socketio.AsyncClient()
                await asyncio.wait_for(
                    test_client.connect(self.backend_url),
                    timeout=2.0
                )
                await test_client.disconnect()
                print("[E2EHarness] Server healthy (Socket.IO connection successful)")
                return
            except Exception:
                pass

            # Check if process died
            if self._server_process and self._server_process.poll() is not None:
                # Output already captured by reader threads, just report exit code
                raise RuntimeError(
                    f"Server process died with exit code {self._server_process.returncode}. "
                    "Check console output above for details."
                )

            await asyncio.sleep(poll_interval)

        raise TimeoutError(f"Server not healthy after {timeout}s")

    async def _start_browser(self) -> None:
        """Start Playwright browser and navigate to GUI."""
        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self._headless)
        self._page = await self._browser.new_page()

        # Navigate to frontend (Next.js serves the UI)
        await self._page.goto(self.frontend_url)
        print("[E2EHarness] Browser connected to frontend")

    async def _connect_socket(self) -> None:
        """Connect Socket.IO client for event interception."""
        self._sio = socketio.AsyncClient()

        # Register catch-all handler for event recording
        @self._sio.on("*")
        async def catch_all(event: str, data):
            if event not in self._received_events:
                self._received_events[event] = []
            self._received_events[event].append(data)
            await self._event_queue.put((event, data))

        # Register specific event handlers for common events
        for event_name in [
            "state_changed",
            "transcription",
            "response",
            "tts_status",
            "tool_invoked",
            "tool_result",
            "orchestrator_ready",
            "orchestrator_error",
            "shutdown_complete",
            "health_status",
            "pipeline_error",
            "launch_progress",
            "launch_error",
        ]:
            self._register_event_handler(event_name)

        await self._sio.connect(self.backend_url)
        print("[E2EHarness] Socket.IO connected to backend")

    def _register_event_handler(self, event_name: str) -> None:
        """Register a handler for a specific event."""
        @self._sio.on(event_name)
        async def handler(data):
            if event_name not in self._received_events:
                self._received_events[event_name] = []
            self._received_events[event_name].append(data)
            await self._event_queue.put((event_name, data))

    async def inject_transcription(self, text: str) -> dict:
        """
        Inject text as if STT transcribed it.

        DEPRECATED: Use send_message() instead. This method requires SPINDL_TEST_MODE.

        Args:
            text: Transcription text to inject.

        Returns:
            Response from server (success/error status).
        """
        if not self._sio:
            raise RuntimeError("Socket.IO not connected")

        return await self._sio.call("test_inject_transcription", {"text": text})

    async def send_message(self, text: str, skip_tts: bool = False) -> dict:
        """
        Send a text message through the pipeline.

        NANO-031: User-facing text input method. Unlike inject_transcription(),
        this works without SPINDL_TEST_MODE and supports skip_tts for text-only responses.

        Args:
            text: Message text to send.
            skip_tts: If True, skip TTS synthesis (text-only response).

        Returns:
            Response from server (success/error status).
        """
        if not self._sio:
            raise RuntimeError("Socket.IO not connected")

        return await self._sio.call("send_message", {
            "text": text,
            "skip_tts": skip_tts,
        })

    async def wait_for_event(
        self,
        event_name: str,
        timeout: float = 5.0,
        predicate: Optional[callable] = None,
    ) -> dict:
        """
        Wait for a specific Socket.IO event.

        Args:
            event_name: Name of the event to wait for.
            timeout: Maximum time to wait in seconds.
            predicate: Optional function to filter events (returns True to accept).

        Returns:
            Event data dict.

        Raises:
            TimeoutError: If event not received within timeout.
        """
        deadline = asyncio.get_event_loop().time() + timeout

        while asyncio.get_event_loop().time() < deadline:
            remaining = deadline - asyncio.get_event_loop().time()
            try:
                event, data = await asyncio.wait_for(
                    self._event_queue.get(),
                    timeout=remaining,
                )
                if event == event_name:
                    if predicate is None or predicate(data):
                        return data
            except asyncio.TimeoutError:
                break

        raise TimeoutError(f"Event '{event_name}' not received within {timeout}s")

    def get_events(self, event_name: str) -> list:
        """Get all received events of a specific type."""
        return self._received_events.get(event_name, [])

    def get_all_events(self) -> dict:
        """Get all received events (for debugging)."""
        return dict(self._received_events)

    def dump_events(self) -> None:
        """Print all received events (for debugging test failures)."""
        print("\n[E2EHarness] === Received Events ===")
        for event_name, events in self._received_events.items():
            print(f"  {event_name}: {len(events)} event(s)")
            for i, data in enumerate(events):
                print(f"    [{i}] {data}")
        print("[E2EHarness] === End Events ===\n")

    def clear_events(self) -> None:
        """Clear all recorded events."""
        self._received_events.clear()
        # Drain the queue
        while not self._event_queue.empty():
            try:
                self._event_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def stop(self) -> None:
        """Stop the test environment and clean up resources."""
        # Disconnect Socket.IO first (gracefully)
        if self._sio:
            try:
                if self._sio.connected:
                    await self._sio.disconnect()
                    print("[E2EHarness] Socket.IO disconnected")
            except Exception as e:
                print(f"[E2EHarness] Socket.IO disconnect error (ignored): {e}")

        # Close browser
        if self._browser:
            try:
                await self._browser.close()
            except Exception as e:
                print(f"[E2EHarness] Browser close error (ignored): {e}")
        if self._playwright:
            try:
                await self._playwright.stop()
            except Exception as e:
                print(f"[E2EHarness] Playwright stop error (ignored): {e}")
        print("[E2EHarness] Browser closed")

        # Terminate server process - be aggressive
        if self._server_process:
            pid = self._server_process.pid
            try:
                if sys.platform == "win32":
                    # Windows: kill entire process tree
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(pid)],
                        capture_output=True,
                        timeout=5,
                    )
                else:
                    # Unix: send SIGTERM then SIGKILL
                    self._server_process.terminate()
                    try:
                        self._server_process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        self._server_process.kill()
                        self._server_process.wait(timeout=2)
            except Exception as e:
                print(f"[E2EHarness] Server termination error (ignored): {e}")

            # Remove from atexit tracking
            _active_server_pids.discard(pid)
            print(f"[E2EHarness] Server stopped (PID: {pid})")

        # Terminate frontend process
        if self._frontend_process:
            pid = self._frontend_process.pid
            try:
                if sys.platform == "win32":
                    # Windows: kill entire process tree
                    subprocess.run(
                        ["taskkill", "/F", "/T", "/PID", str(pid)],
                        capture_output=True,
                        timeout=5,
                    )
                else:
                    # Unix: send SIGTERM then SIGKILL
                    self._frontend_process.terminate()
                    try:
                        self._frontend_process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        self._frontend_process.kill()
                        self._frontend_process.wait(timeout=2)
            except Exception as e:
                print(f"[E2EHarness] Frontend termination error (ignored): {e}")

            # Remove from atexit tracking
            _active_server_pids.discard(pid)
            print(f"[E2EHarness] Frontend stopped (PID: {pid})")

        # Clean up Next.js dev lock file (safety net for orphan processes)
        project_root = Path(__file__).parent.parent.parent
        nextjs_lock = project_root / "gui" / ".next" / "dev" / "lock"
        if nextjs_lock.exists():
            try:
                nextjs_lock.unlink()
                print("[E2EHarness] Removed Next.js dev lock file")
            except Exception as e:
                print(f"[E2EHarness] Failed to remove Next.js lock (ignored): {e}")

        # NANO-081: No config restoration needed — real config was never touched.

    async def __aenter__(self) -> "E2EHarness":
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.stop()
