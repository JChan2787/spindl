"""
Standalone GUI Server for first-time configuration.

Runs the GUI server without requiring an orchestrator, allowing users
to configure services through the web interface before starting them.

Usage:
    python scripts/run_gui_standalone.py                    # Default port 8765
    python scripts/run_gui_standalone.py --port 8080        # Custom port
    python scripts/run_gui_standalone.py --config spindl.yaml # Custom config path

This script is for GUI-first mode:
1. GUI server starts (no orchestrator, no services)
2. User opens /launcher in browser
3. User configures services
4. User clicks "Start Services"
5. Services launch via Socket.IO trigger
6. Orchestrator initializes (NANO-027 Phase 4)
7. orchestrator_ready event emitted
8. User redirected to Dashboard

For headless mode (no GUI configuration needed), use:
    python scripts/launcher.py
"""

import argparse
import asyncio
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

# Load .env file for API keys
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


BANNER = """
===============================================================
              SPINDL GUI-FIRST MODE
         Service Configuration via Web Interface
===============================================================
"""

# Global state for orchestrator (set after services launch)
_orchestrator: Optional["VoiceAgentOrchestrator"] = None
_event_bridge: Optional["EventBridge"] = None


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SpindL Standalone GUI Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config",
        default="./config/spindl.yaml",
        help="Path to configuration file (default: ./config/spindl.yaml)",
    )

    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="GUI server host (default: 127.0.0.1)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="GUI server port (default: 8765)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    return parser.parse_args()


async def initialize_orchestrator(gui_server, config_path: Path) -> bool:
    """
    Initialize the orchestrator after services are ready.

    Called from the services_ready callback when all services have started.
    This is an async function so it can properly await the emit calls.

    Returns:
        True if orchestrator started successfully, False otherwise.
    """
    global _orchestrator, _event_bridge

    print("[GUI-STANDALONE] Services ready, initializing orchestrator...", flush=True)

    try:
        from spindl import VoiceAgentOrchestrator, OrchestratorConfig
        from spindl.gui import EventBridge
        from spindl.core import EventType

        # Load configuration
        config = OrchestratorConfig.from_yaml(str(config_path))

        # Resolve relative paths
        base_dir = Path(__file__).parent.parent
        personas_dir = Path(config.personas_dir)
        if not personas_dir.is_absolute():
            personas_dir = base_dir / personas_dir
        config.personas_dir = str(personas_dir)

        conversations_dir = Path(config.conversations_dir)
        if not conversations_dir.is_absolute():
            conversations_dir = base_dir / conversations_dir
        config.conversations_dir = str(conversations_dir)

        # Create and start orchestrator
        print("[GUI-STANDALONE] Creating orchestrator...", flush=True)
        _orchestrator = VoiceAgentOrchestrator(config)
        _orchestrator.start()

        # Get persona for logging
        persona_name = _orchestrator.persona.get("name", config.persona_id) if _orchestrator.persona else config.persona_id
        print(f"[GUI-STANDALONE] Orchestrator started. Persona: {persona_name}", flush=True)

        # Attach orchestrator to GUI server
        gui_server.attach(_orchestrator)

        # Create and start event bridge
        # Use gui_server.event_loop (the uvicorn loop) instead of asyncio.get_running_loop()
        # because this function runs in a separate launcher thread with its own event loop
        if _orchestrator.event_bus:
            _event_bridge = EventBridge(_orchestrator.event_bus, gui_server)
            if gui_server.event_loop:
                _event_bridge.set_event_loop(gui_server.event_loop)
            else:
                print("[GUI-STANDALONE] WARNING: No event loop captured, EventBridge may not emit correctly", flush=True)
            _event_bridge.start()
            print("[GUI-STANDALONE] EventBridge started", flush=True)

        # Emit orchestrator_ready event
        await gui_server.emit_orchestrator_ready(persona_name)

        return True

    except Exception as e:
        import traceback
        print(f"[GUI-STANDALONE] Failed to initialize orchestrator: {e}", flush=True)
        traceback.print_exc()

        # Emit error event
        await gui_server.emit_orchestrator_error(str(e))

        return False


async def run_server(gui_server, host: str, port: int) -> None:
    """Run the GUI server with uvicorn."""
    import uvicorn

    # Capture the event loop reference for cross-thread EventBridge scheduling.
    # This must happen here, inside asyncio.run(), so we get the uvicorn loop.
    # The EventBridge (created later from a different thread) will use this loop.
    gui_server.set_event_loop(asyncio.get_running_loop())

    config = uvicorn.Config(
        gui_server.app,
        host=host,
        port=port,
        log_level="info",
        access_log=False,
    )
    server = uvicorn.Server(config)

    # Pass server reference to GUIServer for shutdown control (NANO-028)
    gui_server.set_uvicorn_server(server)

    await server.serve()


def main() -> int:
    """Main entry point."""
    args = parse_args()

    print(BANNER)

    # Resolve config path
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = Path(__file__).parent.parent / config_path

    print(f"[GUI-STANDALONE] Config path: {config_path}")
    print(f"[GUI-STANDALONE] Server will start on http://{args.host}:{args.port}")

    # Create GUI server in standalone mode (no orchestrator)
    try:
        from spindl.gui import GUIServer

        gui_server = GUIServer(
            host=args.host,
            port=args.port,
            config_path=str(config_path),
        )
        # Note: NOT calling gui_server.attach(orchestrator) - that happens after services launch
        # The server will operate in standalone mode until services are ready

        # Set callback for when services are ready (NANO-027 Phase 4)
        async def on_services_ready():
            """Called when all services have started successfully."""
            await initialize_orchestrator(gui_server, config_path)

        gui_server.set_services_ready_callback(on_services_ready)
        print("[GUI-STANDALONE] Services-ready callback registered", flush=True)

    except ImportError as e:
        print(f"[GUI-STANDALONE] Could not import GUI modules: {e}", file=sys.stderr)
        print("[GUI-STANDALONE] Install with: pip install python-socketio[asyncio] uvicorn", file=sys.stderr)
        return 1

    # Graceful shutdown handler
    shutdown_event = asyncio.Event()

    def signal_handler(sig, frame):
        print("\n[GUI-STANDALONE] Shutdown requested...")
        shutdown_event.set()

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Run server
    print(f"[GUI-STANDALONE] Starting server...")
    print(f"[GUI-STANDALONE] Open http://{args.host}:{args.port}/launcher to configure services")

    try:
        asyncio.run(run_server(gui_server, args.host, args.port))
    except KeyboardInterrupt:
        pass
    finally:
        # Cleanup
        global _orchestrator, _event_bridge

        if _event_bridge:
            print("[GUI-STANDALONE] Stopping EventBridge...", flush=True)
            _event_bridge.stop()

        if _orchestrator and _orchestrator.is_running:
            print("[GUI-STANDALONE] Stopping orchestrator...", flush=True)
            _orchestrator.stop()

        # Shutdown any launched services
        gui_server.shutdown_services()

    print("[GUI-STANDALONE] Server stopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
