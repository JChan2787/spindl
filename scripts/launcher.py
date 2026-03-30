#!/usr/bin/env python3
"""
SpindL Unified Launcher

Single-command startup for all spindl services with config-driven
flexibility, health checks, and graceful shutdown.

Usage:
    python scripts/launcher.py                    # Start all services
    python scripts/launcher.py --config path.yaml # Custom config
    python scripts/launcher.py --debug            # Debug mode
    python scripts/launcher.py --dry-run          # Validate only
    python scripts/launcher.py --only llm,stt     # Start specific services
    python scripts/launcher.py --skip orchestrator # Skip specific services
"""

import argparse
import signal
import sys
import time
from pathlib import Path

# Add src to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from spindl.launcher import (
    LauncherConfig,
    load_launcher_config,
    LogAggregator,
    ServiceRunner,
)


BANNER = """
╔═══════════════════════════════════════════════════════════════╗
║                      SPINDL LAUNCHER                            ║
║              Unified Service Startup System                   ║
╚═══════════════════════════════════════════════════════════════╝
"""

SUCCESS_BANNER = """
╔═══════════════════════════════════════════════════════════════╗
║              ALL SERVICES STARTED SUCCESSFULLY                ║
║                    Press Ctrl+C to shutdown                   ║
╚═══════════════════════════════════════════════════════════════╝
"""


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SpindL Unified Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--config",
        default="./config/spindl.yaml",
        help="Path to configuration file (default: ./config/spindl.yaml)",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging and pass --debug to services with pass_debug=True",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and print startup plan without executing",
    )

    parser.add_argument(
        "--only",
        type=str,
        help="Only start specified services (comma-separated, e.g., --only llm,stt)",
    )

    parser.add_argument(
        "--skip",
        type=str,
        help="Skip specified services (comma-separated, e.g., --skip orchestrator)",
    )

    parser.add_argument(
        "-v", "--verbosity",
        type=str,
        choices=["quiet", "minimal", "normal", "verbose", "debug"],
        help="Log verbosity preset (overrides config file). "
             "quiet=errors only, minimal=chat+context+errors, "
             "normal=default, verbose=most logs, debug=everything",
    )

    return parser.parse_args()


# Verbosity preset definitions (NANO-017 Session 4)
# Each preset defines service log levels and suppress patterns
VERBOSITY_PRESETS: dict[str, dict] = {
    "quiet": {
        # Errors only - complete silence otherwise
        "service_levels": {
            "stt": "error",
            "tts": "error",
            "llm": "error",
            "vlm": "error",
            "orchestrator": "error",
        },
        "suppress_patterns": [
            ".*",  # Suppress everything that isn't an error
        ],
    },
    "minimal": {
        # Chat log ([User], [Persona]) + context + errors
        # This is the "clean console" experience
        "service_levels": {
            "stt": "error",
            "tts": "error",
            "llm": "error",
            "vlm": "error",
            "orchestrator": "info",
        },
        "suppress_patterns": [
            "Connection from",
            "-> success:",
            "slot \\d+:",
            "sampling params",
            "sampler constr",
            "sampler chain",
        ],
    },
    "normal": {
        # Chat + state changes + service status (default)
        "service_levels": {
            "stt": "warning",
            "tts": "warning",
            "llm": "warning",
            "vlm": "error",
            "orchestrator": "info",
        },
        "suppress_patterns": [
            "Connection from",
            "-> success:",
            "slot \\d+:",
            "sampling params",
            "sampler constr",
            "sampler chain",
        ],
    },
    "verbose": {
        # Everything except per-request spam
        "service_levels": {
            "stt": "info",
            "tts": "info",
            "llm": "info",
            "vlm": "info",
            "orchestrator": "info",
        },
        "suppress_patterns": [
            "slot \\d+:",
            "sampling params",
        ],
    },
    "debug": {
        # Everything - no filtering
        "service_levels": {},
        "suppress_patterns": [],
    },
}


def filter_services(
    config: LauncherConfig,
    only: str | None,
    skip: str | None,
) -> LauncherConfig:
    """
    Filter services based on --only and --skip flags.

    Modifies the config in-place by disabling services.
    """
    only_set = set(only.split(",")) if only else None
    skip_set = set(skip.split(",")) if skip else set()

    for name, svc in config.services.items():
        # Skip if not in --only list (when specified)
        if only_set is not None and name not in only_set:
            svc.enabled = False
            continue

        # Skip if in --skip list
        if name in skip_set:
            svc.enabled = False

    return config


def print_startup_plan(config: LauncherConfig) -> None:
    """Print the startup plan for dry-run mode."""
    print("\n[DRY-RUN] Startup Plan:")
    print("-" * 40)

    order = config.get_startup_order()
    for i, name in enumerate(order, 1):
        svc = config.services[name]
        status = "ENABLED" if svc.enabled else "DISABLED"
        deps = ", ".join(svc.depends_on) if svc.depends_on else "none"
        health = svc.health_check.type

        print(f"  {i}. {name}")
        print(f"     Status:      {status}")
        print(f"     Platform:    {svc.platform}")
        print(f"     Dependencies: {deps}")
        print(f"     Health check: {health}")
        if svc.enabled:
            print(f"     Command:     {svc.command[:60]}...")
        print()

    print("-" * 40)
    enabled_count = sum(1 for s in config.services.values() if s.enabled)
    print(f"[DRY-RUN] {enabled_count}/{len(config.services)} services would be started")
    print("[DRY-RUN] Configuration is valid.")


def main() -> int:
    """Main entry point."""
    args = parse_args()

    print(BANNER)

    # Load configuration
    try:
        config = load_launcher_config(args.config)
        print(f"[LAUNCHER] Loaded config from {args.config}")
        print(f"[LAUNCHER] Found {len(config.services)} services")
    except FileNotFoundError as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        return 1
    except ValueError as e:
        print(f"[ERROR] Configuration error: {e}", file=sys.stderr)
        return 1

    # Apply --only and --skip filters
    config = filter_services(config, args.only, args.skip)

    # Get startup order (validates dependencies)
    try:
        startup_order = config.get_startup_order()
    except ValueError as e:
        print(f"[ERROR] Dependency error: {e}", file=sys.stderr)
        return 1

    # Dry-run mode: just print the plan and exit
    if args.dry_run:
        print_startup_plan(config)
        return 0

    # Set up logging with per-service filtering (NANO-017)
    # CLI --verbosity overrides config file, --debug forces debug level
    if args.debug:
        log_level = "debug"
        service_levels = {}  # No filtering in debug mode
        suppress_patterns = []
    elif args.verbosity:
        preset = VERBOSITY_PRESETS[args.verbosity]
        log_level = "debug" if args.verbosity == "debug" else config.log_level
        service_levels = preset["service_levels"]
        suppress_patterns = preset["suppress_patterns"]
        print(f"[LAUNCHER] Using verbosity preset: {args.verbosity}")
    else:
        log_level = config.log_level
        service_levels = config.service_levels
        suppress_patterns = config.suppress_patterns

    logger = LogAggregator(
        log_file=config.log_file,
        log_level=log_level,
        service_levels=service_levels,
        suppress_patterns=suppress_patterns,
    )

    # Create service runner with provider configs (NANO-015, NANO-019, NANO-021, NANO-023)
    runner = ServiceRunner(
        logger=logger,
        default_health_timeout=config.health_check_timeout,
        debug_mode=args.debug,
        tts_provider_config=config.tts_provider_config,
        llm_provider_config=config.llm_provider_config,
        vision_provider_config=config.vision_provider_config,
        pipeline_max_context=config.pipeline_max_context,
    )

    # Shutdown handler
    shutdown_requested = False

    def handle_shutdown(signum, frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            print("\n[LAUNCHER] Force shutdown...", file=sys.stderr)
            sys.exit(1)
        shutdown_requested = True
        print("\n[LAUNCHER] Shutdown requested (Ctrl+C again to force)...")
        runner.shutdown_all()
        logger.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    # Start services in order
    logger.log_launcher(f"Starting {len(startup_order)} services...")

    for name in startup_order:
        svc_config = config.services[name]

        if not svc_config.enabled:
            logger.log_launcher(f"Skipping disabled service: {name}")
            continue

        success = runner.start_service(svc_config)

        if not success:
            logger.log_launcher(f"ABORTING: Service {name} failed to start", level="error")
            runner.shutdown_all()
            logger.close()
            return 1

    # All services started
    print(SUCCESS_BANNER)
    logger.log_launcher("All services running. Waiting for Ctrl+C...")

    # Wait forever (until signal)
    try:
        while True:
            time.sleep(1)

            # Check if any service died unexpectedly
            for name in startup_order:
                svc = config.services[name]
                if svc.enabled and not runner.is_service_running(name):
                    logger.log_launcher(
                        f"Service {name} exited unexpectedly!",
                        level="error",
                    )
                    runner.shutdown_all()
                    logger.close()
                    return 1
    except KeyboardInterrupt:
        pass  # Handled by signal handler

    return 0


if __name__ == "__main__":
    sys.exit(main())
