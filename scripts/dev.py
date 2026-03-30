"""
SpindL development server.

Starts the backend GUI server and the Next.js frontend dev server
in a single command. Ctrl+C stops both — including all child processes
(llama-server, whisper-server, TTS, avatar, subtitles).

Usage:
    python scripts/dev.py
    python scripts/dev.py --port 8765        # Custom backend port
    python scripts/dev.py --config spindl.yaml  # Custom config path

Prerequisites (one-time):
    pip install -e ".[dev]"
    cd gui && npm install
"""

import os
import signal
import subprocess
import sys
import time

import psutil


def _kill_process_tree(pid: int, name: str, timeout: float = 5.0) -> None:
    """
    Kill a process and all its children using psutil.

    On Unix, sends SIGTERM first (allows graceful cleanup), then SIGKILL
    for survivors. On Windows, TerminateProcess is the only option — but
    we walk the full tree so nothing orphans.

    Args:
        pid: Root process ID.
        name: Label for log messages.
        timeout: Seconds to wait for graceful termination before force-killing.
    """
    try:
        parent = psutil.Process(pid)
    except psutil.NoSuchProcess:
        return

    children = parent.children(recursive=True)
    all_procs = children + [parent]

    # First pass: graceful terminate (SIGTERM on Unix, TerminateProcess on Windows)
    for proc in all_procs:
        try:
            proc.terminate()
        except psutil.NoSuchProcess:
            pass

    gone, alive = psutil.wait_procs(all_procs, timeout=timeout)

    # Second pass: force kill survivors
    for proc in alive:
        try:
            proc.kill()
        except psutil.NoSuchProcess:
            pass

    gone2, still_alive = psutil.wait_procs(alive, timeout=2)
    gone.extend(gone2)

    total_killed = len(gone) + len(still_alive)
    force_count = len(alive)

    if force_count:
        print(f"[dev] {name}: {total_killed} processes stopped ({force_count} force-killed)")
    elif total_killed:
        print(f"[dev] {name}: {total_killed} processes stopped gracefully")


def _shutdown(procs: list[tuple[str, subprocess.Popen]]) -> None:
    """
    Shut down all managed processes and their full trees.

    On Unix: sends SIGINT to the backend first so its finally block can
    run shutdown_services() / orchestrator.stop() for a cleaner exit.
    Falls back to tree kill if it doesn't exit in time.

    On Windows: tree kill immediately — SIGINT delivery to child processes
    is unreliable, and TerminateProcess skips finally blocks anyway.
    """
    for name, proc in procs:
        if proc.poll() is not None:
            print(f"[dev] {name} already exited (code: {proc.returncode})")
            continue

        if sys.platform != "win32" and name == "backend":
            # Unix: send SIGINT so the backend's finally block runs
            try:
                os.kill(proc.pid, signal.SIGINT)
                proc.wait(timeout=8)
                print(f"[dev] {name} shut down gracefully")
                continue
            except (subprocess.TimeoutExpired, OSError):
                print(f"[dev] {name} didn't exit on SIGINT, killing tree...")

        # Tree kill — catches all children (services, avatar, subtitles)
        _kill_process_tree(proc.pid, name, timeout=5.0)


def main():
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    gui_dir = os.path.join(project_root, "gui")

    # Pass any CLI args through to the backend server
    backend_args = sys.argv[1:]

    backend_cmd = [
        sys.executable, "-u",
        os.path.join(project_root, "scripts", "run_gui_standalone.py"),
        *backend_args,
    ]

    # Use npm.cmd on Windows, npm elsewhere
    npm = "npm.cmd" if sys.platform == "win32" else "npm"
    frontend_cmd = [npm, "run", "dev"]

    procs: list[tuple[str, subprocess.Popen]] = []

    try:
        print("[dev] Starting backend GUI server...")
        backend = subprocess.Popen(
            backend_cmd,
            cwd=project_root,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        procs.append(("backend", backend))

        # Give the backend a moment to bind the port before starting frontend
        time.sleep(2)

        print("[dev] Starting frontend dev server...")
        frontend = subprocess.Popen(
            frontend_cmd,
            cwd=gui_dir,
            stdout=sys.stdout,
            stderr=sys.stderr,
        )
        procs.append(("frontend", frontend))

        print("[dev] Ready — open http://localhost:3000")
        print("[dev] Press Ctrl+C to stop both servers.\n")

        # Wait for either process to exit
        while True:
            for name, proc in procs:
                ret = proc.poll()
                if ret is not None:
                    print(f"\n[dev] {name} exited with code {ret}. Shutting down...")
                    raise KeyboardInterrupt
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[dev] Stopping...")
        _shutdown(procs)
        print("[dev] Stopped.")


if __name__ == "__main__":
    main()
