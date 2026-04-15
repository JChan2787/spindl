"""
Session-domain Socket.IO handlers for the SpindL GUI.

Extracted from server.py (NANO-113). Handles:
- Session list (request_sessions)
- Session detail (request_session_detail)
- Chat history (request_chat_history)
- Session resume (resume_session)
- Session create (create_session)
- Session delete (delete_session)
- Session summary generation (generate_session_summary)

Also exposes emit_sessions() and emit_session_detail() as standalone
async helpers — other modules call these when they need to refresh
the session list (e.g. after persona switch).
"""

import asyncio
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .server import GUIServer


async def emit_sessions(
    server: "GUIServer",
    sid: Optional[str] = None,
    persona_filter: Optional[str] = None,
) -> None:
    """Emit list of conversation sessions."""
    from spindl.history import jsonl_store

    sio = server.sio

    # NANO-048: Resolve conversations_dir from orchestrator (preferred) or pre-launch config
    conv_dir_str = None
    if server._orchestrator:
        conv_dir_str = server._orchestrator._config.conversations_dir
    elif server._conversations_dir:
        conv_dir_str = server._conversations_dir

    if not conv_dir_str:
        return

    conversations_dir = Path(conv_dir_str)
    sessions = []

    if conversations_dir.exists():
        for filepath in sorted(
            (p for p in conversations_dir.glob("*.jsonl") if ".snapshot." not in p.name),
            key=lambda p: p.stat().st_mtime, reverse=True,
        ):
            # Parse filename: {persona}_{timestamp}.jsonl
            parts = filepath.stem.rsplit("_", 2)
            if len(parts) >= 2:
                persona = parts[0]
                if persona_filter and persona != persona_filter:
                    continue

                stat = filepath.stat()

                # Count turns (read file to get actual count)
                try:
                    turns = jsonl_store.read_turns(filepath)
                    turn_count = len(turns)
                    visible_count = len([t for t in turns if not t.get("hidden", False)])
                except Exception:
                    turn_count = 0
                    visible_count = 0

                sessions.append({
                    "filepath": str(filepath),
                    "persona": persona,
                    "timestamp": filepath.stem.split("_", 1)[1] if "_" in filepath.stem else "",
                    "turn_count": turn_count,
                    "visible_count": visible_count,
                    "file_size": stat.st_size,
                })

    # Include the orchestrator's current session so the frontend can badge it
    active_session = None
    if server._orchestrator and server._orchestrator.session_file:
        active_session = str(server._orchestrator.session_file)

    payload = {"sessions": sessions, "active_session": active_session}

    if sid:
        await sio.emit("session_list", payload, to=sid)
    else:
        await sio.emit("session_list", payload)


async def emit_session_detail(
    server: "GUIServer", sid: str, filepath: str
) -> None:
    """Emit detailed session data."""
    from spindl.history import jsonl_store

    sio = server.sio

    try:
        path = Path(filepath)
        if not path.exists():
            await sio.emit("error", {"message": "Session not found"}, to=sid)
            return

        turns = jsonl_store.read_turns(path)
        await sio.emit(
            "session_detail",
            {"filepath": filepath, "turns": turns},
            to=sid,
        )
    except Exception as e:
        print(f"[GUI] Error reading session: {e}", flush=True)
        await sio.emit("error", {"message": str(e)}, to=sid)


def register_session_handlers(server: "GUIServer") -> None:
    """Register session-domain Socket.IO event handlers."""
    sio = server.sio

    @sio.event
    async def request_sessions(sid: str, data: dict) -> None:
        """Client requests session list."""
        await emit_sessions(server, sid, data.get("persona"))

    @sio.event
    async def request_session_detail(sid: str, data: dict) -> None:
        """Client requests session details."""
        filepath = data.get("filepath")
        if filepath:
            await emit_session_detail(server, sid, filepath)

    @sio.event
    async def request_chat_history(sid: str, data: dict) -> None:
        """Client requests chat history for the active session (NANO-073a)."""
        from spindl.history import jsonl_store

        if not server._orchestrator or not server._orchestrator.session_file:
            await sio.emit(
                "chat_history",
                {"turns": []},
                to=sid,
            )
            return

        try:
            filepath = Path(server._orchestrator.session_file)
            if not filepath.exists():
                await sio.emit(
                    "chat_history",
                    {"turns": []},
                    to=sid,
                )
                return

            visible = jsonl_store.read_visible_turns(filepath)
            # Map to frontend-friendly format with metadata (NANO-075)
            turns = []
            for t in visible:
                role = t.get("role")
                if role in ("user", "assistant"):
                    # NANO-109: display_content holds raw LLM output (with
                    # formatting); content holds cleaned text for LLM replay.
                    # Chat display should show the raw version when available.
                    #
                    # NANO-111 Phase 2.5 / Session 639: when barge_in_truncated
                    # is set, display_content is the PRE-barge full generation
                    # (preserved for inspection) and content is the truncated
                    # form the user actually heard. Chat bubble must show the
                    # truncated form to match what was spoken.
                    if t.get("barge_in_truncated"):
                        display_text = t.get("content", "")
                    else:
                        display_text = t.get("display_content") or t.get("content", "")
                    turn = {
                        "role": role,
                        "text": display_text,
                        "timestamp": t.get("timestamp", ""),
                    }
                    if t.get("barge_in_truncated"):
                        turn["barge_in_truncated"] = True
                    # NANO-075: Forward metadata for hydration survival
                    if role == "user":
                        input_mod = t.get("input_modality")
                        if input_mod:
                            turn["input_modality"] = input_mod
                    elif role == "assistant":
                        reasoning = t.get("reasoning")
                        if reasoning:
                            turn["reasoning"] = reasoning
                        stimulus = t.get("stimulus_source")
                        if stimulus:
                            turn["stimulus_source"] = stimulus
                        codex = t.get("activated_codex_entries")
                        if codex:
                            turn["activated_codex_entries"] = codex
                        memories = t.get("retrieved_memories")
                        if memories:
                            turn["retrieved_memories"] = memories
                        # NANO-094: Emotion classifier metadata
                        emotion = t.get("emotion")
                        if emotion:
                            turn["emotion"] = emotion
                            turn["emotion_confidence"] = t.get("emotion_confidence")
                    turns.append(turn)

            # Cap at 200 most recent to avoid DOM bloat
            if len(turns) > 200:
                turns = turns[-200:]

            await sio.emit(
                "chat_history",
                {"turns": turns},
                to=sid,
            )
        except Exception as e:
            print(f"[GUI] Error reading chat history: {e}", flush=True)
            await sio.emit(
                "chat_history",
                {"turns": []},
                to=sid,
            )

    @sio.event
    async def resume_session(sid: str, data: dict) -> None:
        """Client requests to resume a specific session by filename."""
        filename = data.get("filename")
        if not filename or not server._orchestrator:
            return

        # Resolve filename to full path in conversations directory
        conv_dir_str = server._orchestrator._config.conversations_dir
        if not conv_dir_str:
            return
        resolved = Path(conv_dir_str) / filename
        if not resolved.exists():
            print(f"[GUI] Session resume failed: {filename} not found", flush=True)
            await sio.emit(
                "session_resumed",
                {"filepath": str(resolved), "success": False, "error": "Session file not found"},
                to=sid,
            )
            return

        success = server._orchestrator.load_session(str(resolved))
        if success:
            print(f"[GUI] Session resumed: {filename}", flush=True)
            await sio.emit(
                "session_resumed",
                {"filepath": str(resolved), "success": True},
                to=sid,
            )
            # Refresh session list so active_session badge updates
            await emit_sessions(server)
        else:
            print(f"[GUI] Session resume failed: {filename}", flush=True)
            await sio.emit(
                "session_resumed",
                {"filepath": str(resolved), "success": False, "error": "Failed to load session"},
                to=sid,
            )

    @sio.event
    async def create_session(sid: str, data: dict | None = None) -> None:
        """Client requests a new session for the current persona (NANO-071)."""
        if not server._orchestrator:
            await sio.emit(
                "session_created",
                {"success": False, "error": "Services not running"},
                to=sid,
            )
            return

        success = server._orchestrator.create_new_session()
        if success:
            new_filepath = str(server._orchestrator.session_file) if server._orchestrator.session_file else None
            # Touch the file so it exists on disk for emit_sessions glob
            if new_filepath:
                Path(new_filepath).parent.mkdir(parents=True, exist_ok=True)
                Path(new_filepath).touch()
            print(f"[GUI] New session created: {Path(new_filepath).name if new_filepath else 'unknown'}", flush=True)
            await sio.emit(
                "session_created",
                {"success": True, "filepath": new_filepath},
                to=sid,
            )
            # Refresh session list for all clients
            await emit_sessions(server)
        else:
            await sio.emit(
                "session_created",
                {"success": False, "error": "Failed to create new session"},
                to=sid,
            )

    @sio.event
    async def delete_session(sid: str, data: dict) -> None:
        """Client requests to delete a session file (NANO-064a)."""
        filepath = data.get("filepath")
        if not filepath:
            return

        try:
            path = Path(filepath)
            if not path.exists():
                await sio.emit(
                    "session_deleted",
                    {"filepath": filepath, "success": False, "error": "Session not found"},
                    to=sid,
                )
                return

            # Don't allow deleting the active session (only when services are running)
            if (
                server._orchestrator
                and server._orchestrator.session_file
                and path == server._orchestrator.session_file
            ):
                await sio.emit(
                    "session_deleted",
                    {"filepath": filepath, "success": False, "error": "Cannot delete active session"},
                    to=sid,
                )
                return

            path.unlink()
            # NANO-076: Clean up snapshot sidecar if it exists
            from spindl.history.snapshot_store import delete_sidecar
            delete_sidecar(path)
            print(f"[GUI] Session deleted: {path.name}", flush=True)
            await sio.emit(
                "session_deleted",
                {"filepath": filepath, "success": True},
                to=sid,
            )
            # Refresh session list for all clients
            await emit_sessions(server)

        except Exception as e:
            print(f"[GUI] Session delete error: {e}", flush=True)
            await sio.emit(
                "session_deleted",
                {"filepath": filepath, "success": False, "error": str(e)},
                to=sid,
            )

    @sio.event
    async def generate_session_summary(sid: str, data: dict) -> None:
        """Client requests session summary generation (NANO-043 Phase 4)."""
        filepath = data.get("filepath")
        if filepath and server._orchestrator:
            try:
                # Run in executor to avoid blocking event loop (LLM call)
                loop = asyncio.get_event_loop()
                summary = await loop.run_in_executor(
                    None,
                    server._orchestrator.generate_session_summary,
                    filepath,
                )
                session_name = Path(filepath).name
                if summary:
                    print(f"[GUI] Session summary generated: {session_name}", flush=True)
                else:
                    print(f"[GUI] Session summary failed: {session_name}", flush=True)
                await sio.emit(
                    "session_summary_generated",
                    {
                        "filepath": filepath,
                        "success": summary is not None,
                        "summary_preview": summary[:200] if summary else None,
                        "error": None if summary else "Failed to generate summary",
                    },
                    to=sid,
                )
            except Exception as e:
                print(f"[GUI] Session summary error: {e}", flush=True)
                await sio.emit(
                    "session_summary_generated",
                    {
                        "filepath": filepath,
                        "success": False,
                        "summary_preview": None,
                        "error": str(e),
                    },
                    to=sid,
                )
