"""
Memory-domain Socket.IO handlers for the SpindL GUI.

Extracted from server.py (NANO-113). Handles:
- Memory config (RAG settings, curation)
- Memory CRUD (general, global, flashcards, summaries)
- Memory search and promotion
"""

import asyncio
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .server import GUIServer


def register_memory_handlers(server: "GUIServer") -> None:
    """Register memory-domain Socket.IO event handlers."""
    sio = server.sio

    # ============================================================
    # NANO-065: Memory Config — Socket Handler
    # ============================================================

    @sio.event
    async def set_memory_config(sid: str, data: dict) -> None:
        """Client updates memory/RAG configuration."""
        if server._orchestrator:
            updated = False

            # Use ellipsis as sentinel to distinguish "not provided" from "set to None"
            top_k = None
            relevance_threshold = ...
            dedup_threshold = ...
            reflection_interval = None
            reflection_prompt = ...
            reflection_system_message = ...
            reflection_delimiter = None

            if "top_k" in data:
                top_k = int(data["top_k"])
                updated = True
            if "relevance_threshold" in data:
                val = data["relevance_threshold"]
                relevance_threshold = float(val) if val is not None else None
                updated = True
            if "dedup_threshold" in data:
                val = data["dedup_threshold"]
                dedup_threshold = float(val) if val is not None else None
                updated = True
            if "reflection_interval" in data:
                reflection_interval = int(data["reflection_interval"])
                updated = True
            if "reflection_prompt" in data:
                val = data["reflection_prompt"]
                # Empty string or null → None (use built-in default)
                reflection_prompt = val if val else None
                updated = True
            if "reflection_system_message" in data:
                val = data["reflection_system_message"]
                reflection_system_message = val if val else None
                updated = True
            if "reflection_delimiter" in data:
                val = data["reflection_delimiter"]
                if val:  # Don't accept empty delimiter
                    reflection_delimiter = str(val)
                    updated = True

            if updated:
                server._orchestrator.update_memory_config(
                    top_k=top_k,
                    relevance_threshold=relevance_threshold,
                    dedup_threshold=dedup_threshold,
                    reflection_interval=reflection_interval,
                    reflection_prompt=reflection_prompt,
                    reflection_system_message=reflection_system_message,
                    reflection_delimiter=reflection_delimiter,
                )

                config = server._orchestrator._config
                print(
                    f"[GUI] Memory: top_k={config.memory_config.rag_top_k}, "
                    f"relevance_threshold={config.memory_config.relevance_threshold}, "
                    f"dedup_threshold={config.memory_config.dedup_threshold}, "
                    f"reflection_interval={config.memory_config.reflection_interval}",
                    flush=True,
                )

                # Persist to YAML
                persisted = False
                if server._config_path:
                    try:
                        config.save_to_yaml(server._config_path)
                        persisted = True
                        print(f"[GUI] Memory config persisted to {Path(server._config_path).name}", flush=True)
                    except Exception as e:
                        print(f"[GUI] Failed to persist Memory config: {e}", flush=True)

                await sio.emit(
                    "memory_config_updated",
                    {
                        "top_k": config.memory_config.rag_top_k,
                        "relevance_threshold": config.memory_config.relevance_threshold,
                        "dedup_threshold": config.memory_config.dedup_threshold,
                        "reflection_interval": config.memory_config.reflection_interval,
                        "reflection_prompt": config.memory_config.reflection_prompt,
                        "reflection_system_message": config.memory_config.reflection_system_message,
                        "reflection_delimiter": config.memory_config.reflection_delimiter,
                        "enabled": config.memory_config.enabled,
                        "persisted": persisted,
                    },
                    to=sid,
                )

    # ============================================================
    # NANO-102: Memory Curation Config — Socket Handler
    # ============================================================

    @sio.event
    async def set_curation_config(sid: str, data: dict) -> None:
        """Client updates memory curation configuration (NANO-102)."""
        if server._orchestrator:
            config = server._orchestrator._config
            curation = config.memory_config.curation
            updated = False

            if "enabled" in data:
                curation.enabled = bool(data["enabled"])
                updated = True
            if "api_key" in data:
                curation.api_key = data["api_key"]
                updated = True
            if "model" in data:
                curation.model = str(data["model"])
                updated = True
            if "prompt" in data:
                curation.prompt = data["prompt"]
                updated = True
            if "timeout" in data:
                curation.timeout = float(data["timeout"])
                updated = True

            if updated:
                print(
                    f"[GUI] Curation: enabled={curation.enabled}, "
                    f"model={curation.model}",
                    flush=True,
                )

                # Persist to YAML
                persisted = False
                if server._config_path:
                    try:
                        config.save_to_yaml(server._config_path)
                        persisted = True
                        print(f"[GUI] Curation config persisted to {Path(server._config_path).name}", flush=True)
                    except Exception as e:
                        print(f"[GUI] Failed to persist Curation config: {e}", flush=True)

                await sio.emit(
                    "curation_config_updated",
                    {
                        "enabled": curation.enabled,
                        "api_key": curation.api_key,
                        "model": curation.model,
                        "prompt": curation.prompt,
                        "timeout": curation.timeout,
                        "persisted": persisted,
                    },
                    to=sid,
                )

    # ============================================================
    # NANO-043 Phase 6: Memory CRUD Handlers
    # ============================================================

    @sio.event
    async def request_memory_counts(sid: str, data: dict) -> None:
        """Client requests memory collection counts."""
        if not server._orchestrator or not server._orchestrator.memory_store:
            await sio.emit(
                "memory_counts",
                {"global": 0, "general": 0, "flashcards": 0, "summaries": 0, "enabled": False},
                to=sid,
            )
            return

        try:
            counts = server._orchestrator.memory_store.counts
            await sio.emit(
                "memory_counts",
                {**counts, "enabled": True},
                to=sid,
            )
        except Exception as e:
            print(f"[GUI] Memory counts error: {e}", flush=True)
            await sio.emit(
                "memory_counts",
                {"global": 0, "general": 0, "flashcards": 0, "summaries": 0, "enabled": True, "error": str(e)},
                to=sid,
            )

    @sio.event
    async def request_memories(sid: str, data: dict) -> None:
        """Client requests all memories from a specific collection."""
        collection = data.get("collection")
        if not collection or not server._orchestrator or not server._orchestrator.memory_store:
            await sio.emit(
                "memory_list",
                {
                    "collection": collection or "unknown",
                    "memories": [],
                    "error": "Memory system not available" if not (server._orchestrator and server._orchestrator.memory_store) else None,
                },
                to=sid,
            )
            return

        try:
            memories = server._orchestrator.memory_store.get_all(collection)
            await sio.emit(
                "memory_list",
                {"collection": collection, "memories": memories},
                to=sid,
            )
        except Exception as e:
            print(f"[GUI] Memory list error ({collection}): {e}", flush=True)
            await sio.emit(
                "memory_list",
                {"collection": collection, "memories": [], "error": str(e)},
                to=sid,
            )

    @sio.event
    async def add_general_memory(sid: str, data: dict) -> None:
        """Client adds a new general memory."""
        content = data.get("content", "").strip()
        if not content:
            await sio.emit(
                "memory_added",
                {"success": False, "error": "Content is required", "collection": "general"},
                to=sid,
            )
            return

        if not server._orchestrator or not server._orchestrator.memory_store:
            await sio.emit(
                "memory_added",
                {"success": False, "error": "Memory system not available", "collection": "general"},
                to=sid,
            )
            return

        try:
            metadata = {"type": "general", "source": "gui_manual"}
            loop = asyncio.get_event_loop()
            doc_id = await loop.run_in_executor(
                None,
                server._orchestrator.memory_store.add_general,
                content,
                metadata,
            )
            print(f"[GUI] General memory added: {doc_id}", flush=True)
            await sio.emit(
                "memory_added",
                {
                    "success": True,
                    "collection": "general",
                    "memory": {"id": doc_id, "content": content, "metadata": metadata},
                },
                to=sid,
            )
        except Exception as e:
            print(f"[GUI] Add memory error: {e}", flush=True)
            await sio.emit(
                "memory_added",
                {"success": False, "error": str(e), "collection": "general"},
                to=sid,
            )

    @sio.event
    async def edit_general_memory(sid: str, data: dict) -> None:
        """Client edits a general memory (delete + re-add with new embedding)."""
        doc_id = data.get("id")
        new_content = data.get("content", "").strip()

        if not doc_id or not new_content:
            await sio.emit(
                "memory_edited",
                {"success": False, "error": "ID and content are required"},
                to=sid,
            )
            return

        if not server._orchestrator or not server._orchestrator.memory_store:
            await sio.emit(
                "memory_edited",
                {"success": False, "error": "Memory system not available"},
                to=sid,
            )
            return

        store = server._orchestrator.memory_store
        try:
            # Get original metadata before deleting
            originals = store.get_all("general")
            original = next((m for m in originals if m["id"] == doc_id), None)
            original_meta = original["metadata"] if original else {}

            # Delete old
            store.delete("general", doc_id)

            # Re-add with new content, preserving original timestamp
            metadata = {
                **original_meta,
                "type": "general",
                "source": "gui_manual",
                "edited_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            loop = asyncio.get_event_loop()
            new_id = await loop.run_in_executor(
                None,
                store.add_general,
                new_content,
                metadata,
            )

            print(f"[GUI] General memory edited: {doc_id} -> {new_id}", flush=True)
            await sio.emit(
                "memory_edited",
                {
                    "success": True,
                    "old_id": doc_id,
                    "memory": {"id": new_id, "content": new_content, "metadata": metadata},
                },
                to=sid,
            )
        except Exception as e:
            print(f"[GUI] Edit memory error: {e}", flush=True)
            await sio.emit(
                "memory_edited",
                {"success": False, "error": str(e)},
                to=sid,
            )

    @sio.event
    async def add_global_memory(sid: str, data: dict) -> None:
        """Client adds a new global memory (cross-character, NANO-105)."""
        content = data.get("content", "").strip()
        if not content:
            await sio.emit(
                "memory_added",
                {"success": False, "error": "Content is required", "collection": "global"},
                to=sid,
            )
            return

        if not server._orchestrator or not server._orchestrator.memory_store:
            await sio.emit(
                "memory_added",
                {"success": False, "error": "Memory system not available", "collection": "global"},
                to=sid,
            )
            return

        try:
            metadata = {"type": "global", "source": "gui_manual"}
            loop = asyncio.get_event_loop()
            doc_id = await loop.run_in_executor(
                None,
                server._orchestrator.memory_store.add_global,
                content,
                metadata,
            )
            print(f"[GUI] Global memory added: {doc_id}", flush=True)
            await sio.emit(
                "memory_added",
                {
                    "success": True,
                    "collection": "global",
                    "memory": {"id": doc_id, "content": content, "metadata": metadata},
                },
                to=sid,
            )
        except Exception as e:
            print(f"[GUI] Add global memory error: {e}", flush=True)
            await sio.emit(
                "memory_added",
                {"success": False, "error": str(e), "collection": "global"},
                to=sid,
            )

    @sio.event
    async def edit_global_memory(sid: str, data: dict) -> None:
        """Client edits a global memory (NANO-105)."""
        doc_id = data.get("id")
        new_content = data.get("content", "").strip()

        if not doc_id or not new_content:
            await sio.emit(
                "memory_edited",
                {"success": False, "error": "ID and content are required"},
                to=sid,
            )
            return

        if not server._orchestrator or not server._orchestrator.memory_store:
            await sio.emit(
                "memory_edited",
                {"success": False, "error": "Memory system not available"},
                to=sid,
            )
            return

        store = server._orchestrator.memory_store
        try:
            loop = asyncio.get_event_loop()
            new_id = await loop.run_in_executor(
                None,
                store.edit_global,
                doc_id,
                new_content,
            )

            print(f"[GUI] Global memory edited: {doc_id} -> {new_id}", flush=True)
            metadata = {"type": "global", "source": "gui_manual", "edited_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            await sio.emit(
                "memory_edited",
                {
                    "success": True,
                    "old_id": doc_id,
                    "memory": {"id": new_id, "content": new_content, "metadata": metadata},
                },
                to=sid,
            )
        except Exception as e:
            print(f"[GUI] Edit global memory error: {e}", flush=True)
            await sio.emit(
                "memory_edited",
                {"success": False, "error": str(e)},
                to=sid,
            )

    @sio.event
    async def delete_memory(sid: str, data: dict) -> None:
        """Client deletes a memory from any collection."""
        collection = data.get("collection")
        doc_id = data.get("id")

        if not collection or not doc_id:
            await sio.emit(
                "memory_deleted",
                {"success": False, "error": "Collection and ID are required"},
                to=sid,
            )
            return

        if not server._orchestrator or not server._orchestrator.memory_store:
            await sio.emit(
                "memory_deleted",
                {"success": False, "error": "Memory system not available"},
                to=sid,
            )
            return

        try:
            success = server._orchestrator.memory_store.delete(collection, doc_id)
            print(f"[GUI] Memory deleted: {collection}/{doc_id} (success={success})", flush=True)
            await sio.emit(
                "memory_deleted",
                {"success": success, "collection": collection, "id": doc_id},
                to=sid,
            )
        except Exception as e:
            print(f"[GUI] Delete memory error: {e}", flush=True)
            await sio.emit(
                "memory_deleted",
                {"success": False, "error": str(e)},
                to=sid,
            )

    @sio.event
    async def promote_memory(sid: str, data: dict) -> None:
        """Promote a flash card or summary to general memory."""
        source_collection = data.get("source_collection")
        doc_id = data.get("id")
        delete_source = data.get("delete_source", False)

        if not source_collection or not doc_id:
            await sio.emit(
                "memory_promoted",
                {"success": False, "error": "Source collection and ID required"},
                to=sid,
            )
            return

        if source_collection not in ("flashcards", "summaries"):
            await sio.emit(
                "memory_promoted",
                {"success": False, "error": "Can only promote from flashcards or summaries"},
                to=sid,
            )
            return

        if not server._orchestrator or not server._orchestrator.memory_store:
            await sio.emit(
                "memory_promoted",
                {"success": False, "error": "Memory system not available"},
                to=sid,
            )
            return

        store = server._orchestrator.memory_store
        try:
            # Get the source document
            all_docs = store.get_all(source_collection)
            source_doc = next((m for m in all_docs if m["id"] == doc_id), None)
            if not source_doc:
                await sio.emit(
                    "memory_promoted",
                    {"success": False, "error": "Source memory not found"},
                    to=sid,
                )
                return

            # Add to general with promotion metadata
            metadata = {
                **source_doc.get("metadata", {}),
                "type": "general",
                "source": f"promoted_from_{source_collection}",
                "promoted_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            }
            loop = asyncio.get_event_loop()
            new_id = await loop.run_in_executor(
                None,
                store.add_general,
                source_doc["content"],
                metadata,
            )

            # Optionally delete from source
            if delete_source:
                store.delete(source_collection, doc_id)

            print(
                f"[GUI] Memory promoted: {source_collection}/{doc_id} -> general/{new_id}"
                f" (source {'deleted' if delete_source else 'kept'})",
                flush=True,
            )
            await sio.emit(
                "memory_promoted",
                {
                    "success": True,
                    "source_collection": source_collection,
                    "source_id": doc_id,
                    "new_id": new_id,
                    "deleted_source": delete_source,
                },
                to=sid,
            )
        except Exception as e:
            print(f"[GUI] Promote memory error: {e}", flush=True)
            await sio.emit(
                "memory_promoted",
                {"success": False, "error": str(e)},
                to=sid,
            )

    @sio.event
    async def search_memories(sid: str, data: dict) -> None:
        """Semantic search across all memory collections."""
        query_text = data.get("query", "").strip()
        top_k = data.get("top_k", 10)

        if not query_text:
            await sio.emit(
                "memory_search_results",
                {"results": [], "query": ""},
                to=sid,
            )
            return

        if not server._orchestrator or not server._orchestrator.memory_store:
            await sio.emit(
                "memory_search_results",
                {"results": [], "query": query_text, "error": "Memory system not available"},
                to=sid,
            )
            return

        try:
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                server._orchestrator.memory_store.query,
                query_text,
                top_k,
            )
            await sio.emit(
                "memory_search_results",
                {"results": results, "query": query_text},
                to=sid,
            )
        except Exception as e:
            print(f"[GUI] Memory search error: {e}", flush=True)
            await sio.emit(
                "memory_search_results",
                {"results": [], "query": query_text, "error": str(e)},
                to=sid,
            )

    @sio.event
    async def clear_flashcards(sid: str, data: dict) -> None:
        """Clear all flash cards."""
        if not server._orchestrator or not server._orchestrator.memory_store:
            await sio.emit(
                "flashcards_cleared",
                {"success": False, "error": "Memory system not available"},
                to=sid,
            )
            return

        try:
            server._orchestrator.memory_store.clear_flash_cards()
            print("[GUI] Flash cards cleared", flush=True)
            await sio.emit(
                "flashcards_cleared",
                {"success": True},
                to=sid,
            )
        except Exception as e:
            print(f"[GUI] Clear flashcards error: {e}", flush=True)
            await sio.emit(
                "flashcards_cleared",
                {"success": False, "error": str(e)},
                to=sid,
            )
