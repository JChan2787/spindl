"use client";

import { useEffect, useCallback } from "react";
import { SessionList, SessionViewer } from "@/components/sessions";
import { useSessionStore, useConnectionStore } from "@/lib/stores";
import { getSocket } from "@/lib/socket";

export default function SessionsPage() {
  const { connected: isConnected } = useConnectionStore();
  const {
    selectedSession,
    setLoading,
    clearActionResult,
  } = useSessionStore();

  const socket = getSocket();

  // Load sessions on mount and when connected
  useEffect(() => {
    if (isConnected) {
      setLoading(true);
      socket.emit("request_sessions", {});
    }
  }, [isConnected, socket, setLoading]);

  // Load session detail when selected
  useEffect(() => {
    if (selectedSession && isConnected) {
      socket.emit("request_session_detail", { filepath: selectedSession.filepath });
    }
  }, [selectedSession, isConnected, socket]);

  // Clear action feedback after 3 seconds
  const { lastAction } = useSessionStore();
  useEffect(() => {
    if (lastAction.type) {
      const timer = setTimeout(() => {
        clearActionResult();
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [lastAction, clearActionResult]);

  const handleRefresh = useCallback(() => {
    setLoading(true);
    socket.emit("request_sessions", {});
  }, [socket, setLoading]);

  const handleCreateSession = useCallback(() => {
    socket.emit("create_session");
  }, [socket]);

  const handleResume = useCallback(
    (filepath: string) => {
      // Send filename only — backend resolves to full path
      const filename = filepath.split(/[\\/]/).pop() ?? filepath;
      socket.emit("resume_session", { filename });
    },
    [socket]
  );

  const handleDelete = useCallback(
    (filepath: string) => {
      socket.emit("delete_session", { filepath });
    },
    [socket]
  );

  const handleGenerateSummary = useCallback(
    (filepath: string) => {
      socket.emit("generate_session_summary", { filepath });
    },
    [socket]
  );

  const handleExport = useCallback((filepath: string) => {
    // Create a download link for the JSONL file
    // Extract the filename from the path
    const filename = filepath.split(/[\\/]/).pop() || "session.jsonl";

    // Get the session detail from store and create a blob
    const { selectedSessionTurns } = useSessionStore.getState();
    if (selectedSessionTurns.length > 0) {
      const jsonl = selectedSessionTurns
        .map((turn) => JSON.stringify(turn))
        .join("\n");
      const blob = new Blob([jsonl], { type: "application/jsonl" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  }, []);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold">Sessions</h1>
        <p className="text-muted-foreground">
          Browse and manage conversation sessions
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-1">
          <SessionList onRefresh={handleRefresh} onCreateSession={handleCreateSession} />
        </div>
        <div className="lg:col-span-2">
          <SessionViewer
            onResume={handleResume}
            onDelete={handleDelete}
            onExport={handleExport}
            onGenerateSummary={handleGenerateSummary}
          />
        </div>
      </div>
    </div>
  );
}
