"use client";

import { useRouter } from "next/navigation";
import { useConnectionStore, useAgentStore } from "@/lib/stores";
import { Button } from "@/components/ui/button";
import { Loader2 } from "lucide-react";

interface StatusOverlayProps {
  /** When true, only show the "Connection Lost" state (skip pre-launch). */
  disconnectedOnly?: boolean;
}

export function StatusOverlay({ disconnectedOnly = false }: StatusOverlayProps) {
  const router = useRouter();
  const { connected, connecting } = useConnectionStore();
  const { config } = useAgentStore();

  const isDisconnected = !connected;

  // In disconnected-only mode (Launcher), only show when backend is gone
  if (disconnectedOnly) {
    if (connected) return null;
  } else {
    // Dashboard mode: hide when orchestrator is live (connected + config loaded)
    if (connected && config) return null;
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-background/80 backdrop-blur-sm">
      <div className="flex flex-col items-center gap-6 text-center max-w-md px-6">
        {/* SpindL Icon */}
        <img
          src="/spindl-icon.png"
          alt="SpindL"
          className={`w-20 h-20 rounded-2xl ${isDisconnected ? "animate-pulse" : ""}`}
        />

        {isDisconnected ? (
          <>
            <div className="space-y-2">
              <h2 className="text-2xl font-bold">Connection Lost</h2>
              <p className="text-muted-foreground">
                The backend is unreachable. Restart the backend server and this
                page will reconnect automatically.
              </p>
            </div>
            {connecting && (
              <div className="flex items-center gap-2 text-sm text-muted-foreground">
                <Loader2 className="h-4 w-4 animate-spin" />
                Reconnecting...
              </div>
            )}
          </>
        ) : (
          <>
            <div className="space-y-2">
              <h2 className="text-2xl font-bold">Services Not Running</h2>
              <p className="text-muted-foreground">
                Head to the Launcher to start your services.
              </p>
            </div>
            <Button onClick={() => router.push("/launcher")}>
              Go to Launcher
            </Button>
          </>
        )}
      </div>
    </div>
  );
}
