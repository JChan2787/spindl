"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Gamepad2, Network, CheckCircle2, XCircle, Loader2 } from "lucide-react";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { CollapsibleCard } from "@/components/ui/collapsible-card";
import { useSettingsStore, selectEffectiveStimuliConfig } from "@/lib/stores";
import { getSocket } from "@/lib/socket";
import type { GameStateConnectionResult } from "@/types/events";

type TestResult = GameStateConnectionResult | null;

export function GameStateBridge() {
  const store = useSettingsStore();
  const effectiveConfig = selectEffectiveStimuliConfig(store);
  const { updatePendingStimuli, setSavingStimuli } = store;
  const setGameStateStatus = useSettingsStore((s) => s.setGameStateStatus);
  const socket = getSocket();

  const debounceRef = useRef<NodeJS.Timeout | null>(null);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<TestResult>(null);

  const emitChanges = useCallback(
    (changes: Partial<import("@/types/events").SetStimuliConfigPayload>) => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
      debounceRef.current = setTimeout(() => {
        setSavingStimuli(true);
        socket.emit("set_stimuli_config", changes);
      }, 300);
    },
    [socket, setSavingStimuli]
  );

  useEffect(() => {
    return () => {
      if (debounceRef.current) {
        clearTimeout(debounceRef.current);
      }
    };
  }, []);

  // Local state for text inputs — emit on blur
  const [localHost, setLocalHost] = useState(effectiveConfig.game_state_host);
  const hostSyncedRef = useRef(effectiveConfig.game_state_host);
  const [localPort, setLocalPort] = useState(String(effectiveConfig.game_state_port));
  const portSyncedRef = useRef(String(effectiveConfig.game_state_port));

  // Sync local state when backend config changes
  useEffect(() => {
    if (effectiveConfig.game_state_host !== hostSyncedRef.current) {
      setLocalHost(effectiveConfig.game_state_host);
      hostSyncedRef.current = effectiveConfig.game_state_host;
    }
    const portStr = String(effectiveConfig.game_state_port);
    if (portStr !== portSyncedRef.current) {
      setLocalPort(portStr);
      portSyncedRef.current = portStr;
    }
  }, [effectiveConfig.game_state_host, effectiveConfig.game_state_port]);

  const handleHostBlur = useCallback(() => {
    if (localHost !== hostSyncedRef.current) {
      updatePendingStimuli({ game_state_host: localHost });
      emitChanges({ game_state_host: localHost });
      hostSyncedRef.current = localHost;
    }
  }, [localHost, updatePendingStimuli, emitChanges]);

  const handlePortBlur = useCallback(() => {
    if (localPort !== portSyncedRef.current) {
      const portNum = parseInt(localPort, 10) || 53817;
      updatePendingStimuli({ game_state_port: portNum });
      emitChanges({ game_state_port: portNum });
      portSyncedRef.current = localPort;
    }
  }, [localPort, updatePendingStimuli, emitChanges]);

  // Test connection handler
  const handleTestConnection = useCallback(() => {
    setTesting(true);
    setTestResult(null);
    const portNum = parseInt(localPort, 10) || 53817;
    socket.emit("test_game_state_connection", {
      host: localHost,
      port: portNum,
    });
  }, [socket, localHost, localPort]);

  // Listen for test result — persist success to store so Dashboard toggle unlocks
  useEffect(() => {
    const handler = (data: GameStateConnectionResult) => {
      setTesting(false);
      setTestResult(data);
      if (data.success) {
        setGameStateStatus({
          connected: true,
          protocol_version: data.protocol_version ?? null,
          buffer_count: 0,
          recent_lines: [],
          enabled: false,
          dialogue_enabled: false,
          current_summary: "",
          gameplay_enabled: false,
          gameplay_event_buffer_count: 0,
          gameplay_snapshot_probability: 0,
        });
        try { localStorage.setItem("game_bridge_verified", "true"); } catch {}
      }
    };
    socket.on("game_state_connection_result", handler);
    return () => { socket.off("game_state_connection_result", handler); };
  }, [socket, setGameStateStatus]);

  // Clear test result when connection params change
  useEffect(() => {
    setTestResult(null);
  }, [localHost, localPort]);

  const hasAddress = localHost.trim() !== "";

  return (
    <CollapsibleCard
      id="game-state-bridge"
      title="Game Bridge"
      icon={<Gamepad2 className="h-4 w-4" />}
    >
      <div className="space-y-3">
        <p className="text-xs text-muted-foreground">
          TCP connection to the SpindL Game Bridge (SPNDL-001). The bridge
          sends in-game dialogue and gameplay events for AI co-host commentary.
        </p>

        {/* NANO-133: Game Integration Profile */}
        <div className="space-y-1">
          <Label className="text-xs">Game Integration</Label>
          <select
            value={effectiveConfig.game_state_profile}
            onChange={(e) => {
              updatePendingStimuli({ game_state_profile: e.target.value });
              emitChanges({ game_state_profile: e.target.value });
            }}
            className="flex h-8 w-full rounded-md border border-input bg-background px-3 py-1 text-xs shadow-sm transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-ring"
          >
            <option value="none">None</option>
            <option value="pragmata">Pragmata</option>
          </select>
          <p className="text-[10px] text-muted-foreground">
            Selects game-specific tools for the LLM (e.g. hack status queries).
          </p>
        </div>

        <div className="grid grid-cols-[1fr_80px] gap-2">
          <div className="space-y-1">
            <Label className="flex items-center gap-2 text-xs">
              <Network className="h-3 w-3" />
              Host
            </Label>
            <Input
              value={localHost}
              onChange={(e) => setLocalHost(e.target.value)}
              onBlur={handleHostBlur}
              placeholder="127.0.0.1"
              className="text-xs h-8"
            />
          </div>
          <div className="space-y-1">
            <Label className="text-xs">Port</Label>
            <Input
              value={localPort}
              onChange={(e) => setLocalPort(e.target.value)}
              onBlur={handlePortBlur}
              placeholder="53817"
              className="text-xs h-8"
            />
          </div>
        </div>

        {/* Test Connection */}
        <div className="flex items-center gap-3 pt-1">
          <Button
            variant="outline"
            size="sm"
            onClick={handleTestConnection}
            disabled={!hasAddress || testing}
            className="text-xs h-7"
          >
            {testing ? (
              <>
                <Loader2 className="h-3 w-3 mr-1.5 animate-spin" />
                Testing...
              </>
            ) : (
              "Test Connection"
            )}
          </Button>

          {testResult && (
            <span className="flex items-center gap-1 text-xs">
              {testResult.success ? (
                <>
                  <CheckCircle2 className="h-3.5 w-3.5 text-green-500" />
                  <span className="text-green-500">
                    Connected{testResult.protocol_version ? ` (v${testResult.protocol_version})` : ""}
                  </span>
                </>
              ) : (
                <>
                  <XCircle className="h-3.5 w-3.5 text-red-500" />
                  <span className="text-red-500">{testResult.error}</span>
                </>
              )}
            </span>
          )}
        </div>
      </div>
    </CollapsibleCard>
  );
}
