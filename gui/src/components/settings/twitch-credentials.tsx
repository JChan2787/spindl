"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Tv, Hash, Key, CheckCircle2, XCircle, Loader2 } from "lucide-react";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { CollapsibleCard } from "@/components/ui/collapsible-card";
import { useSettingsStore, selectEffectiveStimuliConfig } from "@/lib/stores";
import { getSocket } from "@/lib/socket";

type TestResult = { success: boolean; error: string | null } | null;

export function TwitchCredentials() {
  const store = useSettingsStore();
  const effectiveConfig = selectEffectiveStimuliConfig(store);
  const { updatePendingStimuli, setSavingStimuli } = store;
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
  const [localChannel, setLocalChannel] = useState(effectiveConfig.twitch_channel);
  const channelSyncedRef = useRef(effectiveConfig.twitch_channel);
  const [localAppId, setLocalAppId] = useState(effectiveConfig.twitch_app_id);
  const appIdSyncedRef = useRef(effectiveConfig.twitch_app_id);
  const [localAppSecret, setLocalAppSecret] = useState(effectiveConfig.twitch_app_secret);
  const appSecretSyncedRef = useRef(effectiveConfig.twitch_app_secret);

  // Sync local state when backend config changes
  useEffect(() => {
    if (effectiveConfig.twitch_channel !== channelSyncedRef.current) {
      setLocalChannel(effectiveConfig.twitch_channel);
      channelSyncedRef.current = effectiveConfig.twitch_channel;
    }
    if (effectiveConfig.twitch_app_id !== appIdSyncedRef.current) {
      setLocalAppId(effectiveConfig.twitch_app_id);
      appIdSyncedRef.current = effectiveConfig.twitch_app_id;
    }
    if (effectiveConfig.twitch_app_secret !== appSecretSyncedRef.current) {
      setLocalAppSecret(effectiveConfig.twitch_app_secret);
      appSecretSyncedRef.current = effectiveConfig.twitch_app_secret;
    }
  }, [effectiveConfig.twitch_channel, effectiveConfig.twitch_app_id, effectiveConfig.twitch_app_secret]);

  const handleChannelBlur = useCallback(() => {
    if (localChannel !== channelSyncedRef.current) {
      updatePendingStimuli({ twitch_channel: localChannel });
      emitChanges({ twitch_channel: localChannel });
      channelSyncedRef.current = localChannel;
    }
  }, [localChannel, updatePendingStimuli, emitChanges]);

  const handleAppIdBlur = useCallback(() => {
    if (localAppId !== appIdSyncedRef.current) {
      updatePendingStimuli({ twitch_app_id: localAppId });
      emitChanges({ twitch_app_id: localAppId });
      appIdSyncedRef.current = localAppId;
    }
  }, [localAppId, updatePendingStimuli, emitChanges]);

  const handleAppSecretBlur = useCallback(() => {
    if (localAppSecret !== appSecretSyncedRef.current) {
      updatePendingStimuli({ twitch_app_secret: localAppSecret });
      emitChanges({ twitch_app_secret: localAppSecret });
      appSecretSyncedRef.current = localAppSecret;
    }
  }, [localAppSecret, updatePendingStimuli, emitChanges]);

  // Test connection handler
  const handleTestConnection = useCallback(() => {
    setTesting(true);
    setTestResult(null);
    socket.emit("test_twitch_credentials", {
      app_id: localAppId,
      app_secret: localAppSecret,
      channel: localChannel,
    });
  }, [socket, localAppId, localAppSecret, localChannel]);

  // Listen for test result
  const setStimuliConfig = useSettingsStore((s) => s.setStimuliConfig);

  useEffect(() => {
    const handler = (data: { success: boolean; error: string | null; has_credentials?: boolean }) => {
      setTesting(false);
      setTestResult(data);
      // Push credential gate flag into store so dashboard toggle unlocks
      if (data.success && data.has_credentials) {
        const current = selectEffectiveStimuliConfig(useSettingsStore.getState());
        setStimuliConfig({ ...current, twitch_has_credentials: true });
      }
    };
    socket.on("twitch_credentials_result", handler);
    return () => { socket.off("twitch_credentials_result", handler); };
  }, [socket, setStimuliConfig]);

  // Clear test result when credentials change
  useEffect(() => {
    setTestResult(null);
  }, [localAppId, localAppSecret, localChannel]);

  const hasCredentials =
    localAppId.trim() !== "" && localAppSecret.trim() !== "";

  return (
    <CollapsibleCard
      id="twitch-credentials"
      title="Twitch Integration"
      icon={<Tv className="h-4 w-4" />}
    >
      <div className="space-y-3">
        <p className="text-xs text-muted-foreground">
          Twitch developer credentials for chat integration. Register an app at{" "}
          <span className="font-mono">dev.twitch.tv</span> to obtain these values.
        </p>

        <div className="space-y-1">
          <Label className="flex items-center gap-2 text-xs">
            <Hash className="h-3 w-3" />
            Channel
          </Label>
          <Input
            value={localChannel}
            onChange={(e) => setLocalChannel(e.target.value)}
            onBlur={handleChannelBlur}
            placeholder="login name only — no URLs"
            className="text-xs h-8"
          />
        </div>

        <div className="space-y-1">
          <Label className="flex items-center gap-2 text-xs">
            <Key className="h-3 w-3" />
            App ID
          </Label>
          <Input
            value={localAppId}
            onChange={(e) => setLocalAppId(e.target.value)}
            onBlur={handleAppIdBlur}
            placeholder="Twitch app ID"
            className="text-xs h-8"
          />
        </div>

        <div className="space-y-1">
          <Label className="flex items-center gap-2 text-xs">
            <Key className="h-3 w-3" />
            App Secret
          </Label>
          <Input
            type="password"
            value={localAppSecret}
            onChange={(e) => setLocalAppSecret(e.target.value)}
            onBlur={handleAppSecretBlur}
            placeholder="Twitch app secret"
            className="text-xs h-8"
          />
        </div>

        {/* Test Connection */}
        <div className="flex items-center gap-3 pt-1">
          <Button
            variant="outline"
            size="sm"
            onClick={handleTestConnection}
            disabled={!hasCredentials || testing}
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
                  <span className="text-green-500">Connected</span>
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
