"use client";

import { useEffect } from "react";
import {
  VADSettings,
  ProviderDisplay,
  PipelineSettings,
  AvatarSettings,
  BaseAnimationsSettings,
  EmotionClassifierSettings,
  TwitchCredentials,
  GameStateBridge,
  Qwen3TTSSettings,
} from "@/components/settings";
import { useLauncherStore } from "@/lib/stores/launcher-store";
import type { HydrateConfig } from "@/lib/stores/launcher-store";

export default function SettingsPage() {
  const { hydrate, setIsLoading } = useLauncherStore();

  useEffect(() => {
    async function loadConfig() {
      try {
        const response = await fetch("/api/launcher/write-config");
        const result = await response.json();
        if (result.exists && result.config) {
          hydrate(result.config as HydrateConfig);
        } else {
          setIsLoading(false);
        }
      } catch {
        setIsLoading(false);
      }
    }
    loadConfig();
  }, [hydrate, setIsLoading]);

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-2xl font-bold">Settings</h1>
        <p className="text-muted-foreground">
          Runtime configuration for voice, pipeline, providers, and avatar.
        </p>
      </div>

      <VADSettings />
      <PipelineSettings />
      <ProviderDisplay />
      <Qwen3TTSSettings />
      <AvatarSettings />
      <BaseAnimationsSettings />
      <EmotionClassifierSettings />
      <TwitchCredentials />
      <GameStateBridge />
    </div>
  );
}
