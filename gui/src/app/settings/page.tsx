"use client";

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

export default function SettingsPage() {
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
