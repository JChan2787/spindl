"use client";

import { useCallback, useEffect } from "react";
import { User, Clock, Subtitles, Pin, Gamepad2, Loader2, Download } from "lucide-react";
import { Label } from "@/components/ui/label";
import { CollapsibleCard } from "@/components/ui/collapsible-card";
import { getSocket } from "@/lib/socket";
import { useSettingsStore, useConnectionStore } from "@/lib/stores";

export function AvatarSettings() {
  const avatarConfig = useSettingsStore((s) => s.avatarConfig);
  const connected = useConnectionStore((s) => s.connected);

  // Check install status on mount
  useEffect(() => {
    if (connected) {
      const socket = getSocket();
      socket.emit("check_tauri_install", {});
    }
  }, [connected]);

  const handleInstall = useCallback(() => {
    const socket = getSocket();
    socket.emit("install_tauri_apps", {});
  }, []);

  const handleToggle = useCallback(() => {
    const socket = getSocket();
    socket.emit("set_avatar_config", { enabled: !avatarConfig.enabled });
  }, [avatarConfig.enabled]);

  const handleSubtitlesToggle = useCallback(() => {
    const socket = getSocket();
    socket.emit("set_avatar_config", { subtitles_enabled: !avatarConfig.subtitles_enabled });
  }, [avatarConfig.subtitles_enabled]);

  const handleStreamDeckToggle = useCallback(() => {
    const socket = getSocket();
    socket.emit("set_avatar_config", { stream_deck_enabled: !avatarConfig.stream_deck_enabled });
  }, [avatarConfig.stream_deck_enabled]);

  const handleSubtitleFadeChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const socket = getSocket();
      const value = parseFloat(e.target.value);
      socket.emit("set_avatar_config", { subtitle_fade_delay: value });
    },
    [],
  );

  const handleFadeDelayChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const socket = getSocket();
      const value = parseFloat(e.target.value);
      socket.emit("set_avatar_config", { expression_fade_delay: value });
    },
    [],
  );

  const handleAvatarOnTopToggle = useCallback(() => {
    const socket = getSocket();
    socket.emit("set_avatar_config", { avatar_always_on_top: !avatarConfig.avatar_always_on_top });
  }, [avatarConfig.avatar_always_on_top]);

  const handleSubtitleOnTopToggle = useCallback(() => {
    const socket = getSocket();
    socket.emit("set_avatar_config", { subtitle_always_on_top: !avatarConfig.subtitle_always_on_top });
  }, [avatarConfig.subtitle_always_on_top]);

  const notInstalled = !avatarConfig.tauri_installed;
  const installing = avatarConfig.tauri_installing;

  return (
    <CollapsibleCard
      id="avatar"
      title="Avatar"
      icon={<User className="h-4 w-4" />}
    >
      {/* NANO-110: Install banner — shown when binaries don't exist */}
      {notInstalled && (
        <div className="rounded-md border border-border bg-muted/30 p-3 space-y-2">
          {installing ? (
            <>
              <div className="flex items-center gap-2">
                <Loader2 className="h-4 w-4 animate-spin text-primary" />
                <span className="text-sm font-medium">Installing overlay apps...</span>
              </div>
              <p className="text-xs text-muted-foreground">
                {avatarConfig.tauri_install_message || "Preparing build..."}
              </p>
            </>
          ) : (
            <>
              <p className="text-xs text-muted-foreground">
                Overlay windows (avatar, subtitles, stream deck) need to be compiled before first use.
                This is a one-time install that takes a few minutes.
              </p>
              <button
                onClick={handleInstall}
                className="flex items-center gap-2 rounded-md bg-primary px-3 py-1.5 text-xs font-medium text-primary-foreground hover:bg-primary/90 transition-colors"
              >
                <Download className="h-3 w-3" />
                Install Overlay Apps
              </button>
            </>
          )}
        </div>
      )}

      {/* Enable toggle */}
      <div className="flex items-center justify-between">
        <Label className="flex items-center gap-2 text-sm">
          Enable Avatar Bridge
          {avatarConfig.enabled && (
            <span className="flex items-center gap-1 ml-1">
              <span
                className={`inline-block h-2 w-2 rounded-full ${
                  avatarConfig.avatar_connected ? "bg-green-500" : "bg-muted-foreground/40"
                }`}
              />
              <span className="text-xs text-muted-foreground">
                {avatarConfig.avatar_connected ? "Connected" : "Disconnected"}
              </span>
            </span>
          )}
        </Label>
        <button
          onClick={handleToggle}
          disabled={notInstalled}
          className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
            avatarConfig.enabled ? "bg-primary" : "bg-muted"
          }`}
        >
          <span
            className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${
              avatarConfig.enabled ? "translate-x-4" : "translate-x-0.5"
            }`}
          />
        </button>
      </div>

      {/* Subtitles toggle (NANO-100) */}
      <div className="flex items-center justify-between">
        <Label className="flex items-center gap-2 text-sm">
          <Subtitles className="h-3.5 w-3.5" />
          Show Subtitles
        </Label>
        <button
          onClick={handleSubtitlesToggle}
          disabled={notInstalled || !avatarConfig.enabled}
          className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
            avatarConfig.subtitles_enabled ? "bg-primary" : "bg-muted"
          }`}
        >
          <span
            className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${
              avatarConfig.subtitles_enabled ? "translate-x-4" : "translate-x-0.5"
            }`}
          />
        </button>
      </div>

      {/* Stream Deck toggle (NANO-110) */}
      <div className="flex items-center justify-between">
        <Label className="flex items-center gap-2 text-sm">
          <Gamepad2 className="h-3.5 w-3.5" />
          Show Stream Deck
        </Label>
        <button
          onClick={handleStreamDeckToggle}
          disabled={notInstalled}
          className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
            avatarConfig.stream_deck_enabled ? "bg-primary" : "bg-muted"
          }`}
        >
          <span
            className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${
              avatarConfig.stream_deck_enabled ? "translate-x-4" : "translate-x-0.5"
            }`}
          />
        </button>
      </div>

      {/* Subtitle fade delay slider (NANO-100) */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label className="flex items-center gap-2 text-sm">
            <Subtitles className="h-3.5 w-3.5" />
            Subtitle Fade Delay
          </Label>
          <span className="text-sm font-mono text-muted-foreground">
            {avatarConfig.subtitle_fade_delay.toFixed(1)}s
          </span>
        </div>
        <input
          type="range"
          min={0}
          max={5}
          step={0.1}
          value={avatarConfig.subtitle_fade_delay}
          onChange={handleSubtitleFadeChange}
          disabled={!avatarConfig.subtitles_enabled}
          className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer accent-primary disabled:opacity-50 disabled:cursor-not-allowed"
        />
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>0s</span>
          <span>5s</span>
        </div>
        <p className="text-xs text-muted-foreground">
          Time after TTS ends before subtitle text fades out
        </p>
      </div>

      {/* Expression fade delay slider */}
      <div className="space-y-2">
        <div className="flex items-center justify-between">
          <Label className="flex items-center gap-2 text-sm">
            <Clock className="h-3.5 w-3.5" />
            Expression Fade Delay
          </Label>
          <span className="text-sm font-mono text-muted-foreground">
            {avatarConfig.expression_fade_delay.toFixed(1)}s
          </span>
        </div>
        <input
          type="range"
          min={0}
          max={5}
          step={0.1}
          value={avatarConfig.expression_fade_delay}
          onChange={handleFadeDelayChange}
          disabled={!avatarConfig.enabled}
          className="w-full h-2 bg-muted rounded-lg appearance-none cursor-pointer accent-primary disabled:opacity-50 disabled:cursor-not-allowed"
        />
        <div className="flex justify-between text-xs text-muted-foreground">
          <span>0s</span>
          <span>5s</span>
        </div>
        <p className="text-xs text-muted-foreground">
          Time after TTS ends before face and body expressions fade to neutral
        </p>
      </div>

      {/* Always On Top toggles */}
      <div className="flex items-center justify-between">
        <Label className="flex items-center gap-2 text-sm">
          <Pin className="h-3.5 w-3.5" />
          Avatar Always On Top
        </Label>
        <button
          onClick={handleAvatarOnTopToggle}
          disabled={!avatarConfig.enabled}
          className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
            avatarConfig.avatar_always_on_top ? "bg-primary" : "bg-muted"
          }`}
        >
          <span
            className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${
              avatarConfig.avatar_always_on_top ? "translate-x-4" : "translate-x-0.5"
            }`}
          />
        </button>
      </div>

      <div className="flex items-center justify-between">
        <Label className="flex items-center gap-2 text-sm">
          <Pin className="h-3.5 w-3.5" />
          Subtitle Always On Top
        </Label>
        <button
          onClick={handleSubtitleOnTopToggle}
          disabled={!avatarConfig.subtitles_enabled}
          className={`relative inline-flex h-5 w-9 items-center rounded-full transition-colors disabled:opacity-50 disabled:cursor-not-allowed ${
            avatarConfig.subtitle_always_on_top ? "bg-primary" : "bg-muted"
          }`}
        >
          <span
            className={`inline-block h-3.5 w-3.5 transform rounded-full bg-white transition-transform ${
              avatarConfig.subtitle_always_on_top ? "translate-x-4" : "translate-x-0.5"
            }`}
          />
        </button>
      </div>
    </CollapsibleCard>
  );
}
