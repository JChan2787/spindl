"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Music, Upload, X } from "lucide-react";
import { Label } from "@/components/ui/label";
import { CollapsibleCard } from "@/components/ui/collapsible-card";
import { getSocket } from "@/lib/socket";
import {
  fetchBaseAnimations,
  uploadBaseAnimation,
  clearBaseAnimation,
} from "@/lib/stores";
import type { BaseAnimationsConfig } from "@/lib/stores";

const SLOTS = ["idle", "happy", "sad", "angry", "curious"] as const;
type Slot = (typeof SLOTS)[number];

export function BaseAnimationsSettings() {
  const [config, setConfig] = useState<BaseAnimationsConfig | null>(null);
  const [uploading, setUploading] = useState<Slot | null>(null);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const activeSlotRef = useRef<Slot | null>(null);

  // Load config on mount
  useEffect(() => {
    fetchBaseAnimations()
      .then(setConfig)
      .catch((e) => setError(e.message));
  }, []);

  const handleBrowse = useCallback((slot: Slot) => {
    activeSlotRef.current = slot;
    fileInputRef.current?.click();
  }, []);

  const handleFileSelected = useCallback(
    async (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0];
      const slot = activeSlotRef.current;
      if (!file || !slot) return;

      // Reset input so the same file can be re-selected
      e.target.value = "";

      setUploading(slot);
      setError(null);

      try {
        const result = await uploadBaseAnimation(slot, file);
        setConfig((prev) =>
          prev ? { ...prev, [slot]: result.clip_name } : prev
        );

        // Notify avatar renderer to rescan animations
        const socket = getSocket();
        socket.emit("avatar_rescan_animations");
      } catch (err) {
        setError(err instanceof Error ? err.message : "Upload failed");
      } finally {
        setUploading(null);
      }
    },
    []
  );

  const handleClear = useCallback(async (slot: Slot) => {
    setError(null);

    try {
      await clearBaseAnimation(slot);
      setConfig((prev) => (prev ? { ...prev, [slot]: null } : prev));

      // Notify avatar renderer to rescan animations
      const socket = getSocket();
      socket.emit("avatar_rescan_animations");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Clear failed");
    }
  }, []);

  if (!config) {
    return null;
  }

  return (
    <CollapsibleCard
      id="base-animations"
      title="Base Animations"
      icon={<Music className="h-4 w-4" />}
    >
      <p className="text-xs text-muted-foreground mb-4">
        Pick Mixamo FBX files (Without Skin, 30fps). These become the global
        defaults for all characters without per-character overrides.
      </p>

      {error && (
        <p className="text-xs text-red-400 mb-3">{error}</p>
      )}

      <div className="space-y-3">
        {SLOTS.map((slot) => {
          const clipName = config[slot];
          const isUploading = uploading === slot;

          return (
            <div key={slot} className="flex items-center gap-3">
              <Label className="w-14 text-sm capitalize shrink-0">
                {slot}:
              </Label>

              <div className="flex-1 rounded-md border border-border bg-background px-3 py-1.5 text-sm text-muted-foreground truncate">
                {clipName ?? "None"}
              </div>

              <button
                type="button"
                className="inline-flex items-center gap-1.5 rounded-md border border-border px-3 py-1.5 text-xs hover:bg-accent disabled:opacity-50"
                onClick={() => handleBrowse(slot)}
                disabled={isUploading}
              >
                <Upload className="h-3 w-3" />
                {isUploading ? "Uploading…" : "Browse"}
              </button>

              {clipName && (
                <button
                  type="button"
                  className="inline-flex items-center gap-1 rounded-md border border-border px-2 py-1.5 text-xs text-red-400 hover:bg-red-400/10"
                  onClick={() => handleClear(slot)}
                >
                  <X className="h-3 w-3" />
                  Clear
                </button>
              )}
            </div>
          );
        })}
      </div>

      {/* Hidden file input shared across all slots */}
      <input
        ref={fileInputRef}
        type="file"
        accept=".fbx"
        className="hidden"
        onChange={handleFileSelected}
      />
    </CollapsibleCard>
  );
}
