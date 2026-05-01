"use client";

import { useState, useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { Textarea } from "@/components/ui/textarea";
import { Separator } from "@/components/ui/separator";
import type { BlockInfo } from "@/types/events";
import { Settings2, Zap, FileText, Clock } from "lucide-react";

/** IDs of blocks whose content is injected after preprocessing (deferred) */
const DEFERRED_BLOCK_IDS = new Set([
  "codex_context",
  "rag_context",
  "recent_history",
]);

/** Voice state trigger definitions with labels and defaults */
const VOICE_STATE_TRIGGERS = [
  {
    key: "barge_in",
    label: "Barge-In (User Interrupts)",
    defaultText: "The User interrupted you mid-sentence.",
  },
  {
    key: "empty_transcription",
    label: "Empty Transcription (Sound, No Words)",
    defaultText: "The User made a sound but no words were detected.",
  },
  {
    key: "error",
    label: "Error",
    defaultText: "An error occurred. Acknowledge briefly and continue.",
  },
] as const;

interface BlockDetailProps {
  /** The selected block's metadata */
  block: BlockInfo;
  /** Current override content (from pending or server config). null = no override */
  currentOverride: string | null;
  /** Token count for this block from the latest snapshot */
  tokenCount: number | null;
  /** Rendered content from the latest prompt snapshot */
  currentContent: string | null;
  /** Callback to set/clear an override. null content = remove override */
  onSetOverride: (blockId: string, content: string | null) => void;
  /** Per-trigger overrides for voice_state block (compound keys resolved) */
  triggerOverrides?: Record<string, string | null>;
}

/** Per-trigger override editor for the voice_state block */
function VoiceStateTriggerEditor({
  triggerOverrides,
  onSetOverride,
}: {
  triggerOverrides: Record<string, string | null>;
  onSetOverride: (blockId: string, content: string | null) => void;
}) {
  const [localEdits, setLocalEdits] = useState<Record<string, string>>({});
  const [enabledTriggers, setEnabledTriggers] = useState<Record<string, boolean>>({});

  // Sync enabled state from server overrides on mount / change
  useEffect(() => {
    const enabled: Record<string, boolean> = {};
    for (const trigger of VOICE_STATE_TRIGGERS) {
      enabled[trigger.key] = triggerOverrides[trigger.key] != null;
    }
    setEnabledTriggers(enabled);
    // Seed local edits from current overrides
    const edits: Record<string, string> = {};
    for (const trigger of VOICE_STATE_TRIGGERS) {
      edits[trigger.key] = triggerOverrides[trigger.key] ?? "";
    }
    setLocalEdits(edits);
  }, [triggerOverrides]);

  const compoundKey = (triggerKey: string) => `voice_state:${triggerKey}`;

  const handleToggle = (triggerKey: string, checked: boolean) => {
    setEnabledTriggers((prev) => ({ ...prev, [triggerKey]: checked }));
    if (!checked) {
      setLocalEdits((prev) => ({ ...prev, [triggerKey]: "" }));
      onSetOverride(compoundKey(triggerKey), null);
    }
  };

  const handleApply = (triggerKey: string) => {
    const trimmed = (localEdits[triggerKey] ?? "").trim();
    onSetOverride(compoundKey(triggerKey), trimmed || null);
  };

  const handleRevert = (triggerKey: string) => {
    setLocalEdits((prev) => ({
      ...prev,
      [triggerKey]: triggerOverrides[triggerKey] ?? "",
    }));
  };

  return (
    <div className="space-y-4">
      {VOICE_STATE_TRIGGERS.map((trigger) => {
        const isEnabled = enabledTriggers[trigger.key] ?? false;
        const serverVal = triggerOverrides[trigger.key] ?? "";
        const localVal = localEdits[trigger.key] ?? "";
        const hasUnsaved = isEnabled && localVal !== serverVal;

        return (
          <div
            key={trigger.key}
            className="rounded-md border border-border/50 p-3 space-y-2"
          >
            <div className="flex items-center justify-between">
              <Label className="text-sm font-medium">{trigger.label}</Label>
              <Switch
                checked={isEnabled}
                onCheckedChange={(checked) =>
                  handleToggle(trigger.key, checked)
                }
              />
            </div>

            {/* Always show the default */}
            <div className="rounded-md bg-muted/20 p-2">
              <Label className="text-xs text-muted-foreground">Default</Label>
              <p className="text-xs font-mono mt-1 text-muted-foreground">
                {trigger.defaultText}
              </p>
            </div>

            {isEnabled && (
              <div className="space-y-2">
                <Textarea
                  value={localVal}
                  onChange={(e) =>
                    setLocalEdits((prev) => ({
                      ...prev,
                      [trigger.key]: e.target.value,
                    }))
                  }
                  placeholder={trigger.defaultText}
                  className="min-h-[80px] font-mono text-sm"
                />
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">
                    {localVal.length.toLocaleString()} characters
                  </span>
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handleRevert(trigger.key)}
                      disabled={!hasUnsaved}
                    >
                      Revert
                    </Button>
                    <Button
                      size="sm"
                      onClick={() => handleApply(trigger.key)}
                      disabled={!hasUnsaved}
                    >
                      Apply
                    </Button>
                  </div>
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

export function BlockDetail({
  block,
  currentOverride,
  tokenCount,
  currentContent,
  onSetOverride,
  triggerOverrides,
}: BlockDetailProps) {
  const [overrideEnabled, setOverrideEnabled] = useState(
    currentOverride !== null
  );
  const [editContent, setEditContent] = useState(currentOverride ?? "");

  // Sync local state when selected block or override changes
  useEffect(() => {
    setOverrideEnabled(currentOverride !== null);
    setEditContent(currentOverride ?? "");
  }, [currentOverride, block.id]);

  const isDeferred = DEFERRED_BLOCK_IDS.has(block.id);
  const isVoiceState = block.id === "voice_state";
  const hasUnsavedEdit =
    overrideEnabled && editContent !== (currentOverride ?? "");

  const handleToggleOverride = (checked: boolean) => {
    setOverrideEnabled(checked);
    if (!checked) {
      // Clear override
      setEditContent("");
      onSetOverride(block.id, null);
    }
  };

  const handleApply = () => {
    const trimmed = editContent.trim();
    onSetOverride(block.id, trimmed || null);
  };

  const handleRevert = () => {
    setEditContent(currentOverride ?? "");
  };

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="text-sm font-medium flex items-center gap-2">
            <Settings2 className="h-4 w-4" />
            {block.label}
          </CardTitle>
          <div className="flex items-center gap-2">
            {block.section_header && (
              <Badge variant="outline" className="text-xs">
                {block.section_header}
              </Badge>
            )}
            {block.is_static ? (
              <Badge variant="secondary" className="text-xs">
                <FileText className="h-3 w-3 mr-1" />
                Static
              </Badge>
            ) : isDeferred ? (
              <Badge variant="secondary" className="text-xs">
                <Clock className="h-3 w-3 mr-1" />
                Deferred
              </Badge>
            ) : (
              <Badge variant="secondary" className="text-xs">
                <Zap className="h-3 w-3 mr-1" />
                Provider
              </Badge>
            )}
          </div>
        </div>
        {tokenCount !== null && (
          <div className="text-xs text-muted-foreground mt-1">
            {tokenCount.toLocaleString()} tokens in latest snapshot
          </div>
        )}
      </CardHeader>

      <CardContent className="space-y-4">
        {/* Block info */}
        <div className="rounded-md bg-muted/30 p-3 text-sm text-muted-foreground">
          {block.is_static ? (
            <p>
              Static content block. Content is defined in the prompt template
              and does not come from a provider.
            </p>
          ) : isDeferred ? (
            <p>
              Deferred injection block. Content is populated after
              preprocessing (e.g., RAG retrieval, codex lookup, or history
              formatting) and injected into the prompt at runtime.
            </p>
          ) : isVoiceState ? (
            <p>
              Voice state context — injected when the user interrupts
              mid-sentence, makes a sound with no words, or an error occurs.
              Override individual triggers below.
            </p>
          ) : (
            <p>
              Provider-sourced block. Content is generated by the{" "}
              <span className="font-mono text-foreground">{block.id}</span>{" "}
              provider at prompt build time.
            </p>
          )}
          {block.content_wrapper && (
            <p className="mt-2 text-xs">
              Wrapper:{" "}
              <code className="px-1 py-0.5 rounded bg-muted font-mono">
                {block.content_wrapper}
              </code>
            </p>
          )}
        </div>

        {/* Current content from latest snapshot */}
        {currentContent !== null && currentContent.length > 0 && (
          <>
            <div className="space-y-2">
              <Label className="text-xs text-muted-foreground">
                Current Content
              </Label>
              <div className="rounded-md border bg-muted/20 p-3 max-h-[400px] overflow-y-auto">
                <pre className="text-xs font-mono whitespace-pre-wrap break-words">
                  {currentContent}
                </pre>
              </div>
            </div>
          </>
        )}

        <Separator />

        {/* Voice state: per-trigger override editor */}
        {isVoiceState && (
          <div className="space-y-3">
            <Label className="text-sm font-medium">Trigger Overrides</Label>
            <VoiceStateTriggerEditor
              triggerOverrides={triggerOverrides ?? {}}
              onSetOverride={onSetOverride}
            />
          </div>
        )}

        {/* Generic override editor (hidden for voice_state) */}
        {!isVoiceState && (
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <Label htmlFor="override-toggle" className="text-sm font-medium">
                User Override
              </Label>
              <Switch
                id="override-toggle"
                checked={overrideEnabled}
                onCheckedChange={handleToggleOverride}
                disabled={isDeferred}
              />
            </div>

            {isDeferred && (
              <p className="text-xs text-muted-foreground">
                Deferred blocks are populated at runtime and cannot be overridden.
              </p>
            )}

            {overrideEnabled && !isDeferred && (
              <div className="space-y-2">
                <Textarea
                  value={editContent}
                  onChange={(e) => setEditContent(e.target.value)}
                  placeholder="Enter override content for this block..."
                  className="min-h-[200px] font-mono text-sm"
                />
                <div className="flex items-center justify-between">
                  <span className="text-xs text-muted-foreground">
                    {editContent.length.toLocaleString()} characters
                  </span>
                  <div className="flex gap-2">
                    <Button
                      variant="outline"
                      size="sm"
                      onClick={handleRevert}
                      disabled={!hasUnsavedEdit}
                    >
                      Revert
                    </Button>
                    <Button
                      size="sm"
                      onClick={handleApply}
                      disabled={!hasUnsavedEdit}
                    >
                      Apply
                    </Button>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}
      </CardContent>
    </Card>
  );
}
