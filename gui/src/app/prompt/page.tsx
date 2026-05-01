"use client";

import { useEffect } from "react";
import {
  usePromptStore,
  useConnectionStore,
  useBlockEditorStore,
  useSettingsStore,
} from "@/lib/stores";
import {
  PromptViewer,
  TokenBreakdown,
  BlockList,
  BlockDetail,
  InjectionWrappersCard,
  MessageArrayPreview,
} from "@/components/prompt";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Toggle } from "@/components/ui/toggle";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import {
  WifiOff,
  Loader2,
  Eye,
  Pencil,
  Save,
  Undo2,
  RotateCcw,
  MousePointerClick,
  FileSearch,
} from "lucide-react";
import { getSocket } from "@/lib/socket";
import type { SetBlockConfigPayload } from "@/types/events";

export default function PromptCompositionPage() {
  const connected = useConnectionStore((s) => s.connected);
  const currentSnapshot = usePromptStore((s) => s.currentSnapshot);
  const blockConfig = useBlockEditorStore((s) => s.blockConfig);
  const isSaving = useBlockEditorStore((s) => s.isSaving);
  const selectedBlockId = useBlockEditorStore((s) => s.selectedBlockId);
  const editMode = useBlockEditorStore((s) => s.editMode);
  const pendingOverrides = useBlockEditorStore((s) => s.pendingOverrides);
  const pendingDisabled = useBlockEditorStore((s) => s.pendingDisabled);
  const pendingOrder = useBlockEditorStore((s) => s.pendingOrder);
  // NANO-114: active provider's role-array capability locks the recent_history
  // block position (history gets spliced into the message array, not the
  // system prompt, so drag-to-reorder is meaningless for that block).
  const supportsRoleHistory = useSettingsStore(
    (s) => s.llmConfig.supports_role_history,
  );

  const {
    selectBlock,
    toggleEditMode,
    toggleBlockEnabled,
    setOverride,
    clearPending,
    setSaving,
    setPendingOrder,
    hasPendingChanges,
  } = useBlockEditorStore.getState();

  // Request block config on mount
  useEffect(() => {
    if (connected) {
      const socket = getSocket();
      socket.emit("request_block_config", {});
    }
  }, [connected]);

  // --- Not connected ---
  if (!connected) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold">Prompt Workshop</h1>
          <p className="text-muted-foreground">
            Configure prompt blocks and overrides
          </p>
        </div>
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <WifiOff className="h-12 w-12 text-muted-foreground mb-4" />
            <p className="text-lg font-medium">Not Connected</p>
            <p className="text-sm text-muted-foreground">
              Connect to the orchestrator to view prompt data
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  // --- Loading block config ---
  if (!blockConfig) {
    return (
      <div className="space-y-6">
        <div>
          <h1 className="text-2xl font-bold">Prompt Workshop</h1>
          <p className="text-muted-foreground">
            Configure prompt blocks and overrides
          </p>
        </div>
        <Card>
          <CardContent className="flex flex-col items-center justify-center py-12">
            <Loader2 className="h-12 w-12 text-muted-foreground mb-4 animate-spin" />
            <p className="text-lg font-medium">Loading Block Config</p>
            <p className="text-sm text-muted-foreground">
              Waiting for block configuration from the orchestrator...
            </p>
          </CardContent>
        </Card>
      </div>
    );
  }

  // --- Effective state (pending overrides server) ---
  const effectiveDisabled = pendingDisabled ?? blockConfig.disabled;
  const effectiveOverrides = { ...blockConfig.overrides, ...pendingOverrides };
  const blocks = blockConfig.blocks;
  const effectiveOrder = pendingOrder ?? blockConfig.order;
  const orderedBlocks = effectiveOrder.length > 0
    ? effectiveOrder.map((id) => blocks.find((b) => b.id === id)).filter((b): b is typeof blocks[number] => b != null)
    : blocks;
  const tokenData = currentSnapshot?.token_breakdown.blocks ?? null;
  const selectedBlock = blocks.find((b) => b.id === selectedBlockId) ?? null;
  const pending = hasPendingChanges();

  const handleSave = () => {
    const socket = getSocket();
    setSaving(true);

    const payload: SetBlockConfigPayload = {};
    if (pendingDisabled !== null) {
      payload.disabled = pendingDisabled;
    }
    if (Object.keys(pendingOverrides).length > 0) {
      payload.overrides = effectiveOverrides;
    }
    if (pendingOrder !== null) {
      payload.order = pendingOrder;
    }

    socket.emit("set_block_config", payload);
  };

  const handleReset = () => {
    const socket = getSocket();
    setSaving(true);
    socket.emit("reset_block_config", {});
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Prompt Workshop</h1>
          <p className="text-muted-foreground">
            {editMode
              ? "Configure blocks, overrides, and injection wrappers"
              : "View prompt structure and token distribution"}
          </p>
        </div>
        <div className="flex items-center gap-2">
          {pending && (
            <Badge variant="secondary" className="text-xs">
              Unsaved Changes
            </Badge>
          )}

          <Toggle
            variant="outline"
            pressed={editMode}
            onPressedChange={() => toggleEditMode()}
            aria-label="Toggle edit mode"
          >
            {editMode ? (
              <Eye className="h-4 w-4" />
            ) : (
              <Pencil className="h-4 w-4" />
            )}
            {editMode ? "View" : "Edit"}
          </Toggle>

          {editMode && (
            <>
              <Button
                variant="outline"
                size="sm"
                onClick={() => clearPending()}
                disabled={!pending}
              >
                <Undo2 className="h-4 w-4 mr-1" />
                Discard
              </Button>
              <Button
                size="sm"
                onClick={handleSave}
                disabled={!pending || isSaving}
              >
                {isSaving ? (
                  <Loader2 className="h-4 w-4 mr-1 animate-spin" />
                ) : (
                  <Save className="h-4 w-4 mr-1" />
                )}
                {isSaving ? "Saving..." : "Save"}
              </Button>
              <Button variant="destructive" size="sm" onClick={handleReset}>
                <RotateCcw className="h-4 w-4 mr-1" />
                Reset All
              </Button>
            </>
          )}
        </div>
      </div>

      {/* Main 2-column layout */}
      <div className="grid gap-4 lg:grid-cols-3">
        {/* Left: Block List (1/3) */}
        <div className="lg:col-span-1">
          <BlockList
            blocks={orderedBlocks}
            tokenData={tokenData}
            disabledBlocks={effectiveDisabled}
            selectedBlockId={selectedBlockId}
            editMode={editMode}
            pendingOverrides={pendingOverrides}
            onSelectBlock={(id) => selectBlock(id)}
            onToggleBlock={(id) => toggleBlockEnabled(id)}
            onReorder={(order) => setPendingOrder(order)}
            lockedBlockIds={
              supportsRoleHistory
                ? new Set(["recent_history"])
                : undefined
            }
            lockReason={
              supportsRoleHistory
                ? "spliced to message array"
                : undefined
            }
          />
        </div>

        {/* Right: Conditional Panel (2/3) */}
        <div className="lg:col-span-2 max-h-[calc(100vh-8rem)] overflow-y-auto">
          {editMode ? (
            <Tabs defaultValue="blocks">
              <TabsList>
                <TabsTrigger value="blocks">Blocks</TabsTrigger>
                <TabsTrigger value="wrappers">Injection Wrappers</TabsTrigger>
              </TabsList>
              <TabsContent value="blocks">
                {selectedBlock ? (
                  <BlockDetail
                    block={selectedBlock}
                    currentOverride={effectiveOverrides[selectedBlock.id] ?? null}
                    tokenCount={
                      tokenData?.find((t) => t.id === selectedBlock.id)?.tokens ??
                      null
                    }
                    currentContent={
                      tokenData?.find((t) => t.id === selectedBlock.id)?.content ??
                      null
                    }
                    onSetOverride={(blockId, content) =>
                      setOverride(blockId, content)
                    }
                  />
                ) : (
                  <Card>
                    <CardContent className="flex flex-col items-center justify-center py-12">
                      <MousePointerClick className="h-12 w-12 text-muted-foreground mb-4" />
                      <p className="text-lg font-medium">No Block Selected</p>
                      <p className="text-sm text-muted-foreground">
                        Click a block in the list to view details and configure
                        overrides
                      </p>
                    </CardContent>
                  </Card>
                )}
              </TabsContent>
              <TabsContent value="wrappers">
                <InjectionWrappersCard />
              </TabsContent>
            </Tabs>
          ) : currentSnapshot ? (
            <Tabs defaultValue="breakdown">
              <TabsList>
                <TabsTrigger value="breakdown">Token Breakdown</TabsTrigger>
                <TabsTrigger value="prompt">Raw Prompt</TabsTrigger>
                {supportsRoleHistory && (
                  <TabsTrigger value="messages">Message Array</TabsTrigger>
                )}
              </TabsList>
              <TabsContent value="breakdown">
                <TokenBreakdown
                  breakdown={currentSnapshot.token_breakdown}
                />
              </TabsContent>
              <TabsContent value="prompt">
                <PromptViewer
                  messages={currentSnapshot.messages}
                  inputModality={currentSnapshot.input_modality}
                  stateTrigger={currentSnapshot.state_trigger}
                  timestamp={currentSnapshot.timestamp}
                />
              </TabsContent>
              {supportsRoleHistory && (
                <TabsContent value="messages">
                  <MessageArrayPreview messages={currentSnapshot.messages} />
                </TabsContent>
              )}
            </Tabs>
          ) : (
            <Card>
              <CardContent className="flex flex-col items-center justify-center py-12">
                <FileSearch className="h-12 w-12 text-muted-foreground mb-4" />
                <p className="text-lg font-medium">No Prompt Snapshot</p>
                <p className="text-sm text-muted-foreground">
                  Speak to the agent to generate a prompt snapshot
                </p>
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
