"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog";
import { useCharacterStore, useAgentStore } from "@/lib/stores";
import {
  User,
  Save,
  Trash2,
  Upload,
  Loader2,
  X,
  Plus,
  AlertCircle,
} from "lucide-react";
import { CharacterCodexTab } from "@/components/codex";
import { AvatarCropModal } from "./avatar-crop-modal";
import { FieldStatusBadge } from "./field-status-badge";
import { fetchAvatarExtended } from "@/lib/stores/character-store";
import { getSocket } from "@/lib/socket";
import type { CharacterCardData, SpindlExtensions, CharacterBook, CropSettings } from "@/types/events";

interface CharacterEditorProps {
  onSave: (card: CharacterCardData) => void;
  onDelete: () => void;
  onCancel: () => void;
  onAvatarUpload: (imageData: string, originalData?: string, cropSettings?: CropSettings) => void;
  isNew?: boolean;
  isSaving?: boolean;
  pendingAvatarPreview?: string | null;
}

export function CharacterEditor({
  onSave,
  onDelete,
  onCancel,
  onAvatarUpload,
  isNew = false,
  isSaving = false,
  pendingAvatarPreview,
}: CharacterEditorProps) {
  const {
    selectedCharacterId,
    selectedCharacterHasAvatar,
    editedCard,
    hasUnsavedChanges,
    avatarCache,
    setEditedCard,
    lastAction,
    activeCharacterId,
  } = useCharacterStore();

  // NANO-036 Phase 2: Agent state gating for save button
  const { state: agentState } = useAgentStore();
  const canSave = agentState === "idle" || agentState === "listening";

  const fileInputRef = useRef<HTMLInputElement>(null);
  const vrmFileInputRef = useRef<HTMLInputElement>(null);
  const [newTag, setNewTag] = useState("");
  const [newRule, setNewRule] = useState("");
  const [isUploadingVrm, setIsUploadingVrm] = useState(false);
  // NANO-098: Available animations for dropdowns
  const [availableAnimations, setAvailableAnimations] = useState<{ name: string; source: string }[]>([]);
  // NANO-041b: Track active tab to hide top Save on Codex tab
  const [activeTab, setActiveTab] = useState("prompt");

  // NANO-041a: Crop modal state for re-editing existing avatars
  const [showCropModal, setShowCropModal] = useState(false);
  const [selectedImageForCrop, setSelectedImageForCrop] = useState<string | null>(null);
  const [initialCropSettings, setInitialCropSettings] = useState<CropSettings | undefined>(undefined);

  // NANO-098: Snapshot saved expression composites for live preview revert
  const savedExpressionsRef = useRef<Record<string, Record<string, number>> | undefined>(undefined);
  const hasPreviewedRef = useRef(false);
  // NANO-098 Session 3: Track animation preview for revert on cancel/unmount
  const savedAnimationClipRef = useRef<string | null>(null);
  const hasPreviewedAnimationRef = useRef(false);

  // Snapshot on mount (or when editedCard changes identity)
  useEffect(() => {
    if (editedCard) {
      const nano = (editedCard.data.extensions?.spindl as SpindlExtensions) || {};
      savedExpressionsRef.current = nano.avatar_expressions ? { ...nano.avatar_expressions } : undefined;
      hasPreviewedRef.current = false;
      savedAnimationClipRef.current = nano.avatar_animations?.default ?? null;
      hasPreviewedAnimationRef.current = false;
    }
  }, [editedCard?.spec_version]); // only re-snapshot on card load, not every edit

  // Revert composites + animation on unmount (cancel / navigate away)
  useEffect(() => {
    return () => {
      if (hasPreviewedRef.current) {
        try {
          const socket = getSocket();
          socket.emit("preview_avatar_expressions", {
            expressions: savedExpressionsRef.current ?? {},
          });
        } catch { /* socket may not be connected */ }
      }
      if (hasPreviewedAnimationRef.current) {
        try {
          const socket = getSocket();
          socket.emit("preview_avatar_animation", {
            clip: savedAnimationClipRef.current,
          });
        } catch { /* socket may not be connected */ }
      }
    };
  }, []);

  // NANO-098: Fetch available animations for dropdown population
  useEffect(() => {
    if (selectedCharacterId && !isNew) {
      fetch(`/api/characters/${encodeURIComponent(selectedCharacterId)}/animations`)
        .then((r) => r.json())
        .then((data) => setAvailableAnimations(data.animations ?? []))
        .catch(() => setAvailableAnimations([]));
    } else {
      setAvailableAnimations([]);
    }
  }, [selectedCharacterId, isNew]);

  // NANO-098: Wrap save to update snapshots so unmount revert is a no-op
  const handleSaveWithExpressionSnapshot = useCallback(() => {
    if (!editedCard) return;
    const nano = (editedCard.data.extensions?.spindl as SpindlExtensions) || {};
    savedExpressionsRef.current = nano.avatar_expressions ? { ...nano.avatar_expressions } : undefined;
    hasPreviewedRef.current = false;
    savedAnimationClipRef.current = nano.avatar_animations?.default ?? null;
    hasPreviewedAnimationRef.current = false;
    onSave(editedCard);
    // NANO-098 Session 3: Push animation config directly to renderer on save.
    // This avoids the race condition of reload_avatar_model reading card.json
    // before the save API finishes writing it.
    if (selectedCharacterId && selectedCharacterId === activeCharacterId) {
      try {
        const socket = getSocket();
        socket.emit("update_avatar_animation_config", {
          animations: nano.avatar_animations ?? null,
        });
      } catch { /* socket may not be connected */ }
    }
  }, [editedCard, onSave, selectedCharacterId, activeCharacterId]);

  // Get avatar URL — prioritize pending avatar for creation flow
  const avatarUrl = pendingAvatarPreview
    || (selectedCharacterId ? avatarCache[selectedCharacterId] : null);

  // Helper to update card data
  const updateData = useCallback(
    (field: keyof NonNullable<CharacterCardData>["data"], value: unknown) => {
      if (!editedCard) return;
      setEditedCard({
        ...editedCard,
        data: {
          ...editedCard.data,
          [field]: value,
        },
      });
    },
    [editedCard, setEditedCard]
  );

  // Helper to update spindl extensions
  const updateSpindl = useCallback(
    (field: keyof SpindlExtensions, value: unknown) => {
      if (!editedCard) return;
      const currentExtensions = editedCard.data.extensions || {};
      const currentNano = (currentExtensions.spindl as SpindlExtensions) || {};
      setEditedCard({
        ...editedCard,
        data: {
          ...editedCard.data,
          extensions: {
            ...currentExtensions,
            spindl: {
              ...currentNano,
              [field]: value,
            },
          },
        },
      });
    },
    [editedCard, setEditedCard]
  );

  // Get spindl extensions
  const getSpindl = (): SpindlExtensions => {
    if (!editedCard?.data.extensions?.spindl) {
      return {};
    }
    return editedCard.data.extensions.spindl as SpindlExtensions;
  };

  // NANO-097: VRM file upload handler
  const handleVrmSelect = useCallback(async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || !selectedCharacterId) return;
    setIsUploadingVrm(true);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const response = await fetch(
        `/api/characters/${encodeURIComponent(selectedCharacterId)}/vrm`,
        { method: "POST", body: formData }
      );
      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.error || `Upload failed: ${response.statusText}`);
      }
      const result = await response.json();
      updateSpindl("avatar_vrm", result.vrm_filename);
      // Tell server to push new model to connected avatar clients
      getSocket().emit("reload_avatar_model", { character_id: selectedCharacterId });
    } catch (err) {
      console.error("[CharacterEditor] VRM upload failed:", err);
    } finally {
      setIsUploadingVrm(false);
      // Reset input so re-selecting the same file triggers onChange
      if (vrmFileInputRef.current) vrmFileInputRef.current.value = "";
    }
  }, [selectedCharacterId, updateSpindl]);

  // NANO-097: VRM file removal handler
  const handleVrmRemove = useCallback(async () => {
    if (!selectedCharacterId) return;
    setIsUploadingVrm(true);
    try {
      const response = await fetch(
        `/api/characters/${encodeURIComponent(selectedCharacterId)}/vrm`,
        { method: "DELETE" }
      );
      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        throw new Error(err.error || `Delete failed: ${response.statusText}`);
      }
      updateSpindl("avatar_vrm", undefined);
      // Tell avatar renderer to revert to default model
      if (selectedCharacterId === activeCharacterId) {
        try {
          getSocket().emit("reload_avatar_model", { character_id: selectedCharacterId });
        } catch { /* socket may not be connected */ }
      }
    } catch (err) {
      console.error("[CharacterEditor] VRM removal failed:", err);
    } finally {
      setIsUploadingVrm(false);
    }
  }, [selectedCharacterId, updateSpindl]);

  // NANO-041a: Route file selection through crop modal
  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result as string;
      setSelectedImageForCrop(dataUrl);
      setInitialCropSettings(undefined);
      setShowCropModal(true);
    };
    reader.readAsDataURL(file);

    // Reset to allow re-selecting same file
    e.target.value = "";
  }, []);

  // NANO-041a: Avatar click — check for original + settings, or open file picker
  const handleAvatarClick = useCallback(async () => {
    if (isNew) return; // During creation, avatar is already set

    if (selectedCharacterId) {
      try {
        const extendedData = await fetchAvatarExtended(selectedCharacterId);

        if (extendedData.original_data && extendedData.crop_settings) {
          // Re-edit: open crop modal with original at saved position
          setSelectedImageForCrop(extendedData.original_data);
          setInitialCropSettings(extendedData.crop_settings as CropSettings);
          setShowCropModal(true);
        } else {
          // No original — trigger file picker for new upload
          fileInputRef.current?.click();
        }
      } catch {
        // Fallback to file picker
        fileInputRef.current?.click();
      }
    }
  }, [isNew, selectedCharacterId]);

  // NANO-041a: Crop confirmed — upload to backend for existing characters
  const handleCropConfirm = useCallback((
    croppedData: string,
    originalData: string,
    settings: CropSettings,
  ) => {
    onAvatarUpload(croppedData, originalData, settings);
    setShowCropModal(false);
    setSelectedImageForCrop(null);
    setInitialCropSettings(undefined);
  }, [onAvatarUpload]);

  // Handle adding tags
  const addTag = () => {
    if (!newTag.trim() || !editedCard) return;
    const currentTags = editedCard.data.tags || [];
    if (!currentTags.includes(newTag.trim())) {
      updateData("tags", [...currentTags, newTag.trim()]);
    }
    setNewTag("");
  };

  // Handle removing tags
  const removeTag = (tag: string) => {
    if (!editedCard) return;
    updateData(
      "tags",
      (editedCard.data.tags || []).filter((t) => t !== tag)
    );
  };

  // Handle adding rules
  const addRule = () => {
    if (!newRule.trim()) return;
    const nano = getSpindl();
    const currentRules = nano.rules || [];
    updateSpindl("rules", [...currentRules, newRule.trim()]);
    setNewRule("");
  };

  // Handle removing rules
  const removeRule = (index: number) => {
    const nano = getSpindl();
    const currentRules = nano.rules || [];
    updateSpindl(
      "rules",
      currentRules.filter((_, i) => i !== index)
    );
  };

  if (!editedCard) {
    return (
      <Card>
        <CardContent className="flex items-center justify-center h-64 text-muted-foreground">
          Select a character to edit or create a new one.
        </CardContent>
      </Card>
    );
  }

  const nano = getSpindl();

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <div className="flex items-center gap-3">
          <div className="relative group">
            <Avatar className="h-16 w-16">
              {avatarUrl ? (
                <AvatarImage src={avatarUrl} alt={editedCard.data.name} />
              ) : null}
              <AvatarFallback>
                <User className="h-8 w-8" />
              </AvatarFallback>
            </Avatar>
            <button
              className="absolute inset-0 flex items-center justify-center bg-black/50 rounded-full opacity-0 group-hover:opacity-100 transition-opacity"
              onClick={handleAvatarClick}
            >
              <Upload className="h-5 w-5 text-white" />
            </button>
            <input
              ref={fileInputRef}
              type="file"
              accept="image/png,image/jpeg,image/webp"
              className="hidden"
              onChange={handleFileSelect}
            />
            {/* NANO-041a: Crop modal for re-editing existing avatars */}
            {selectedImageForCrop && (
              <AvatarCropModal
                image={selectedImageForCrop}
                open={showCropModal}
                onOpenChange={(open) => {
                  setShowCropModal(open);
                  if (!open) {
                    setSelectedImageForCrop(null);
                    setInitialCropSettings(undefined);
                  }
                }}
                onConfirm={handleCropConfirm}
                initialSettings={initialCropSettings}
              />
            )}
          </div>
          <div>
            <CardTitle className="text-lg">
              {isNew ? "New Character" : editedCard.data.name}
            </CardTitle>
            {hasUnsavedChanges && (
              <span data-testid="unsaved-changes-indicator" className="text-xs text-yellow-500">Unsaved changes</span>
            )}
          </div>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={() => {
            // Revert animation preview on cancel
            if (hasPreviewedAnimationRef.current) {
              try {
                getSocket().emit("preview_avatar_animation", { clip: savedAnimationClipRef.current });
              } catch { /* socket may not be connected */ }
              hasPreviewedAnimationRef.current = false;
            }
            // Revert expression preview on cancel
            if (hasPreviewedRef.current) {
              try {
                getSocket().emit("preview_avatar_expressions", { expressions: savedExpressionsRef.current ?? {} });
              } catch { /* socket may not be connected */ }
              hasPreviewedRef.current = false;
            }
            onCancel();
          }}>
            <X className="h-4 w-4 mr-1" />
            Cancel
          </Button>
          {!isNew && (
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button variant="destructive" size="sm">
                  <Trash2 className="h-4 w-4 mr-1" />
                  Delete
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Delete Character</AlertDialogTitle>
                  <AlertDialogDescription>
                    Are you sure you want to delete &quot;{editedCard.data.name}&quot;?
                    This action cannot be undone.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction onClick={onDelete}>Delete</AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          )}
          {/* NANO-039: Hide top Save on Codex tab — codex entry Save handles persistence */}
          {activeTab !== "codex" && (
            <Button
              size="sm"
              data-testid="save-button"
              onClick={handleSaveWithExpressionSnapshot}
              disabled={isSaving || !hasUnsavedChanges || !canSave}
              title={
                !canSave
                  ? `Cannot save while agent is ${agentState.replace("_", " ")}`
                  : undefined
              }
            >
              {isSaving ? (
                <Loader2 className="h-4 w-4 mr-1 animate-spin" />
              ) : (
                <Save className="h-4 w-4 mr-1" />
              )}
              Save
            </Button>
          )}
        </div>
      </CardHeader>

      {lastAction.error && (
        <div className="mx-6 mb-2 p-2 bg-destructive/10 border border-destructive/20 rounded-md flex items-center gap-2 text-sm text-destructive">
          <AlertCircle className="h-4 w-4" />
          {lastAction.error}
        </div>
      )}

      {/* NANO-036 Phase 2: Agent state indicator when save is blocked */}
      {hasUnsavedChanges && !canSave && (
        <div data-testid="save-disabled-banner" className="mx-6 mb-2 p-2 bg-yellow-500/10 border border-yellow-500/20 rounded-md flex items-center gap-2 text-sm text-yellow-600 dark:text-yellow-500">
          <AlertCircle className="h-4 w-4" />
          Save disabled while agent is {agentState.replace("_", " ")}
        </div>
      )}

      <CardContent>
        {/* NANO-041b: 3 function-based tabs (Prompt / Metadata / Codex) */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <TabsList className="w-full">
            <TabsTrigger value="prompt" className="flex-1">
              Prompt
            </TabsTrigger>
            <TabsTrigger value="metadata" className="flex-1">
              Metadata
            </TabsTrigger>
            <TabsTrigger value="codex" className="flex-1">
              Codex
            </TabsTrigger>
          </TabsList>

          <ScrollArea className="h-[calc(100vh-380px)] mt-4">
            {/* ============================================================
                PROMPT TAB — Everything that affects character behavior
                ============================================================ */}
            <TabsContent value="prompt" className="space-y-4 pr-4">
              {/* Name */}
              <div className="space-y-1">
                <Label htmlFor="name">Name *</Label>
                <FieldStatusBadge variant="live" />
                <Input
                  id="name"
                  data-testid="character-name-input"
                  value={editedCard.data.name}
                  onChange={(e) => updateData("name", e.target.value)}
                  placeholder="Character name"
                />
              </div>

              {/* Personality */}
              <div className="space-y-1">
                <Label htmlFor="personality">Personality</Label>
                <FieldStatusBadge variant="live" />
                <Textarea
                  id="personality"
                  value={editedCard.data.personality}
                  onChange={(e) => updateData("personality", e.target.value)}
                  placeholder="Character's personality traits..."
                  rows={4}
                />
              </div>

              {/* Description (migrated from appearance — now the canonical source for [PERSONA_APPEARANCE]) */}
              <div className="space-y-1">
                <Label htmlFor="description">Description</Label>
                <FieldStatusBadge variant="live" />
                <Textarea
                  id="description"
                  value={editedCard.data.description}
                  onChange={(e) => updateData("description", e.target.value)}
                  placeholder="Physical description, visual traits, mannerisms..."
                  rows={3}
                />
              </div>

              {/* Rules */}
              <div className="space-y-1">
                <Label>Rules</Label>
                <FieldStatusBadge variant="live" />
                <div className="flex gap-2">
                  <Input
                    value={newRule}
                    onChange={(e) => setNewRule(e.target.value)}
                    placeholder="Add a rule..."
                    onKeyDown={(e) =>
                      e.key === "Enter" && (e.preventDefault(), addRule())
                    }
                  />
                  <Button variant="outline" onClick={addRule}>
                    <Plus className="h-4 w-4" />
                  </Button>
                </div>
                {(nano.rules || []).length > 0 && (
                  <ul className="mt-2 space-y-1">
                    {(nano.rules || []).map((rule, i) => (
                      <li
                        key={i}
                        className="flex items-center justify-between p-2 bg-muted rounded-md text-sm"
                      >
                        <span>{rule}</span>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={() => removeRule(i)}
                        >
                          <X className="h-3 w-3" />
                        </Button>
                      </li>
                    ))}
                  </ul>
                )}
              </div>

              {/* Scenario */}
              <div className="space-y-1">
                <Label htmlFor="scenario">Scenario</Label>
                <FieldStatusBadge variant="live" description="Injected into prompt as scenario context" />
                <Textarea
                  id="scenario"
                  value={editedCard.data.scenario}
                  onChange={(e) => updateData("scenario", e.target.value)}
                  placeholder="The scenario or setting for conversations..."
                  rows={3}
                />
              </div>

              {/* First Message */}
              <div className="space-y-1">
                <Label htmlFor="first_mes">First Message</Label>
                <FieldStatusBadge variant="live" />
                <Textarea
                  id="first_mes"
                  value={editedCard.data.first_mes}
                  onChange={(e) => updateData("first_mes", e.target.value)}
                  placeholder="The character's opening message..."
                  rows={4}
                />
              </div>

              {/* Alternate Greetings */}
              <div className="space-y-1">
                <Label>Alternate Greetings</Label>
                <FieldStatusBadge variant="live" />
                {(editedCard.data.alternate_greetings || []).map((greeting, i) => (
                  <div key={i} className="flex gap-2 items-start mt-2">
                    <Textarea
                      value={greeting}
                      onChange={(e) => {
                        const updated = [...(editedCard.data.alternate_greetings || [])];
                        updated[i] = e.target.value;
                        updateData("alternate_greetings", updated);
                      }}
                      placeholder={`Alternate greeting ${i + 1}...`}
                      rows={2}
                      className="flex-1"
                    />
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={() => {
                        const updated = (editedCard.data.alternate_greetings || []).filter((_, idx) => idx !== i);
                        updateData("alternate_greetings", updated);
                      }}
                    >
                      <X className="h-3 w-3" />
                    </Button>
                  </div>
                ))}
                <Button
                  variant="outline"
                  size="sm"
                  className="mt-2"
                  onClick={() => updateData("alternate_greetings", [...(editedCard.data.alternate_greetings || []), ""])}
                >
                  <Plus className="h-4 w-4 mr-1" />
                  Add Greeting
                </Button>
              </div>

              {/* Message Examples */}
              <div className="space-y-1">
                <Label htmlFor="mes_example">Message Examples</Label>
                <FieldStatusBadge variant="live" />
                <Textarea
                  id="mes_example"
                  value={editedCard.data.mes_example}
                  onChange={(e) => updateData("mes_example", e.target.value)}
                  placeholder="Example conversations to guide the AI..."
                  rows={6}
                />
              </div>

              {/* Voice */}
              <div className="space-y-1">
                <Label htmlFor="voice">TTS Voice</Label>
                <FieldStatusBadge variant="tts" />
                <Input
                  id="voice"
                  value={nano.voice || ""}
                  onChange={(e) => updateSpindl("voice", e.target.value)}
                  placeholder="Provider default"
                />
                <p className="text-xs text-muted-foreground">
                  Voice ID for the active TTS provider. Leave empty for provider default.
                </p>
              </div>

              {/* Language */}
              <div className="space-y-1">
                <Label htmlFor="language">Language</Label>
                <FieldStatusBadge variant="tts" />
                <Input
                  id="language"
                  value={nano.language || ""}
                  onChange={(e) => updateSpindl("language", e.target.value)}
                  placeholder="Provider default"
                />
                <p className="text-xs text-muted-foreground">
                  Language code for the active TTS provider. Leave empty for provider default.
                </p>
              </div>
            </TabsContent>

            {/* ============================================================
                METADATA TAB — Card info, ST V2 compat, power-user overrides
                ============================================================ */}
            <TabsContent value="metadata" className="space-y-4 pr-4">
              {/* Character ID (read-only) */}
              <div className="space-y-1">
                <Label htmlFor="nano_id">Character ID</Label>
                <FieldStatusBadge variant="metadata" description="Character directory name (read-only)" />
                <Input
                  id="nano_id"
                  value={nano.id || ""}
                  readOnly
                  className="bg-muted cursor-default"
                />
              </div>

              {/* Tags */}
              <div className="space-y-1">
                <Label>Tags</Label>
                <FieldStatusBadge variant="metadata" description="Not sent to model — card organization" />
                <div className="flex gap-2">
                  <Input
                    value={newTag}
                    onChange={(e) => setNewTag(e.target.value)}
                    placeholder="Add a tag..."
                    onKeyDown={(e) => e.key === "Enter" && (e.preventDefault(), addTag())}
                  />
                  <Button variant="outline" onClick={addTag}>
                    <Plus className="h-4 w-4" />
                  </Button>
                </div>
                {editedCard.data.tags.length > 0 && (
                  <div className="flex flex-wrap gap-1 mt-2">
                    {editedCard.data.tags.map((tag) => (
                      <Badge
                        key={tag}
                        variant="secondary"
                        className="cursor-pointer"
                        onClick={() => removeTag(tag)}
                      >
                        {tag}
                        <X className="h-3 w-3 ml-1" />
                      </Badge>
                    ))}
                  </div>
                )}
              </div>

              {/* Creator */}
              <div className="grid grid-cols-2 gap-4">
                <div className="space-y-1">
                  <Label htmlFor="creator">Creator</Label>
                  <FieldStatusBadge variant="metadata" description="Not sent to model — attribution" />
                  <Input
                    id="creator"
                    value={editedCard.data.creator}
                    onChange={(e) => updateData("creator", e.target.value)}
                    placeholder="Your name"
                  />
                </div>
                <div className="space-y-1">
                  <Label htmlFor="version">Version</Label>
                  <FieldStatusBadge variant="metadata" description="Not sent to model — versioning" />
                  <Input
                    id="version"
                    value={editedCard.data.character_version}
                    onChange={(e) => updateData("character_version", e.target.value)}
                    placeholder="1.0"
                  />
                </div>
              </div>

              {/* Creator Notes */}
              <div className="space-y-1">
                <Label htmlFor="creator_notes">Creator Notes</Label>
                <FieldStatusBadge variant="metadata" description="Not sent to model — author notes" />
                <Textarea
                  id="creator_notes"
                  value={editedCard.data.creator_notes}
                  onChange={(e) => updateData("creator_notes", e.target.value)}
                  placeholder="Notes for other users..."
                  rows={2}
                />
              </div>

              {/* System Prompt (ST V2 compat) */}
              <div className="space-y-1">
                <Label htmlFor="system_prompt">System Prompt</Label>
                <FieldStatusBadge variant="metadata" description="Not sent to model — ST V2 compatibility" />
                <Textarea
                  id="system_prompt"
                  value={editedCard.data.system_prompt}
                  onChange={(e) => updateData("system_prompt", e.target.value)}
                  placeholder="ST V2 system prompt field (not used by spindl pipeline)..."
                  rows={4}
                />
              </div>

              {/* Post History Instructions (ST V2 compat) */}
              <div className="space-y-1">
                <Label htmlFor="post_history">Post-History Instructions</Label>
                <FieldStatusBadge variant="metadata" description="Not sent to model — ST V2 compatibility" />
                <Textarea
                  id="post_history"
                  value={editedCard.data.post_history_instructions}
                  onChange={(e) =>
                    updateData("post_history_instructions", e.target.value)
                  }
                  placeholder="ST V2 post-history instructions (not used by spindl pipeline)..."
                  rows={3}
                />
              </div>

              {/* Summarization Prompt */}
              <div className="space-y-1">
                <Label htmlFor="summarization_prompt">Summarization Prompt</Label>
                <FieldStatusBadge variant="summarizer" />
                <Textarea
                  id="summarization_prompt"
                  value={nano.summarization_prompt || ""}
                  onChange={(e) =>
                    updateSpindl("summarization_prompt", e.target.value)
                  }
                  placeholder="Custom prompt for conversation summarization..."
                  rows={3}
                />
              </div>

              {/* NANO-097: Avatar VRM Model */}
              {!isNew && (
                <div className="space-y-1">
                  <Label>Avatar Model (VRM)</Label>
                  <FieldStatusBadge variant="metadata" description="3D avatar model for SpindL Avatar renderer" />
                  <div className="flex items-center gap-2">
                    <Input
                      readOnly
                      value={nano.avatar_vrm || "None"}
                      className="flex-1 bg-muted cursor-default"
                    />
                    <input
                      ref={vrmFileInputRef}
                      type="file"
                      accept=".vrm"
                      className="hidden"
                      onChange={handleVrmSelect}
                    />
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      disabled={isUploadingVrm}
                      onClick={() => vrmFileInputRef.current?.click()}
                    >
                      {isUploadingVrm ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Upload className="h-4 w-4" />
                      )}
                      <span className="ml-1">Choose VRM...</span>
                    </Button>
                    {nano.avatar_vrm && (
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        disabled={isUploadingVrm}
                        onClick={handleVrmRemove}
                      >
                        <X className="h-4 w-4" />
                      </Button>
                    )}
                  </div>
                </div>
              )}

              {/* NANO-098: Expression Composites — per-character blend shape overrides */}
              {!isNew && (
                <div className="space-y-2">
                  <Label>Expression Composites</Label>
                  <FieldStatusBadge variant="metadata" description="Custom blend shape composites for moods this VRM lacks natively (e.g. curious)" />
                  <p className="text-xs text-muted-foreground">
                    Build the &quot;curious&quot; expression from available blend shapes. Tune per-VRM using the avatar sandbox.
                  </p>
                  <div className="grid grid-cols-3 gap-x-4 gap-y-2">
                    {(["aa", "ih", "ee", "oh", "ou", "happy", "sad", "angry", "relaxed"] as const).map((channel) => {
                      const composites = nano.avatar_expressions ?? {};
                      const surprised = composites["curious"] ?? {};
                      const value = surprised[channel] ?? 0;
                      return (
                        <div key={channel} className="flex items-center gap-2">
                          <label className="text-xs text-muted-foreground w-12 text-right">{channel}</label>
                          <input
                            type="range"
                            min={0}
                            max={1}
                            step={0.01}
                            value={value}
                            onChange={(e) => {
                              const newVal = parseFloat(e.target.value);
                              const existing = nano.avatar_expressions ?? {};
                              const existingCurious = existing["curious"] ?? {};
                              const updated = { ...existingCurious, [channel]: newVal };
                              // Remove zero-value keys to keep the data clean
                              if (newVal === 0) delete updated[channel];
                              const newExpressions = {
                                ...existing,
                                curious: Object.keys(updated).length > 0 ? updated : undefined,
                              };
                              updateSpindl("avatar_expressions", newExpressions);
                              // NANO-098: Live preview — only if editing the active character
                              if (selectedCharacterId && selectedCharacterId === activeCharacterId) {
                                try {
                                  const socket = getSocket();
                                  const cleanExpressions: Record<string, Record<string, number>> = {};
                                  for (const [k, v] of Object.entries(newExpressions)) {
                                    if (v != null) cleanExpressions[k] = v;
                                  }
                                  socket.emit("preview_avatar_expressions", {
                                    expressions: cleanExpressions,
                                    previewMood: "curious",
                                  });
                                  hasPreviewedRef.current = true;
                                } catch { /* socket may not be connected */ }
                              }
                            }}
                            className="flex-1 h-1.5 accent-purple-500"
                          />
                          <span className="text-xs text-muted-foreground w-8">{value.toFixed(2)}</span>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* NANO-098 Session 3: Animation Config — emotion-driven clip selection */}
              {!isNew && (
                <div className="space-y-2">
                  <Label>Animations</Label>
                  <FieldStatusBadge variant="metadata" description="Emotion-driven animation clip selection — crossfade body clips based on classifier confidence" />
                  <p className="text-xs text-muted-foreground">
                    Map emotion confidence thresholds to Mixamo animation clips. Drop FBX files in the character&apos;s
                    <code className="mx-1 px-1 bg-muted rounded">animations/</code> folder or the global pool.
                  </p>

                  {/* Default Idle */}
                  <div className="flex items-center gap-2">
                    <label className="text-xs text-muted-foreground w-20 text-right">Default Idle</label>
                    <select
                      className="flex-1 h-8 rounded-md border border-input bg-background px-2 text-sm"
                      value={(nano.avatar_animations?.default) ?? ""}
                      onChange={(e) => {
                        const val = e.target.value || undefined;
                        const existing = nano.avatar_animations ?? {};
                        updateSpindl("avatar_animations", { ...existing, default: val });
                        // Live preview: play the selected clip on the avatar
                        if (selectedCharacterId && selectedCharacterId === activeCharacterId) {
                          try {
                            getSocket().emit("preview_avatar_animation", { clip: val ?? null });
                            hasPreviewedAnimationRef.current = true;
                          } catch { /* socket may not be connected */ }
                        }
                      }}
                    >
                      <option value="">None</option>
                      {availableAnimations.map((a) => (
                        <option key={a.name} value={a.name}>
                          {a.name} ({a.source})
                        </option>
                      ))}
                    </select>
                  </div>

                  {/* Per-emotion rows */}
                  {(["amused", "melancholy", "annoyed", "curious"] as const).map((mood) => {
                    const emotions = nano.avatar_animations?.emotions ?? {};
                    const entry = emotions[mood] ?? { threshold: 0.75, clip: "" };
                    return (
                      <div key={mood} className="flex items-center gap-2">
                        <label className="text-xs text-muted-foreground w-20 text-right">{mood}</label>
                        <input
                          type="range"
                          min={0}
                          max={1}
                          step={0.01}
                          value={entry.threshold}
                          onChange={(e) => {
                            const newThreshold = parseFloat(e.target.value);
                            const existing = nano.avatar_animations ?? {};
                            const existingEmotions = existing.emotions ?? {};
                            const updated = {
                              ...existing,
                              emotions: {
                                ...existingEmotions,
                                [mood]: { ...entry, threshold: newThreshold },
                              },
                            };
                            updateSpindl("avatar_animations", updated);
                          }}
                          className="w-20 h-1.5 accent-purple-500"
                        />
                        <span className="text-xs text-muted-foreground w-8">{entry.threshold.toFixed(2)}</span>
                        <select
                          className="flex-1 h-8 rounded-md border border-input bg-background px-2 text-sm"
                          value={entry.clip}
                          onChange={(e) => {
                            const newClip = e.target.value;
                            const existing = nano.avatar_animations ?? {};
                            const existingEmotions = existing.emotions ?? {};
                            if (!newClip) {
                              // Remove this emotion entry entirely
                              const { [mood]: _, ...rest } = existingEmotions;
                              updateSpindl("avatar_animations", {
                                ...existing,
                                emotions: Object.keys(rest).length > 0 ? rest : undefined,
                              });
                            } else {
                              updateSpindl("avatar_animations", {
                                ...existing,
                                emotions: {
                                  ...existingEmotions,
                                  [mood]: { ...entry, clip: newClip },
                                },
                              });
                            }
                            // Live preview: play the selected emotion clip on the avatar
                            if (selectedCharacterId && selectedCharacterId === activeCharacterId) {
                              try {
                                getSocket().emit("preview_avatar_animation", { clip: newClip || null });
                                hasPreviewedAnimationRef.current = true;
                              } catch { /* socket may not be connected */ }
                            }
                          }}
                        >
                          <option value="">None</option>
                          {availableAnimations.map((a) => (
                            <option key={a.name} value={a.name}>
                              {a.name} ({a.source})
                            </option>
                          ))}
                        </select>
                      </div>
                    );
                  })}
                </div>
              )}
            </TabsContent>

            {/* ============================================================
                CODEX TAB — Character book entries (unchanged)
                ============================================================ */}
            <TabsContent value="codex" className="space-y-4 pr-4">
              <CharacterCodexTab
                characterBook={editedCard.data.character_book}
                onChange={(book: CharacterBook) => updateData("character_book", book)}
                onCharacterSave={(updatedBook: CharacterBook) => {
                  // NANO-039: Cascade codex entry save to character persistence
                  const updatedCard = {
                    ...editedCard,
                    data: {
                      ...editedCard.data,
                      character_book: updatedBook,
                    },
                  };
                  onSave(updatedCard);
                }}
              />
            </TabsContent>
          </ScrollArea>
        </Tabs>
      </CardContent>
    </Card>
  );
}
