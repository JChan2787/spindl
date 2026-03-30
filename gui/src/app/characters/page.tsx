"use client";

import { useEffect, useCallback, useState, useRef } from "react";
import { CharacterList, CharacterEditor, AvatarCropModal, ImportDialog, ExportDialog } from "@/components/characters";
import {
  useCharacterStore,
  createEmptyCharacterCard,
  fetchCharacters,
  fetchCharacterDetail,
  createCharacterApi,
  updateCharacterApi,
  deleteCharacterApi,
  uploadAvatarApi,
  reloadCharacter,
} from "@/lib/stores";
import { Button } from "@/components/ui/button";
import { Upload, Download } from "lucide-react";
import type { CropSettings } from "@/types/events";

export default function CharactersPage() {
  const {
    selectedCharacterId,
    editedCard,
    setLoading,
    setCharacters,
    selectCharacter,
    setCharacterDetail,
    setEditedCard,
    setActionResult,
    setActionError,
    clearActionResult,
    lastAction,
    selectedCharacterCard,
    removeCharacter,
    setAvatar,
    resetEditor,
  } = useCharacterStore();

  const [isCreatingNew, setIsCreatingNew] = useState(false);
  const [isSaving, setIsSaving] = useState(false);
  const [importDialogOpen, setImportDialogOpen] = useState(false);
  const [exportDialogOpen, setExportDialogOpen] = useState(false);

  // Avatar-first creation state
  const [pendingAvatar, setPendingAvatar] = useState<{
    cropped: string;
    original: string;
    settings: CropSettings;
  } | null>(null);
  const [showCropModal, setShowCropModal] = useState(false);
  const [selectedImageForCrop, setSelectedImageForCrop] = useState<string | null>(null);
  const creationFileInputRef = useRef<HTMLInputElement>(null);

  // Load characters on mount (no Socket.IO dependency!)
  useEffect(() => {
    const loadCharacters = async () => {
      setLoading(true);
      try {
        const data = await fetchCharacters();
        setCharacters(data.characters, data.active);
      } catch (error) {
        console.error("Failed to load characters:", error);
        setActionError(error instanceof Error ? error.message : "Failed to load characters");
      }
    };
    loadCharacters();
  }, [setLoading, setCharacters, setActionError]);

  // Load character detail when selected
  useEffect(() => {
    if (selectedCharacterId && !isCreatingNew) {
      const loadDetail = async () => {
        try {
          const detail = await fetchCharacterDetail(selectedCharacterId);
          setCharacterDetail(detail);
        } catch (error) {
          console.error("Failed to load character detail:", error);
          setActionError(error instanceof Error ? error.message : "Failed to load character");
        }
      };
      loadDetail();
    }
  }, [selectedCharacterId, isCreatingNew, setCharacterDetail, setActionError]);

  // Clear action feedback after 3 seconds
  useEffect(() => {
    if (lastAction.type && lastAction.success) {
      const timer = setTimeout(() => {
        clearActionResult();
      }, 3000);
      return () => clearTimeout(timer);
    }
  }, [lastAction, clearActionResult]);

  // Reset saving state when action completes
  useEffect(() => {
    if (lastAction.success !== null) {
      setIsSaving(false);
      if (lastAction.success && lastAction.type === "create") {
        // After successful create, select the new character
        if (lastAction.characterId) {
          setIsCreatingNew(false);
          selectCharacter(lastAction.characterId);
        }
      }
    }
  }, [lastAction, selectCharacter]);

  const handleRefresh = useCallback(async () => {
    setLoading(true);
    try {
      const data = await fetchCharacters();
      setCharacters(data.characters, data.active);
    } catch (error) {
      console.error("Failed to refresh characters:", error);
      setActionError(error instanceof Error ? error.message : "Failed to refresh characters");
    }
  }, [setLoading, setCharacters, setActionError]);

  const handleSelect = useCallback(
    (characterId: string) => {
      setIsCreatingNew(false);
      setPendingAvatar(null);
      selectCharacter(characterId);
    },
    [selectCharacter]
  );

  // Avatar-first creation: trigger file picker immediately
  const handleCreate = useCallback(() => {
    creationFileInputRef.current?.click();
  }, []);

  // File selected for new character creation
  const handleCreationFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
      const dataUrl = reader.result as string;
      setSelectedImageForCrop(dataUrl);
      setShowCropModal(true);
    };
    reader.readAsDataURL(file);

    // Reset to allow re-selecting same file
    e.target.value = "";
  }, []);

  // Crop confirmed — show editor form with avatar pre-loaded
  const handleCropConfirm = useCallback((
    croppedData: string,
    originalData: string,
    settings: CropSettings,
  ) => {
    setPendingAvatar({ cropped: croppedData, original: originalData, settings });
    selectCharacter(null);
    setIsCreatingNew(true);
    setEditedCard(createEmptyCharacterCard());
    setShowCropModal(false);
    setSelectedImageForCrop(null);
  }, [selectCharacter, setEditedCard]);

  // Crop modal closed without confirming
  const handleCropModalChange = useCallback((open: boolean) => {
    setShowCropModal(open);
    if (!open) {
      setSelectedImageForCrop(null);
    }
  }, []);

  const handleSave = useCallback(
    async (card: typeof editedCard) => {
      if (!card) return;
      setIsSaving(true);

      try {
        if (isCreatingNew) {
          const result = await createCharacterApi(
            card,
            undefined,
            pendingAvatar || undefined,
          );
          setActionResult("create", result.character_id, true);

          // Populate avatar cache if we had a pending avatar
          if (pendingAvatar) {
            setAvatar(result.character_id, pendingAvatar.cropped);
            setPendingAvatar(null);
          }

          // Refresh character list
          const data = await fetchCharacters();
          setCharacters(data.characters, data.active);
        } else if (selectedCharacterId) {
          const result = await updateCharacterApi(selectedCharacterId, card);

          // NANO-036 Phase 3: Trigger backend hot-reload after successful save
          const reloadResult = await reloadCharacter();
          if (reloadResult.success) {
            setActionResult("update", result.character_id, true);
          } else {
            // Save succeeded but reload pending (agent busy or socket disconnected)
            setActionResult(
              "update",
              result.character_id,
              true,
              `Saved. Reload pending: ${reloadResult.error || "agent busy"}`
            );
          }

          // Refresh the detail to get the saved state
          const detail = await fetchCharacterDetail(selectedCharacterId);
          setCharacterDetail(detail);
          // Also refresh the list in case name changed
          const data = await fetchCharacters();
          setCharacters(data.characters, data.active);
        }
      } catch (error) {
        console.error("Failed to save character:", error);
        const errorMessage = error instanceof Error ? error.message : "Failed to save character";
        setActionResult(
          isCreatingNew ? "create" : "update",
          selectedCharacterId || "",
          false,
          errorMessage
        );
      }
    },
    [isCreatingNew, selectedCharacterId, pendingAvatar, setActionResult, setCharacters, setCharacterDetail, setAvatar]
  );

  const handleDelete = useCallback(async () => {
    if (!selectedCharacterId) return;

    try {
      await deleteCharacterApi(selectedCharacterId);
      setActionResult("delete", selectedCharacterId, true);
      removeCharacter(selectedCharacterId);
      // Refresh character list
      const data = await fetchCharacters();
      setCharacters(data.characters, data.active);
    } catch (error) {
      console.error("Failed to delete character:", error);
      setActionResult(
        "delete",
        selectedCharacterId,
        false,
        error instanceof Error ? error.message : "Failed to delete character"
      );
    }
  }, [selectedCharacterId, setActionResult, removeCharacter, setCharacters]);

  const handleCancel = useCallback(() => {
    setIsCreatingNew(false);
    setPendingAvatar(null);
    resetEditor();
  }, [resetEditor]);

  const handleAvatarUpload = useCallback(
    async (imageData: string, originalData?: string, cropSettings?: CropSettings) => {
      if (!selectedCharacterId) return;

      try {
        await uploadAvatarApi(selectedCharacterId, imageData, originalData, cropSettings);
        setActionResult("avatar", selectedCharacterId, true);
        // Update avatar cache
        setAvatar(selectedCharacterId, imageData);
      } catch (error) {
        console.error("Failed to upload avatar:", error);
        setActionResult(
          "avatar",
          selectedCharacterId,
          false,
          error instanceof Error ? error.message : "Failed to upload avatar"
        );
      }
    },
    [selectedCharacterId, setActionResult, setAvatar]
  );

  const handleImportComplete = useCallback(async () => {
    // Refresh character list after import
    setLoading(true);
    try {
      const data = await fetchCharacters();
      setCharacters(data.characters, data.active);
    } catch (error) {
      console.error("Failed to refresh characters:", error);
    }
  }, [setLoading, setCharacters]);

  const handleExport = useCallback(() => {
    if (selectedCharacterId) {
      setExportDialogOpen(true);
    }
  }, [selectedCharacterId]);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">Characters</h1>
          <p className="text-muted-foreground">
            Manage your character cards for SpindL
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            onClick={() => setImportDialogOpen(true)}
          >
            <Upload className="h-4 w-4 mr-2" />
            Import
          </Button>
          <Button
            variant="outline"
            onClick={handleExport}
            disabled={!selectedCharacterId || isCreatingNew}
          >
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-1">
          <CharacterList
            onSelect={handleSelect}
            onCreate={handleCreate}
            onRefresh={handleRefresh}
          />
        </div>
        <div className="lg:col-span-2">
          <CharacterEditor
            onSave={handleSave}
            onDelete={handleDelete}
            onCancel={handleCancel}
            onAvatarUpload={handleAvatarUpload}
            isNew={isCreatingNew}
            isSaving={isSaving}
            pendingAvatarPreview={pendingAvatar?.cropped}
          />
        </div>
      </div>

      {/* Hidden file input for avatar-first creation */}
      <input
        ref={creationFileInputRef}
        type="file"
        accept="image/png,image/jpeg,image/webp"
        className="hidden"
        onChange={handleCreationFileSelect}
      />

      {/* Avatar Crop Modal (creation flow) */}
      {selectedImageForCrop && (
        <AvatarCropModal
          image={selectedImageForCrop}
          open={showCropModal}
          onOpenChange={handleCropModalChange}
          onConfirm={handleCropConfirm}
        />
      )}

      {/* Import/Export Dialogs */}
      <ImportDialog
        open={importDialogOpen}
        onOpenChange={setImportDialogOpen}
        onImportComplete={handleImportComplete}
      />
      <ExportDialog
        open={exportDialogOpen}
        onOpenChange={setExportDialogOpen}
        characterId={selectedCharacterId}
        characterName={selectedCharacterCard?.data?.name ?? "Character"}
      />
    </div>
  );
}
