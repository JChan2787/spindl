"use client";

import { useEffect, useCallback, useState } from "react";
import { useRouter, useParams } from "next/navigation";
import { CharacterEditor } from "@/components/characters";
import { Button } from "@/components/ui/button";
import { useCharacterStore, useConnectionStore, reloadCharacter } from "@/lib/stores";
import { getSocket } from "@/lib/socket";
import { ArrowLeft, Loader2 } from "lucide-react";

export default function CharacterEditPage() {
  const router = useRouter();
  const params = useParams();
  const characterId = params.id as string;

  const { connected: isConnected } = useConnectionStore();
  const {
    selectedCharacterId,
    editedCard,
    isLoadingDetail,
    selectCharacter,
    setActionResult,
    clearActionResult,
    lastAction,
  } = useCharacterStore();

  const socket = getSocket();
  const [isSaving, setIsSaving] = useState(false);

  // Select this character on mount
  useEffect(() => {
    if (characterId && characterId !== selectedCharacterId) {
      selectCharacter(characterId);
    }
  }, [characterId, selectedCharacterId, selectCharacter]);

  // Load character detail when selected and connected
  useEffect(() => {
    if (selectedCharacterId && isConnected) {
      socket.emit("request_character", { character_id: selectedCharacterId });
    }
  }, [selectedCharacterId, isConnected, socket]);

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
      if (lastAction.success && lastAction.type === "delete") {
        // After successful delete, go back to list
        router.push("/characters");
      }
      // NANO-036 Phase 3: Trigger backend hot-reload after successful update
      if (lastAction.success && lastAction.type === "update") {
        reloadCharacter().then((result) => {
          if (!result.success) {
            console.log("[CharacterEdit] Reload pending:", result.error);
          }
        });
      }
    }
  }, [lastAction, router]);

  const handleSave = useCallback(
    (card: typeof editedCard) => {
      if (!card || !selectedCharacterId) return;
      setIsSaving(true);
      socket.emit("update_character", {
        character_id: selectedCharacterId,
        card,
      });
    },
    [selectedCharacterId, socket]
  );

  const handleDelete = useCallback(() => {
    if (!selectedCharacterId) return;
    socket.emit("delete_character", { character_id: selectedCharacterId });
  }, [selectedCharacterId, socket]);

  const handleCancel = useCallback(() => {
    router.push("/characters");
  }, [router]);

  const handleAvatarUpload = useCallback(
    (imageData: string) => {
      if (!selectedCharacterId) return;
      socket.emit("upload_avatar", {
        character_id: selectedCharacterId,
        image_data: imageData,
      });
    },
    [selectedCharacterId, socket]
  );

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-4">
        <Button variant="ghost" size="sm" onClick={() => router.push("/characters")}>
          <ArrowLeft className="h-4 w-4 mr-1" />
          Back to Characters
        </Button>
      </div>

      {isLoadingDetail ? (
        <div className="flex items-center justify-center py-16 text-muted-foreground">
          <Loader2 className="h-6 w-6 animate-spin mr-2" />
          Loading character...
        </div>
      ) : (
        <CharacterEditor
          onSave={handleSave}
          onDelete={handleDelete}
          onCancel={handleCancel}
          onAvatarUpload={handleAvatarUpload}
          isNew={false}
          isSaving={isSaving}
        />
      )}
    </div>
  );
}
