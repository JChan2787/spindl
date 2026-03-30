"use client";

import { useEffect } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { CharacterCard } from "./character-card";
import { useCharacterStore, fetchAvatar } from "@/lib/stores";
import { RefreshCw, Plus, Loader2 } from "lucide-react";

interface CharacterListProps {
  onSelect: (characterId: string) => void;
  onCreate: () => void;
  onRefresh: () => void;
}

export function CharacterList({ onSelect, onCreate, onRefresh }: CharacterListProps) {
  const {
    characters,
    activeCharacterId,
    selectedCharacterId,
    isLoading,
    avatarCache,
    setAvatar,
  } = useCharacterStore();

  // Request avatars for characters that have them but aren't in cache
  useEffect(() => {
    const loadAvatars = async () => {
      for (const char of characters) {
        if (char.has_avatar && !(char.id in avatarCache)) {
          try {
            const data = await fetchAvatar(char.id);
            setAvatar(char.id, data.image_data);
          } catch (error) {
            console.error(`Failed to load avatar for ${char.id}:`, error);
            // Mark as loaded (with null) to prevent retry loops
            setAvatar(char.id, null);
          }
        }
      }
    };
    loadAvatars();
  }, [characters, avatarCache, setAvatar]);

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-lg">Characters</CardTitle>
        <div className="flex gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={onRefresh}
            disabled={isLoading}
          >
            {isLoading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <RefreshCw className="h-4 w-4" />
            )}
          </Button>
          <Button size="sm" onClick={onCreate}>
            <Plus className="h-4 w-4 mr-1" />
            New
          </Button>
        </div>
      </CardHeader>
      <CardContent>
        <ScrollArea className="h-[calc(100vh-280px)]">
          {isLoading && characters.length === 0 ? (
            <div className="flex items-center justify-center py-8 text-muted-foreground">
              <Loader2 className="h-6 w-6 animate-spin mr-2" />
              Loading characters...
            </div>
          ) : characters.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              <p>No characters found.</p>
              <p className="text-sm mt-1">Create your first character to get started.</p>
            </div>
          ) : (
            <div className="space-y-2 pr-4">
              {characters.map((character) => (
                <CharacterCard
                  key={character.id}
                  character={character}
                  isActive={character.id === activeCharacterId}
                  isSelected={character.id === selectedCharacterId}
                  avatarUrl={avatarCache[character.id]}
                  onClick={() => onSelect(character.id)}
                />
              ))}
            </div>
          )}
        </ScrollArea>
      </CardContent>
    </Card>
  );
}
