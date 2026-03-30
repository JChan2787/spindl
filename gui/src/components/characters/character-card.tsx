"use client";

import { useCallback } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { cn } from "@/lib/utils";
import { User, Volume2, Play, Loader2 } from "lucide-react";
import { getSocket } from "@/lib/socket";
import { useCharacterStore } from "@/lib/stores";
import type { CharacterInfo } from "@/types/events";

interface CharacterCardProps {
  character: CharacterInfo;
  isActive?: boolean;
  isSelected?: boolean;
  avatarUrl?: string | null;
  onClick?: () => void;
}

export function CharacterCard({
  character,
  isActive = false,
  isSelected = false,
  avatarUrl,
  onClick,
}: CharacterCardProps) {
  const { isSwitchingCharacter, setSwitchingCharacter } = useCharacterStore();

  const handleSetActive = useCallback(
    (e: React.MouseEvent) => {
      e.stopPropagation();
      if (isActive || isSwitchingCharacter) return;
      setSwitchingCharacter(true);
      const socket = getSocket();
      socket.emit("set_persona", { persona_id: character.id });
    },
    [isActive, isSwitchingCharacter, setSwitchingCharacter, character.id]
  );

  return (
    <Card
      className={cn(
        "cursor-pointer transition-all hover:border-primary/50",
        isActive && "border-green-500 ring-1 ring-green-500",
        isSelected && !isActive && "border-primary bg-muted/30"
      )}
      onClick={onClick}
    >
      <CardContent className="p-4">
        <div className="flex items-start gap-3">
          <Avatar className="h-12 w-12">
            {avatarUrl ? (
              <AvatarImage src={avatarUrl} alt={character.name} />
            ) : null}
            <AvatarFallback>
              <User className="h-6 w-6" />
            </AvatarFallback>
          </Avatar>

          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2">
              <h3 className="font-semibold truncate">{character.name}</h3>
              {isActive && (
                <Badge variant="default" className="text-xs">
                  Active
                </Badge>
              )}
            </div>

            {character.description && (
              <p className="text-sm text-muted-foreground line-clamp-2 mt-1">
                {character.description}
              </p>
            )}

            <div className="flex items-center justify-between mt-2">
              <div className="flex items-center gap-2">
                {character.voice && (
                  <div className="flex items-center gap-1 text-xs text-muted-foreground">
                    <Volume2 className="h-3 w-3" />
                    <span>{character.voice}</span>
                  </div>
                )}
                {character.tags.length > 0 && (
                  <div className="flex gap-1">
                    {character.tags.slice(0, 2).map((tag) => (
                      <Badge key={tag} variant="secondary" className="text-xs">
                        {tag}
                      </Badge>
                    ))}
                    {character.tags.length > 2 && (
                      <Badge variant="secondary" className="text-xs">
                        +{character.tags.length - 2}
                      </Badge>
                    )}
                  </div>
                )}
              </div>

              {/* NANO-077: Set Active button */}
              {!isActive && (
                <Button
                  variant="outline"
                  size="sm"
                  className="h-7 text-xs"
                  disabled={isSwitchingCharacter}
                  onClick={handleSetActive}
                >
                  {isSwitchingCharacter ? (
                    <Loader2 className="h-3 w-3 animate-spin mr-1" />
                  ) : (
                    <Play className="h-3 w-3 mr-1" />
                  )}
                  Set Active
                </Button>
              )}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
