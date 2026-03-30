"use client";

import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Textarea } from "@/components/ui/textarea";
import { Plus, X, Loader2 } from "lucide-react";

interface MemoryAddFormProps {
  content: string;
  onContentChange: (content: string) => void;
  onSave: () => void;
  onCancel: () => void;
  isSaving: boolean;
  isGlobal?: boolean;
}

export function MemoryAddForm({
  content,
  onContentChange,
  onSave,
  onCancel,
  isSaving,
  isGlobal = false,
}: MemoryAddFormProps) {
  const canSave = content.trim().length > 0 && !isSaving;

  return (
    <Card className="h-full">
      <CardHeader className="flex flex-row items-center justify-between py-3 px-4">
        <CardTitle className="flex items-center gap-2 text-base">
          <Plus className="h-5 w-5" />
          {isGlobal ? "Add Global Memory" : "Add Character Memory"}
        </CardTitle>
        <Button variant="ghost" size="icon" onClick={onCancel} className="h-8 w-8">
          <X className="h-4 w-4" />
        </Button>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="space-y-2">
          <label className="text-sm font-medium text-muted-foreground">
            Memory Content
          </label>
          <Textarea
            value={content}
            onChange={(e) => onContentChange(e.target.value)}
            placeholder={isGlobal
              ? "Enter a fact shared across all characters (e.g., user preferences, world knowledge)..."
              : "Enter a fact or knowledge for this character to remember permanently..."
            }
            className="min-h-[200px] resize-y"
            autoFocus
          />
          <p className="text-xs text-muted-foreground">
            {isGlobal
              ? "Global memories are cross-character facts. They are included in RAG retrieval for all characters."
              : "Character memories are permanent, per-character facts. They are included in RAG retrieval for this character."
            }
          </p>
        </div>
        <div className="flex gap-2">
          <Button onClick={onSave} disabled={!canSave} className="flex-1">
            {isSaving ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Plus className="h-4 w-4 mr-2" />
            )}
            Save Memory
          </Button>
          <Button variant="outline" onClick={onCancel}>
            Cancel
          </Button>
        </div>
      </CardContent>
    </Card>
  );
}
