"use client";

import { useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ScrollArea } from "@/components/ui/scroll-area";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
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
import { Save, Trash2, X, Plus, Loader2, AlertCircle } from "lucide-react";
import type { CharacterBookEntry } from "@/types/events";

interface CodexEntryFormProps {
  entry: CharacterBookEntry;
  isNew: boolean;
  isSaving: boolean;
  error: string | null;
  onChange: (entry: CharacterBookEntry) => void;
  onSave: () => void;
  onDelete?: () => void;
  onCancel: () => void;
}

export function CodexEntryForm({
  entry,
  isNew,
  isSaving,
  error,
  onChange,
  onSave,
  onDelete,
  onCancel,
}: CodexEntryFormProps) {
  const [newKey, setNewKey] = useState("");
  const [newSecondaryKey, setNewSecondaryKey] = useState("");

  // Update a single field
  const updateField = <K extends keyof CharacterBookEntry>(
    field: K,
    value: CharacterBookEntry[K]
  ) => {
    onChange({ ...entry, [field]: value });
  };

  // Add a primary key
  const addKey = () => {
    if (!newKey.trim()) return;
    const keys = entry.keys || [];
    if (!keys.includes(newKey.trim())) {
      updateField("keys", [...keys, newKey.trim()]);
    }
    setNewKey("");
  };

  // Remove a primary key
  const removeKey = (key: string) => {
    updateField("keys", (entry.keys || []).filter((k) => k !== key));
  };

  // Add a secondary key
  const addSecondaryKey = () => {
    if (!newSecondaryKey.trim()) return;
    const secondaryKeys = entry.secondary_keys || [];
    if (!secondaryKeys.includes(newSecondaryKey.trim())) {
      updateField("secondary_keys", [...secondaryKeys, newSecondaryKey.trim()]);
    }
    setNewSecondaryKey("");
  };

  // Remove a secondary key
  const removeSecondaryKey = (key: string) => {
    updateField(
      "secondary_keys",
      (entry.secondary_keys || []).filter((k) => k !== key)
    );
  };

  // Get selective logic from extensions
  const getSelectiveLogic = (): string => {
    return (entry.extensions?.selective_logic as string) || "AND_ANY";
  };

  // Set selective logic
  const setSelectiveLogic = (value: string) => {
    updateField("extensions", {
      ...entry.extensions,
      selective_logic: value,
    });
  };

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-lg">
          {isNew ? "New Codex Entry" : entry.name || `Entry #${entry.id}`}
        </CardTitle>
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={onCancel}>
            <X className="h-4 w-4 mr-1" />
            Cancel
          </Button>
          {!isNew && onDelete && (
            <AlertDialog>
              <AlertDialogTrigger asChild>
                <Button variant="destructive" size="sm">
                  <Trash2 className="h-4 w-4 mr-1" />
                  Delete
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Delete Codex Entry</AlertDialogTitle>
                  <AlertDialogDescription>
                    Are you sure you want to delete this entry? This action
                    cannot be undone.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction onClick={onDelete}>Delete</AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          )}
          <Button size="sm" onClick={onSave} disabled={isSaving}>
            {isSaving ? (
              <Loader2 className="h-4 w-4 mr-1 animate-spin" />
            ) : (
              <Save className="h-4 w-4 mr-1" />
            )}
            Save
          </Button>
        </div>
      </CardHeader>

      {error && (
        <div className="mx-6 mb-2 p-2 bg-destructive/10 border border-destructive/20 rounded-md flex items-center gap-2 text-sm text-destructive">
          <AlertCircle className="h-4 w-4" />
          {error}
        </div>
      )}

      <CardContent>
        <Tabs defaultValue="basic" className="w-full">
          <TabsList className="w-full">
            <TabsTrigger value="basic" className="flex-1">
              Basic
            </TabsTrigger>
            <TabsTrigger value="keys" className="flex-1">
              Keywords
            </TabsTrigger>
            <TabsTrigger value="timing" className="flex-1">
              Timing
            </TabsTrigger>
            <TabsTrigger value="advanced" className="flex-1">
              Advanced
            </TabsTrigger>
          </TabsList>

          <ScrollArea className="h-[calc(100vh-380px)] mt-4">
            {/* Basic Tab */}
            <TabsContent value="basic" className="space-y-4 pr-4">
              {/* Name */}
              <div className="space-y-2">
                <Label htmlFor="name">Entry Name</Label>
                <Input
                  id="name"
                  value={entry.name || ""}
                  onChange={(e) => updateField("name", e.target.value)}
                  placeholder="A descriptive name for this entry"
                />
              </div>

              {/* Content */}
              <div className="space-y-2">
                <Label htmlFor="content">Content *</Label>
                <Textarea
                  id="content"
                  value={entry.content}
                  onChange={(e) => updateField("content", e.target.value)}
                  placeholder="The text to inject when this entry activates..."
                  rows={8}
                />
                <p className="text-xs text-muted-foreground">
                  This content will be injected into the prompt when keywords match.
                </p>
              </div>

              {/* Comment */}
              <div className="space-y-2">
                <Label htmlFor="comment">Comment</Label>
                <Textarea
                  id="comment"
                  value={entry.comment || ""}
                  onChange={(e) => updateField("comment", e.target.value)}
                  placeholder="Internal notes (not sent to LLM)..."
                  rows={2}
                />
              </div>

              {/* Enabled */}
              <div className="flex items-center space-x-2">
                <Switch
                  id="enabled"
                  checked={entry.enabled}
                  onCheckedChange={(checked) => updateField("enabled", checked)}
                />
                <Label htmlFor="enabled">Enabled</Label>
              </div>

              {/* Constant */}
              <div className="flex items-center space-x-2">
                <Switch
                  id="constant"
                  checked={entry.constant || false}
                  onCheckedChange={(checked) => updateField("constant", checked)}
                />
                <Label htmlFor="constant">Constant</Label>
                <span className="text-xs text-muted-foreground ml-2">
                  (Always active, ignores keywords)
                </span>
              </div>
            </TabsContent>

            {/* Keywords Tab */}
            <TabsContent value="keys" className="space-y-4 pr-4">
              {/* Primary Keys */}
              <div className="space-y-2">
                <Label>Primary Keywords</Label>
                <div className="flex gap-2">
                  <Input
                    value={newKey}
                    onChange={(e) => setNewKey(e.target.value)}
                    placeholder="Add a keyword..."
                    onKeyDown={(e) =>
                      e.key === "Enter" && (e.preventDefault(), addKey())
                    }
                  />
                  <Button variant="outline" onClick={addKey}>
                    <Plus className="h-4 w-4" />
                  </Button>
                </div>
                {(entry.keys || []).length > 0 && (
                  <div className="flex flex-wrap gap-1 mt-2">
                    {(entry.keys || []).map((key) => (
                      <Badge
                        key={key}
                        variant="secondary"
                        className="cursor-pointer"
                        onClick={() => removeKey(key)}
                      >
                        {key}
                        <X className="h-3 w-3 ml-1" />
                      </Badge>
                    ))}
                  </div>
                )}
                <p className="text-xs text-muted-foreground">
                  Entry activates if ANY primary keyword matches. Use /regex/ for regex patterns.
                </p>
              </div>

              {/* Selective (Secondary Keys) */}
              <div className="flex items-center space-x-2">
                <Switch
                  id="selective"
                  checked={entry.selective || false}
                  onCheckedChange={(checked) => updateField("selective", checked)}
                />
                <Label htmlFor="selective">Use Secondary Keywords</Label>
              </div>

              {entry.selective && (
                <>
                  {/* Secondary Key Logic */}
                  <div className="space-y-2">
                    <Label>Secondary Key Logic</Label>
                    <Select
                      value={getSelectiveLogic()}
                      onValueChange={setSelectiveLogic}
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="AND_ANY">
                          AND ANY (primary AND any secondary)
                        </SelectItem>
                        <SelectItem value="AND_ALL">
                          AND ALL (primary AND all secondary)
                        </SelectItem>
                        <SelectItem value="NOT_ANY">
                          NOT ANY (primary AND NOT any secondary)
                        </SelectItem>
                        <SelectItem value="NOT_ALL">
                          NOT ALL (primary AND NOT all secondary)
                        </SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  {/* Secondary Keys */}
                  <div className="space-y-2">
                    <Label>Secondary Keywords</Label>
                    <div className="flex gap-2">
                      <Input
                        value={newSecondaryKey}
                        onChange={(e) => setNewSecondaryKey(e.target.value)}
                        placeholder="Add a secondary keyword..."
                        onKeyDown={(e) =>
                          e.key === "Enter" &&
                          (e.preventDefault(), addSecondaryKey())
                        }
                      />
                      <Button variant="outline" onClick={addSecondaryKey}>
                        <Plus className="h-4 w-4" />
                      </Button>
                    </div>
                    {(entry.secondary_keys || []).length > 0 && (
                      <div className="flex flex-wrap gap-1 mt-2">
                        {(entry.secondary_keys || []).map((key) => (
                          <Badge
                            key={key}
                            variant="outline"
                            className="cursor-pointer"
                            onClick={() => removeSecondaryKey(key)}
                          >
                            {key}
                            <X className="h-3 w-3 ml-1" />
                          </Badge>
                        ))}
                      </div>
                    )}
                  </div>
                </>
              )}

              {/* Case Sensitive */}
              <div className="flex items-center space-x-2">
                <Switch
                  id="case_sensitive"
                  checked={entry.case_sensitive || false}
                  onCheckedChange={(checked) =>
                    updateField("case_sensitive", checked)
                  }
                />
                <Label htmlFor="case_sensitive">Case Sensitive</Label>
              </div>
            </TabsContent>

            {/* Timing Tab */}
            <TabsContent value="timing" className="space-y-4 pr-4">
              {/* Sticky */}
              <div className="space-y-2">
                <Label htmlFor="sticky">Sticky Duration</Label>
                <Input
                  id="sticky"
                  type="number"
                  min="0"
                  value={entry.sticky ?? ""}
                  onChange={(e) =>
                    updateField(
                      "sticky",
                      e.target.value ? parseInt(e.target.value) : undefined
                    )
                  }
                  placeholder="Number of turns"
                />
                <p className="text-xs text-muted-foreground">
                  Entry remains active for N turns after triggering. Leave empty for no stickiness.
                </p>
              </div>

              {/* Cooldown */}
              <div className="space-y-2">
                <Label htmlFor="cooldown">Cooldown Duration</Label>
                <Input
                  id="cooldown"
                  type="number"
                  min="0"
                  value={entry.cooldown ?? ""}
                  onChange={(e) =>
                    updateField(
                      "cooldown",
                      e.target.value ? parseInt(e.target.value) : undefined
                    )
                  }
                  placeholder="Number of turns"
                />
                <p className="text-xs text-muted-foreground">
                  Entry cannot re-activate for N turns after triggering. Leave empty for no cooldown.
                </p>
              </div>

              {/* Delay */}
              <div className="space-y-2">
                <Label htmlFor="delay">Activation Delay</Label>
                <Input
                  id="delay"
                  type="number"
                  min="0"
                  value={entry.delay ?? ""}
                  onChange={(e) =>
                    updateField(
                      "delay",
                      e.target.value ? parseInt(e.target.value) : undefined
                    )
                  }
                  placeholder="Number of turns"
                />
                <p className="text-xs text-muted-foreground">
                  Entry cannot activate until turn N. Leave empty for immediate availability.
                </p>
              </div>
            </TabsContent>

            {/* Advanced Tab */}
            <TabsContent value="advanced" className="space-y-4 pr-4">
              {/* Position */}
              <div className="space-y-2">
                <Label>Insertion Position</Label>
                <Select
                  value={entry.position || "after_char"}
                  onValueChange={(value) =>
                    updateField("position", value as "before_char" | "after_char")
                  }
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="before_char">Before Character Definition</SelectItem>
                    <SelectItem value="after_char">After Character Definition</SelectItem>
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">
                  Where in the system prompt this entry&apos;s content appears.
                </p>
              </div>

              {/* Priority */}
              <div className="space-y-2">
                <Label htmlFor="priority">Priority</Label>
                <Input
                  id="priority"
                  type="number"
                  value={entry.priority ?? 10}
                  onChange={(e) =>
                    updateField(
                      "priority",
                      e.target.value ? parseInt(e.target.value) : undefined
                    )
                  }
                  placeholder="10"
                />
                <p className="text-xs text-muted-foreground">
                  Higher priority entries are processed first and take precedence in budget.
                </p>
              </div>

              {/* Insertion Order */}
              <div className="space-y-2">
                <Label htmlFor="insertion_order">Insertion Order</Label>
                <Input
                  id="insertion_order"
                  type="number"
                  value={entry.insertion_order}
                  onChange={(e) =>
                    updateField(
                      "insertion_order",
                      e.target.value ? parseInt(e.target.value) : 0
                    )
                  }
                  placeholder="0"
                />
                <p className="text-xs text-muted-foreground">
                  Lower values appear earlier in the prompt. Used for ordering within the same position.
                </p>
              </div>

              {/* ID (read-only for existing entries) */}
              {!isNew && entry.id !== undefined && (
                <div className="space-y-2">
                  <Label>Entry ID</Label>
                  <Input value={entry.id.toString()} disabled />
                  <p className="text-xs text-muted-foreground">
                    Auto-assigned identifier. Cannot be changed.
                  </p>
                </div>
              )}
            </TabsContent>
          </ScrollArea>
        </Tabs>
      </CardContent>
    </Card>
  );
}
