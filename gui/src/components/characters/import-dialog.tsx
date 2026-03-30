"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Badge } from "@/components/ui/badge";
import { validateImportApi, importCharacterApi, importPngCharacterApi } from "@/lib/stores";
import { Upload, FileJson, FileImage, AlertCircle, CheckCircle, Loader2 } from "lucide-react";

interface ImportPreview {
  name: string;
  description: string;
  has_personality: boolean;
  has_system_prompt: boolean;
  has_codex: boolean;
  codex_count: number;
  has_spindl: boolean;
  tags: string[];
}

interface ValidationResult {
  valid: boolean;
  preview?: ImportPreview;
  errors?: string[];
  warnings?: string[];
}

interface ImportDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  onImportComplete: () => void;
}

export function ImportDialog({ open, onOpenChange, onImportComplete }: ImportDialogProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [jsonInput, setJsonInput] = useState("");
  const [customId, setCustomId] = useState("");
  const [overwrite, setOverwrite] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [isImporting, setIsImporting] = useState(false);
  const [validation, setValidation] = useState<ValidationResult | null>(null);
  const [importError, setImportError] = useState<string | null>(null);
  const [existsError, setExistsError] = useState(false);

  // PNG-specific state
  const [pngFile, setPngFile] = useState<File | null>(null);
  const [pngPreviewUrl, setPngPreviewUrl] = useState<string | null>(null);
  const [isPngMode, setIsPngMode] = useState(false);

  // Reset state when dialog opens/closes
  useEffect(() => {
    if (!open) {
      setJsonInput("");
      setCustomId("");
      setOverwrite(false);
      setValidation(null);
      setImportError(null);
      setExistsError(false);
      setIsValidating(false);
      setIsImporting(false);
      setPngFile(null);
      setIsPngMode(false);
      if (pngPreviewUrl) {
        URL.revokeObjectURL(pngPreviewUrl);
        setPngPreviewUrl(null);
      }
    }
  }, [open]); // eslint-disable-line react-hooks/exhaustive-deps

  // Validate JSON using REST API
  const doValidate = useCallback(async (content: string) => {
    setIsValidating(true);
    setValidation(null);
    setImportError(null);
    setExistsError(false);

    try {
      const result = await validateImportApi(content);
      setValidation(result);
    } catch (error) {
      console.error("Validation failed:", error);
      setValidation({
        valid: false,
        errors: [error instanceof Error ? error.message : "Validation failed"],
      });
    } finally {
      setIsValidating(false);
    }
  }, []);

  // Handle file upload
  const handleFileSelect = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    const filename = file.name.toLowerCase();

    if (filename.endsWith(".png")) {
      // PNG mode — store file, create preview URL, extract JSON for validation
      setPngFile(file);
      setIsPngMode(true);
      setJsonInput("");
      setValidation(null);
      setImportError(null);
      setExistsError(false);

      // Create preview URL
      const previewUrl = URL.createObjectURL(file);
      setPngPreviewUrl(previewUrl);

      // Read the PNG to extract JSON for validation/preview
      const reader = new FileReader();
      reader.onload = (e) => {
        const buffer = e.target?.result as ArrayBuffer;
        const jsonStr = extractCharaFromPngClient(new Uint8Array(buffer));
        if (jsonStr) {
          setJsonInput(jsonStr);
          doValidate(jsonStr);
        } else {
          setValidation({
            valid: false,
            errors: ["PNG file does not contain character data. Expected a tEXt chunk with key 'chara' (SillyTavern format)."],
          });
        }
      };
      reader.readAsArrayBuffer(file);
    } else {
      // JSON mode
      setPngFile(null);
      setIsPngMode(false);
      if (pngPreviewUrl) {
        URL.revokeObjectURL(pngPreviewUrl);
        setPngPreviewUrl(null);
      }

      const reader = new FileReader();
      reader.onload = (e) => {
        const content = e.target?.result as string;
        setJsonInput(content);
        setValidation(null);
        setImportError(null);
        setExistsError(false);
        doValidate(content);
      };
      reader.readAsText(file);
    }

    // Reset file input
    event.target.value = "";
  }, [doValidate, pngPreviewUrl]);

  // Validate current input
  const handleValidate = useCallback(() => {
    if (!jsonInput.trim()) return;
    doValidate(jsonInput);
  }, [jsonInput, doValidate]);

  // Perform import using REST API
  const handleImport = useCallback(async () => {
    if (!validation?.valid) return;

    setIsImporting(true);
    setImportError(null);

    try {
      let result;

      if (isPngMode && pngFile) {
        // PNG import via FormData
        result = await importPngCharacterApi(
          pngFile,
          customId.trim() || undefined,
          overwrite
        );
      } else {
        // JSON import (existing path)
        if (!jsonInput.trim()) return;
        result = await importCharacterApi(
          jsonInput,
          customId.trim() || undefined,
          overwrite
        );
      }

      if (result.error) {
        setImportError(result.error);
        if (result.exists) {
          setExistsError(true);
        }
      } else if (result.success) {
        onImportComplete();
        onOpenChange(false);
      }
    } catch (error) {
      console.error("Import failed:", error);
      setImportError(error instanceof Error ? error.message : "Import failed");
    } finally {
      setIsImporting(false);
    }
  }, [jsonInput, validation, customId, overwrite, onImportComplete, onOpenChange, isPngMode, pngFile]);

  // Trigger file picker
  const handleBrowse = useCallback(() => {
    fileInputRef.current?.click();
  }, []);

  // Clear PNG and return to JSON mode
  const handleClearPng = useCallback(() => {
    setPngFile(null);
    setIsPngMode(false);
    setJsonInput("");
    setValidation(null);
    setImportError(null);
    setExistsError(false);
    if (pngPreviewUrl) {
      URL.revokeObjectURL(pngPreviewUrl);
      setPngPreviewUrl(null);
    }
  }, [pngPreviewUrl]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Upload className="h-5 w-5" />
            Import Character
          </DialogTitle>
          <DialogDescription>
            Import a SillyTavern Character Card V2 file. Supports JSON files and PNG cards (Chub/Tavern format).
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          {/* File upload */}
          <div className="flex items-center gap-2">
            <Button variant="outline" onClick={handleBrowse} disabled={isImporting}>
              {isPngMode ? (
                <FileImage className="h-4 w-4 mr-2" />
              ) : (
                <FileJson className="h-4 w-4 mr-2" />
              )}
              Choose File
            </Button>
            <span className="text-sm text-muted-foreground">
              .json or .png{!isPngMode && " — or paste JSON below"}
            </span>
            <input
              ref={fileInputRef}
              type="file"
              accept=".json,.png"
              className="hidden"
              onChange={handleFileSelect}
            />
          </div>

          {/* PNG preview */}
          {isPngMode && pngPreviewUrl && (
            <div className="flex items-start gap-4 rounded-md border p-4">
              <img
                src={pngPreviewUrl}
                alt="Character avatar"
                className="w-24 h-24 rounded-md object-cover border"
              />
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-1">
                  <Badge variant="secondary">
                    <FileImage className="h-3 w-3 mr-1" />
                    PNG Card
                  </Badge>
                  <span className="text-sm text-muted-foreground truncate">
                    {pngFile?.name}
                  </span>
                </div>
                <p className="text-xs text-muted-foreground">
                  Avatar image will be saved automatically on import.
                </p>
                <Button
                  variant="ghost"
                  size="sm"
                  className="mt-1 h-7 text-xs"
                  onClick={handleClearPng}
                  disabled={isImporting}
                >
                  Clear
                </Button>
              </div>
            </div>
          )}

          {/* JSON input — only show if not in PNG mode */}
          {!isPngMode && (
            <div className="space-y-2">
              <Label htmlFor="json-input">Character Card JSON</Label>
              <Textarea
                id="json-input"
                placeholder='{"spec": "chara_card_v2", "data": {...}}'
                value={jsonInput}
                onChange={(e) => {
                  setJsonInput(e.target.value);
                  setValidation(null);
                  setImportError(null);
                  setExistsError(false);
                }}
                className="font-mono text-sm h-48"
                disabled={isImporting}
              />
            </div>
          )}

          {/* Validate button */}
          {jsonInput.trim() && !validation && !isPngMode && (
            <Button
              variant="secondary"
              onClick={handleValidate}
              disabled={isValidating}
            >
              {isValidating ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Validating...
                </>
              ) : (
                "Validate"
              )}
            </Button>
          )}

          {/* Validation errors */}
          {validation && !validation.valid && (
            <div className="rounded-md bg-destructive/10 p-3 text-destructive">
              <div className="flex items-center gap-2 font-medium">
                <AlertCircle className="h-4 w-4" />
                Validation Failed
              </div>
              <ul className="mt-2 text-sm list-disc list-inside">
                {validation.errors?.map((error, i) => (
                  <li key={i}>{error}</li>
                ))}
              </ul>
            </div>
          )}

          {/* Validation success + preview */}
          {validation?.valid && validation.preview && (
            <div className="rounded-md border p-4 space-y-3">
              <div className="flex items-center gap-2 text-green-600 dark:text-green-400 font-medium">
                <CheckCircle className="h-4 w-4" />
                Valid Character Card
              </div>

              <div className="space-y-2">
                <div>
                  <span className="font-medium">Name:</span>{" "}
                  <span>{validation.preview.name}</span>
                </div>

                {validation.preview.description && (
                  <div>
                    <span className="font-medium">Description:</span>{" "}
                    <span className="text-muted-foreground">
                      {validation.preview.description}
                    </span>
                  </div>
                )}

                <div className="flex flex-wrap gap-2">
                  {validation.preview.has_personality && (
                    <Badge variant="secondary">Personality</Badge>
                  )}
                  {validation.preview.has_system_prompt && (
                    <Badge variant="secondary">System Prompt</Badge>
                  )}
                  {validation.preview.has_codex && (
                    <Badge variant="secondary">
                      Codex ({validation.preview.codex_count} entries)
                    </Badge>
                  )}
                  {validation.preview.has_spindl && (
                    <Badge variant="outline">SpindL extensions</Badge>
                  )}
                  {isPngMode && (
                    <Badge variant="secondary">
                      <FileImage className="h-3 w-3 mr-1" />
                      Avatar included
                    </Badge>
                  )}
                </div>

                {validation.preview.tags.length > 0 && (
                  <div className="flex flex-wrap gap-1">
                    {validation.preview.tags.map((tag) => (
                      <Badge key={tag} variant="outline" className="text-xs">
                        {tag}
                      </Badge>
                    ))}
                  </div>
                )}
              </div>

              {/* Warnings */}
              {validation.warnings && validation.warnings.length > 0 && (
                <div className="rounded-md bg-yellow-500/10 p-2 text-yellow-600 dark:text-yellow-400 text-sm">
                  <ul className="list-disc list-inside">
                    {validation.warnings.map((warning, i) => (
                      <li key={i}>{warning}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          )}

          {/* Import options */}
          {validation?.valid && (
            <div className="space-y-4 pt-2">
              <div className="space-y-2">
                <Label htmlFor="custom-id">Custom Character ID (optional)</Label>
                <Input
                  id="custom-id"
                  placeholder={validation.preview?.name.toLowerCase().replace(/\s+/g, "_") || "leave empty to auto-generate"}
                  value={customId}
                  onChange={(e) => setCustomId(e.target.value)}
                  disabled={isImporting}
                />
                <p className="text-xs text-muted-foreground">
                  Override the auto-generated ID. Useful for avoiding conflicts.
                </p>
              </div>

              <div className="flex items-center space-x-2">
                <Switch
                  id="overwrite"
                  checked={overwrite}
                  onCheckedChange={setOverwrite}
                  disabled={isImporting}
                />
                <Label htmlFor="overwrite" className="text-sm">
                  Overwrite if character already exists
                </Label>
              </div>
            </div>
          )}

          {/* Import error */}
          {importError && (
            <div className="rounded-md bg-destructive/10 p-3 text-destructive">
              <div className="flex items-center gap-2 font-medium">
                <AlertCircle className="h-4 w-4" />
                Import Failed
              </div>
              <p className="mt-1 text-sm">{importError}</p>
              {existsError && (
                <p className="mt-2 text-sm">
                  Enable "Overwrite" above to replace the existing character.
                </p>
              )}
            </div>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)} disabled={isImporting}>
            Cancel
          </Button>
          <Button
            onClick={handleImport}
            disabled={!validation?.valid || isImporting}
          >
            {isImporting ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Importing...
              </>
            ) : (
              <>
                <Upload className="h-4 w-4 mr-2" />
                Import Character
              </>
            )}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

// ============================================
// Client-side PNG tEXt chunk extraction
// (for validation preview before upload)
// ============================================

const PNG_SIG = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]);

function extractCharaFromPngClient(data: Uint8Array): string | null {
  // Verify PNG signature
  for (let i = 0; i < 8; i++) {
    if (data[i] !== PNG_SIG[i]) return null;
  }

  const view = new DataView(data.buffer, data.byteOffset, data.byteLength);
  let offset = 8;

  while (offset < data.length) {
    if (offset + 8 > data.length) break;

    const length = view.getUint32(offset, false);
    const chunkType = String.fromCharCode(
      data[offset + 4], data[offset + 5], data[offset + 6], data[offset + 7]
    );

    if (offset + 12 + length > data.length) break;

    if (chunkType === "tEXt") {
      const chunkData = data.subarray(offset + 8, offset + 8 + length);
      const nullIdx = chunkData.indexOf(0x00);

      if (nullIdx !== -1) {
        const keyword = new TextDecoder("latin1").decode(chunkData.subarray(0, nullIdx));

        if (keyword === "chara" || keyword === "ccv3") {
          const textBytes = chunkData.subarray(nullIdx + 1);
          const b64Str = new TextDecoder("latin1").decode(textBytes);
          try {
            return atob(b64Str);
          } catch {
            // Try raw UTF-8
            try {
              return new TextDecoder("utf-8").decode(textBytes);
            } catch {
              // Skip
            }
          }
        }
      }
    } else if (chunkType === "IEND") {
      break;
    }

    offset += 12 + length;
  }

  return null;
}
