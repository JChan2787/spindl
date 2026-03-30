"use client";

import { useState, useCallback, useEffect, useRef } from "react";
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
import { Switch } from "@/components/ui/switch";
import { exportCharacterApi, exportPngCharacterApi } from "@/lib/stores";
import { Download, Copy, Check, AlertCircle, Loader2, FileJson, FileImage } from "lucide-react";

type ExportFormat = "json" | "png";

interface ExportDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  characterId: string | null;
  characterName: string;
}

export function ExportDialog({ open, onOpenChange, characterId, characterName }: ExportDialogProps) {
  const [includeSpindl, setIncludeSpindl] = useState(true);
  const [includeCodex, setIncludeCodex] = useState(true);
  const [exportFormat, setExportFormat] = useState<ExportFormat>("json");
  const [isExporting, setIsExporting] = useState(false);
  const [exportedJson, setExportedJson] = useState<string | null>(null);
  const [exportedPngBlob, setExportedPngBlob] = useState<Blob | null>(null);
  const [exportFilename, setExportFilename] = useState<string | null>(null);
  const [exportError, setExportError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  // Track if we've done the initial export on open
  const initialExportDone = useRef(false);

  // Perform export using REST API
  const doExport = useCallback(async () => {
    if (!characterId) return;

    setIsExporting(true);
    setExportError(null);

    try {
      if (exportFormat === "png") {
        const result = await exportPngCharacterApi(characterId, includeSpindl, includeCodex);
        if (result.error) {
          setExportError(result.error);
          setExportedPngBlob(null);
          setExportedJson(null);
        } else if (result.blob && result.filename) {
          setExportedPngBlob(result.blob);
          setExportFilename(result.filename);
          setExportedJson(null);
          setExportError(null);
        }
      } else {
        const result = await exportCharacterApi(characterId, includeSpindl, includeCodex);
        if (result.error) {
          setExportError(result.error);
          setExportedJson(null);
          setExportedPngBlob(null);
        } else if (result.success && result.json_data && result.filename) {
          setExportedJson(result.json_data);
          setExportFilename(result.filename);
          setExportedPngBlob(null);
          setExportError(null);
        }
      }
    } catch (error) {
      console.error("Export failed:", error);
      setExportError(error instanceof Error ? error.message : "Export failed");
    } finally {
      setIsExporting(false);
    }
  }, [characterId, includeSpindl, includeCodex, exportFormat]);

  // Reset state when dialog opens/closes or character changes
  useEffect(() => {
    if (open && characterId) {
      setExportedJson(null);
      setExportedPngBlob(null);
      setExportFilename(null);
      setExportError(null);
      setCopied(false);
      initialExportDone.current = false;
    } else if (!open) {
      setExportedJson(null);
      setExportedPngBlob(null);
      setExportFilename(null);
      setExportError(null);
      setCopied(false);
      setIsExporting(false);
      initialExportDone.current = false;
    }
  }, [open, characterId]);

  // Auto-export on open (only once)
  useEffect(() => {
    if (open && characterId && !initialExportDone.current) {
      initialExportDone.current = true;
      doExport();
    }
  }, [open, characterId, doExport]);

  // Re-export when options change (only if we already have exported content)
  useEffect(() => {
    if (open && characterId && (exportedJson || exportedPngBlob) && initialExportDone.current) {
      doExport();
    }
    // Only trigger on option changes, not on content changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [includeSpindl, includeCodex, exportFormat]);

  // Copy to clipboard (JSON only)
  const handleCopy = useCallback(async () => {
    if (!exportedJson) return;

    try {
      await navigator.clipboard.writeText(exportedJson);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Fallback for older browsers
      const textarea = document.createElement("textarea");
      textarea.value = exportedJson;
      document.body.appendChild(textarea);
      textarea.select();
      document.execCommand("copy");
      document.body.removeChild(textarea);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  }, [exportedJson]);

  // Download as file
  const handleDownload = useCallback(() => {
    if (!exportFilename) return;

    let blob: Blob;
    if (exportFormat === "png" && exportedPngBlob) {
      blob = exportedPngBlob;
    } else if (exportedJson) {
      blob = new Blob([exportedJson], { type: "application/json" });
    } else {
      return;
    }

    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = exportFilename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }, [exportedJson, exportedPngBlob, exportFilename, exportFormat]);

  const hasExportedContent = exportFormat === "png" ? !!exportedPngBlob : !!exportedJson;

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-2xl max-h-[90vh] overflow-y-auto">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Download className="h-5 w-5" />
            Export Character
          </DialogTitle>
          <DialogDescription>
            Export &ldquo;{characterName}&rdquo; as a SillyTavern Character Card V2 file.
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-4">
          {/* Format selector */}
          <div className="space-y-3 rounded-md border p-4">
            <Label className="text-base font-medium">Export Format</Label>

            <div className="flex gap-2">
              <Button
                variant={exportFormat === "json" ? "default" : "outline"}
                size="sm"
                onClick={() => setExportFormat("json")}
                disabled={isExporting}
              >
                <FileJson className="h-4 w-4 mr-1" />
                JSON
              </Button>
              <Button
                variant={exportFormat === "png" ? "default" : "outline"}
                size="sm"
                onClick={() => setExportFormat("png")}
                disabled={isExporting}
              >
                <FileImage className="h-4 w-4 mr-1" />
                PNG (Tavern Card)
              </Button>
            </div>

            {exportFormat === "png" && (
              <p className="text-xs text-muted-foreground">
                Embeds the character data into the avatar image. Compatible with SillyTavern, Chub.ai, and other tools that accept PNG character cards.
              </p>
            )}
          </div>

          {/* Export options */}
          <div className="space-y-4 rounded-md border p-4">
            <Label className="text-base font-medium">Export Options</Label>

            <div className="flex items-center space-x-2">
              <Switch
                id="include-spindl"
                checked={includeSpindl}
                onCheckedChange={setIncludeSpindl}
                disabled={isExporting}
              />
              <Label htmlFor="include-spindl" className="text-sm">
                Include SpindL extensions
              </Label>
            </div>
            <p className="text-xs text-muted-foreground ml-9">
              Voice, language, rules, generation settings. Disable for pure ST compatibility.
            </p>

            <div className="flex items-center space-x-2">
              <Switch
                id="include-codex"
                checked={includeCodex}
                onCheckedChange={setIncludeCodex}
                disabled={isExporting}
              />
              <Label htmlFor="include-codex" className="text-sm">
                Include codex (character_book) entries
              </Label>
            </div>
            <p className="text-xs text-muted-foreground ml-9">
              Lorebook entries embedded in the character card.
            </p>
          </div>

          {/* Loading state */}
          {isExporting && (
            <div className="flex items-center justify-center py-8 text-muted-foreground">
              <Loader2 className="h-6 w-6 animate-spin mr-2" />
              Generating export...
            </div>
          )}

          {/* Export error */}
          {exportError && (
            <div className="rounded-md bg-destructive/10 p-3 text-destructive">
              <div className="flex items-center gap-2 font-medium">
                <AlertCircle className="h-4 w-4" />
                Export Failed
              </div>
              <p className="mt-1 text-sm">{exportError}</p>
            </div>
          )}

          {/* JSON preview (JSON format only) */}
          {exportedJson && !isExporting && exportFormat === "json" && (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <Label htmlFor="json-output">
                  <FileJson className="h-4 w-4 inline mr-1" />
                  {exportFilename}
                </Label>
                <div className="flex gap-2">
                  <Button variant="outline" size="sm" onClick={handleCopy}>
                    {copied ? (
                      <>
                        <Check className="h-4 w-4 mr-1" />
                        Copied
                      </>
                    ) : (
                      <>
                        <Copy className="h-4 w-4 mr-1" />
                        Copy
                      </>
                    )}
                  </Button>
                  <Button variant="outline" size="sm" onClick={handleDownload}>
                    <Download className="h-4 w-4 mr-1" />
                    Download
                  </Button>
                </div>
              </div>
              <Textarea
                id="json-output"
                value={exportedJson}
                readOnly
                className="font-mono text-sm h-64"
              />
              <p className="text-xs text-muted-foreground">
                {new Blob([exportedJson]).size.toLocaleString()} bytes
              </p>
            </div>
          )}

          {/* PNG export success */}
          {exportedPngBlob && !isExporting && exportFormat === "png" && (
            <div className="rounded-md border p-4 space-y-3">
              <div className="flex items-center gap-2 text-green-600 dark:text-green-400 font-medium">
                <FileImage className="h-4 w-4" />
                PNG ready for download
              </div>
              <p className="text-sm text-muted-foreground">
                {exportFilename} — {exportedPngBlob.size.toLocaleString()} bytes
              </p>
            </div>
          )}
        </div>

        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Close
          </Button>
          {hasExportedContent && (
            <Button onClick={handleDownload}>
              <Download className="h-4 w-4 mr-2" />
              Download {exportFormat === "png" ? "PNG" : "JSON"}
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
