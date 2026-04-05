"use client";

import { Mic } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Button } from "@/components/ui/button";
import { ChevronDown, MicOff } from "lucide-react";
import { useState } from "react";
import { useLauncherStore, type STTPlatform, type EnvironmentType, type STTProviderType } from "@/lib/stores";

interface FieldRowProps {
  label: string;
  children: React.ReactNode;
  error?: string;
}

function FieldRow({ label, children, error }: FieldRowProps) {
  return (
    <div className="space-y-1.5">
      <Label className="text-sm">{label}</Label>
      {children}
      {error && <p className="text-xs text-destructive">{error}</p>}
    </div>
  );
}

// ============================================
// Parakeet-specific fields
// ============================================

function ParakeetSTTFields() {
  const sttParakeet = useLauncherStore((s) => s.sttParakeet);
  const updateSTTParakeet = useLauncherStore((s) => s.updateSTTParakeet);
  const validationErrors = useLauncherStore((s) => s.validationErrors);

  const showEnvNameOrPath = sttParakeet.envType === "conda" || sttParakeet.envType === "venv";
  const showCustomActivation = sttParakeet.envType === "other";

  return (
    <>
      {/* Platform Selection */}
      <div className="grid gap-4 md:grid-cols-2">
        <FieldRow label="Platform">
          <Select
            value={sttParakeet.platform}
            onValueChange={(v) => updateSTTParakeet({ platform: v as STTPlatform })}
          >
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="native">Native (Windows/Linux)</SelectItem>
              <SelectItem value="wsl">WSL (Windows Subsystem for Linux)</SelectItem>
            </SelectContent>
          </Select>
        </FieldRow>

        {sttParakeet.platform === "wsl" && (
          <FieldRow label="WSL Distro">
            <Input
              placeholder="Ubuntu"
              value={sttParakeet.wslDistro}
              onChange={(e) => updateSTTParakeet({ wslDistro: e.target.value })}
            />
          </FieldRow>
        )}
      </div>

      {/* Server Script Path */}
      <FieldRow
        label="Server Script Path"
        error={validationErrors["stt.serverScriptPath"]}
      >
        <Input
          placeholder="/path/to/nemo_server.py"
          value={sttParakeet.serverScriptPath}
          onChange={(e) => updateSTTParakeet({ serverScriptPath: e.target.value })}
        />
      </FieldRow>

      {/* Environment Configuration */}
      <div className="grid gap-4 md:grid-cols-2">
        <FieldRow label="Environment Type">
          <Select
            value={sttParakeet.envType}
            onValueChange={(v) => updateSTTParakeet({ envType: v as EnvironmentType })}
          >
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="conda">Conda</SelectItem>
              <SelectItem value="venv">Python venv</SelectItem>
              <SelectItem value="system">System Python</SelectItem>
              <SelectItem value="other">Custom Activation</SelectItem>
            </SelectContent>
          </Select>
        </FieldRow>

        {showEnvNameOrPath && (
          <FieldRow
            label={sttParakeet.envType === "conda" ? "Conda Environment Name" : "Venv Path"}
            error={validationErrors["stt.envNameOrPath"]}
          >
            <Input
              placeholder={sttParakeet.envType === "conda" ? "nemo" : "/path/to/venv"}
              value={sttParakeet.envNameOrPath}
              onChange={(e) => updateSTTParakeet({ envNameOrPath: e.target.value })}
            />
          </FieldRow>
        )}
      </div>

      {showCustomActivation && (
        <FieldRow
          label="Custom Activation Command"
          error={validationErrors["stt.customActivation"]}
        >
          <Input
            placeholder="source /path/to/activate"
            value={sttParakeet.customActivation}
            onChange={(e) => updateSTTParakeet({ customActivation: e.target.value })}
          />
        </FieldRow>
      )}
    </>
  );
}

// ============================================
// Whisper-specific fields
// ============================================

function WhisperSTTFields() {
  const sttWhisper = useLauncherStore((s) => s.sttWhisper);
  const updateSTTWhisper = useLauncherStore((s) => s.updateSTTWhisper);
  const validationErrors = useLauncherStore((s) => s.validationErrors);

  return (
    <>
      {/* Binary Path */}
      <FieldRow
        label="Whisper Server Binary"
        error={validationErrors["stt.binaryPath"]}
      >
        <Input
          placeholder="whisper-server"
          value={sttWhisper.binaryPath}
          onChange={(e) => updateSTTWhisper({ binaryPath: e.target.value })}
        />
      </FieldRow>

      {/* Model Path */}
      <FieldRow
        label="Model Path"
        error={validationErrors["stt.modelPath"]}
      >
        <Input
          placeholder="/path/to/ggml-model.bin"
          value={sttWhisper.modelPath}
          onChange={(e) => updateSTTWhisper({ modelPath: e.target.value })}
        />
      </FieldRow>

      {/* Language + Threads + No GPU */}
      <div className="grid gap-4 md:grid-cols-3">
        <FieldRow label="Language">
          <Input
            placeholder="en"
            value={sttWhisper.language}
            onChange={(e) => updateSTTWhisper({ language: e.target.value })}
          />
        </FieldRow>

        <FieldRow label="Threads">
          <Input
            type="number"
            value={sttWhisper.threads}
            onChange={(e) => updateSTTWhisper({ threads: parseInt(e.target.value) || 4 })}
          />
        </FieldRow>

        <div className="space-y-1.5">
          <Label className="text-sm">No GPU</Label>
          <div className="flex items-center h-9">
            <input
              type="checkbox"
              checked={sttWhisper.noGpu}
              onChange={(e) => updateSTTWhisper({ noGpu: e.target.checked })}
              className="h-4 w-4"
            />
            <span className="ml-2 text-sm text-muted-foreground">Disable GPU acceleration</span>
          </div>
        </div>
      </div>
    </>
  );
}

// ============================================
// Command Preview (provider-conditional)
// ============================================

function CommandPreview() {
  const sttProvider = useLauncherStore((s) => s.sttProvider);
  const sttParakeet = useLauncherStore((s) => s.sttParakeet);
  const sttWhisper = useLauncherStore((s) => s.sttWhisper);

  if (sttProvider === "whisper") {
    const parts = [
      sttWhisper.binaryPath || "<binary>",
      "-m", sttWhisper.modelPath || "<model>",
      "--host", sttWhisper.host,
      "--port", String(sttWhisper.port),
      "-l", sttWhisper.language || "en",
      "-t", String(sttWhisper.threads),
    ];
    if (sttWhisper.noGpu) parts.push("--no-gpu");

    return (
      <span className="break-all text-muted-foreground">
        {parts.join(" ")}
      </span>
    );
  }

  // Parakeet preview
  const parts: string[] = [];

  if (sttParakeet.platform === "wsl") {
    parts.push(`wsl -d ${sttParakeet.wslDistro || "Ubuntu"} --`);
  }

  switch (sttParakeet.envType) {
    case "conda":
      parts.push(
        `conda run --live-stream -n ${sttParakeet.envNameOrPath || "<env_name>"}`
      );
      break;
    case "venv":
      parts.push(`source ${sttParakeet.envNameOrPath || "<venv_path>"}/bin/activate &&`);
      break;
    case "other":
      if (sttParakeet.customActivation) {
        parts.push(`${sttParakeet.customActivation} &&`);
      }
      break;
    case "system":
      break;
  }

  parts.push(`python3 -u ${sttParakeet.serverScriptPath || "<script_path>"}`);

  return (
    <span className="break-all text-muted-foreground">
      {parts.join(" ")}
    </span>
  );
}

// ============================================
// Main STT Section
// ============================================

export function STTSection() {
  const sttEnabled = useLauncherStore((s) => s.sttEnabled);
  const setSTTEnabled = useLauncherStore((s) => s.setSTTEnabled);
  const sttProvider = useLauncherStore((s) => s.sttProvider);
  const setSTTProvider = useLauncherStore((s) => s.setSTTProvider);
  // Active provider's network fields for the shared host/port/timeout row
  const activeConfig = useLauncherStore((s) =>
    s.sttProvider === "parakeet" ? s.sttParakeet : s.sttWhisper
  );
  const updateActive = useLauncherStore((s) =>
    s.sttProvider === "parakeet" ? s.updateSTTParakeet : s.updateSTTWhisper
  );
  const [advancedOpen, setAdvancedOpen] = useState(false);

  return (
    <Card className={!sttEnabled ? "opacity-60" : ""}>
      <CardHeader>
        <CardTitle className="flex items-center justify-between text-base font-medium">
          <div className="flex items-center gap-2">
            {sttEnabled ? (
              <Mic className="h-5 w-5" />
            ) : (
              <MicOff className="h-5 w-5" />
            )}
            STT Configuration
          </div>
          <div className="flex items-center gap-2">
            <Label
              htmlFor="stt-enabled"
              className="text-sm font-normal text-muted-foreground"
            >
              {sttEnabled ? "Enabled" : "Disabled"}
            </Label>
            <Switch
              id="stt-enabled"
              checked={sttEnabled}
              onCheckedChange={setSTTEnabled}
            />
          </div>
        </CardTitle>
      </CardHeader>
      {sttEnabled && <CardContent className="space-y-4">
        {/* Provider Selection */}
        <FieldRow label="STT Provider">
          <Select
            value={sttProvider}
            onValueChange={(v) => setSTTProvider(v as STTProviderType)}
          >
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="parakeet">Parakeet (NeMo)</SelectItem>
              <SelectItem value="whisper">Whisper.cpp</SelectItem>
            </SelectContent>
          </Select>
        </FieldRow>

        {/* Shared Network Settings — reads from the active provider's sub-config */}
        <div className="grid gap-4 md:grid-cols-3">
          <FieldRow label="Host">
            <Input
              value={activeConfig.host}
              onChange={(e) => updateActive({ host: e.target.value })}
            />
          </FieldRow>

          <FieldRow label="Port">
            <Input
              type="number"
              value={activeConfig.port}
              onChange={(e) => updateActive({ port: parseInt(e.target.value) || 5555 })}
            />
          </FieldRow>

          <FieldRow label="Timeout (s)">
            <Input
              type="number"
              value={activeConfig.timeout}
              onChange={(e) => updateActive({ timeout: parseInt(e.target.value) || 30 })}
            />
          </FieldRow>
        </div>

        {/* Provider-specific fields */}
        {sttProvider === "parakeet" ? <ParakeetSTTFields /> : <WhisperSTTFields />}

        {/* Command Preview */}
        <Collapsible open={advancedOpen} onOpenChange={setAdvancedOpen}>
          <CollapsibleTrigger asChild>
            <Button variant="ghost" size="sm" className="gap-2 p-0 h-auto">
              <ChevronDown
                className={`h-4 w-4 transition-transform ${
                  advancedOpen ? "rotate-180" : ""
                }`}
              />
              <span className="text-xs text-muted-foreground">
                Preview Generated Command
              </span>
            </Button>
          </CollapsibleTrigger>
          <CollapsibleContent className="pt-2">
            <div className="rounded-lg bg-muted p-3 font-mono text-xs">
              <CommandPreview />
            </div>
          </CollapsibleContent>
        </Collapsible>
      </CardContent>}
    </Card>
  );
}
