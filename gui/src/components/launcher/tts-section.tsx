"use client";

import { Volume2, Cloud, HardDrive } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useLauncherStore, type EnvironmentType, type TTSProvider } from "@/lib/stores";

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

function LocalTTSFields() {
  const { ttsLocal, updateTTSLocal, validationErrors } = useLauncherStore();

  // Determine which environment fields to show
  const showEnvNameOrPath = ttsLocal.envType === "conda" || ttsLocal.envType === "venv";
  const showCustomActivation = ttsLocal.envType === "other";

  return (
    <div className="space-y-4">
      {/* Provider Selection */}
      <FieldRow label="Provider">
        <Select
          value={ttsLocal.provider}
          onValueChange={(v) => updateTTSLocal({ provider: v as TTSProvider })}
        >
          <SelectTrigger className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="kokoro">Kokoro</SelectItem>
          </SelectContent>
        </Select>
      </FieldRow>

      {/* Network Settings */}
      <div className="grid gap-4 md:grid-cols-3">
        <FieldRow label="Host">
          <Input
            value={ttsLocal.host}
            onChange={(e) => updateTTSLocal({ host: e.target.value })}
          />
        </FieldRow>

        <FieldRow label="Port">
          <Input
            type="number"
            value={ttsLocal.port}
            onChange={(e) => updateTTSLocal({ port: parseInt(e.target.value) || 5556 })}
          />
        </FieldRow>

        <FieldRow label="Timeout (s)">
          <Input
            type="number"
            value={ttsLocal.timeout}
            onChange={(e) => updateTTSLocal({ timeout: parseInt(e.target.value) || 30 })}
          />
        </FieldRow>
      </div>

      {/* Voice Settings */}
      <div className="grid gap-4 md:grid-cols-2">
        <FieldRow label="Voice">
          <Input
            placeholder="Provider default"
            value={ttsLocal.voice}
            onChange={(e) => updateTTSLocal({ voice: e.target.value })}
          />
          <p className="text-xs text-muted-foreground">
            Voice ID for TTS provider. Leave empty for provider default.
          </p>
        </FieldRow>

        <FieldRow label="Language">
          <Input
            placeholder="Provider default"
            value={ttsLocal.language}
            onChange={(e) => updateTTSLocal({ language: e.target.value })}
          />
          <p className="text-xs text-muted-foreground">
            Language code for TTS provider. Leave empty for provider default.
          </p>
        </FieldRow>
      </div>

      {/* Models Directory */}
      <FieldRow label="Models Directory">
        <Input
          placeholder="./tts/models"
          value={ttsLocal.modelsDirectory}
          onChange={(e) => updateTTSLocal({ modelsDirectory: e.target.value })}
        />
      </FieldRow>

      {/* Device Selection */}
      <FieldRow label="Device">
        <Select
          value={ttsLocal.device || "cuda"}
          onValueChange={(v) => updateTTSLocal({ device: v })}
        >
          <SelectTrigger className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="cuda">Auto (GPU)</SelectItem>
            <SelectItem value="cpu">CPU</SelectItem>
            <SelectItem value="cuda:0">CUDA:0</SelectItem>
            <SelectItem value="cuda:1">CUDA:1</SelectItem>
          </SelectContent>
        </Select>
        <p className="text-xs text-muted-foreground">
          Use CPU to avoid CUDA contention when using tensor split across multiple GPUs.
        </p>
      </FieldRow>

      {/* Environment Configuration */}
      <div className="grid gap-4 md:grid-cols-2">
        <FieldRow label="Environment Type">
          <Select
            value={ttsLocal.envType}
            onValueChange={(v) => updateTTSLocal({ envType: v as EnvironmentType })}
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

        {/* Env Name or Path - shown for conda/venv */}
        {showEnvNameOrPath && (
          <FieldRow
            label={ttsLocal.envType === "conda" ? "Conda Environment Name" : "Venv Path"}
            error={validationErrors["ttsLocal.envNameOrPath"]}
          >
            <Input
              placeholder={ttsLocal.envType === "conda" ? "spindl" : "/path/to/venv"}
              value={ttsLocal.envNameOrPath}
              onChange={(e) => updateTTSLocal({ envNameOrPath: e.target.value })}
            />
          </FieldRow>
        )}
      </div>

      {/* Custom Activation - shown for 'other' */}
      {showCustomActivation && (
        <FieldRow
          label="Custom Activation Command"
          error={validationErrors["ttsLocal.customActivation"]}
        >
          <Input
            placeholder="source /path/to/activate"
            value={ttsLocal.customActivation}
            onChange={(e) => updateTTSLocal({ customActivation: e.target.value })}
          />
        </FieldRow>
      )}
    </div>
  );
}

function CloudTTSPlaceholder() {
  return (
    <div className="rounded-lg bg-muted/50 p-6 text-center">
      <p className="text-sm text-muted-foreground">
        Cloud TTS providers (ElevenLabs, OpenAI) coming in a future update.
      </p>
      <p className="mt-2 text-xs text-muted-foreground/70">
        For now, use the Local tab with a supported TTS provider.
      </p>
    </div>
  );
}

export function TTSSection() {
  const { ttsProviderType, setTTSProviderType } = useLauncherStore();

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base font-medium">
          <Volume2 className="h-5 w-5" />
          TTS Configuration
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs
          value={ttsProviderType}
          onValueChange={(v) => setTTSProviderType(v as "local" | "cloud")}
        >
          <TabsList className="mb-4">
            <TabsTrigger value="local" className="gap-2">
              <HardDrive className="h-4 w-4" />
              Local
            </TabsTrigger>
            <TabsTrigger value="cloud" className="gap-2">
              <Cloud className="h-4 w-4" />
              Cloud
            </TabsTrigger>
          </TabsList>

          <TabsContent value="local">
            <LocalTTSFields />
          </TabsContent>

          <TabsContent value="cloud">
            <CloudTTSPlaceholder />
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
