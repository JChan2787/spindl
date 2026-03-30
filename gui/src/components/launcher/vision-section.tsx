"use client";

import { Eye, EyeOff, Cloud, HardDrive, Brain } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useLauncherStore, type VLMModelType } from "@/lib/stores";

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

function LocalVLMFields() {
  const { vlmLocal, updateVLMLocal, validationErrors } = useLauncherStore();

  return (
    <div className="space-y-4">
      <FieldRow label="Model Type">
        <Select
          value={vlmLocal.modelType}
          onValueChange={(v) => updateVLMLocal({ modelType: v as VLMModelType })}
        >
          <SelectTrigger className="w-full">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="gemma3">Gemma 3</SelectItem>
            <SelectItem value="qwen2_vl">Qwen2 VL</SelectItem>
            <SelectItem value="llava">LLaVA</SelectItem>
            <SelectItem value="minicpm_v">MiniCPM-V</SelectItem>
          </SelectContent>
        </Select>
      </FieldRow>

      <div className="grid gap-4 md:grid-cols-2">
        <FieldRow
          label="Executable Path"
          error={validationErrors["vlmLocal.executablePath"]}
        >
          <Input
            placeholder="C:\path\to\llama-server.exe"
            value={vlmLocal.executablePath}
            onChange={(e) => updateVLMLocal({ executablePath: e.target.value })}
          />
        </FieldRow>

        <FieldRow
          label="Model Path"
          error={validationErrors["vlmLocal.modelPath"]}
        >
          <Input
            placeholder="C:\models\vision-model.gguf"
            value={vlmLocal.modelPath}
            onChange={(e) => updateVLMLocal({ modelPath: e.target.value })}
          />
        </FieldRow>
      </div>

      <FieldRow label="MMProj Path (if applicable)">
        <Input
          placeholder="C:\models\mmproj.gguf"
          value={vlmLocal.mmprojPath}
          onChange={(e) => updateVLMLocal({ mmprojPath: e.target.value })}
        />
      </FieldRow>

      <div className="grid gap-4 md:grid-cols-5">
        <FieldRow label="Host">
          <Input
            value={vlmLocal.host}
            onChange={(e) => updateVLMLocal({ host: e.target.value })}
          />
        </FieldRow>

        <FieldRow label="Port">
          <Input
            type="number"
            value={vlmLocal.port}
            onChange={(e) => updateVLMLocal({ port: parseInt(e.target.value) || 5558 })}
          />
        </FieldRow>

        <FieldRow label="Context Size">
          <Input
            type="number"
            value={vlmLocal.contextSize}
            onChange={(e) => updateVLMLocal({ contextSize: parseInt(e.target.value) || 8192 })}
          />
        </FieldRow>

        <FieldRow label="GPU Layers">
          <Input
            type="number"
            value={vlmLocal.gpuLayers}
            onChange={(e) => updateVLMLocal({ gpuLayers: parseInt(e.target.value) || 99 })}
          />
        </FieldRow>

        <FieldRow label="Device">
          <Input
            placeholder="CUDA0"
            value={vlmLocal.device}
            onChange={(e) => updateVLMLocal({ device: e.target.value })}
            disabled={!!vlmLocal.tensorSplit}
          />
        </FieldRow>
      </div>

      <FieldRow label="Tensor Split (multi-GPU)">
        <Input
          placeholder="e.g. 0.7,0.3 for 70%/30% across two GPUs"
          value={vlmLocal.tensorSplit}
          onChange={(e) => updateVLMLocal({ tensorSplit: e.target.value })}
        />
        {vlmLocal.tensorSplit && (
          <p className="text-xs text-muted-foreground">
            Overrides Device — model layers distributed across GPUs by ratio
          </p>
        )}
      </FieldRow>

      <FieldRow label="Extra Arguments">
        <Input
          placeholder='--cache-ram 0'
          value={vlmLocal.extraArgs}
          onChange={(e) => updateVLMLocal({ extraArgs: e.target.value })}
        />
      </FieldRow>

      <div className="grid gap-4 md:grid-cols-2">
        <FieldRow label="Timeout (s)">
          <Input
            type="number"
            value={vlmLocal.timeout}
            onChange={(e) => updateVLMLocal({ timeout: parseInt(e.target.value) || 30 })}
          />
        </FieldRow>

        <FieldRow label="Max Tokens">
          <Input
            type="number"
            value={vlmLocal.maxTokens}
            onChange={(e) => updateVLMLocal({ maxTokens: parseInt(e.target.value) || 300 })}
          />
        </FieldRow>
      </div>
    </div>
  );
}

function CloudVLMFields() {
  const { vlmCloud, updateVLMCloud, validationErrors } = useLauncherStore();

  return (
    <div className="space-y-4">
      <div className="grid gap-4 md:grid-cols-2">
        <FieldRow label="API Key" error={validationErrors["vlmCloud.apiKey"]}>
          <Input
            type="password"
            placeholder="sk-... or ${ENV_VAR}"
            value={vlmCloud.apiKey}
            onChange={(e) => updateVLMCloud({ apiKey: e.target.value })}
          />
        </FieldRow>

        <FieldRow label="Model" error={validationErrors["vlmCloud.model"]}>
          <Input
            placeholder="gpt-4o"
            value={vlmCloud.model}
            onChange={(e) => updateVLMCloud({ model: e.target.value })}
          />
        </FieldRow>
      </div>

      <FieldRow label="Base URL">
        <Input
          placeholder="https://api.openai.com/v1"
          value={vlmCloud.baseUrl}
          onChange={(e) => updateVLMCloud({ baseUrl: e.target.value })}
        />
      </FieldRow>

      <div className="grid gap-4 md:grid-cols-3">
        <FieldRow label="Context Size">
          <Input
            type="number"
            value={vlmCloud.contextSize}
            onChange={(e) => updateVLMCloud({ contextSize: parseInt(e.target.value) || 8192 })}
          />
        </FieldRow>

        <FieldRow label="Timeout (s)">
          <Input
            type="number"
            value={vlmCloud.timeout}
            onChange={(e) => updateVLMCloud({ timeout: parseInt(e.target.value) || 30 })}
          />
        </FieldRow>

        <FieldRow label="Max Tokens">
          <Input
            type="number"
            value={vlmCloud.maxTokens}
            onChange={(e) => updateVLMCloud({ maxTokens: parseInt(e.target.value) || 300 })}
          />
        </FieldRow>
      </div>
    </div>
  );
}

export function VisionSection() {
  const {
    vlmEnabled,
    setVLMEnabled,
    useLLMForVision,
    setUseLLMForVision,
    vlmProviderType,
    setVLMProviderType,
  } = useLauncherStore();

  return (
    <Card className={!vlmEnabled ? "opacity-60" : ""}>
      <CardHeader>
        <CardTitle className="flex items-center justify-between text-base font-medium">
          <div className="flex items-center gap-2">
            {vlmEnabled ? (
              <Eye className="h-5 w-5" />
            ) : (
              <EyeOff className="h-5 w-5" />
            )}
            Vision Configuration
          </div>
          <div className="flex items-center gap-2">
            <Label
              htmlFor="vlm-enabled"
              className="text-sm font-normal text-muted-foreground"
            >
              {vlmEnabled ? "Enabled" : "Disabled"}
            </Label>
            <Switch
              id="vlm-enabled"
              checked={vlmEnabled}
              onCheckedChange={setVLMEnabled}
            />
          </div>
        </CardTitle>
      </CardHeader>
      {vlmEnabled && (
        <CardContent className="space-y-4">
          {/* Use LLM for Vision Toggle */}
          <div className="flex items-center justify-between rounded-lg border p-4">
            <div className="space-y-0.5">
              <Label className="text-sm font-medium flex items-center gap-2">
                <Brain className="h-4 w-4" />
                Does your LLM support vision?
              </Label>
              <p className="text-xs text-muted-foreground">
                If yes, VLM will use the same endpoint as your LLM
              </p>
            </div>
            <Switch
              checked={useLLMForVision}
              onCheckedChange={setUseLLMForVision}
            />
          </div>

          {/* VLM Configuration - only shown if not using LLM for vision */}
          {!useLLMForVision && (
            <Tabs
              value={vlmProviderType}
              onValueChange={(v) => setVLMProviderType(v as "local" | "cloud")}
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
                <LocalVLMFields />
              </TabsContent>

              <TabsContent value="cloud">
                <CloudVLMFields />
              </TabsContent>
            </Tabs>
          )}

          {/* Info message when using LLM for vision */}
          {useLLMForVision && (
            <div className="rounded-lg bg-muted/50 p-4 text-sm text-muted-foreground">
              Vision requests will be routed to your LLM endpoint. Make sure your
              chosen model supports multimodal inputs.
            </div>
          )}
        </CardContent>
      )}
    </Card>
  );
}
