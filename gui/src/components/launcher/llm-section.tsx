"use client";

import { useState, useCallback, useEffect, useRef } from "react";
import { Brain, Cloud, HardDrive } from "lucide-react";
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
import {
  ModelCombobox,
  type ModelOption,
} from "@/components/ui/model-combobox";
import { useLauncherStore } from "@/lib/stores";
import { CLOUD_PROVIDERS, CLOUD_PROVIDER_IDS, type CloudProvider } from "@/lib/constants/cloud-providers";

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

function LocalLLMFields() {
  const { llmLocal, updateLLMLocal, validationErrors, useLLMForVision } = useLauncherStore();

  return (
    <div className="space-y-4">
      <div className="grid gap-4 md:grid-cols-2">
        <FieldRow
          label="Executable Path"
          error={validationErrors["llmLocal.executablePath"]}
        >
          <Input
            placeholder="C:\path\to\llama-server.exe"
            value={llmLocal.executablePath}
            onChange={(e) => updateLLMLocal({ executablePath: e.target.value })}
          />
        </FieldRow>

        <FieldRow
          label="Model Path"
          error={validationErrors["llmLocal.modelPath"]}
        >
          <Input
            placeholder="C:\models\model.gguf"
            value={llmLocal.modelPath}
            onChange={(e) => updateLLMLocal({ modelPath: e.target.value })}
          />
        </FieldRow>
      </div>

      <div className="grid gap-4 md:grid-cols-5">
        <FieldRow label="Host">
          <Input
            value={llmLocal.host}
            onChange={(e) => updateLLMLocal({ host: e.target.value })}
          />
        </FieldRow>

        <FieldRow label="Port">
          <Input
            type="number"
            value={llmLocal.port}
            onChange={(e) => updateLLMLocal({ port: parseInt(e.target.value) || 5557 })}
          />
        </FieldRow>

        <FieldRow label="Context Size">
          <Input
            type="number"
            value={llmLocal.contextSize}
            onChange={(e) => updateLLMLocal({ contextSize: parseInt(e.target.value) || 8192 })}
          />
          {useLLMForVision && (
            <p className="text-xs text-muted-foreground">
              Unified vision: {Math.floor(llmLocal.contextSize / 2).toLocaleString()} per slot (2 slots: chat + vision)
            </p>
          )}
        </FieldRow>

        <FieldRow label="GPU Layers">
          <Input
            type="number"
            value={llmLocal.gpuLayers}
            onChange={(e) => updateLLMLocal({ gpuLayers: parseInt(e.target.value) || 99 })}
          />
        </FieldRow>

        <FieldRow label="Device">
          <Input
            placeholder="CUDA0"
            value={llmLocal.device}
            onChange={(e) => updateLLMLocal({ device: e.target.value })}
            disabled={!!llmLocal.tensorSplit}
          />
        </FieldRow>
      </div>

      <FieldRow label="Tensor Split (multi-GPU)">
        <Input
          placeholder="e.g. 0.7,0.3 for 70%/30% across two GPUs"
          value={llmLocal.tensorSplit}
          onChange={(e) => updateLLMLocal({ tensorSplit: e.target.value })}
        />
        {llmLocal.tensorSplit && (
          <p className="text-xs text-muted-foreground">
            Overrides Device — model layers distributed across GPUs by ratio
          </p>
        )}
      </FieldRow>

      <FieldRow label="Extra Arguments">
        <Input
          placeholder='-fa on --no-mmap'
          value={llmLocal.extraArgs}
          onChange={(e) => updateLLMLocal({ extraArgs: e.target.value })}
        />
      </FieldRow>

      {useLLMForVision && (
        <FieldRow label="MMProj Path" error={validationErrors["llmLocal.mmprojPath"]}>
          <Input
            placeholder="C:\models\mmproj.gguf"
            value={llmLocal.mmprojPath}
            onChange={(e) => updateLLMLocal({ mmprojPath: e.target.value })}
          />
          <p className="text-xs text-muted-foreground">
            Required for unified vision mode — multimodal projection weights for this model
          </p>
        </FieldRow>
      )}

      <div className="grid gap-4 md:grid-cols-4">
        <FieldRow label="Timeout (s)">
          <Input
            type="number"
            value={llmLocal.timeout}
            onChange={(e) => updateLLMLocal({ timeout: parseInt(e.target.value) || 30 })}
          />
        </FieldRow>

        <FieldRow label="Temperature">
          <Input
            type="number"
            step="0.1"
            value={llmLocal.temperature}
            onChange={(e) => updateLLMLocal({ temperature: parseFloat(e.target.value) || 0.7 })}
          />
        </FieldRow>

        <FieldRow label="Max Tokens">
          <Input
            type="number"
            value={llmLocal.maxTokens}
            onChange={(e) => updateLLMLocal({ maxTokens: parseInt(e.target.value) || 256 })}
          />
        </FieldRow>

        <FieldRow label="Top P">
          <Input
            type="number"
            step="0.05"
            value={llmLocal.topP}
            onChange={(e) => updateLLMLocal({ topP: parseFloat(e.target.value) || 0.95 })}
          />
        </FieldRow>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <FieldRow label="Reasoning Format">
          <Select
            value={llmLocal.reasoningFormat || "none"}
            onValueChange={(v) => updateLLMLocal({ reasoningFormat: v === "none" ? "" : v })}
          >
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="none">None</SelectItem>
              <SelectItem value="deepseek">DeepSeek</SelectItem>
            </SelectContent>
          </Select>
          <p className="text-xs text-muted-foreground">
            For models with &lt;think&gt; blocks (Qwen3, DeepSeek). Leave &quot;None&quot; for standard models.
          </p>
        </FieldRow>

        <FieldRow label="Reasoning Budget">
          <Input
            type="number"
            value={llmLocal.reasoningBudget}
            onChange={(e) => {
              const parsed = parseInt(e.target.value);
              updateLLMLocal({ reasoningBudget: isNaN(parsed) ? -1 : parsed });
            }}
          />
          <p className="text-xs text-muted-foreground">
            -1 = unlimited, 0 = disabled, &gt;0 = token limit
          </p>
        </FieldRow>
      </div>
    </div>
  );
}

function CloudLLMFields() {
  const { llmCloud, updateLLMCloud, validationErrors, savedProviderKeys, setSavedProviderKeys } = useLauncherStore();

  const [models, setModels] = useState<ModelOption[]>([]);
  const [isLoadingModels, setIsLoadingModels] = useState(false);
  const [modelsError, setModelsError] = useState<string | null>(null);
  const hasFetchedRef = useRef(false);

  const fetchModels = useCallback(async () => {
    if (!llmCloud.apiKey || llmCloud.provider !== "openrouter") return;

    setIsLoadingModels(true);
    setModelsError(null);

    try {
      const response = await fetch("/api/launcher/fetch-models", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          provider: "openrouter",
          apiKey: llmCloud.apiKey,
          apiUrl: llmCloud.apiUrl,
        }),
      });

      const data = await response.json();

      if (!data.success) {
        setModelsError(data.error || "Failed to fetch models");
        setModels([]);
      } else {
        setModels(data.models);
      }
    } catch (err) {
      setModelsError(err instanceof Error ? err.message : "Network error");
      setModels([]);
    } finally {
      setIsLoadingModels(false);
    }
  }, [llmCloud.apiKey, llmCloud.apiUrl, llmCloud.provider]);

  useEffect(() => {
    if (llmCloud.provider === "openrouter" && llmCloud.apiKey && !hasFetchedRef.current) {
      hasFetchedRef.current = true;
      fetchModels();
    }
  }, [llmCloud.provider, llmCloud.apiKey, fetchModels]);

  // NANO-063: Registry-driven provider switch with key isolation
  const handleProviderChange = (newProvider: CloudProvider) => {
    // Stash current provider's key before switching
    const updatedKeys: Record<string, string> = { ...savedProviderKeys, [llmCloud.provider]: llmCloud.apiKey };
    setSavedProviderKeys(updatedKeys);

    const entry = CLOUD_PROVIDERS[newProvider];
    updateLLMCloud({
      provider: newProvider,
      apiUrl: entry.defaultUrl,
      model: entry.defaultModel,
      apiKey: updatedKeys[newProvider] || "",
    });

    // Reset model fetch state for clean slate
    hasFetchedRef.current = false;
    setModels([]);
    setModelsError(null);
  };

  const isOpenRouter = llmCloud.provider === "openrouter";

  return (
    <div className="space-y-4">
      <div className="grid gap-4 md:grid-cols-2">
        <FieldRow label="Provider">
          <Select
            value={llmCloud.provider}
            onValueChange={(v) => handleProviderChange(v as CloudProvider)}
          >
            <SelectTrigger className="w-full">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {CLOUD_PROVIDER_IDS.map((id) => (
                <SelectItem key={id} value={id}>
                  {CLOUD_PROVIDERS[id].label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </FieldRow>

        <FieldRow label="API Key" error={validationErrors["llmCloud.apiKey"]}>
          <Input
            type="password"
            placeholder="sk-..."
            value={llmCloud.apiKey}
            onChange={(e) => updateLLMCloud({ apiKey: e.target.value })}
          />
        </FieldRow>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        <FieldRow label="API URL">
          <Input
            value={llmCloud.apiUrl}
            onChange={(e) => updateLLMCloud({ apiUrl: e.target.value })}
          />
        </FieldRow>

        <FieldRow label="Model" error={validationErrors["llmCloud.model"]}>
          {isOpenRouter ? (
            <ModelCombobox
              value={llmCloud.model}
              onValueChange={(model) => updateLLMCloud({ model })}
              models={models}
              isLoading={isLoadingModels}
              error={modelsError}
              onRefresh={fetchModels}
              placeholder={llmCloud.apiKey ? "Select a model..." : "Enter API key first"}
              disabled={!llmCloud.apiKey}
            />
          ) : (
            <Input
              placeholder="deepseek-chat"
              value={llmCloud.model}
              onChange={(e) => updateLLMCloud({ model: e.target.value })}
            />
          )}
        </FieldRow>
      </div>

      <div className="grid gap-4 md:grid-cols-4">
        <FieldRow label="Context Size">
          <Input
            type="number"
            value={llmCloud.contextSize}
            onChange={(e) => updateLLMCloud({ contextSize: parseInt(e.target.value) || 32768 })}
          />
        </FieldRow>

        <FieldRow label="Timeout (s)">
          <Input
            type="number"
            value={llmCloud.timeout}
            onChange={(e) => updateLLMCloud({ timeout: parseInt(e.target.value) || 60 })}
          />
        </FieldRow>

        <FieldRow label="Temperature">
          <Input
            type="number"
            step="0.1"
            value={llmCloud.temperature}
            onChange={(e) => updateLLMCloud({ temperature: parseFloat(e.target.value) || 0.7 })}
          />
        </FieldRow>

        <FieldRow label="Max Tokens">
          <Input
            type="number"
            value={llmCloud.maxTokens}
            onChange={(e) => updateLLMCloud({ maxTokens: parseInt(e.target.value) || 256 })}
          />
        </FieldRow>
      </div>
    </div>
  );
}

export function LLMSection() {
  const { llmProviderType, setLLMProviderType } = useLauncherStore();

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base font-medium">
          <Brain className="h-5 w-5" />
          LLM Configuration
        </CardTitle>
      </CardHeader>
      <CardContent>
        <Tabs
          value={llmProviderType}
          onValueChange={(v) => setLLMProviderType(v as "local" | "cloud")}
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
            <LocalLLMFields />
          </TabsContent>

          <TabsContent value="cloud">
            <CloudLLMFields />
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
