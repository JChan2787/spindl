"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Eye, Brain, ChevronDown, ChevronRight, Loader2, Play, RotateCcw } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Badge } from "@/components/ui/badge";
import { useSettingsStore } from "@/lib/stores";
import { getSocket } from "@/lib/socket";

/** Display labels for VLM provider values (NANO-079: "llm" removed — unified is a toggle) */
const PROVIDER_LABELS: Record<string, string> = {
  llama: "Local (llama)",
  openai: "Cloud (OpenAI-compat)",
};

export function VLMConfig() {
  const {
    vlmConfig, isSavingVLM, setSavingVLM,
    toolsConfig,
    localVLMConfig, updateLocalVLMConfig,
    isLaunchingVLM, setLaunchingVLM,
    vlmServerRunning, vlmLaunchError, setVLMLaunchError,
    llmConfig,
    setUnifiedVLM,
  } = useSettingsStore();
  const socket = getSocket();

  const [pendingProvider, setPendingProvider] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [localExpanded, setLocalExpanded] = useState(false);

  // Cloud VLM config state (Case 3)
  const [cloudApiKey, setCloudApiKey] = useState("");
  const [cloudModel, setCloudModel] = useState("");
  const [cloudBaseUrl, setCloudBaseUrl] = useState("");

  const unifiedVLM = vlmConfig.unified_vlm;
  // Effective provider: pending swap takes visual priority
  const activeProvider = pendingProvider ?? vlmConfig.provider;
  const toolsEnabled = toolsConfig.master_enabled;
  const isLocalProvider = activeProvider === "llama";
  const isLocalLLM = llmConfig.provider === "llama";

  // Available providers — filter out "llm" (unified is a toggle, not a dropdown option)
  const availableProviders = vlmConfig.available_providers.filter((p) => p !== "llm");

  // Request VLM config on mount + hydrate cloud fields once from initial response
  const cloudHydratedRef = useRef(false);
  useEffect(() => {
    const handleVlmConfig = (event: { cloud_config?: { api_key: string; model: string; base_url: string } }) => {
      if (event.cloud_config && !cloudHydratedRef.current) {
        setCloudApiKey(event.cloud_config.api_key || "");
        setCloudModel(event.cloud_config.model || "");
        setCloudBaseUrl(event.cloud_config.base_url || "");
        cloudHydratedRef.current = true;
      }
    };
    socket.on("vlm_config_updated", handleVlmConfig);
    socket.emit("request_vlm_config", {});
    return () => { socket.off("vlm_config_updated", handleVlmConfig); };
  }, [socket]);

  // Auto-expand local config when switching to llama
  useEffect(() => {
    if (isLocalProvider && !vlmServerRunning && !unifiedVLM) {
      setLocalExpanded(true);
    }
  }, [isLocalProvider, vlmServerRunning, unifiedVLM]);

  const handleUnifiedToggle = useCallback(
    (checked: boolean) => {
      // Toggle only sets local UI state — no backend call.
      // The actual swap happens via Apply/Launch buttons.
      setError(null);
      setUnifiedVLM(checked);
    },
    [setUnifiedVLM]
  );

  const handleApplyUnified = useCallback(() => {
    setError(null);
    setSavingVLM(true);
    socket.emit("set_vlm_provider", { provider: "llm", config: {} });

    const handleResponse = (event: { error?: string }) => {
      if (event.error) {
        setError(event.error);
        setUnifiedVLM(false);
      }
      socket.off("vlm_config_updated", handleResponse);
    };
    socket.on("vlm_config_updated", handleResponse);
  }, [socket, setSavingVLM, setUnifiedVLM]);

  const handleProviderChange = useCallback(
    (provider: string) => {
      if (provider === vlmConfig.provider || pendingProvider) return;

      // Don't emit set_vlm_provider on dropdown change — show config panel first.
      // For llama: user fills fields, hits Launch Server.
      // For openai: user fills fields, hits Apply.
      setError(null);
      setPendingProvider(null);
      if (provider === "llama") {
        setVLMLaunchError(null);
        setLocalExpanded(true);
      }
      useSettingsStore.getState().setVLMConfig({
        ...vlmConfig,
        provider,
      });
    },
    [vlmConfig, setVLMLaunchError]
  );

  const handleApplyCloudVLM = useCallback(() => {
    setError(null);
    setSavingVLM(true);
    socket.emit("set_vlm_provider", {
      provider: "openai",
      config: {
        api_key: cloudApiKey,
        model: cloudModel,
        base_url: cloudBaseUrl,
      },
    });

    const handleResponse = (event: { error?: string }) => {
      if (event.error) {
        setError(event.error);
      }
      socket.off("vlm_config_updated", handleResponse);
    };
    socket.on("vlm_config_updated", handleResponse);
  }, [socket, setSavingVLM, cloudApiKey, cloudModel, cloudBaseUrl]);

  const handleLaunchVLM = useCallback(() => {
    setLaunchingVLM(true);
    setVLMLaunchError(null);
    setError(null);
    socket.emit("launch_vlm_server", { config: localVLMConfig });
  }, [socket, localVLMConfig, setLaunchingVLM, setVLMLaunchError]);

  // Launch button state
  const launchButtonLabel = isLaunchingVLM
    ? "Launching..."
    : vlmServerRunning
      ? "Relaunch Server"
      : "Launch Server";

  const LaunchIcon = isLaunchingVLM
    ? Loader2
    : vlmServerRunning
      ? RotateCcw
      : Play;

  const canLaunch = !isLaunchingVLM && localVLMConfig.executable_path && localVLMConfig.model_path;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-sm font-medium">
          <Eye className="h-4 w-4" />
          VLM Provider
          {isSavingVLM && (
            <span className="text-xs text-muted-foreground">(saving...)</span>
          )}
          {isLaunchingVLM && (
            <span className="text-xs text-muted-foreground">(launching...)</span>
          )}
          <Badge
            variant={vlmConfig.healthy ? "default" : "secondary"}
            className="ml-auto text-xs"
          >
            {vlmConfig.healthy ? "Healthy" : "Offline"}
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Unified VLM toggle — mirrors Launcher vision-section.tsx */}
        <div className="flex items-center justify-between rounded-lg border p-3">
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
            checked={unifiedVLM}
            onCheckedChange={handleUnifiedToggle}
          />
        </div>

        {/* Unified info message + Apply button */}
        {unifiedVLM && (
          <div className="space-y-3">
            <div className="rounded-lg bg-muted/50 p-3 text-sm text-muted-foreground">
              Vision requests will route through your LLM endpoint.
              {isLocalLLM
                ? " Make sure your LLM server was launched with an mmproj file."
                : " Make sure your chosen model supports multimodal inputs."}
            </div>
            <Button
              size="sm"
              className="w-full"
              onClick={handleApplyUnified}
              disabled={isSavingVLM}
            >
              Apply Unified Vision
            </Button>
          </div>
        )}

        {/* Provider selection — only shown when unified is off */}
        {!unifiedVLM && (
          <>
            <div className="space-y-2">
              <Label className="text-sm">Provider</Label>
              <Select
                value={activeProvider}
                onValueChange={handleProviderChange}
              >
                <SelectTrigger className="w-full">
                  <SelectValue placeholder="Select VLM provider" />
                </SelectTrigger>
                <SelectContent>
                  {availableProviders.map((p) => (
                    <SelectItem key={p} value={p}>
                      {PROVIDER_LABELS[p] ?? p}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            {/* Cloud VLM config fields (Case 3) */}
            {activeProvider === "openai" && (
              <div className="space-y-3 border-l-2 border-muted pl-3">
                <div className="space-y-2">
                  <Label className="text-xs">API Key</Label>
                  <Input
                    type="password"
                    value={cloudApiKey}
                    onChange={(e) => setCloudApiKey(e.target.value)}
                    placeholder="sk-... or ${ENV_VAR}"
                    className="text-xs h-8"
                  />
                </div>
                <div className="space-y-2">
                  <Label className="text-xs">Model</Label>
                  <Input
                    value={cloudModel}
                    onChange={(e) => setCloudModel(e.target.value)}
                    placeholder="gpt-4o"
                    className="text-xs h-8"
                  />
                </div>
                <div className="space-y-2">
                  <Label className="text-xs">Base URL</Label>
                  <Input
                    value={cloudBaseUrl}
                    onChange={(e) => setCloudBaseUrl(e.target.value)}
                    placeholder="https://api.openai.com/v1"
                    className="text-xs h-8"
                  />
                </div>
                <Button
                  size="sm"
                  className="w-full"
                  onClick={handleApplyCloudVLM}
                  disabled={isSavingVLM}
                >
                  Apply
                </Button>
              </div>
            )}

            {/* Local VLM section (Case 1) */}
            {isLocalProvider && (
              <>
                <div className="space-y-2">
                  <Label className="text-sm">Model</Label>
                  <p className="text-sm text-muted-foreground truncate">
                    {vlmServerRunning
                      ? vlmConfig.provider === "llama" ? localVLMConfig.model_path || "Running" : "Running"
                      : "Server not running"}
                  </p>
                </div>

                <button
                  type="button"
                  className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors w-full"
                  onClick={() => setLocalExpanded(!localExpanded)}
                >
                  {localExpanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
                  Local Server Config
                  {vlmServerRunning && (
                    <span className="ml-auto text-xs text-green-500">Running</span>
                  )}
                </button>

                {localExpanded && (
                  <div className="space-y-3 border-l-2 border-muted pl-3">
                    <div className="space-y-2">
                      <Label className="text-xs">Model Type</Label>
                      <Select
                        value={localVLMConfig.model_type || "gemma3"}
                        onValueChange={(v) => updateLocalVLMConfig({ model_type: v })}
                      >
                        <SelectTrigger className="text-xs h-8">
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="gemma3">Gemma 3</SelectItem>
                          <SelectItem value="qwen2_vl">Qwen2 VL</SelectItem>
                          <SelectItem value="llava">LLaVA</SelectItem>
                          <SelectItem value="minicpm_v">MiniCPM-V</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <Label className="text-xs">Executable Path</Label>
                      <Input
                        value={localVLMConfig.executable_path || ""}
                        onChange={(e) => updateLocalVLMConfig({ executable_path: e.target.value })}
                        placeholder="/path/to/llama-server"
                        className="text-xs h-8"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label className="text-xs">Model Path</Label>
                      <Input
                        value={localVLMConfig.model_path || ""}
                        onChange={(e) => updateLocalVLMConfig({ model_path: e.target.value })}
                        placeholder="/path/to/vision-model.gguf"
                        className="text-xs h-8"
                      />
                    </div>
                    <div className="space-y-2">
                      <Label className="text-xs">MMProj Path</Label>
                      <Input
                        value={localVLMConfig.mmproj_path || ""}
                        onChange={(e) => updateLocalVLMConfig({ mmproj_path: e.target.value })}
                        placeholder="/path/to/mmproj.gguf"
                        className="text-xs h-8"
                      />
                    </div>

                    <div className="grid grid-cols-2 gap-2">
                      <div className="space-y-1">
                        <Label className="text-xs">Host</Label>
                        <Input
                          value={localVLMConfig.host || "127.0.0.1"}
                          onChange={(e) => updateLocalVLMConfig({ host: e.target.value })}
                          className="text-xs h-8"
                        />
                      </div>
                      <div className="space-y-1">
                        <Label className="text-xs">Port</Label>
                        <Input
                          type="number"
                          value={localVLMConfig.port ?? 5558}
                          onChange={(e) => updateLocalVLMConfig({ port: parseInt(e.target.value) || 5558 })}
                          className="text-xs h-8"
                        />
                      </div>
                      <div className="space-y-1">
                        <Label className="text-xs">GPU Layers</Label>
                        <Input
                          type="number"
                          value={localVLMConfig.gpu_layers ?? 99}
                          onChange={(e) => updateLocalVLMConfig({ gpu_layers: parseInt(e.target.value) || 99 })}
                          className="text-xs h-8"
                        />
                      </div>
                      <div className="space-y-1">
                        <Label className="text-xs">Context Size</Label>
                        <Input
                          type="number"
                          value={localVLMConfig.context_size ?? 8192}
                          onChange={(e) => updateLocalVLMConfig({ context_size: parseInt(e.target.value) || 8192 })}
                          className="text-xs h-8"
                        />
                      </div>
                    </div>

                    <div className="space-y-2">
                      <Label className="text-xs">Tensor Split (multi-GPU)</Label>
                      <Input
                        value={localVLMConfig.tensor_split || ""}
                        onChange={(e) => updateLocalVLMConfig({ tensor_split: e.target.value })}
                        placeholder="e.g. 0.5,0.5 for 50%/50% across two GPUs"
                        className="text-xs h-8"
                      />
                      {localVLMConfig.tensor_split && (
                        <p className="text-xs text-muted-foreground">
                          Overrides Device — model layers distributed across GPUs by ratio
                        </p>
                      )}
                    </div>

                    <div className="space-y-2">
                      <Label className="text-xs">Extra Args</Label>
                      <Input
                        value={localVLMConfig.extra_args || ""}
                        onChange={(e) => updateLocalVLMConfig({ extra_args: e.target.value })}
                        placeholder="--cache-ram 0"
                        className="text-xs h-8"
                      />
                    </div>

                    <Button
                      size="sm"
                      className="w-full"
                      disabled={!canLaunch}
                      onClick={handleLaunchVLM}
                    >
                      <LaunchIcon className={`h-3 w-3 mr-1 ${isLaunchingVLM ? "animate-spin" : ""}`} />
                      {launchButtonLabel}
                    </Button>

                    {vlmLaunchError && (
                      <p className="text-xs text-destructive">{vlmLaunchError}</p>
                    )}
                  </div>
                )}
              </>
            )}
          </>
        )}

        {/* Error display */}
        {error && (
          <p className="text-xs text-destructive">{error}</p>
        )}

        {!toolsEnabled && (
          <p className="text-xs text-muted-foreground">
            Tools are disabled — VLM will activate when tools are enabled.
          </p>
        )}
      </CardContent>
    </Card>
  );
}
