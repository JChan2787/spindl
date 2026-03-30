"use client";

import { useCallback, useEffect, useState } from "react";
import { Cpu, ChevronDown, ChevronRight, Loader2, Play, RotateCcw } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { Button } from "@/components/ui/button";
import { ModelCombobox, type ModelOption } from "@/components/ui/model-combobox";
import { useSettingsStore } from "@/lib/stores";
import { getSocket } from "@/lib/socket";
import type { OpenRouterModelsEvent } from "@/types/events";

export function LLMConfig() {
  const {
    llmConfig, isSavingLLM, setSavingLLM,
    localLLMConfig, updateLocalLLMConfig,
    isLaunchingLLM, setLaunchingLLM,
    llmServerRunning, llmLaunchError, setLLMLaunchError,
    vlmConfig,
  } = useSettingsStore();
  const socket = getSocket();

  // Local state for the model input (editable for cloud providers)
  const [pendingModel, setPendingModel] = useState<string | null>(null);
  const [pendingProvider, setPendingProvider] = useState<string | null>(null);
  const [pendingContextSize, setPendingContextSize] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [localExpanded, setLocalExpanded] = useState(false);

  // OpenRouter model list
  const [orModels, setOrModels] = useState<ModelOption[]>([]);
  const [orLoading, setOrLoading] = useState(false);
  const [orError, setOrError] = useState<string | null>(null);

  // Effective provider: pending swap takes visual priority, reverts on error
  const activeProvider = pendingProvider ?? llmConfig.provider;
  const isLocalProvider = activeProvider === "llama";
  const isOpenRouter = activeProvider === "openrouter";
  const displayModel = pendingModel ?? llmConfig.model;
  const unifiedVLM = vlmConfig.unified_vlm;

  // Auto-expand local config when switching to llama
  useEffect(() => {
    if (isLocalProvider && !llmServerRunning) {
      setLocalExpanded(true);
    }
  }, [isLocalProvider, llmServerRunning]);

  // Fetch OpenRouter models
  const fetchOrModels = useCallback(() => {
    setOrLoading(true);
    setOrError(null);
    socket.emit("request_openrouter_models", {});
  }, [socket]);

  // Listen for OpenRouter models response
  useEffect(() => {
    const handler = (event: OpenRouterModelsEvent) => {
      setOrLoading(false);
      if (event.error) {
        setOrError(event.error);
      } else if (event.models) {
        setOrModels(
          event.models.map((m) => ({
            id: m.id,
            name: m.name,
            context_length: m.context_length,
            pricing: null,
          }))
        );
      }
    };
    socket.on("openrouter_models", handler);
    return () => { socket.off("openrouter_models", handler); };
  }, [socket]);

  // Fetch models when switching to OpenRouter
  useEffect(() => {
    if (isOpenRouter && orModels.length === 0) {
      fetchOrModels();
    }
  }, [isOpenRouter, orModels.length, fetchOrModels]);

  const handleProviderChange = useCallback(
    (provider: string) => {
      if (provider === llmConfig.provider || pendingProvider) return;

      // For llama: don't emit set_llm_provider immediately — show config panel
      if (provider === "llama") {
        setPendingProvider(null);
        // Just visually switch to show local config; launch button handles the rest
        setError(null);
        setLLMLaunchError(null);
        // Update llmConfig.provider locally via the store's setLLMConfig
        // so activeProvider reflects "llama" without a server call
        useSettingsStore.getState().setLLMConfig({
          ...llmConfig,
          provider: "llama",
        });
        setLocalExpanded(true);
        return;
      }

      setError(null);
      setPendingModel(null);
      setPendingProvider(provider);
      setSavingLLM(true);
      socket.emit("set_llm_provider", {
        provider,
        config: {},
      });

      // Listen for response — revert pending on error, clear on success
      const handleResponse = (event: { error?: string }) => {
        if (event.error) {
          setError(event.error);
          setPendingProvider(null);
        } else {
          setPendingProvider(null);
        }
        socket.off("llm_config_updated", handleResponse);
      };
      socket.on("llm_config_updated", handleResponse);
    },
    [llmConfig, pendingProvider, socket, setSavingLLM, setLLMLaunchError]
  );

  const handleModelChange = useCallback(
    (model: string) => {
      if (model === llmConfig.model) return;

      setError(null);
      setSavingLLM(true);
      socket.emit("set_llm_provider", {
        provider: llmConfig.provider,
        config: { model },
      });

      const handleResponse = (event: { error?: string }) => {
        if (event.error) {
          setError(event.error);
        }
        socket.off("llm_config_updated", handleResponse);
      };
      socket.on("llm_config_updated", handleResponse);
    },
    [llmConfig.model, llmConfig.provider, socket, setSavingLLM]
  );

  const handleModelSubmit = useCallback(() => {
    if (!pendingModel || pendingModel === llmConfig.model) {
      setPendingModel(null);
      return;
    }

    setError(null);
    setSavingLLM(true);
    socket.emit("set_llm_provider", {
      provider: llmConfig.provider,
      config: { model: pendingModel },
    });
    setPendingModel(null);

    const handleResponse = (event: { error?: string }) => {
      if (event.error) {
        setError(event.error);
      }
      socket.off("llm_config_updated", handleResponse);
    };
    socket.on("llm_config_updated", handleResponse);
  }, [pendingModel, llmConfig.model, llmConfig.provider, socket, setSavingLLM]);

  // NANO-096: Update context_size for cloud providers
  const handleContextSizeSubmit = useCallback(() => {
    if (pendingContextSize === null || pendingContextSize === llmConfig.context_size) {
      setPendingContextSize(null);
      return;
    }

    setError(null);
    setSavingLLM(true);
    socket.emit("set_llm_provider", {
      provider: llmConfig.provider,
      config: { context_size: pendingContextSize },
    });
    setPendingContextSize(null);

    const handleResponse = (event: { error?: string }) => {
      if (event.error) {
        setError(event.error);
      }
      socket.off("llm_config_updated", handleResponse);
    };
    socket.on("llm_config_updated", handleResponse);
  }, [pendingContextSize, llmConfig.context_size, llmConfig.provider, socket, setSavingLLM]);

  const handleLaunchLLM = useCallback(() => {
    setLaunchingLLM(true);
    setLLMLaunchError(null);
    setError(null);
    // NANO-079: Include unified_vision flag when unified VLM is active
    const config = unifiedVLM
      ? { ...localLLMConfig, unified_vision: true }
      : localLLMConfig;
    socket.emit("launch_llm_server", { config });
  }, [socket, localLLMConfig, unifiedVLM, setLaunchingLLM, setLLMLaunchError]);

  // Determine launch button state
  const launchButtonLabel = isLaunchingLLM
    ? "Launching..."
    : llmServerRunning
      ? "Relaunch Server"
      : "Launch Server";

  const LaunchIcon = isLaunchingLLM
    ? Loader2
    : llmServerRunning
      ? RotateCcw
      : Play;

  const canLaunch = !isLaunchingLLM && localLLMConfig.executable_path && localLLMConfig.model_path
    && (!unifiedVLM || localLLMConfig.mmproj_path);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-sm font-medium">
          <Cpu className="h-4 w-4" />
          LLM Provider
          {isSavingLLM && (
            <span className="text-xs text-muted-foreground">(saving...)</span>
          )}
          {isLaunchingLLM && (
            <span className="text-xs text-muted-foreground">(launching...)</span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Provider dropdown */}
        <div className="space-y-2">
          <Label className="text-sm">Provider</Label>
          <Select
            value={activeProvider}
            onValueChange={handleProviderChange}
          >
            <SelectTrigger className="w-full">
              <SelectValue placeholder="Select provider" />
            </SelectTrigger>
            <SelectContent>
              {llmConfig.available_providers.map((p) => (
                <SelectItem key={p} value={p}>
                  {p}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>

        {/* Model display/input — cloud providers */}
        {!isLocalProvider && (
          <div className="space-y-2">
            <Label className="text-sm">Model</Label>
            {isOpenRouter ? (
              <ModelCombobox
                value={llmConfig.model}
                onValueChange={handleModelChange}
                models={orModels}
                isLoading={orLoading}
                error={orError}
                onRefresh={fetchOrModels}
                placeholder="Search models..."
                disabled={isSavingLLM}
              />
            ) : (
              <div className="flex gap-2">
                <Input
                  value={displayModel}
                  onChange={(e) => setPendingModel(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") handleModelSubmit();
                  }}
                  placeholder="e.g., google/gemini-2.5-pro"
                  className="flex-1"
                />
                {pendingModel && pendingModel !== llmConfig.model && (
                  <Button size="sm" onClick={handleModelSubmit}>
                    Apply
                  </Button>
                )}
              </div>
            )}
          </div>
        )}

        {/* Local LLM section */}
        {isLocalProvider && (
          <>
            {/* Server status + model display */}
            <div className="space-y-2">
              <Label className="text-sm">Model</Label>
              <p className="text-sm text-muted-foreground truncate">
                {llmServerRunning
                  ? llmConfig.model || localLLMConfig.model_path || "Running"
                  : "Server not running"}
              </p>
            </div>

            {/* Collapsible local config */}
            <button
              type="button"
              className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors w-full"
              onClick={() => setLocalExpanded(!localExpanded)}
            >
              {localExpanded ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
              Local Server Config
              {llmServerRunning && (
                <span className="ml-auto text-xs text-green-500">Running</span>
              )}
            </button>

            {localExpanded && (
              <div className="space-y-3 border-l-2 border-muted pl-3">
                {/* Paths — full width */}
                <div className="space-y-2">
                  <Label className="text-xs">Executable Path</Label>
                  <Input
                    value={localLLMConfig.executable_path || ""}
                    onChange={(e) => updateLocalLLMConfig({ executable_path: e.target.value })}
                    placeholder="/path/to/llama-server"
                    className="text-xs h-8"
                  />
                </div>
                <div className="space-y-2">
                  <Label className="text-xs">Model Path</Label>
                  <Input
                    value={localLLMConfig.model_path || ""}
                    onChange={(e) => updateLocalLLMConfig({ model_path: e.target.value })}
                    placeholder="/path/to/model.gguf"
                    className="text-xs h-8"
                  />
                </div>

                {/* NANO-079: Conditional mmproj field when unified VLM is active */}
                {unifiedVLM && (
                  <div className="space-y-2">
                    <Label className="text-xs">MMProj Path (required for unified vision)</Label>
                    <Input
                      value={localLLMConfig.mmproj_path || ""}
                      onChange={(e) => updateLocalLLMConfig({ mmproj_path: e.target.value })}
                      placeholder="/path/to/mmproj.gguf"
                      className="text-xs h-8"
                    />
                  </div>
                )}

                {/* Server params — grid */}
                <div className="grid grid-cols-2 gap-2">
                  <div className="space-y-1">
                    <Label className="text-xs">Host</Label>
                    <Input
                      value={localLLMConfig.host || "127.0.0.1"}
                      onChange={(e) => updateLocalLLMConfig({ host: e.target.value })}
                      className="text-xs h-8"
                    />
                  </div>
                  <div className="space-y-1">
                    <Label className="text-xs">Port</Label>
                    <Input
                      type="number"
                      value={localLLMConfig.port ?? 5557}
                      onChange={(e) => updateLocalLLMConfig({ port: parseInt(e.target.value) || 5557 })}
                      className="text-xs h-8"
                    />
                  </div>
                  <div className="space-y-1">
                    <Label className="text-xs">GPU Layers</Label>
                    <Input
                      type="number"
                      value={localLLMConfig.gpu_layers ?? 99}
                      onChange={(e) => updateLocalLLMConfig({ gpu_layers: parseInt(e.target.value) || 99 })}
                      className="text-xs h-8"
                    />
                  </div>
                  <div className="space-y-1">
                    <Label className="text-xs">Context Size</Label>
                    <Input
                      type="number"
                      value={localLLMConfig.context_size ?? 8192}
                      onChange={(e) => updateLocalLLMConfig({ context_size: parseInt(e.target.value) || 8192 })}
                      className="text-xs h-8"
                    />
                  </div>
                </div>

                {/* Tensor split */}
                <div className="space-y-2">
                  <Label className="text-xs">Tensor Split (multi-GPU)</Label>
                  <Input
                    value={localLLMConfig.tensor_split || ""}
                    onChange={(e) => updateLocalLLMConfig({ tensor_split: e.target.value })}
                    placeholder="e.g. 0.5,0.5 for 50%/50% across two GPUs"
                    className="text-xs h-8"
                  />
                  {localLLMConfig.tensor_split && (
                    <p className="text-xs text-muted-foreground">
                      Overrides Device — model layers distributed across GPUs by ratio
                    </p>
                  )}
                </div>

                {/* Extra args */}
                <div className="space-y-2">
                  <Label className="text-xs">Extra Args</Label>
                  <Input
                    value={localLLMConfig.extra_args || ""}
                    onChange={(e) => updateLocalLLMConfig({ extra_args: e.target.value })}
                    placeholder="e.g. --chat-template-file /path/to/template"
                    className="text-xs h-8"
                  />
                </div>

                {/* Reasoning config */}
                <div className="grid grid-cols-2 gap-2">
                  <div className="space-y-1">
                    <Label className="text-xs">Reasoning Format</Label>
                    <Select
                      value={localLLMConfig.reasoning_format || "none"}
                      onValueChange={(v) => updateLocalLLMConfig({ reasoning_format: v === "none" ? "" : v })}
                    >
                      <SelectTrigger className="text-xs h-8">
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="none">None</SelectItem>
                        <SelectItem value="deepseek">DeepSeek</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-1">
                    <Label className="text-xs">Reasoning Budget</Label>
                    <Input
                      type="number"
                      value={localLLMConfig.reasoning_budget ?? -1}
                      onChange={(e) => {
                        const parsed = parseInt(e.target.value);
                        updateLocalLLMConfig({ reasoning_budget: isNaN(parsed) ? -1 : parsed });
                      }}
                      className="text-xs h-8"
                    />
                  </div>
                </div>

                {/* Launch button */}
                <Button
                  size="sm"
                  className="w-full"
                  disabled={!canLaunch}
                  onClick={handleLaunchLLM}
                >
                  <LaunchIcon className={`h-3 w-3 mr-1 ${isLaunchingLLM ? "animate-spin" : ""}`} />
                  {launchButtonLabel}
                </Button>

                {/* Launch error */}
                {llmLaunchError && (
                  <p className="text-xs text-destructive">{llmLaunchError}</p>
                )}
              </div>
            )}
          </>
        )}

        {/* Context size — editable for cloud, read-only for local (NANO-096) */}
        {!isLocalProvider ? (
          <div className="space-y-1">
            <Label className="text-xs">Context Size</Label>
            <div className="flex gap-2">
              <Input
                type="number"
                value={pendingContextSize ?? llmConfig.context_size ?? 8192}
                onChange={(e) => setPendingContextSize(parseInt(e.target.value) || 8192)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") handleContextSizeSubmit();
                }}
                className="text-xs h-8 flex-1"
              />
              {pendingContextSize !== null && pendingContextSize !== llmConfig.context_size && (
                <Button size="sm" className="h-8" onClick={handleContextSizeSubmit}>
                  Apply
                </Button>
              )}
            </div>
          </div>
        ) : llmConfig.context_size ? (
          <div className="flex items-center justify-between">
            <Label className="text-sm text-muted-foreground">Context</Label>
            <span className="text-sm text-muted-foreground">
              {llmConfig.context_size.toLocaleString()} tokens
            </span>
          </div>
        ) : null}

        {/* Error display */}
        {error && (
          <p className="text-xs text-destructive">{error}</p>
        )}

        <p className="text-xs text-muted-foreground">
          {isLocalProvider
            ? "Configure and launch a local llama-server, or relaunch with a different model."
            : "Switch providers mid-conversation. Local llama requires server restart for model changes."}
        </p>
      </CardContent>
    </Card>
  );
}
