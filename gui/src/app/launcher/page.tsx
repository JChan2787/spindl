"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Rocket, Play, CheckCircle, AlertCircle, Loader2, Server, Cpu, Mic, Volume2, Database } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { LLMSection, VisionSection, STTSection, TTSSection, EmbeddingSection } from "@/components/launcher";
import { StatusOverlay } from "@/components/dashboard/status-overlay";
import { useLauncherStore, selectIsFormComplete, selectHasValidationErrors, useConnectionStore, useAgentStore } from "@/lib/stores";
import { getSocket } from "@/lib/socket";
import type { HydrateConfig, LaunchStatus, OrchestratorStatus } from "@/lib/stores/launcher-store";

interface SaveStatus {
  type: "success" | "error" | null;
  message: string;
}

// Map service names to icons
const SERVICE_ICONS: Record<string, React.ReactNode> = {
  llm: <Cpu className="h-4 w-4" />,
  vlm: <Cpu className="h-4 w-4" />,
  stt: <Mic className="h-4 w-4" />,
  tts: <Volume2 className="h-4 w-4" />,
  embedding: <Database className="h-4 w-4" />,
  orchestrator: <Server className="h-4 w-4" />,
};

export default function LauncherPage() {
  const router = useRouter();
  const store = useLauncherStore();
  const isFormComplete = selectIsFormComplete(store);
  const hasErrors = selectHasValidationErrors(store);
  const { connected: isConnected } = useConnectionStore();
  const servicesRunning = useAgentStore((s) => s.health) !== null;
  const { isStarting, isLoading, hydrate, setIsLoading, launchProgress, resetLaunch } = store;
  const [saveStatus, setSaveStatus] = useState<SaveStatus>({ type: null, message: "" });

  // Hydrate store from existing config on mount
  useEffect(() => {
    async function loadConfig() {
      try {
        const response = await fetch("/api/launcher/write-config");
        const result = await response.json();

        if (result.exists && result.config) {
          hydrate(result.config as HydrateConfig);
        } else {
          // No config exists, just mark as loaded with defaults
          setIsLoading(false);
        }
      } catch (error) {
        console.error("Failed to load config:", error);
        setIsLoading(false);
      }
    }

    loadConfig();
  }, [hydrate, setIsLoading]);

  // Redirect to dashboard when orchestrator is ready (not just services complete)
  useEffect(() => {
    if (launchProgress.orchestratorStatus === "ready") {
      const timer = setTimeout(() => {
        router.push("/");
      }, 2000);
      return () => clearTimeout(timer);
    }
  }, [launchProgress.orchestratorStatus, router]);

  const handleStartServices = async () => {
    store.setIsStarting(true);
    setSaveStatus({ type: null, message: "" });
    resetLaunch();

    try {
      // Step 1: Save configuration to spindl.yaml
      const response = await fetch("/api/launcher/write-config", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          llmProviderType: store.llmProviderType,
          llmLocal: store.llmLocal,
          llmCloud: store.llmCloud,
          vlmEnabled: store.vlmEnabled,
          useLLMForVision: store.useLLMForVision,
          vlmProviderType: store.vlmProviderType,
          vlmLocal: store.vlmLocal,
          vlmCloud: store.vlmCloud,
          sttEnabled: store.sttEnabled,
          sttProvider: store.sttProvider,
          sttParakeet: store.sttParakeet,
          sttWhisper: store.sttWhisper,
          ttsEnabled: store.ttsEnabled,
          ttsProviderType: store.ttsProviderType,
          ttsLocal: store.ttsLocal,
          embedding: store.embedding,
          historyMode: store.historyMode,
        }),
      });

      const result = await response.json();

      if (!result.success) {
        setSaveStatus({
          type: "error",
          message: result.error || "Failed to save configuration",
        });
        store.setIsStarting(false);
        return;
      }

      setSaveStatus({
        type: "success",
        message: "Configuration saved. Starting services...",
      });

      // Step 2: Trigger service launch via Socket.IO
      if (isConnected) {
        const socket = getSocket();
        socket.emit("start_services", {});
        // Note: isStarting will be set to false by setLaunchComplete or setLaunchError
      } else {
        setSaveStatus({
          type: "error",
          message: "Not connected to server. Cannot start services.",
        });
        store.setIsStarting(false);
      }
    } catch (error) {
      setSaveStatus({
        type: "error",
        message: error instanceof Error ? error.message : "Network error",
      });
      store.setIsStarting(false);
    }
  };

  // Calculate progress percentage based on launch status and orchestrator status
  const getProgressPercent = (): number => {
    // Services launch: 0-70%
    // Orchestrator init: 70-100%
    switch (launchProgress.status) {
      case "idle": return 0;
      case "starting": return 10;
      case "loading_config": return 20;
      case "config_loaded": return 30;
      case "launching": return 50;
      case "complete":
        // Services done, now check orchestrator
        switch (launchProgress.orchestratorStatus) {
          case "pending": return 70;
          case "initializing": return 85;
          case "ready": return 100;
          case "error": return 70;
          default: return 70;
        }
      case "error": return 0;
      default: return 0;
    }
  };

  // Show loading state while hydrating from config
  if (isLoading) {
    return (
      <div className="flex h-[50vh] items-center justify-center">
        <div className="flex flex-col items-center gap-3">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
          <p className="text-sm text-muted-foreground">Loading configuration...</p>
        </div>
      </div>
    );
  }

  // Show launch progress view when launching
  const isLaunching = launchProgress.status !== "idle" && launchProgress.status !== "error";
  const hasLaunchError = launchProgress.status === "error" || launchProgress.orchestratorStatus === "error";
  const isFullyReady = launchProgress.orchestratorStatus === "ready";

  return (
    <div className="space-y-6">
      <StatusOverlay disconnectedOnly />

      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="flex items-center gap-2 text-2xl font-bold">
            <Rocket className="h-6 w-6" />
            Service Launcher
          </h1>
          <p className="text-muted-foreground">
            Configure and launch SpindL services
          </p>
        </div>
        <Button
          size="lg"
          disabled={!isFormComplete || hasErrors || isStarting || isLaunching || servicesRunning}
          onClick={handleStartServices}
          data-testid="start-services-button"
        >
          {isStarting || isLaunching ? (
            <Loader2 className="h-4 w-4 animate-spin" />
          ) : (
            <Play className="h-4 w-4" />
          )}
          {isStarting ? "Saving..." : isLaunching ? "Launching..." : servicesRunning ? "Services Running" : "Start Services"}
        </Button>
      </div>

      {/* Save Status */}
      {saveStatus.type && !isLaunching && (
        <div
          className={`flex items-center gap-2 rounded-lg p-3 ${
            saveStatus.type === "success"
              ? "bg-green-500/10 text-green-500"
              : "bg-destructive/10 text-destructive"
          }`}
        >
          {saveStatus.type === "success" ? (
            <CheckCircle className="h-4 w-4" />
          ) : (
            <AlertCircle className="h-4 w-4" />
          )}
          <span className="text-sm">{saveStatus.message}</span>
        </div>
      )}

      {/* Launch Progress Panel */}
      {(isLaunching || hasLaunchError || launchProgress.status === "complete") && (
        <Card className={hasLaunchError ? "border-destructive" : isFullyReady ? "border-green-500" : ""}>
          <CardHeader className="pb-3">
            <CardTitle className="flex items-center gap-2 text-lg">
              {hasLaunchError ? (
                <>
                  <AlertCircle className="h-5 w-5 text-destructive" />
                  {launchProgress.orchestratorStatus === "error" ? "Orchestrator Failed" : "Launch Failed"}
                </>
              ) : isFullyReady ? (
                <>
                  <CheckCircle className="h-5 w-5 text-green-500" />
                  System Ready
                </>
              ) : launchProgress.status === "complete" ? (
                <>
                  <Loader2 className="h-5 w-5 animate-spin" />
                  Initializing Orchestrator
                </>
              ) : (
                <>
                  <Loader2 className="h-5 w-5 animate-spin" />
                  Launching Services
                </>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Progress Bar */}
            {!hasLaunchError && (
              <Progress value={getProgressPercent()} className="h-2" />
            )}

            {/* Status Message */}
            <p className={`text-sm ${hasLaunchError ? "text-destructive" : "text-muted-foreground"}`}>
              {launchProgress.orchestratorError || launchProgress.error || launchProgress.message || "Initializing..."}
            </p>

            {/* Current Service */}
            {launchProgress.currentService && !hasLaunchError && (
              <div className="flex items-center gap-2 text-sm">
                {SERVICE_ICONS[launchProgress.currentService] || <Server className="h-4 w-4" />}
                <span className="font-medium">{launchProgress.currentService}</span>
                <Loader2 className="h-3 w-3 animate-spin text-muted-foreground" />
              </div>
            )}

            {/* Launched Services */}
            {launchProgress.launchedServices.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {launchProgress.launchedServices.map((service) => (
                  <div
                    key={service}
                    className="flex items-center gap-1.5 rounded-full bg-green-500/10 px-3 py-1 text-xs text-green-500"
                  >
                    <CheckCircle className="h-3 w-3" />
                    {service}
                  </div>
                ))}
              </div>
            )}

            {/* Orchestrator Status */}
            {launchProgress.status === "complete" && launchProgress.orchestratorStatus === "initializing" && (
              <div className="flex items-center gap-2 text-sm">
                <Server className="h-4 w-4" />
                <span className="font-medium">orchestrator</span>
                <Loader2 className="h-3 w-3 animate-spin text-muted-foreground" />
              </div>
            )}

            {/* Orchestrator Ready Badge */}
            {launchProgress.orchestratorStatus === "ready" && launchProgress.orchestratorPersona && (
              <div className="flex items-center gap-1.5 rounded-full bg-green-500/10 px-3 py-1 text-xs text-green-500 w-fit">
                <CheckCircle className="h-3 w-3" />
                orchestrator ({launchProgress.orchestratorPersona})
              </div>
            )}

            {/* Redirect Notice */}
            {isFullyReady && (
              <p className="text-sm text-muted-foreground">
                Redirecting to dashboard...
              </p>
            )}

            {/* Retry Button on Error */}
            {hasLaunchError && (
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  resetLaunch();
                  setSaveStatus({ type: null, message: "" });
                }}
              >
                Try Again
              </Button>
            )}
          </CardContent>
        </Card>
      )}

      {/* Configuration Sections - hide when launching */}
      {!isLaunching && launchProgress.status !== "complete" && !isFullyReady && (
        <div className="space-y-6">
          <LLMSection />
          <VisionSection />
          <STTSection />
          <TTSSection />
          <EmbeddingSection />
        </div>
      )}
    </div>
  );
}
