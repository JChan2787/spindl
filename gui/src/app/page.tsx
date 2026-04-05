"use client";

import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { useAgentStore, useConnectionStore } from "@/lib/stores";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Progress } from "@/components/ui/progress";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
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
import { getSocket } from "@/lib/socket";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuItem,
} from "@/components/ui/dropdown-menu";
import {
  Mic,
  MicOff,
  Brain,
  RefreshCw,
  RotateCcw,
  Wrench,
  CheckCircle2,
  XCircle,
  Loader2,
  Power,
  MoreVertical,
} from "lucide-react";
import { CharacterPortrait } from "@/components/dashboard/character-portrait";
import { ChatHistory } from "@/components/dashboard/chat-history";
import { ChatInput } from "@/components/dashboard/chat-input";
import { ReasoningSettings } from "@/components/dashboard/reasoning-settings";
import { MemorySettings } from "@/components/dashboard/memory-settings";
import { GenerationSettings } from "@/components/dashboard/generation-settings";
import { ToolsSettings } from "@/components/dashboard/tools-settings";
import { LLMConfig } from "@/components/dashboard/llm-config";
import { VLMConfig } from "@/components/dashboard/vlm-config";
import { PatienceBar } from "@/components/dashboard/patience-bar";
import { StimuliSettings } from "@/components/dashboard/stimuli-settings";
import { TwitchChatCard } from "@/components/dashboard/twitch-chat-card";
import { VTubeStudioCard } from "@/components/dashboard/vtubestudio-card";
import { StatusOverlay } from "@/components/dashboard/status-overlay";

const stateLabels: Record<string, string> = {
  idle: "Idle",
  listening: "Listening",
  user_speaking: "User Speaking",
  processing: "Processing",
  system_speaking: "Speaking",
};

export default function DashboardPage() {
  const router = useRouter();
  const { connected } = useConnectionStore();
  const {
    stateTransitions,
    tokenUsage,
    health,
    isListeningPaused,
    setListeningPaused,
    toolActivities,
    shutdownStatus,
    shutdownMessage,
    resetShutdown,
  } = useAgentStore();

  const [configChecked, setConfigChecked] = useState(false);
  const [shutdownDialogOpen, setShutdownDialogOpen] = useState(false);

  // NANO-027 Phase 4: Check if config needs setup, redirect to launcher
  useEffect(() => {
    async function checkConfig() {
      try {
        const response = await fetch("/api/launcher/check-config");
        const result = await response.json();

        if (result.needsSetup) {
          console.log("[Dashboard] Config needs setup, redirecting to /launcher");
          router.replace("/launcher");
          return;
        }

        setConfigChecked(true);
      } catch (error) {
        console.error("[Dashboard] Failed to check config:", error);
        // On error, show dashboard anyway (might be running in attached mode)
        setConfigChecked(true);
      }
    }

    checkConfig();
  }, [router]);

  // NANO-028: Redirect to launcher when shutdown completes
  useEffect(() => {
    if (shutdownStatus === "complete") {
      console.log("[Dashboard] Shutdown complete, redirecting to /launcher");
      resetShutdown();
      router.replace("/launcher");
    }
  }, [shutdownStatus, router, resetShutdown]);

  const handleToggleListening = () => {
    const socket = getSocket();
    if (isListeningPaused) {
      socket.emit("resume_listening");
      setListeningPaused(false);
    } else {
      socket.emit("pause_listening");
      setListeningPaused(true);
    }
  };

  const handleRefreshHealth = () => {
    const socket = getSocket();
    socket.emit("request_health", {});
  };

  const handleRequestState = () => {
    const socket = getSocket();
    socket.emit("request_state", {});
  };

  // NANO-028: Handle shutdown backend
  const handleShutdown = () => {
    const socket = getSocket();
    socket.emit("shutdown_backend");
    setShutdownDialogOpen(false);
  };


  // Show loading while checking config
  if (!configChecked) {
    return (
      <div className="flex h-[50vh] items-center justify-center">
        <div className="flex flex-col items-center gap-3">
          <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
          <p className="text-sm text-muted-foreground">Checking configuration...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <StatusOverlay />

      {/* NANO-069: Header with health badges + controls */}
      <div className="flex flex-col gap-2">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">Dashboard</h1>
            <p className="text-muted-foreground text-sm">
              Real-time voice pipeline monitoring
            </p>
          </div>
          <div className="flex items-center gap-2">
            {/* Context Window — inline in header */}
            <div className="hidden md:flex items-center gap-1.5 pr-2 mr-2 border-r border-border">
              <Brain className="size-3.5 text-muted-foreground" />
              {tokenUsage ? (
                <>
                  <span className="text-sm font-medium">
                    {tokenUsage.percent.toFixed(1)}%
                  </span>
                  <Progress value={tokenUsage.percent} className="w-20 h-1.5" />
                  <span className="text-xs text-muted-foreground">
                    {tokenUsage.total.toLocaleString()} / {tokenUsage.max.toLocaleString()}
                  </span>
                </>
              ) : (
                <span className="text-xs text-muted-foreground">No data</span>
              )}
            </div>

            {/* Health badges — hidden on narrow viewports */}
            <div className="hidden md:flex items-center gap-1.5">
              {connected && health ? (
                <>
                  <Badge
                    variant={health.stt === "disabled" ? "secondary" : health.stt ? "default" : "destructive"}
                    className="text-xs"
                  >
                    STT {health.stt === "disabled" ? "OFF" : health.stt ? "OK" : "DOWN"}
                  </Badge>
                  <Badge
                    variant={health.tts === "disabled" ? "secondary" : health.tts ? "default" : "destructive"}
                    className="text-xs"
                  >
                    TTS {health.tts === "disabled" ? "OFF" : health.tts ? "OK" : "DOWN"}
                  </Badge>
                  <Badge variant={health.llm ? "default" : "destructive"} className="text-xs">
                    LLM {health.llm ? "OK" : "DOWN"}
                  </Badge>
                  {health.vlm !== undefined && (
                    <Badge variant={health.vlm ? "default" : "secondary"} className="text-xs">
                      VLM {health.vlm ? "OK" : "OFF"}
                    </Badge>
                  )}
                  {health.embedding !== undefined && (
                    <Badge variant={health.embedding ? "default" : "secondary"} className="text-xs">
                      EMB {health.embedding ? "OK" : "OFF"}
                    </Badge>
                  )}
                  {health.mic !== undefined && health.stt !== "disabled" && (
                    <Badge
                      variant={health.mic === "ok" ? "default" : health.mic === "restarting" ? "outline" : "destructive"}
                      className="text-xs"
                    >
                      MIC {health.mic === "ok" ? "OK" : health.mic === "restarting" ? "..." : "DOWN"}
                    </Badge>
                  )}
                </>
              ) : (
                <Badge variant="outline" className="text-xs">
                  {connected ? "Loading..." : "Disconnected"}
                </Badge>
              )}
            </div>

            {/* Three-dot dropdown: Refresh Health / Refresh State */}
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="icon" className="size-9">
                  <MoreVertical className="size-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem onClick={handleRefreshHealth} disabled={!connected}>
                  <RefreshCw className="size-4" />
                  Refresh Health
                </DropdownMenuItem>
                <DropdownMenuItem onClick={handleRequestState} disabled={!connected}>
                  <RotateCcw className="size-4" />
                  Refresh State
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>

            {/* Mic toggle (NANO-112: disabled when STT is off) */}
            <Button
              variant={isListeningPaused ? "default" : "ghost"}
              size="icon"
              className="size-9 rounded-full"
              onClick={handleToggleListening}
              disabled={!connected || health?.stt === "disabled"}
              title={health?.stt === "disabled" ? "STT Disabled" : isListeningPaused ? "Resume Listening" : "Pause Listening"}
            >
              {isListeningPaused || health?.stt === "disabled" ? (
                <Mic className="size-4" />
              ) : (
                <MicOff className="size-4" />
              )}
            </Button>

            {/* Shutdown */}
            <AlertDialog open={shutdownDialogOpen} onOpenChange={setShutdownDialogOpen}>
              <AlertDialogTrigger asChild>
                <Button
                  variant="ghost"
                  size="icon"
                  className="size-9 rounded-full border border-destructive/30 hover:bg-destructive hover:text-white"
                  disabled={!connected || shutdownStatus === "shutting_down"}
                  title="Shutdown Backend"
                >
                  {shutdownStatus === "shutting_down" ? (
                    <Loader2 className="size-4 animate-spin" />
                  ) : (
                    <Power className="size-4" />
                  )}
                </Button>
              </AlertDialogTrigger>
              <AlertDialogContent>
                <AlertDialogHeader>
                  <AlertDialogTitle>Shutdown Backend?</AlertDialogTitle>
                  <AlertDialogDescription>
                    This will stop the orchestrator and all running services (STT, TTS, LLM, VLM).
                    <br /><br />
                    You&apos;ll be redirected to the Launcher to restart.
                  </AlertDialogDescription>
                </AlertDialogHeader>
                <AlertDialogFooter>
                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                  <AlertDialogAction
                    onClick={handleShutdown}
                    className="bg-destructive text-white hover:bg-destructive/90"
                  >
                    Shutdown
                  </AlertDialogAction>
                </AlertDialogFooter>
              </AlertDialogContent>
            </AlertDialog>
          </div>
        </div>

        {/* Shutdown error — inline below header */}
        {shutdownStatus === "error" && shutdownMessage && (
          <p className="text-xs text-destructive">
            Error: {shutdownMessage}
          </p>
        )}
      </div>

      {/* NANO-069: Hero section — Portrait */}
      <Card className="flex items-center justify-center p-6">
        <CharacterPortrait />
      </Card>

      {/* NANO-073a: Unified Chat History */}
      <Card>
        <CardContent className="p-0">
          <ChatHistory />
        </CardContent>
      </Card>

      {/* NANO-073c: Unified input bar with voice overlay + stimulus indicator */}
      <ChatInput />

      {/* NANO-056: Idle timer progress bar — standalone card, only renders when stimuli enabled */}
      <PatienceBar />

      {/* LLM Provider (NANO-065b) — runtime provider/model swap */}
      <LLMConfig />

      {/* Reasoning Settings (NANO-042) — only renders when reasoning_format is configured */}
      <ReasoningSettings />

      {/* Memory Settings (NANO-043) — only renders when memory.enabled */}
      <MemorySettings />

      {/* Stimuli Settings (NANO-056) — always visible (master toggle inside) */}
      <StimuliSettings />

      {/* Twitch Chat (NANO-056b) — standalone card, credential-gated toggle */}
      <TwitchChatCard />

      {/* Generation Parameters (NANO-053) — always visible */}
      <GenerationSettings />

      {/* Tools Settings (NANO-065a) — always visible (master toggle inside) */}
      <ToolsSettings />

      {/* VLM Provider (NANO-065c) — runtime VLM provider swap */}
      <VLMConfig />

      {/* Tool Activity (NANO-025 Phase 7) */}
      {toolActivities.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="text-sm font-medium flex items-center gap-2">
              <Wrench className="h-4 w-4" />
              Tool Activity
              {toolActivities.some((t) => t.status === "running") && (
                <Loader2 className="h-4 w-4 animate-spin text-yellow-500" />
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            <div className="space-y-2 max-h-[200px] overflow-y-auto">
              {toolActivities.map((activity) => (
                <div
                  key={activity.id}
                  className="flex items-center justify-between text-sm border-b border-border pb-2 last:border-0"
                >
                  <div className="flex items-center gap-2">
                    {activity.status === "running" && (
                      <Loader2 className="h-4 w-4 animate-spin text-yellow-500" />
                    )}
                    {activity.status === "complete" && (
                      <CheckCircle2 className="h-4 w-4 text-green-500" />
                    )}
                    {activity.status === "error" && (
                      <XCircle className="h-4 w-4 text-red-500" />
                    )}
                    <span className="font-medium">{activity.tool_name}</span>
                    <Badge variant="outline" className="text-xs">
                      Iteration {activity.iteration}
                    </Badge>
                  </div>
                  <div className="flex items-center gap-2">
                    {activity.duration_ms !== undefined && (
                      <span className="text-xs text-muted-foreground">
                        {activity.duration_ms}ms
                      </span>
                    )}
                    {activity.status !== "running" && activity.result_summary && (
                      <span
                        className="text-xs text-muted-foreground max-w-[200px] truncate"
                        title={activity.result_summary}
                      >
                        {activity.result_summary.slice(0, 50)}
                        {activity.result_summary.length > 50 ? "..." : ""}
                      </span>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}


      {/* VTubeStudio (NANO-060b) — disabled pending avatar integration (NANO-092) */}
      <VTubeStudioCard />

      {/* State Transitions */}
      <Card>
        <CardHeader>
          <CardTitle className="text-sm font-medium">Recent State Transitions</CardTitle>
        </CardHeader>
        <CardContent>
          {stateTransitions.length > 0 ? (
            <div className="space-y-2 max-h-[200px] overflow-y-auto">
              {stateTransitions.map((t) => (
                <div
                  key={t.id}
                  className="flex items-center justify-between text-sm border-b border-border pb-2 last:border-0"
                >
                  <div className="flex items-center gap-2">
                    <span className="text-muted-foreground">
                      {stateLabels[t.from] || t.from}
                    </span>
                    <span className="text-muted-foreground">→</span>
                    <span className="font-medium">
                      {stateLabels[t.to] || t.to}
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="text-xs">
                      {t.trigger}
                    </Badge>
                    <span className="text-xs text-muted-foreground">
                      {new Date(t.timestamp).toLocaleTimeString()}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <p className="text-sm text-muted-foreground">No transitions yet</p>
          )}
        </CardContent>
      </Card>
    </div>
  );
}
