"use client";

import { useCallback, useRef, useEffect } from "react";
import { Cpu, Volume2, Mic, Eye, Database, GitBranch } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { CollapsibleCard } from "@/components/ui/collapsible-card";
import { useSettingsStore, useAgentStore } from "@/lib/stores";
import { getSocket } from "@/lib/socket";

interface ProviderCardProps {
  title: string;
  icon: React.ReactNode;
  name: string | null;
  config?: Record<string, unknown> | null;
  isConfigured: boolean;
  isHealthy?: boolean | string | null;
}

function ProviderCard({ title, icon, name, config, isConfigured, isHealthy }: ProviderCardProps) {
  // NANO-112: Handle "disabled" status from health check
  const isDisabled = isHealthy === "disabled";
  // When health data is available, use it. Otherwise fall back to config presence.
  const hasHealth = isHealthy !== null && isHealthy !== undefined;
  const badgeVariant = isDisabled
    ? "secondary"
    : hasHealth
      ? (isHealthy === true) ? "default" : isConfigured ? "destructive" : "secondary"
      : isConfigured ? "secondary" : "secondary";
  const badgeLabel = isDisabled
    ? "Disabled"
    : hasHealth
      ? (isHealthy === true) ? "Active" : isConfigured ? "Unreachable" : "Not Configured"
      : isConfigured ? "Configured" : "Not Configured";

  return (
    <div className="p-3 rounded-md border border-border bg-muted/30">
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2">
          {icon}
          <span className="font-medium text-sm">{title}</span>
        </div>
        <Badge variant={badgeVariant}>
          {badgeLabel}
        </Badge>
      </div>
      {name && (
        <p className="text-sm text-muted-foreground">
          Provider: <span className="font-mono">{name}</span>
        </p>
      )}
      {config && Object.keys(config).length > 0 && (
        <div className="mt-2 text-xs text-muted-foreground font-mono">
          {Object.entries(config).map(([key, value]) => (
            <div key={key}>
              {key}: {maskSensitiveValue(key, String(value))}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function maskSensitiveValue(key: string, value: string): string {
  const sensitiveKeys = ["api_key", "token", "secret", "password", "key"];
  if (sensitiveKeys.some((k) => key.toLowerCase().includes(k))) {
    return "••••••••";
  }
  return value;
}

export function ProviderDisplay() {
  const { providers, generationConfig, setGenerationConfig } = useSettingsStore();
  const health = useAgentStore((s) => s.health);
  const socket = getSocket();
  const debounceRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current);
    };
  }, []);

  const handleHistoryModeChange = useCallback(
    (mode: "auto" | "splice" | "flatten") => {
      setGenerationConfig({ ...generationConfig, force_role_history: mode });
      if (debounceRef.current) clearTimeout(debounceRef.current);
      debounceRef.current = setTimeout(() => {
        socket.emit("set_generation_params", { force_role_history: mode });
      }, 300);
    },
    [generationConfig, socket, setGenerationConfig]
  );

  return (
    <CollapsibleCard
      id="providers"
      title="Providers"
      icon={<Cpu className="h-4 w-4" />}
    >
      <ProviderCard
        title="LLM"
        icon={<Cpu className="h-4 w-4 text-blue-500" />}
        name={providers.llm?.name ?? null}
        config={providers.llm?.config}
        isConfigured={!!providers.llm}
        isHealthy={health?.llm ?? null}
      />

      <div className="px-3 pb-3 -mt-1">
        <p className="text-xs font-medium text-muted-foreground mb-2 flex items-center gap-1.5">
          <GitBranch className="h-3 w-3" />
          History Mode
        </p>
        <div className="flex gap-1 rounded-md bg-muted p-1">
          {(["auto", "splice", "flatten"] as const).map((mode) => (
            <button
              key={mode}
              onClick={() => handleHistoryModeChange(mode)}
              className={`flex-1 text-xs py-1 px-2 rounded transition-colors ${
                generationConfig.force_role_history === mode
                  ? "bg-background text-foreground shadow-sm"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              {mode === "auto" ? "Auto" : mode === "splice" ? "Splice" : "Flatten"}
            </button>
          ))}
        </div>
      </div>

      <ProviderCard
        title="TTS"
        icon={<Volume2 className="h-4 w-4 text-green-500" />}
        name={providers.tts?.name ?? null}
        config={providers.tts?.config}
        isConfigured={!!providers.tts}
        isHealthy={health?.tts ?? null}
      />

      <ProviderCard
        title="STT"
        icon={<Mic className="h-4 w-4 text-yellow-500" />}
        name={providers.stt?.name ?? null}
        config={providers.stt?.config}
        isConfigured={!!providers.stt}
        isHealthy={health?.stt ?? null}
      />

      <ProviderCard
        title="VLM"
        icon={<Eye className="h-4 w-4 text-purple-500" />}
        name={providers.vlm?.name ?? null}
        config={providers.vlm?.config}
        isConfigured={!!providers.vlm}
        isHealthy={health?.vlm ?? null}
      />

      <ProviderCard
        title="Embedding"
        icon={<Database className="h-4 w-4 text-cyan-500" />}
        name={providers.embedding?.base_url ?? null}
        isConfigured={!!providers.embedding?.enabled}
        isHealthy={health?.embedding ?? null}
      />

      <p className="text-xs text-muted-foreground pt-2">
        Provider configuration is read-only. Edit <code className="bg-muted px-1 rounded">spindl.yaml</code> and restart to change providers.
      </p>
    </CollapsibleCard>
  );
}
