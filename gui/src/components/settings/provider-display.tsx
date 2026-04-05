"use client";

import { Cpu, Volume2, Mic, Eye, Database } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { CollapsibleCard } from "@/components/ui/collapsible-card";
import { useSettingsStore, useAgentStore } from "@/lib/stores";

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
  const { providers } = useSettingsStore();
  const health = useAgentStore((s) => s.health);

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
