"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Wrench } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";
import { useSettingsStore } from "@/lib/stores";
import { getSocket } from "@/lib/socket";

export function ToolsSettings() {
  const { toolsConfig, isSavingTools, setSavingTools, setToolsConfig } = useSettingsStore();
  const socket = getSocket();
  const [error, setError] = useState<string | null>(null);
  const pendingMasterRef = useRef<boolean | null>(null);

  // Listen for server response — detect toggle rejection
  useEffect(() => {
    const handler = (event: typeof toolsConfig & { error?: string }) => {
      // NANO-089 Phase 4: explicit server error takes priority
      if (event.error) {
        setError(event.error);
        pendingMasterRef.current = null;
        return;
      }
      if (pendingMasterRef.current !== null) {
        const requested = pendingMasterRef.current;
        if (requested && !event.master_enabled) {
          setError("No tools available — configure a VLM provider first.");
        }
        pendingMasterRef.current = null;
      }
    };
    socket.on("tools_config_updated", handler);
    return () => { socket.off("tools_config_updated", handler); };
  }, [socket]);

  const handleMasterToggle = useCallback(
    (checked: boolean) => {
      setError(null);
      pendingMasterRef.current = checked;
      setSavingTools(true);
      socket.emit("set_tools_config", { master_enabled: checked });
    },
    [socket, setSavingTools]
  );

  const handleToolToggle = useCallback(
    (toolName: string, checked: boolean) => {
      // Update local state immediately for per-tool (these don't fail)
      const updatedTools = { ...toolsConfig.tools };
      updatedTools[toolName] = { ...updatedTools[toolName], enabled: checked };
      setToolsConfig({ ...toolsConfig, tools: updatedTools });
      setSavingTools(true);
      socket.emit("set_tools_config", {
        tools: { [toolName]: { enabled: checked } },
      });
    },
    [toolsConfig, socket, setToolsConfig, setSavingTools]
  );

  const toolEntries = Object.entries(toolsConfig.tools);

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-sm font-medium">
          <Wrench className="h-4 w-4" />
          Tool Use
          {isSavingTools && (
            <span className="text-xs text-muted-foreground">(saving...)</span>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex items-center justify-between">
          <Label htmlFor="tools-master-toggle" className="text-sm">
            Enable Tools
          </Label>
          <Switch
            id="tools-master-toggle"
            checked={toolsConfig.master_enabled}
            onCheckedChange={handleMasterToggle}
            disabled={isSavingTools}
          />
        </div>

        {error && (
          <p className="text-xs text-destructive">{error}</p>
        )}

        {toolEntries.length > 0 && (
          <div className="space-y-3 pl-2 border-l-2 border-border ml-1">
            {toolEntries.map(([name, tool]) => (
              <div key={name} className="flex items-center justify-between">
                <Label
                  htmlFor={`tool-toggle-${name}`}
                  className={`text-sm ${!toolsConfig.master_enabled ? "text-muted-foreground" : ""}`}
                >
                  {tool.label}
                </Label>
                <Switch
                  id={`tool-toggle-${name}`}
                  checked={tool.enabled}
                  disabled={!toolsConfig.master_enabled || isSavingTools}
                  onCheckedChange={(checked) => handleToolToggle(name, checked)}
                />
              </div>
            ))}
          </div>
        )}

        <p className="text-xs text-muted-foreground">
          Disabled tools are excluded from the LLM&apos;s function list. Changes apply to the next message.
        </p>
      </CardContent>
    </Card>
  );
}
