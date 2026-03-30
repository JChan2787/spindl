"use client";

import { Database } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";
import { useLauncherStore } from "@/lib/stores";

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

export function EmbeddingSection() {
  const { embedding, updateEmbedding, validationErrors } = useLauncherStore();

  return (
    <Card className={!embedding.enabled ? "opacity-60" : ""}>
      <CardHeader>
        <CardTitle className="flex items-center justify-between text-base font-medium">
          <div className="flex items-center gap-2">
            <Database className="h-5 w-5" />
            Embedding Server (Memory)
          </div>
          <div className="flex items-center gap-2">
            <Label
              htmlFor="embedding-enabled"
              className="text-sm font-normal text-muted-foreground"
            >
              {embedding.enabled ? "Enabled" : "Disabled"}
            </Label>
            <Switch
              id="embedding-enabled"
              checked={embedding.enabled}
              onCheckedChange={(checked) =>
                updateEmbedding({ enabled: checked })
              }
            />
          </div>
        </CardTitle>
      </CardHeader>
      {embedding.enabled && (
        <CardContent className="space-y-4">
          {/* Executable Path + Model Path */}
          <div className="grid gap-4 md:grid-cols-2">
            <FieldRow
              label="Executable Path"
              error={validationErrors["embedding.executablePath"]}
            >
              <Input
                placeholder="/path/to/llama-server"
                value={embedding.executablePath}
                onChange={(e) =>
                  updateEmbedding({ executablePath: e.target.value })
                }
              />
            </FieldRow>
            <FieldRow
              label="Model Path"
              error={validationErrors["embedding.modelPath"]}
            >
              <Input
                placeholder="/path/to/embedding-model.gguf"
                value={embedding.modelPath}
                onChange={(e) =>
                  updateEmbedding({ modelPath: e.target.value })
                }
              />
            </FieldRow>
          </div>

          {/* Network + Hardware Settings */}
          <div className="grid gap-4 md:grid-cols-5">
            <FieldRow label="Host">
              <Input
                value={embedding.host}
                onChange={(e) =>
                  updateEmbedding({ host: e.target.value })
                }
              />
            </FieldRow>
            <FieldRow label="Port">
              <Input
                type="number"
                value={embedding.port}
                onChange={(e) =>
                  updateEmbedding({
                    port: parseInt(e.target.value) || 5559,
                  })
                }
              />
            </FieldRow>
            <FieldRow label="GPU Layers">
              <Input
                type="number"
                value={embedding.gpuLayers}
                onChange={(e) =>
                  updateEmbedding({
                    gpuLayers: parseInt(e.target.value) || 99,
                  })
                }
              />
            </FieldRow>
            <FieldRow label="Context Size">
              <Input
                type="number"
                value={embedding.contextSize}
                onChange={(e) =>
                  updateEmbedding({
                    contextSize: parseInt(e.target.value) || 2048,
                  })
                }
              />
            </FieldRow>
            <FieldRow label="Timeout (s)">
              <Input
                type="number"
                value={embedding.timeout}
                onChange={(e) =>
                  updateEmbedding({
                    timeout: parseInt(e.target.value) || 60,
                  })
                }
              />
            </FieldRow>
          </div>

          {/* Memory Retrieval Settings */}
          <div className="grid gap-4 md:grid-cols-2">
            <FieldRow label="Relevance Threshold (0 = loose, 1 = strict)">
              <Input
                type="number"
                min={0}
                max={1}
                step={0.05}
                value={embedding.relevanceThreshold}
                onChange={(e) =>
                  updateEmbedding({
                    relevanceThreshold: parseFloat(e.target.value) || 0.25,
                  })
                }
              />
            </FieldRow>
            <FieldRow label="Top K Results">
              <Input
                type="number"
                min={1}
                max={20}
                step={1}
                value={embedding.topK}
                onChange={(e) =>
                  updateEmbedding({
                    topK: parseInt(e.target.value) || 5,
                  })
                }
              />
            </FieldRow>
          </div>
        </CardContent>
      )}
    </Card>
  );
}
