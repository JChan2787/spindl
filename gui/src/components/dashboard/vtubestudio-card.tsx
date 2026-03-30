"use client";

// Interactive VTS controls (hotkeys, expressions, position presets) disabled
// pending avatar integration (NANO-092). Re-enable these imports and restore
// callbacks when the VTubeStudio driver is reactivated. See NANO-060b for
// original implementation.
// import { useCallback } from "react";
// import { RefreshCw, Keyboard, Smile, Move } from "lucide-react";
// import { useVTSStore } from "@/lib/stores";
// import { getSocket } from "@/lib/socket";

import { Monitor } from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Label } from "@/components/ui/label";

export function VTubeStudioCard() {
  return (
    <Card className="opacity-60">
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-sm font-medium">
          <Monitor className="h-4 w-4" />
          VTubeStudio
          <Badge variant="secondary" className="text-[10px] px-1.5 py-0">
            Coming Soon
          </Badge>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Disabled toggle */}
        <div className="flex items-center justify-between">
          <Label className="flex items-center gap-2 text-sm text-muted-foreground">
            Enable VTS
          </Label>
          <button
            disabled
            className="relative inline-flex h-5 w-9 items-center rounded-full bg-muted cursor-not-allowed"
          >
            <span className="inline-block h-3.5 w-3.5 transform rounded-full translate-x-0.5 bg-white" />
          </button>
        </div>
        <p className="text-xs text-muted-foreground">
          Avatar integration is under active development. VTubeStudio driver will be re-enabled in a future release.
        </p>
      </CardContent>
    </Card>
  );
}
