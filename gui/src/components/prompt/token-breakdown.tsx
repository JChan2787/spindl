"use client";

import { Progress } from "@/components/ui/progress";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type {
  TokenBreakdown as TokenBreakdownType,
  BlockTokenData,
} from "@/types/events";
import { Coins } from "lucide-react";
import { getBlockColor } from "@/lib/constants/block-colors";

interface TokenBreakdownProps {
  breakdown: TokenBreakdownType;
}

interface SectionBarProps {
  label: string;
  tokens: number;
  total: number;
  color: string;
}

function SectionBar({ label, tokens, total, color }: SectionBarProps) {
  const percent = total > 0 ? (tokens / total) * 100 : 0;
  return (
    <div className="space-y-1">
      <div className="flex justify-between text-sm">
        <span className="text-muted-foreground">{label}</span>
        <span className="font-mono text-xs">
          {tokens.toLocaleString()} ({percent.toFixed(1)}%)
        </span>
      </div>
      <div className="h-2 rounded-full bg-muted overflow-hidden">
        <div
          className={`h-full rounded-full transition-all ${color}`}
          style={{ width: `${percent}%` }}
        />
      </div>
    </div>
  );
}

export function TokenBreakdown({ breakdown }: TokenBreakdownProps) {
  const { total, prompt, completion, system, user, sections } = breakdown;
  const hasBlocks = breakdown.blocks && breakdown.blocks.length > 0;

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between pb-2">
        <CardTitle className="text-sm font-medium">Token Breakdown</CardTitle>
        <Coins className="h-4 w-4 text-muted-foreground" />
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Overview */}
        <div className="grid grid-cols-3 gap-4 text-center">
          <div>
            <div className="text-2xl font-bold">{total.toLocaleString()}</div>
            <div className="text-xs text-muted-foreground">Total</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-blue-500">
              {prompt.toLocaleString()}
            </div>
            <div className="text-xs text-muted-foreground">Prompt</div>
          </div>
          <div>
            <div className="text-2xl font-bold text-green-500">
              {completion.toLocaleString()}
            </div>
            <div className="text-xs text-muted-foreground">Completion</div>
          </div>
        </div>

        {/* System vs User */}
        <div className="space-y-3 pt-2 border-t">
          <div className="text-xs font-medium uppercase text-muted-foreground">
            Prompt Distribution
          </div>
          <SectionBar
            label="System Prompt"
            tokens={system}
            total={prompt}
            color="bg-purple-500"
          />
          <SectionBar
            label="User Input"
            tokens={user}
            total={prompt}
            color="bg-blue-500"
          />
        </div>

        {/* NANO-045b: Per-block breakdown when available, legacy sections otherwise */}
        {hasBlocks ? (
          <div className="space-y-3 pt-2 border-t">
            <div className="text-xs font-medium uppercase text-muted-foreground">
              Prompt Blocks
            </div>
            {breakdown.blocks!
              .filter((b) => b.tokens > 0)
              .map((block) => (
                <SectionBar
                  key={block.id}
                  label={block.label}
                  tokens={block.tokens}
                  total={system}
                  color={getBlockColor(block.id)}
                />
              ))}
          </div>
        ) : (
          <div className="space-y-3 pt-2 border-t">
            <div className="text-xs font-medium uppercase text-muted-foreground">
              System Prompt Sections
            </div>
            <SectionBar
              label="Agent (Name, Appearance, Personality)"
              tokens={sections.agent}
              total={system}
              color="bg-violet-500"
            />
            <SectionBar
              label="Context (Modality, State)"
              tokens={sections.context}
              total={system}
              color="bg-cyan-500"
            />
            <SectionBar
              label="Rules"
              tokens={sections.rules}
              total={system}
              color="bg-amber-500"
            />
            <SectionBar
              label="Conversation (Summary, History)"
              tokens={sections.conversation}
              total={system}
              color="bg-rose-500"
            />
          </div>
        )}
      </CardContent>
    </Card>
  );
}
