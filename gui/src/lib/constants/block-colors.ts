/**
 * Shared block color map for prompt block visualization.
 * Used by token-breakdown.tsx and block-list.tsx.
 * NANO-045b / NANO-045c-2
 */

export const BLOCK_COLORS: Record<string, string> = {
  persona_name: "bg-violet-500",
  persona_appearance: "bg-violet-400",
  persona_personality: "bg-violet-300",
  modality_context: "bg-cyan-500",
  voice_state: "bg-cyan-400",
  codex_context: "bg-cyan-300",
  rag_context: "bg-cyan-200",
  persona_rules: "bg-amber-500",
  modality_rules: "bg-amber-400",
  conversation_summary: "bg-rose-500",
  recent_history: "bg-rose-400",
  closing_instruction: "bg-gray-400",
};

export function getBlockColor(blockId: string): string {
  return BLOCK_COLORS[blockId] || "bg-gray-400";
}

/** Convert bg-* class to border-l-* for left border indicator */
export function getBlockBorderColor(blockId: string): string {
  return getBlockColor(blockId).replace("bg-", "border-l-");
}
