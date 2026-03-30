interface FieldStatusBadgeProps {
  variant: "live" | "pending" | "tts" | "summarizer" | "metadata";
  description?: string;
}

const VARIANT_CONFIG: Record<
  FieldStatusBadgeProps["variant"],
  { dotClass: string; defaultText: string }
> = {
  live: {
    dotClass: "inline-block w-2 h-2 rounded-full bg-green-500 shrink-0",
    defaultText: "Injected into system prompt",
  },
  pending: {
    dotClass: "inline-block w-2 h-2 rounded-full bg-yellow-500 shrink-0",
    defaultText: "Will be injected (pending wiring)",
  },
  tts: {
    dotClass: "inline-block w-2 h-2 rounded-full bg-blue-500 shrink-0",
    defaultText: "Used for text-to-speech",
  },
  summarizer: {
    dotClass: "inline-block w-2 h-2 rounded-full bg-blue-500 shrink-0",
    defaultText: "Used for conversation summarization",
  },
  metadata: {
    dotClass: "inline-block w-2 h-2 rounded-full bg-gray-400 shrink-0",
    defaultText: "Not sent to model",
  },
};

export function FieldStatusBadge({ variant, description }: FieldStatusBadgeProps) {
  const config = VARIANT_CONFIG[variant];
  return (
    <span className="flex items-center gap-1.5 text-xs text-muted-foreground mt-0.5">
      <span className={config.dotClass} />
      {description ?? config.defaultText}
    </span>
  );
}
