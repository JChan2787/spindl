"use client";

import { useState, useCallback, useEffect } from "react";
import { ChevronDown } from "lucide-react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { cn } from "@/lib/utils";

const STORAGE_KEY = "spindl-settings-collapsed";

function getCollapsedState(): Record<string, boolean> {
  if (typeof window === "undefined") return {};
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch {
    return {};
  }
}

function setCollapsedState(id: string, collapsed: boolean) {
  const state = getCollapsedState();
  state[id] = collapsed;
  localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
}

interface CollapsibleCardProps {
  id: string;
  title: React.ReactNode;
  icon: React.ReactNode;
  /** Extra elements in the header row (e.g. saving indicator) */
  headerExtra?: React.ReactNode;
  /** Default collapsed state when no localStorage entry exists */
  defaultCollapsed?: boolean;
  children: React.ReactNode;
}

export function CollapsibleCard({
  id,
  title,
  icon,
  headerExtra,
  defaultCollapsed = false,
  children,
}: CollapsibleCardProps) {
  const [collapsed, setCollapsed] = useState(defaultCollapsed);
  const [hydrated, setHydrated] = useState(false);

  // Hydrate from localStorage after mount to avoid SSR mismatch
  useEffect(() => {
    const stored = getCollapsedState();
    if (id in stored) {
      setCollapsed(stored[id]);
    }
    setHydrated(true);
  }, [id]);

  // Persist to localStorage on user-driven changes (skip initial hydration)
  useEffect(() => {
    if (hydrated) {
      setCollapsedState(id, collapsed);
    }
  }, [id, collapsed, hydrated]);

  const toggle = useCallback(() => {
    setCollapsed((prev) => !prev);
  }, []);

  return (
    <Card>
      <CardHeader
        className="cursor-pointer select-none"
        onClick={toggle}
      >
        <CardTitle className="flex items-center gap-2 text-sm font-medium">
          {icon}
          {title}
          {headerExtra}
          <ChevronDown
            className={cn(
              "h-4 w-4 ml-auto text-muted-foreground transition-transform duration-200",
              collapsed && "-rotate-90"
            )}
          />
        </CardTitle>
      </CardHeader>
      {!collapsed && (
        <CardContent className="space-y-6">
          {children}
        </CardContent>
      )}
    </Card>
  );
}
