"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { useConnectionStore } from "@/lib/stores";
import {
  LayoutDashboard,
  FileSearch,
  MessageSquare,
  Settings,
  Database,
  BookOpen,
  Rocket,
  Users,
} from "lucide-react";

const navItems = [
  {
    label: "Dashboard",
    href: "/",
    icon: LayoutDashboard,
  },
  {
    label: "Launcher",
    href: "/launcher",
    icon: Rocket,
  },
  {
    label: "Prompt Workshop",
    href: "/prompt",
    icon: FileSearch,
  },
  {
    label: "Sessions",
    href: "/sessions",
    icon: MessageSquare,
  },
  {
    label: "Settings",
    href: "/settings",
    icon: Settings,
  },
  {
    label: "Characters",
    href: "/characters",
    icon: Users,
  },
  {
    label: "Memories",
    href: "/memories",
    icon: Database,
  },
  {
    label: "Codex",
    href: "/codex",
    icon: BookOpen,
  },
];

export function Sidebar() {
  const pathname = usePathname();
  const { connected, connecting } = useConnectionStore();

  return (
    <aside className="fixed left-0 top-0 z-40 h-screen w-56 border-r border-border bg-sidebar">
      <div className="flex h-full flex-col">
        {/* Header */}
        <div className="flex h-16 items-center border-b border-border px-4">
          <div className="flex items-center gap-2">
            <img src="/spindl-icon.png" alt="SpindL" className="h-8 w-8 rounded-lg" />
            <div>
              <h1 className="font-semibold text-sidebar-foreground">SpindL</h1>
              <div className="flex items-center gap-1.5">
                <div
                  className={cn(
                    "h-2 w-2 rounded-full",
                    connected
                      ? "bg-green-500"
                      : connecting
                        ? "bg-yellow-500 animate-pulse"
                        : "bg-red-500"
                  )}
                />
                <span className="text-xs text-muted-foreground">
                  {connected ? "Connected" : connecting ? "Connecting..." : "Disconnected"}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 space-y-1 px-2 py-4">
          {navItems.map((item) => {
            const isActive = pathname === item.href;
            const Icon = item.icon;

            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex items-center gap-3 rounded-lg px-3 py-2 text-sm font-medium transition-colors",
                  isActive
                    ? "bg-sidebar-accent text-sidebar-accent-foreground"
                    : "text-sidebar-foreground hover:bg-sidebar-accent/50",
                )}
              >
                <Icon className="h-4 w-4" />
                <span>{item.label}</span>
              </Link>
            );
          })}
        </nav>

        {/* Footer */}
        <div className="border-t border-border p-4">
          <p className="text-xs text-muted-foreground">
            Voice AI Control Panel
          </p>
        </div>
      </div>
    </aside>
  );
}
