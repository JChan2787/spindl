/**
 * Service status hooks (NANO-116 Phase B.4a).
 *
 * Thin selectors over the existing health_status data in agent-store.
 * Replaces scattered `health?.stt === "disabled"` checks across components
 * with a single import.
 *
 * No new WebSocket event — reads from the existing health_status payload
 * that already carries "disabled" | boolean for each service.
 */

import { useAgentStore } from "./agent-store";

type HealthService = "stt" | "tts";

/**
 * Returns true if the service is intentionally disabled in config.
 *
 * Usage: `const sttOff = useServiceDisabled("stt");`
 */
export function useServiceDisabled(service: HealthService): boolean {
  return useAgentStore((s) => s.health?.[service] === "disabled") ?? false;
}

/**
 * Returns the tri-state for a service: "disabled" | "healthy" | "down".
 *
 * Usage: `const sttState = useServiceStatus("stt");`
 */
export function useServiceStatus(
  service: HealthService,
): "disabled" | "healthy" | "down" | "unknown" {
  return useAgentStore((s) => {
    const val = s.health?.[service];
    if (val === undefined || val === null) return "unknown";
    if (val === "disabled") return "disabled";
    return val ? "healthy" : "down";
  });
}

/**
 * Badge variant helper — maps service status to Badge variant prop.
 *
 * Usage: `<Badge variant={serviceBadgeVariant("stt", health)}>`
 */
export function serviceBadgeVariant(
  status: "disabled" | "healthy" | "down" | "unknown",
): "secondary" | "default" | "destructive" | "outline" {
  switch (status) {
    case "disabled":
      return "secondary";
    case "healthy":
      return "default";
    case "down":
      return "destructive";
    case "unknown":
      return "outline";
  }
}

/**
 * Badge label helper — maps service status to display text.
 */
export function serviceBadgeLabel(
  status: "disabled" | "healthy" | "down" | "unknown",
): string {
  switch (status) {
    case "disabled":
      return "OFF";
    case "healthy":
      return "OK";
    case "down":
      return "DOWN";
    case "unknown":
      return "...";
  }
}
