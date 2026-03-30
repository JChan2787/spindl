import React from "react";
import { render, RenderOptions } from "@testing-library/react";

// Wrapper for tests that need providers
interface ProvidersProps {
  children: React.ReactNode;
}

function Providers({ children }: ProvidersProps) {
  // Add providers here as needed (e.g., SocketProvider mock)
  return <>{children}</>;
}

// Custom render function that includes providers
function customRender(
  ui: React.ReactElement,
  options?: Omit<RenderOptions, "wrapper">
) {
  return render(ui, { wrapper: Providers, ...options });
}

// Re-export everything
export * from "@testing-library/react";
export { customRender as render };
