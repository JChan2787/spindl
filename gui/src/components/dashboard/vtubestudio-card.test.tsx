import { describe, it, expect } from "vitest";
import { render, screen } from "@testing-library/react";
import { VTubeStudioCard } from "./vtubestudio-card";

describe("VTubeStudioCard", () => {
  it("should render with Coming Soon badge", () => {
    render(<VTubeStudioCard />);
    expect(screen.getByText("Coming Soon")).toBeInTheDocument();
    expect(screen.getByText("VTubeStudio")).toBeInTheDocument();
  });

  it("should have a disabled toggle", () => {
    render(<VTubeStudioCard />);
    const toggle = screen
      .getByText("Enable VTS")
      .closest("div")
      ?.querySelector("button");
    expect(toggle).toBeInTheDocument();
    expect(toggle).toBeDisabled();
  });

  it("should show development message", () => {
    render(<VTubeStudioCard />);
    expect(
      screen.getByText(
        "Avatar integration is under active development. VTubeStudio driver will be re-enabled in a future release.",
      ),
    ).toBeInTheDocument();
  });

  it("should not show interactive controls", () => {
    render(<VTubeStudioCard />);
    expect(screen.queryByText("Hotkeys")).not.toBeInTheDocument();
    expect(screen.queryByText("Expressions")).not.toBeInTheDocument();
    expect(screen.queryByText("Position Presets")).not.toBeInTheDocument();
  });
});
