"""
Invoke Hack Tool (NANO-134).

LLM-callable tool that sends a hack_walk command to the game bridge,
triggering the golden path walker on the active puzzle grid.
"""

import uuid
from typing import Any

from ...base import Tool, ToolParameter, ToolResult


class InvokeHackTool(Tool):

    def __init__(self) -> None:
        self._game_state_module: Any = None

    @property
    def name(self) -> str:
        return "invoke_hack"

    @property
    def description(self) -> str:
        return (
            "Hack an enemy's puzzle grid by walking the golden path solution. "
            "Use when the user asks you to hack a robot, bot, or enemy during combat. "
            "Only works when the player is already in hack mode (aiming at a hackable enemy). "
            "Returns success or an error explaining why the hack couldn't execute."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return []

    def initialize(self, config: dict) -> None:
        pass

    def set_game_state_module(self, module: Any) -> None:
        """Inject live GameStateModule reference (called by orchestrator)."""
        self._game_state_module = module

    def health_check(self) -> bool:
        if not self._game_state_module:
            return False
        return self._game_state_module.health_check()

    async def execute(self, **kwargs: Any) -> ToolResult:
        if not self._game_state_module:
            return ToolResult(
                success=False,
                output="Game bridge not connected — can't send hack command.",
                error="no_module",
            )

        if not self._game_state_module.connected:
            return ToolResult(
                success=False,
                output="Game bridge not connected — can't send hack command.",
                error="disconnected",
            )

        cmd_id = str(uuid.uuid4())
        command = {
            "command_id": cmd_id,
            "command_type": "hack_walk",
        }

        try:
            response = await self._game_state_module.send_command_and_wait(
                command, timeout=10.0
            )
        except ConnectionError:
            return ToolResult(
                success=False,
                output="Lost connection to game bridge while sending hack command.",
                error="connection_lost",
            )

        if response.get("success"):
            return ToolResult(
                success=True,
                output=(
                    "Hack initiated — walking the golden path now. "
                    "The puzzle grid is being solved."
                ),
                metadata={"command_id": cmd_id},
            )

        error_code = response.get("error_code", "unknown")
        message = response.get("message", "Unknown error")

        error_descriptions = {
            "no_active_hack": (
                "No active hack — the player needs to enter hack mode first "
                "(aim at a hackable enemy)."
            ),
            "not_snake_puzzle": (
                "The active puzzle is an environmental terminal, not a combat hack. "
                "Can't auto-solve this type."
            ),
            "walk_in_progress": "Already walking the golden path — hack in progress.",
            "player_dead": "Can't hack — the player is dead.",
            "golden_path_unavailable": (
                "The puzzle grid hasn't generated its solution yet. "
                "Try again in a moment."
            ),
            "timeout": "The game bridge didn't respond in time.",
            "disconnected": "Lost connection to the game bridge.",
        }

        output = error_descriptions.get(error_code, f"Hack failed: {message}")

        return ToolResult(
            success=False,
            output=output,
            error=error_code,
            metadata={"command_id": cmd_id},
        )
