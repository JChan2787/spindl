"""
Game State Query Tool (NANO-133).

On-demand tool for LLM to query current game state from the bridge.
Returns structured text describing hack status, combat state, or player info.
The LLM calls this when the user asks about in-game activity.
"""

from typing import Any, Optional

from ...base import Tool, ToolParameter, ToolResult


class GameStateQueryTool(Tool):

    def __init__(self) -> None:
        self._game_state_module: Any = None

    @property
    def name(self) -> str:
        return "game_state_query"

    @property
    def description(self) -> str:
        return (
            "Query current game state from the live game bridge. "
            "Use when the user asks about what's happening in-game: "
            "hack status, combat info, or player status."
        )

    @property
    def parameters(self) -> list[ToolParameter]:
        return [
            ToolParameter(
                name="query_type",
                param_type="string",
                description=(
                    "What to report: 'hack_status' for current/last hack, "
                    "'combat_status' for enemy roster and fight state, "
                    "'player_status' for Hugh's HP/weapon/location."
                ),
                required=True,
                enum=["hack_status", "combat_status", "player_status"],
            ),
        ]

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
        query_type = kwargs.get("query_type", "hack_status")

        if not self._game_state_module:
            return ToolResult(
                success=False,
                output="Game bridge not connected — no game state available.",
                error="no_module",
            )

        if not self._game_state_module.connected:
            return ToolResult(
                success=False,
                output="Game bridge not connected — no game state available.",
                error="disconnected",
            )

        state = self._game_state_module.get_hack_state()

        if query_type == "hack_status":
            output = self._format_hack_status(state)
        elif query_type == "combat_status":
            output = self._format_combat_status(state)
        elif query_type == "player_status":
            output = self._format_player_status(state)
        else:
            output = self._format_hack_status(state)

        return ToolResult(success=True, output=output)

    def _format_hack_status(self, state: dict[str, Any]) -> str:
        lines: list[str] = []
        current = state["current"]
        last = state["last_outcome"]
        snapshot = state["snapshot"]

        if current:
            puzzle_type = current.get("puzzle_type", "unknown")
            target = current.get("target_name", "unknown")

            if puzzle_type == "snake":
                w = current.get("grid_width", "?")
                h = current.get("grid_height", "?")
                playable = current.get("grid_playable_cells", "?")
                total = current.get("grid_total_cells", "?")
                obstacles = current.get("obstacle_count", 0)
                hazards = current.get("hazard_count", 0)
                golden = current.get("golden_path_length", -1)

                shape = "irregular shape" if playable != total else "full grid"
                lines.append(f"HACK ACTIVE — Combat (PuzzleSnake grid)")
                lines.append(f"Target: {target}")
                lines.append(f"Grid: {w}x{h}, {playable} playable cells ({shape})")
                lines.append(f"Obstacles: {obstacles} | Hazards: {hazards}")
                if golden == -1:
                    lines.append("Solution path: not yet computed")
                else:
                    lines.append(f"Solution path: {golden} steps")
            else:
                lines.append("HACK ACTIVE — Environmental terminal")
                lines.append(f"Target: {target}")
                lines.append("Puzzle type: Node routing (no grid)")
        else:
            lines.append("NO ACTIVE HACK")
            if last:
                outcome = last.get("outcome", "unknown")
                target = last.get("target_name", "unknown")
                chain = last.get("chain_count", 0)
                puzzle_type = last.get("puzzle_type", "unknown")
                detail = f"target: {target}"
                if chain:
                    detail += f", chain: {chain}"
                lines.append(f"Last result: {outcome.capitalize()} ({detail})")
            else:
                lines.append("No recent hack activity.")

        lines.append(self._format_combat_context(snapshot))
        return "\n".join(lines)

    def _format_combat_status(self, state: dict[str, Any]) -> str:
        snapshot = state["snapshot"]
        if not snapshot:
            return "No game state snapshot available."

        lines: list[str] = []
        in_combat = snapshot.get("in_combat", False)
        hp_ratio = snapshot.get("hp_ratio", 1.0)
        weapon = snapshot.get("weapon_name", "Unknown")
        enemies = snapshot.get("enemies") or []
        is_boss = snapshot.get("is_boss_battle", False)

        if in_combat:
            lines.append(f"IN COMBAT — {len(enemies)} enem{'y' if len(enemies) == 1 else 'ies'} engaged")
        else:
            lines.append("NOT IN COMBAT")

        lines.append(f"Hugh: {int(hp_ratio * 100)}% HP, weapon: {weapon}")

        if enemies:
            lines.append("Enemies:")
            for e in enemies:
                name = e.get("display_name", e.get("name", "Unknown"))
                e_hp = int(e.get("hp_ratio", 1.0) * 100)
                flags = []
                if e.get("is_hackable"):
                    flags.append("hackable")
                if e.get("is_confused"):
                    flags.append("confused")
                if e.get("is_dead"):
                    flags.append("dead")
                flag_str = f" [{', '.join(flags)}]" if flags else ""
                lines.append(f"  - {name}: {e_hp}% HP{flag_str}")

        if is_boss:
            lines.append("** BOSS BATTLE ACTIVE **")

        return "\n".join(lines)

    def _format_player_status(self, state: dict[str, Any]) -> str:
        snapshot = state["snapshot"]
        if not snapshot:
            return "No game state snapshot available."

        hp_ratio = snapshot.get("hp_ratio", 1.0)
        weapon = snapshot.get("weapon_name", "Unknown")
        in_combat = snapshot.get("in_combat", False)
        is_dead = snapshot.get("is_dead", False)
        enemies = snapshot.get("enemies") or []

        status_parts = []
        if is_dead:
            status_parts.append("DEAD")
        else:
            status_parts.append("Alive")
        if in_combat:
            status_parts.append(f"in combat ({len(enemies)} enemies)")
        else:
            status_parts.append("not in combat")

        return (
            f"Hugh: {int(hp_ratio * 100)}% HP, weapon: {weapon}\n"
            f"Status: {', '.join(status_parts)}"
        )

    def _format_combat_context(self, snapshot: Optional[dict[str, Any]]) -> str:
        if not snapshot:
            return "Combat context: No snapshot available"

        in_combat = snapshot.get("in_combat", False)
        if not in_combat:
            enemies = snapshot.get("enemies") or []
            if enemies:
                return f"Combat context: Not in combat, {len(enemies)} enemies nearby"
            return "Combat context: Not in combat, no enemies nearby"

        enemies = snapshot.get("enemies") or []
        hp_ratio = snapshot.get("hp_ratio", 1.0)
        return (
            f"Combat context: In combat, {len(enemies)} "
            f"enem{'y' if len(enemies) == 1 else 'ies'} engaged, "
            f"Hugh at {int(hp_ratio * 100)}% HP"
        )
