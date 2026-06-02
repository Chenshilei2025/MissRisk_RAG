from __future__ import annotations

from agentic_mm_rag.tools.base import ObservationTool


class ToolRegistry:
    """Small explicit registry for observation tools."""

    def __init__(self) -> None:
        self._tools: dict[str, ObservationTool] = {}

    def register(self, tool: ObservationTool) -> None:
        self._tools[tool.tool_name] = tool

    def get(self, tool_name: str) -> ObservationTool:
        try:
            return self._tools[tool_name]
        except KeyError as exc:
            raise KeyError(f"unknown observation tool: {tool_name}") from exc

    def list_tools(self) -> list[str]:
        return sorted(self._tools)
