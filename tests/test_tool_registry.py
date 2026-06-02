from agentic_mm_rag.tools import ToolRegistry


class DummyTool:
    tool_name = "dummy"

    def can_apply(self, unit, state) -> bool:
        return True

    def apply(self, unit, state, action):
        raise NotImplementedError


def test_tool_registry_registers_and_lists_tools() -> None:
    registry = ToolRegistry()
    registry.register(DummyTool())
    assert registry.list_tools() == ["dummy"]
    assert registry.get("dummy").tool_name == "dummy"
