from blackgeorge.core.tool_call import ToolCall
from blackgeorge.tools import execute_tool, tool


def test_tool_schema_inference() -> None:
    @tool()
    def add(a: int, b: int) -> int:
        return a + b

    assert "properties" in add.schema
    assert "a" in add.schema["properties"]
    assert "b" in add.schema["properties"]


def test_tool_execution() -> None:
    @tool()
    def add(a: int, b: int) -> int:
        return a + b

    call = ToolCall(id="1", name="add", arguments={"a": 1, "b": 2})
    result = execute_tool(add, call)
    assert result.error is None
    assert result.content == "3"
