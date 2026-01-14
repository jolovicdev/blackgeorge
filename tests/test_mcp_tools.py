from unittest.mock import AsyncMock, MagicMock

import pytest
from mcp import types as mcp_types
from pydantic import ValidationError

from blackgeorge.tools.mcp import (
    MCPToolProvider,
    _build_input_model_from_schema,
)


def test_build_input_model_from_schema() -> None:
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "URL to fetch"},
            "timeout": {"type": "integer", "default": 30},
        },
        "required": ["url"],
    }
    model = _build_input_model_from_schema("fetch", parameters)
    assert "url" in model.model_fields
    assert "timeout" in model.model_fields
    instance = model(url="http://example.com")
    assert instance.url == "http://example.com"
    assert instance.timeout == 30


def test_build_input_model_optional_fields() -> None:
    parameters = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "count": {"type": "integer"},
        },
        "required": [],
    }
    model = _build_input_model_from_schema("optional_test", parameters)
    instance = model()
    assert instance.name is None
    assert instance.count is None


def test_required_field_with_default_is_required() -> None:
    parameters = {
        "type": "object",
        "properties": {
            "count": {"type": "integer", "default": 5},
        },
        "required": ["count"],
    }
    model = _build_input_model_from_schema("required_default", parameters)
    with pytest.raises(ValidationError):
        model()


@pytest.fixture
def mock_mcp_tools() -> list[mcp_types.Tool]:
    return [
        mcp_types.Tool(
            name="add",
            description="Add two numbers",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "integer"},
                    "b": {"type": "integer"},
                },
                "required": ["a", "b"],
            },
        ),
        mcp_types.Tool(
            name="fetch",
            description="Fetch a URL",
            inputSchema={
                "type": "object",
                "properties": {
                    "url": {"type": "string"},
                },
                "required": ["url"],
            },
        ),
    ]


@pytest.mark.asyncio
async def test_mcp_tool_provider_list_tools(mock_mcp_tools: list[mcp_types.Tool]) -> None:
    provider = MCPToolProvider()
    mock_session = MagicMock()
    mock_session.initialize = AsyncMock()
    mock_session.list_tools = AsyncMock(return_value=MagicMock(tools=mock_mcp_tools))
    provider._session = mock_session
    await provider._discover_tools()
    tools = provider.list_tools()
    assert len(tools) == 2
    tool_names = [t.name for t in tools]
    assert "add" in tool_names
    assert "fetch" in tool_names


@pytest.mark.asyncio
async def test_mcp_tool_provider_call_tool() -> None:
    provider = MCPToolProvider()
    mock_result = MagicMock()
    mock_result.content = [mcp_types.TextContent(type="text", text="Result: 8")]
    mock_result.structuredContent = {"sum": 8}
    mock_session = MagicMock()
    mock_session.call_tool = AsyncMock(return_value=mock_result)
    provider._session = mock_session
    result = await provider.acall_tool("add", {"a": 3, "b": 5})
    assert result.error is None
    assert result.content == "Result: 8"
    assert result.data == {"sum": 8}


@pytest.mark.asyncio
async def test_mcp_tool_provider_sync_call_in_async_context() -> None:
    provider = MCPToolProvider()
    mock_result = MagicMock()
    mock_result.content = [mcp_types.TextContent(type="text", text="Result: 8")]
    mock_result.structuredContent = {"sum": 8}
    mock_session = MagicMock()
    mock_session.call_tool = AsyncMock(return_value=mock_result)
    provider._session = mock_session
    result = provider.call_tool("add", {"a": 3, "b": 5})
    assert result.error is None
    assert result.content == "Result: 8"


@pytest.mark.asyncio
async def test_mcp_tool_provider_call_tool_error() -> None:
    provider = MCPToolProvider()
    mock_session = MagicMock()
    mock_session.call_tool = AsyncMock(side_effect=Exception("Connection failed"))
    provider._session = mock_session
    result = await provider.acall_tool("bad_tool", {})
    assert result.error == "Connection failed"


@pytest.mark.asyncio
async def test_mcp_tool_provider_not_connected() -> None:
    provider = MCPToolProvider()
    result = await provider.acall_tool("any", {})
    assert result.error == "MCP session not connected"


def test_converted_tool_has_input_model(mock_mcp_tools: list[mcp_types.Tool]) -> None:
    provider = MCPToolProvider()
    tool = provider._convert_mcp_tool(mock_mcp_tools[0])
    assert tool.name == "add"
    assert tool.description == "Add two numbers"
    assert tool.external_execution is True
    assert "a" in tool.schema.get("properties", {})
    assert "b" in tool.schema.get("properties", {})
    validated = tool.input_model(a=1, b=2)
    assert validated.a == 1
    assert validated.b == 2
