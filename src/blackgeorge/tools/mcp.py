from dataclasses import dataclass
from typing import Any, cast

from mcp import types as mcp_types
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel, create_model

from blackgeorge.async_utils import run_coroutine_sync
from blackgeorge.tools.base import Tool, ToolResult


def _json_schema_to_pydantic_field(
    name: str,
    schema: dict[str, Any],
) -> tuple[type[Any], Any]:
    json_type = schema.get("type", "string")
    default: Any = ...
    if "default" in schema:
        default = schema["default"]
    type_mapping: dict[str, type[Any]] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list,
        "object": dict,
    }
    py_type = type_mapping.get(json_type, Any)
    return (py_type, default)


def _build_input_model_from_schema(
    tool_name: str,
    parameters: dict[str, Any],
) -> type[BaseModel]:
    properties = parameters.get("properties", {})
    required_fields = set(parameters.get("required", []))
    fields: dict[str, Any] = {}
    for prop_name, prop_schema in properties.items():
        py_type, default = _json_schema_to_pydantic_field(prop_name, prop_schema)
        if prop_name in required_fields:
            fields[prop_name] = (py_type, ...)
        elif default is ...:
            fields[prop_name] = (py_type | None, None)
        else:
            fields[prop_name] = (py_type, default)
    model_name = f"{tool_name.replace('-', '_').replace('.', '_').title()}Input"
    return create_model(model_name, **fields)


@dataclass
class MCPConnection:
    session: ClientSession
    read_stream: Any
    write_stream: Any


class MCPToolProvider:
    def __init__(self) -> None:
        self._session: ClientSession | None = None
        self._context_manager: Any = None
        self._session_context: Any = None
        self._tools: list[Tool] = []
        self._mcp_tools: dict[str, mcp_types.Tool] = {}

    async def connect_stdio(
        self,
        command: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        params = StdioServerParameters(
            command=command,
            args=args or [],
            env=env,
        )
        self._context_manager = stdio_client(params)
        read_stream, write_stream = await self._context_manager.__aenter__()
        self._session_context = ClientSession(read_stream, write_stream)
        self._session = await self._session_context.__aenter__()
        await self._session.initialize()  # type: ignore[misc]
        await self._discover_tools()

    async def connect_sse(self, url: str) -> None:
        self._context_manager = sse_client(url)
        read_stream, write_stream = await self._context_manager.__aenter__()
        self._session_context = ClientSession(read_stream, write_stream)
        self._session = await self._session_context.__aenter__()
        await self._session.initialize()  # type: ignore[misc]
        await self._discover_tools()

    async def connect_streamable_http(
        self,
        url: str,
        http_client: Any | None = None,
    ) -> None:
        self._context_manager = streamable_http_client(url, http_client=http_client)
        read_stream, write_stream, _ = await self._context_manager.__aenter__()
        self._session_context = ClientSession(read_stream, write_stream)
        self._session = await self._session_context.__aenter__()
        await self._session.initialize()  # type: ignore[misc]
        await self._discover_tools()

    async def close(self) -> None:
        if self._session_context is not None:
            await self._session_context.__aexit__(None, None, None)
            self._session_context = None
        if self._context_manager is not None:
            await self._context_manager.__aexit__(None, None, None)
            self._context_manager = None
        self._session = None
        self._tools = []
        self._mcp_tools = {}

    async def _discover_tools(self) -> None:
        if self._session is None:
            return
        result = await self._session.list_tools()
        self._tools = []
        self._mcp_tools = {}
        for mcp_tool in result.tools:
            tool = self._convert_mcp_tool(mcp_tool)
            self._tools.append(tool)
            self._mcp_tools[mcp_tool.name] = mcp_tool

    def _convert_mcp_tool(self, mcp_tool: mcp_types.Tool) -> Tool:
        schema = mcp_tool.inputSchema or {}
        parameters = schema if isinstance(schema, dict) else {}
        input_model = _build_input_model_from_schema(mcp_tool.name, parameters)

        async def call_mcp_tool(**kwargs: Any) -> ToolResult:
            return await self._execute_tool(mcp_tool.name, kwargs)

        return Tool(
            name=mcp_tool.name,
            description=mcp_tool.description or "",
            schema=parameters,
            callable=call_mcp_tool,
            input_model=input_model,
            requires_confirmation=False,
            requires_user_input=False,
            external_execution=True,
        )

    async def _execute_tool(
        self,
        name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        if self._session is None:
            return ToolResult(error="MCP session not connected")
        try:
            result = await self._session.call_tool(name, arguments=arguments)
            content_parts: list[str] = []
            for item in result.content:
                if isinstance(item, mcp_types.TextContent):
                    content_parts.append(item.text)
                elif isinstance(item, mcp_types.ImageContent):
                    content_parts.append(f"[Image: {item.mimeType}]")
                elif isinstance(item, mcp_types.EmbeddedResource):
                    content_parts.append("[Embedded Resource]")
            content = "\n".join(content_parts) if content_parts else None
            data = result.structuredContent if hasattr(result, "structuredContent") else None
            return ToolResult(content=content, data=data)
        except Exception as exc:
            return ToolResult(error=str(exc))

    def list_tools(self) -> list[Tool]:
        return self._tools.copy()

    async def alist_tools(self) -> list[Tool]:
        if self._session is not None:
            await self._discover_tools()
        return self._tools.copy()

    async def acall_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        return await self._execute_tool(name, arguments)

    def call_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        result = run_coroutine_sync(self.acall_tool(name, arguments))
        return cast(ToolResult, result)

    async def __aenter__(self) -> "MCPToolProvider":
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        await self.close()
