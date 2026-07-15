import contextlib
from typing import Any, cast

from jsonschema.validators import validator_for
from mcp import types as mcp_types
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client
from mcp.client.stdio import StdioServerParameters, stdio_client
from mcp.client.streamable_http import streamable_http_client
from pydantic import BaseModel, ConfigDict, create_model, model_validator

from blackgeorge.async_utils import run_coroutine_sync
from blackgeorge.tools.base import Tool, ToolResult


def _json_schema_to_pydantic_field(
    schema: dict[str, Any],
) -> tuple[Any, Any]:
    default = schema.get("default", ...)
    json_type = schema.get("type")
    type_mapping: dict[str, Any] = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": list[Any],
        "object": dict[str, Any],
        "null": type(None),
    }
    if isinstance(json_type, list):
        available = [
            type_mapping[item]
            for item in json_type
            if isinstance(item, str) and item in type_mapping
        ]
        if available:
            field_type = available[0]
            for option in available[1:]:
                field_type |= option
            return field_type, default
        return Any, default
    if not isinstance(json_type, str):
        return Any, default
    return type_mapping.get(json_type, Any), default


def _build_input_model_from_schema(
    tool_name: str,
    parameters: dict[str, Any],
) -> type[BaseModel]:
    validator_class = validator_for(parameters)
    validator_class.check_schema(parameters)
    properties = parameters.get("properties", {})
    required_fields = set(parameters.get("required", []))
    fields: dict[str, Any] = {}
    if not isinstance(properties, dict):
        properties = {}
    for prop_name, prop_schema in properties.items():
        if not isinstance(prop_name, str) or not isinstance(prop_schema, dict):
            continue
        py_type, default = _json_schema_to_pydantic_field(prop_schema)
        if prop_name in required_fields:
            fields[prop_name] = (py_type, ...)
        elif default is ...:
            fields[prop_name] = (py_type | None, None)
        else:
            fields[prop_name] = (py_type, default)

    schema_validator = validator_class(parameters)

    def validate_json_schema(value: Any) -> Any:
        errors = list(schema_validator.iter_errors(value))
        if errors:
            raise ValueError(errors[0].message)
        return value

    model_name = f"{tool_name.replace('-', '_').replace('.', '_').title()}Input"
    validators: dict[str, Any] = {
        "validate_json_schema": model_validator(mode="before")(validate_json_schema)
    }
    return create_model(
        model_name,
        __config__=ConfigDict(extra="allow"),
        __validators__=validators,
        **fields,
    )


class MCPToolProvider:
    def __init__(self) -> None:
        self._session: ClientSession | None = None
        self._context_manager: Any = None
        self._session_context: Any = None
        self._tools: list[Tool] = []

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
        await self._connect(stdio_client(params))

    async def connect_sse(self, url: str) -> None:
        await self._connect(sse_client(url))

    async def connect_streamable_http(
        self,
        url: str,
        http_client: Any | None = None,
    ) -> None:
        await self._connect(streamable_http_client(url, http_client=http_client))

    async def _connect(self, context_manager: Any) -> None:
        await self.close()
        self._context_manager = context_manager
        try:
            streams = await context_manager.__aenter__()
            read_stream, write_stream = streams[0], streams[1]
            self._session_context = ClientSession(read_stream, write_stream)
            self._session = await self._session_context.__aenter__()
            await self._session.initialize()  # type: ignore[misc]
            await self._discover_tools()
        except BaseException:
            with contextlib.suppress(Exception):
                await self.close()
            raise

    async def close(self) -> None:
        session_context = self._session_context
        context_manager = self._context_manager
        self._session_context = None
        self._context_manager = None
        self._session = None
        self._tools = []
        try:
            if session_context is not None:
                await session_context.__aexit__(None, None, None)
        finally:
            if context_manager is not None:
                await context_manager.__aexit__(None, None, None)

    async def _discover_tools(self) -> None:
        if self._session is None:
            return
        result = await self._session.list_tools()
        self._tools = []
        for mcp_tool in result.tools:
            tool = self._convert_mcp_tool(mcp_tool)
            self._tools.append(tool)

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
            if getattr(result, "isError", False) is True:
                return ToolResult(
                    content=content,
                    data=data,
                    error=content or "MCP tool returned an error",
                )
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
        _exc_type: type[BaseException] | None,
        _exc_val: BaseException | None,
        _exc_tb: Any,
    ) -> None:
        await self.close()
