import asyncio
import json
from inspect import iscoroutinefunction
from typing import Any

from pydantic import BaseModel

from blackgeorge.core.tool_call import ToolCall
from blackgeorge.tools.base import Tool, ToolResult


def _to_content(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, BaseModel):
        return value.model_dump_json()
    return json.dumps(value, ensure_ascii=True)


def execute_tool(tool: Tool, call: ToolCall) -> ToolResult:
    for pre_hook in tool.pre:
        pre_hook(call)

    try:
        validated = tool.input_model.model_validate(call.arguments)
        args = validated.model_dump()
        result = tool.callable(**args)
        if isinstance(result, ToolResult):
            tool_result = result
        else:
            tool_result = ToolResult(content=_to_content(result), data=result)
    except Exception as exc:
        tool_result = ToolResult(error=str(exc))

    for post_hook in tool.post:
        post_hook(call, tool_result)

    return tool_result


async def aexecute_tool(tool: Tool, call: ToolCall) -> ToolResult:
    for pre_hook in tool.pre:
        if iscoroutinefunction(pre_hook):
            await pre_hook(call)
        else:
            pre_hook(call)

    try:
        validated = tool.input_model.model_validate(call.arguments)
        args = validated.model_dump()

        if iscoroutinefunction(tool.callable):
            result = await tool.callable(**args)
        else:
            result = await asyncio.to_thread(tool.callable, **args)

        if isinstance(result, ToolResult):
            tool_result = result
        else:
            tool_result = ToolResult(content=_to_content(result), data=result)
    except Exception as exc:
        tool_result = ToolResult(error=str(exc))

    for post_hook in tool.post:
        if iscoroutinefunction(post_hook):
            await post_hook(call, tool_result)
        else:
            post_hook(call, tool_result)

    return tool_result
