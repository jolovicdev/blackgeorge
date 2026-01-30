import json
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from blackgeorge.core.job import Job
from blackgeorge.core.message import Message
from blackgeorge.core.tool_call import ToolCall
from blackgeorge.tools.base import Tool, ToolResult


def render_input(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=True)


def tool_call_payload(tool_call: ToolCall) -> dict[str, Any]:
    return {
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.name,
            "arguments": json.dumps(tool_call.arguments, ensure_ascii=True),
        },
    }


def messages_to_payload(messages: list[Message]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for message in messages:
        item: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.role == "assistant" and message.tool_calls:
            item["tool_calls"] = [tool_call_payload(call) for call in message.tool_calls]
        if message.role == "tool" and message.tool_call_id:
            item["tool_call_id"] = message.tool_call_id
        payload.append(item)
    return payload


def tool_schemas(tools: list[Tool]) -> list[dict[str, Any]]:
    schemas: list[dict[str, Any]] = []
    for tool in tools:
        schemas.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.schema,
                },
            }
        )
    return schemas


def system_message(instructions: str | None, job: Job) -> str | None:
    parts: list[str] = []
    if instructions:
        parts.append(instructions)
    if job.expected_output:
        parts.append(f"Expected output: {job.expected_output}")
    if job.constraints:
        parts.append(f"Constraints: {json.dumps(job.constraints, ensure_ascii=True)}")
    if not parts:
        return None
    return "\n\n".join(parts)


def chunk_content(chunk: Any) -> str | None:
    if isinstance(chunk, dict):
        choices = chunk.get("choices") or []
        if not choices:
            return None
        delta = choices[0].get("delta") or {}
        return delta.get("content")
    choices = getattr(chunk, "choices", [])
    if not choices:
        return None
    delta = getattr(choices[0], "delta", None)
    if delta is None:
        return None
    return getattr(delta, "content", None)


def chunk_usage(chunk: Any) -> dict[str, Any] | None:
    if isinstance(chunk, dict):
        return chunk.get("usage")
    return getattr(chunk, "usage", None)


def structured_content(value: Any) -> str:
    if isinstance(value, BaseModel):
        return value.model_dump_json(warnings=False)
    return json.dumps(value, ensure_ascii=True)


def tool_message(result: ToolResult | dict[str, Any], tool_call: ToolCall) -> Message:
    if isinstance(result, ToolResult):
        if result.content is not None:
            content = result.content
        elif result.data is not None:
            try:
                content = json.dumps(result.data, ensure_ascii=True, default=str)
            except TypeError:
                content = str(result.data)
        elif result.error is not None:
            content = result.error
        else:
            content = ""
    else:
        try:
            content = json.dumps(result, ensure_ascii=True, default=str)
        except TypeError:
            content = str(result)
    return Message(role="tool", content=content, tool_call_id=tool_call.id)


def tool_call_with_result(call: ToolCall, result: ToolResult | dict[str, Any]) -> ToolCall:
    error = result.error if isinstance(result, ToolResult) else None
    return call.model_copy(update={"result": result, "error": error})


def replace_tool_call(tool_calls: list[ToolCall], updated: ToolCall) -> None:
    for index, call in enumerate(tool_calls):
        if call.id == updated.id:
            tool_calls[index] = updated
            return
    raise ValueError(f"Tool call {updated.id} not found")


def tool_call_summaries(tool_calls: list[ToolCall]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for call in tool_calls:
        summaries.append(
            {
                "id": call.id,
                "name": call.name,
                "arguments": call.arguments,
            }
        )
    return summaries


def emit_assistant_message(
    emit: Callable[[str, str, dict[str, Any]], None],
    worker_name: str,
    message: Message,
) -> None:
    payload: dict[str, Any] = {"content": message.content}
    if message.tool_calls:
        payload["tool_calls"] = tool_call_summaries(message.tool_calls)
    emit("assistant.message", worker_name, payload)


def ensure_content(value: str | None) -> str:
    return value or ""
