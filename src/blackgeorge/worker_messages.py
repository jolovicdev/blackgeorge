import json
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from typing import Any, cast

from pydantic import BaseModel

from blackgeorge.core.event_types import EventType
from blackgeorge.core.job import Job
from blackgeorge.core.message import Message
from blackgeorge.core.tool_call import ToolCall
from blackgeorge.tools.base import Tool, ToolResult


def _json_payload(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", warnings=False)
    if is_dataclass(value) and not isinstance(value, type):
        return asdict(value)
    if isinstance(value, dict):
        return {key: _json_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_payload(item) for item in value]
    if isinstance(value, tuple):
        return [_json_payload(item) for item in value]
    return value


def _is_multimodal_content_blocks(value: Any) -> bool:
    if not isinstance(value, list):
        return False
    if len(value) == 0:
        return False
    known_type_fields: dict[str, tuple[str, ...]] = {
        "text": ("text",),
        "image_url": ("image_url",),
        "video_url": ("video_url",),
        "file": ("file",),
        "audio_url": ("audio_url",),
        "input_text": ("text",),
        "input_image": ("image_url", "file_id"),
        "input_audio": ("input_audio",),
    }
    for item in value:
        if not isinstance(item, dict):
            return False
        block_type = item.get("type")
        if not isinstance(block_type, str):
            return False
        required_fields = known_type_fields.get(block_type)
        if required_fields is None:
            return False
        if not any(field in item for field in required_fields):
            return False
    return True


def render_input(value: Any) -> str | list[dict[str, Any]]:
    if isinstance(value, str):
        return value
    if _is_multimodal_content_blocks(value):
        payload = _json_payload(value)
        if isinstance(payload, list):
            return cast(list[dict[str, Any]], payload)
    return json.dumps(_json_payload(value), ensure_ascii=True, default=str)


def tool_call_payload(tool_call: ToolCall) -> dict[str, Any]:
    return {
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.name,
            "arguments": json.dumps(
                _json_payload(tool_call.arguments),
                ensure_ascii=True,
                default=str,
            ),
        },
    }


def messages_to_payload(messages: list[Message]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for message in messages:
        item: dict[str, Any] = {"role": message.role, "content": message.content}
        if message.role == "assistant":
            if message.reasoning_content is not None:
                item["reasoning_content"] = message.reasoning_content
            if message.thinking_blocks is not None:
                item["thinking_blocks"] = message.thinking_blocks
            if message.tool_calls:
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
        parts.append(
            "Constraints: "
            f"{json.dumps(_json_payload(job.constraints), ensure_ascii=True, default=str)}"
        )
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


def chunk_reasoning_content(chunk: Any) -> str | None:
    if isinstance(chunk, dict):
        choices = chunk.get("choices") or []
        if not choices:
            return None
        delta = choices[0].get("delta") or {}
        return delta.get("reasoning_content")
    choices = getattr(chunk, "choices", [])
    if not choices:
        return None
    delta = getattr(choices[0], "delta", None)
    if delta is None:
        return None
    return getattr(delta, "reasoning_content", None)


def chunk_thinking_blocks(chunk: Any) -> list[dict[str, Any]] | None:
    if isinstance(chunk, dict):
        choices = chunk.get("choices") or []
        if not choices:
            return None
        delta = choices[0].get("delta") or {}
        blocks = delta.get("thinking_blocks")
    else:
        choices = getattr(chunk, "choices", [])
        if not choices:
            return None
        delta = getattr(choices[0], "delta", None)
        if delta is None:
            return None
        blocks = getattr(delta, "thinking_blocks", None)
    if isinstance(blocks, dict):
        return [dict(blocks)]
    if isinstance(blocks, list):
        items = [dict(item) for item in blocks if isinstance(item, dict)]
        return items or None
    return None


def structured_content(value: Any) -> str:
    if isinstance(value, BaseModel):
        return value.model_dump_json(warnings=False)
    return json.dumps(_json_payload(value), ensure_ascii=True, default=str)


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
    emit(EventType.ASSISTANT_MESSAGE, worker_name, payload)


def ensure_content(value: str | None) -> str:
    return value or ""
