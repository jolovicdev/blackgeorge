import json
from typing import Any, cast

from pydantic import BaseModel, TypeAdapter

from blackgeorge.core.tool_call import ToolCall
from blackgeorge.utils import new_id


def parse_structured_stream_json(response_schema: Any, content: str) -> Any:
    if isinstance(response_schema, TypeAdapter):
        return response_schema.validate_json(content)
    if isinstance(response_schema, type) and issubclass(response_schema, BaseModel):
        return response_schema.model_validate_json(content)
    return TypeAdapter(response_schema).validate_json(content)


def stream_value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def append_tool_error(current: str | None, message: str) -> str:
    return message if current is None else f"{current}; {message}"


def is_stream_unsupported_error(exc: Exception) -> bool:
    if isinstance(exc, NotImplementedError):
        return True
    message = f"{type(exc).__name__}: {exc}".lower()
    mentions_stream = "stream" in message or "streaming" in message
    unsupported_markers = (
        "unsupported",
        "not support",
        "not supported",
        "not implemented",
        "cannot stream",
    )
    return mentions_stream and any(marker in message for marker in unsupported_markers)


def chunk_tool_call_deltas(chunk: Any) -> tuple[list[Any], bool]:
    choices = stream_value(chunk, "choices", []) or []
    if not choices:
        return [], False
    choice = choices[0]
    delta = stream_value(choice, "delta")
    if delta is not None:
        tool_calls = stream_value(delta, "tool_calls")
        if isinstance(tool_calls, list) and tool_calls:
            return tool_calls, False
    message = stream_value(choice, "message")
    if message is None:
        return [], False
    tool_calls = stream_value(message, "tool_calls", []) or []
    if isinstance(tool_calls, list):
        return tool_calls, True
    return [], False


def streamed_tool_calls(states: list[dict[str, Any]]) -> list[ToolCall]:
    parsed: list[ToolCall] = []
    for state in states:
        error = cast(str | None, state.get("error"))
        name_raw = state.get("name")
        name = name_raw.strip() if isinstance(name_raw, str) else ""
        if not name:
            error = append_tool_error(error, "Missing tool name")

        arguments: dict[str, Any] = {}
        arguments_obj = state.get("arguments_obj")
        if isinstance(arguments_obj, dict):
            arguments = dict(arguments_obj)
        else:
            argument_parts = state.get("arguments_parts")
            argument_text = "".join(argument_parts) if isinstance(argument_parts, list) else ""
            if argument_text:
                try:
                    parsed_arguments = json.loads(argument_text)
                    if isinstance(parsed_arguments, dict):
                        arguments = parsed_arguments
                    else:
                        error = append_tool_error(
                            error, "Tool arguments JSON must decode to object"
                        )
                except json.JSONDecodeError as exc:
                    error = append_tool_error(
                        error,
                        f"Invalid JSON in tool arguments: {exc}. Raw: {argument_text[:100]}",
                    )

        call_id_raw = state.get("id")
        call_id = call_id_raw if isinstance(call_id_raw, str) and call_id_raw else new_id()
        parsed.append(ToolCall(id=call_id, name=name, arguments=arguments, error=error))
    return parsed
