import json
from typing import Any, cast

import litellm

from blackgeorge.adapters.base import BaseModelAdapter, ModelResponse
from blackgeorge.core.tool_call import ToolCall
from blackgeorge.utils import new_id


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _parse_tool_calls(message: Any) -> list[ToolCall]:
    tool_calls = _get(message, "tool_calls", []) or []
    parsed: list[ToolCall] = []

    for call in tool_calls:
        function = _get(call, "function", {})
        name = _get(function, "name")
        arguments_raw = _get(function, "arguments")
        arguments: dict[str, Any] = {}
        error: str | None = None

        if isinstance(arguments_raw, str) and arguments_raw:
            try:
                arguments = json.loads(arguments_raw)
            except json.JSONDecodeError as e:
                error = f"Invalid JSON in tool arguments: {e}. Raw: {arguments_raw[:100]}"
                arguments = {}

        call_id = _get(call, "id") or new_id()
        parsed.append(ToolCall(id=call_id, name=name, arguments=arguments, error=error))

    return parsed


def _parse_response(response: Any) -> ModelResponse:
    choices = _get(response, "choices", [])
    message = _get(choices[0], "message") if choices else None
    content = _get(message, "content") if message else None
    tool_calls = _parse_tool_calls(message) if message else []
    usage = _get(response, "usage", {}) or {}
    return ModelResponse(content=content, tool_calls=tool_calls, usage=usage, raw=response)


class LiteLLMAdapter(BaseModelAdapter):
    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
        stream_options: dict[str, Any] | None,
    ) -> ModelResponse | list[dict[str, Any]]:
        stream_options = stream_options if stream else None
        response = litellm.completion(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            stream_options=stream_options,
        )
        if stream:
            return cast(list[dict[str, Any]], response)
        return _parse_response(response)

    async def acomplete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        tool_choice: str | dict[str, Any] | None,
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
        stream_options: dict[str, Any] | None,
    ) -> ModelResponse | Any:
        stream_options = stream_options if stream else None
        response = await litellm.acompletion(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            stream_options=stream_options,
        )
        if stream:
            return response
        return _parse_response(response)
