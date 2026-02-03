import json
import warnings
from collections.abc import Callable
from typing import Any, cast

import litellm
from pydantic import BaseModel, TypeAdapter

from blackgeorge.adapters.base import BaseModelAdapter, ModelResponse
from blackgeorge.adapters.instructor_client import instructor_clients
from blackgeorge.adapters.litellm_callbacks import (
    _callback_context,
    emit_llm_completed,
    emit_llm_failed,
    emit_llm_started,
)
from blackgeorge.core.tool_call import ToolCall
from blackgeorge.utils import new_id


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _response_content(response: Any) -> str | None:
    choices = _get(response, "choices", [])
    message = _get(choices[0], "message") if choices else None
    return _get(message, "content") if message else None


def _build_json_schema(response_schema: Any) -> dict[str, Any] | None:
    if isinstance(response_schema, TypeAdapter):
        try:
            return response_schema.json_schema()
        except Exception:
            return None
    if isinstance(response_schema, type) and issubclass(response_schema, BaseModel):
        try:
            return response_schema.model_json_schema()
        except Exception:
            return None
    try:
        adapter = TypeAdapter(response_schema)
    except Exception:
        return None
    try:
        return adapter.json_schema()
    except Exception:
        return None


def _response_format(response_schema: Any) -> dict[str, Any] | None:
    schema = _build_json_schema(response_schema)
    if schema is None:
        return None
    name = getattr(response_schema, "__name__", None)
    if name is None:
        inner = getattr(response_schema, "_type", None)
        name = getattr(inner, "__name__", "response_schema")
    return {"type": "json_schema", "json_schema": {"name": name, "schema": schema, "strict": True}}


def _parse_structured_json(response_schema: Any, content: str) -> Any:
    if isinstance(response_schema, TypeAdapter):
        return response_schema.validate_json(content)
    if isinstance(response_schema, type) and issubclass(response_schema, BaseModel):
        return response_schema.model_validate_json(content)
    adapter = TypeAdapter(response_schema)
    return adapter.validate_json(content)


def _structured_retry_prompt(error: Exception) -> str:
    return (
        "Fix validation errors and return only JSON that matches the schema. "
        "Do not output YAML. Do not wrap objects as strings. "
        "If a field is a list, each item must be a JSON object, not a string. "
        f"Validation errors: {error}"
    )


def _is_base_model_schema(response_schema: Any) -> bool:
    return isinstance(response_schema, type) and issubclass(response_schema, BaseModel)


def _structured_json_retry(
    *,
    model: str,
    payload: list[dict[str, Any]],
    response_schema: Any,
    retries: int,
    response_format: dict[str, Any] | None,
) -> Any:
    attempts = 0
    while True:
        try:
            if response_format is None:
                response = litellm.completion(model=model, messages=payload)
            else:
                response = litellm.completion(
                    model=model,
                    messages=payload,
                    response_format=response_format,
                )
            content = _response_content(response)
            if not content:
                raise ValueError("Empty response content")
            return _parse_structured_json(response_schema, content)
        except Exception as exc:
            if attempts >= retries:
                raise
            payload.append({"role": "user", "content": _structured_retry_prompt(exc)})
            attempts += 1


async def _astructured_json_retry(
    *,
    model: str,
    payload: list[dict[str, Any]],
    response_schema: Any,
    retries: int,
    response_format: dict[str, Any] | None,
) -> Any:
    attempts = 0
    while True:
        try:
            if response_format is None:
                response = await litellm.acompletion(model=model, messages=payload)
            else:
                response = await litellm.acompletion(
                    model=model,
                    messages=payload,
                    response_format=response_format,
                )
            content = _response_content(response)
            if not content:
                raise ValueError("Empty response content")
            return _parse_structured_json(response_schema, content)
        except Exception as exc:
            if attempts >= retries:
                raise
            payload.append({"role": "user", "content": _structured_retry_prompt(exc)})
            attempts += 1


def _parse_tool_calls(message: Any) -> list[ToolCall]:
    tool_calls = _get(message, "tool_calls", []) or []
    parsed: list[ToolCall] = []

    for call in tool_calls:
        function = _get(call, "function", {})
        name = _get(function, "name")
        arguments_raw = _get(function, "arguments")
        arguments: dict[str, Any] = {}
        error: str | None = None

        if isinstance(arguments_raw, dict):
            arguments = arguments_raw
        elif isinstance(arguments_raw, str) and arguments_raw:
            try:
                arguments = json.loads(arguments_raw)
            except json.JSONDecodeError as e:
                error = f"Invalid JSON in tool arguments: {e}. Raw: {arguments_raw[:100]}"
                arguments = {}

        call_id = _get(call, "id") or new_id()
        parsed.append(ToolCall(id=call_id, name=name, arguments=arguments, error=error))

    return parsed


def _supports_parallel_function_calling(model: str) -> bool:
    checker = getattr(litellm, "supports_parallel_function_calling", None)
    if checker is None:
        return False
    try:
        return bool(checker(model))
    except TypeError:
        return bool(checker(model=model))


def _parse_response(response: Any) -> ModelResponse:
    choices = _get(response, "choices", [])
    message = _get(choices[0], "message") if choices else None
    content = _get(message, "content") if message else None
    reasoning_content = _get(message, "reasoning_content") if message else None
    tool_calls = _parse_tool_calls(message) if message else []
    usage = _get(response, "usage", {}) or {}
    if isinstance(usage, BaseModel):
        usage = usage.model_dump(mode="json", warnings=False)
    return ModelResponse(
        content=content,
        reasoning_content=reasoning_content,
        tool_calls=tool_calls,
        usage=usage,
        raw=response,
    )


class LiteLLMAdapter(BaseModelAdapter):
    def __init__(self) -> None:
        warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
        warnings.filterwarnings("ignore", message="coroutine.*was never awaited")
        litellm.disable_streaming_logging = True

    def set_callback_context(
        self, run_id: str, emit: Callable[[str, str, dict[str, Any]], None]
    ) -> None:
        _callback_context.set({"run_id": run_id, "emit": emit})

    def clear_callback_context(self) -> None:
        _callback_context.set(None)

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
        thinking: dict[str, Any] | None = None,
        drop_params: bool | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ModelResponse | list[dict[str, Any]]:
        stream_options = stream_options if stream else None
        litellm_params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            "stream_options": stream_options,
        }

        if tools and _supports_parallel_function_calling(model):
            litellm_params["parallel_tool_calls"] = True
        if thinking is not None:
            litellm_params["thinking"] = thinking
        if drop_params is not None:
            litellm_params["drop_params"] = drop_params
        if extra_body is not None:
            litellm_params["extra_body"] = extra_body

        emit_llm_started(model, len(messages), len(tools) if tools else 0)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
                response = litellm.completion(**litellm_params)
            if stream:
                return cast(list[dict[str, Any]], response)
            emit_llm_completed(model, response)
            return _parse_response(response)
        except Exception as exc:
            emit_llm_failed(model, exc)
            raise

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
        thinking: dict[str, Any] | None = None,
        drop_params: bool | None = None,
        extra_body: dict[str, Any] | None = None,
    ) -> ModelResponse | Any:
        stream_options = stream_options if stream else None
        litellm_params: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": tools,
            "tool_choice": tool_choice,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            "stream_options": stream_options,
        }

        if tools and _supports_parallel_function_calling(model):
            litellm_params["parallel_tool_calls"] = True
        if thinking is not None:
            litellm_params["thinking"] = thinking
        if drop_params is not None:
            litellm_params["drop_params"] = drop_params
        if extra_body is not None:
            litellm_params["extra_body"] = extra_body

        emit_llm_started(model, len(messages), len(tools) if tools else 0)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
                response = await litellm.acompletion(**litellm_params)
            if stream:
                return response
            emit_llm_completed(model, response)
            return _parse_response(response)
        except Exception as exc:
            emit_llm_failed(model, exc)
            raise

    def structured_complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        response_schema: Any,
        retries: int,
    ) -> Any:
        payload = list(messages)
        response_format = _response_format(response_schema)
        response_format_accepted = False
        if response_format is not None:
            try:
                response = litellm.completion(
                    model=model,
                    messages=payload,
                    response_format=response_format,
                )
                response_format_accepted = True
                content = _response_content(response)
                if content:
                    return _parse_structured_json(response_schema, content)
            except Exception:
                response_format_accepted = False
        retries = max(retries, 3)
        if not _is_base_model_schema(response_schema):
            manual_response_format = response_format if response_format_accepted else None
            return _structured_json_retry(
                model=model,
                payload=payload,
                response_schema=response_schema,
                retries=retries,
                response_format=manual_response_format,
            )
        client = instructor_clients.get(model, async_client=False)
        attempts = 0
        while True:
            try:
                return client.chat.completions.create(
                    model=model,
                    messages=payload,
                    response_model=response_schema,
                )
            except Exception as exc:
                if attempts >= retries:
                    raise
                payload.append(
                    {
                        "role": "user",
                        "content": _structured_retry_prompt(exc),
                    }
                )
                attempts += 1

    async def astructured_complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        response_schema: Any,
        retries: int,
    ) -> Any:
        payload = list(messages)
        response_format = _response_format(response_schema)
        response_format_accepted = False
        if response_format is not None:
            try:
                response = await litellm.acompletion(
                    model=model,
                    messages=payload,
                    response_format=response_format,
                )
                response_format_accepted = True
                content = _response_content(response)
                if content:
                    return _parse_structured_json(response_schema, content)
            except Exception:
                response_format_accepted = False
        retries = max(retries, 3)
        if not _is_base_model_schema(response_schema):
            manual_response_format = response_format if response_format_accepted else None
            return await _astructured_json_retry(
                model=model,
                payload=payload,
                response_schema=response_schema,
                retries=retries,
                response_format=manual_response_format,
            )
        client = instructor_clients.get(model, async_client=True)
        attempts = 0
        while True:
            try:
                return await client.chat.completions.create(
                    model=model,
                    messages=payload,
                    response_model=response_schema,
                )
            except Exception as exc:
                if attempts >= retries:
                    raise
                payload.append(
                    {
                        "role": "user",
                        "content": _structured_retry_prompt(exc),
                    }
                )
                attempts += 1
