import asyncio
import atexit
import json
import warnings
from collections.abc import AsyncIterator, Callable, Iterator
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

_litellm_runtime_configured = False


def _close_coroutine(coroutine: Any) -> None:
    close = getattr(coroutine, "close", None)
    if callable(close):
        close()


def _safe_litellm_shutdown() -> None:
    try:
        from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER

        flush = getattr(GLOBAL_LOGGING_WORKER, "_flush_on_exit", None)
        if callable(flush):
            flush()
    except Exception:
        pass
    try:
        from litellm.llms.custom_httpx.async_client_cleanup import close_litellm_async_clients
    except Exception:
        return
    cleanup_coroutine = close_litellm_async_clients()
    try:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(cleanup_coroutine)
            cleanup_coroutine = None
        finally:
            loop.close()
    except Exception:
        pass
    finally:
        if cleanup_coroutine is not None:
            _close_coroutine(cleanup_coroutine)


def _patch_logging_worker_enqueue() -> None:
    try:
        from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
    except Exception:
        return
    worker = cast(Any, GLOBAL_LOGGING_WORKER)
    if getattr(worker, "_blackgeorge_enqueue_patch", False):
        return
    original_ensure_initialized_and_enqueue = worker.ensure_initialized_and_enqueue

    def ensure_initialized_and_enqueue(async_coroutine: Any) -> None:
        original_ensure_initialized_and_enqueue(async_coroutine)
        if getattr(worker, "_queue", None) is None:
            _close_coroutine(async_coroutine)

    worker.ensure_initialized_and_enqueue = ensure_initialized_and_enqueue
    worker._blackgeorge_enqueue_patch = True


def _configure_litellm_runtime() -> None:
    global _litellm_runtime_configured
    if _litellm_runtime_configured:
        return
    litellm.disable_streaming_logging = True
    if hasattr(litellm, "_async_client_cleanup_registered"):
        litellm._async_client_cleanup_registered = True
    _patch_logging_worker_enqueue()
    atexit.register(_safe_litellm_shutdown)
    _litellm_runtime_configured = True


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


def _is_json_schema_unavailable_error(error: Exception) -> bool:
    error_str = str(error).lower()
    return "response_format type is unavailable" in error_str


def _build_json_object_prompt(response_schema: Any) -> str:
    schema = _build_json_schema(response_schema)
    if schema is None:
        return ""
    import json

    return f"Respond with valid JSON matching this schema: {json.dumps(schema, indent=2)}"


def _try_json_object_fallback(
    model: str,
    payload: list[dict[str, Any]],
    response_schema: Any,
) -> Any | None:
    schema_prompt = _build_json_object_prompt(response_schema)
    if not schema_prompt:
        return None

    augmented_payload = list(payload)
    augmented_payload.append({"role": "user", "content": schema_prompt})

    try:
        response = litellm.completion(
            model=model,
            messages=augmented_payload,
            response_format={"type": "json_object"},
        )
        content = _response_content(response)
        if content:
            return _parse_structured_json(response_schema, content)
    except Exception:
        pass
    return None


async def _atry_json_object_fallback(
    model: str,
    payload: list[dict[str, Any]],
    response_schema: Any,
) -> Any | None:
    schema_prompt = _build_json_object_prompt(response_schema)
    if not schema_prompt:
        return None

    augmented_payload = list(payload)
    augmented_payload.append({"role": "user", "content": schema_prompt})

    try:
        response = await litellm.acompletion(
            model=model,
            messages=augmented_payload,
            response_format={"type": "json_object"},
        )
        content = _response_content(response)
        if content:
            return _parse_structured_json(response_schema, content)
    except Exception:
        pass
    return None


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
    if not isinstance(tool_calls, list):
        return []
    parsed: list[ToolCall] = []

    for call in tool_calls:
        call_like = isinstance(call, dict) or hasattr(call, "function") or hasattr(call, "id")
        if not call_like:
            continue
        function = _get(call, "function", {})
        function_like = (
            isinstance(function, dict)
            or hasattr(function, "name")
            or hasattr(function, "arguments")
        )
        if not function_like:
            function = {}
        name_raw = _get(function, "name")
        name = name_raw.strip() if isinstance(name_raw, str) else ""
        arguments_raw = _get(function, "arguments")
        arguments: dict[str, Any] = {}
        error: str | None = None
        if not name:
            error = "Missing tool name"

        if isinstance(arguments_raw, dict):
            arguments = arguments_raw
        elif isinstance(arguments_raw, str):
            if arguments_raw:
                try:
                    arguments = json.loads(arguments_raw)
                except json.JSONDecodeError as e:
                    message_text = (
                        f"Invalid JSON in tool arguments: {e}. Raw: {arguments_raw[:100]}"
                    )
                    error = f"{error}; {message_text}" if error else message_text
                    arguments = {}
        elif arguments_raw is not None:
            message_text = f"Unsupported tool arguments type: {type(arguments_raw).__name__}"
            error = f"{error}; {message_text}" if error else message_text

        call_id_raw = _get(call, "id")
        call_id = call_id_raw if isinstance(call_id_raw, str) and call_id_raw else new_id()
        parsed.append(ToolCall(id=call_id, name=name, arguments=arguments, error=error))

    return parsed


def _supports_parallel_function_calling(model: str) -> bool:
    model_cost = getattr(litellm, "model_cost", {})
    if not isinstance(model_cost, dict):
        return False
    candidates = [model]
    model_parts = model.split("/")
    if len(model_parts) > 1:
        for index in range(1, len(model_parts)):
            candidates.append("/".join(model_parts[index:]))
    for candidate in candidates:
        info = model_cost.get(candidate)
        if not isinstance(info, dict):
            continue
        supported = info.get("supports_parallel_function_calling")
        if isinstance(supported, bool):
            return supported
    return False


def _parse_response(response: Any) -> ModelResponse:
    choices = _get(response, "choices", [])
    message = _get(choices[0], "message") if choices else None
    content = _get(message, "content") if message else None
    reasoning_content = _get(message, "reasoning_content") if message else None
    thinking_blocks = _get(message, "thinking_blocks") if message else None
    tool_calls = _parse_tool_calls(message) if message else []
    usage = _get(response, "usage", {}) or {}
    if isinstance(usage, BaseModel):
        usage = usage.model_dump(mode="json", warnings=False)
    return ModelResponse(
        content=content,
        reasoning_content=reasoning_content,
        thinking_blocks=thinking_blocks,
        tool_calls=tool_calls,
        usage=usage,
        raw=response,
    )


def _build_litellm_params(
    *,
    model: str,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]] | None,
    tool_choice: str | dict[str, Any] | None,
    temperature: float | None,
    max_tokens: int | None,
    stream: bool,
    stream_options: dict[str, Any] | None,
    thinking: dict[str, Any] | None,
    drop_params: bool | None,
    extra_body: dict[str, Any] | None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    if tools:
        params["tools"] = tools
        if tool_choice is not None:
            params["tool_choice"] = tool_choice
    if temperature is not None:
        params["temperature"] = temperature
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    if stream and stream_options is not None:
        params["stream_options"] = stream_options
    if thinking is not None:
        params["thinking"] = thinking
    if drop_params is not None:
        params["drop_params"] = drop_params
    if extra_body is not None:
        params["extra_body"] = extra_body
    if tools and _supports_parallel_function_calling(model):
        params["parallel_tool_calls"] = True
    return params


def _stream_usage(chunk: Any) -> dict[str, Any] | None:
    usage = _get(chunk, "usage")
    if isinstance(usage, dict):
        return usage
    if isinstance(usage, BaseModel):
        return usage.model_dump(mode="json", warnings=False)
    return None


class _SyncStreamWithEvents:
    def __init__(self, model: str, stream: Iterator[Any]) -> None:
        self._model = model
        self._stream = stream
        self._usage: dict[str, Any] = {}
        self._closed = False

    def __iter__(self) -> Iterator[Any]:
        try:
            for chunk in self._stream:
                usage = _stream_usage(chunk)
                if usage:
                    self._usage = usage
                yield chunk
        except Exception as exc:
            self._emit_failed(exc)
            raise
        self._emit_completed()

    def close(self) -> None:
        if self._closed:
            return
        try:
            close_fn = getattr(self._stream, "close", None)
            if callable(close_fn):
                close_fn()
        except Exception as exc:
            self._emit_failed(exc)
            raise
        self._emit_completed()

    def _emit_completed(self) -> None:
        if self._closed:
            return
        self._closed = True
        emit_llm_completed(self._model, {"usage": self._usage})

    def _emit_failed(self, exc: Exception) -> None:
        if self._closed:
            return
        self._closed = True
        emit_llm_failed(self._model, exc)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class _AsyncStreamWithEvents:
    def __init__(self, model: str, stream: AsyncIterator[Any]) -> None:
        self._model = model
        self._stream = stream
        self._iterator = stream.__aiter__()
        self._usage: dict[str, Any] = {}
        self._closed = False

    def __aiter__(self) -> "_AsyncStreamWithEvents":
        return self

    async def __anext__(self) -> Any:
        if self._closed:
            raise StopAsyncIteration
        try:
            chunk = await self._iterator.__anext__()
        except StopAsyncIteration:
            self._emit_completed()
            raise
        except Exception as exc:
            self._emit_failed(exc)
            raise
        usage = _stream_usage(chunk)
        if usage:
            self._usage = usage
        return chunk

    async def aclose(self) -> None:
        if self._closed:
            return
        try:
            aclose_fn = getattr(self._stream, "aclose", None)
            if callable(aclose_fn):
                await aclose_fn()
            else:
                close_fn = getattr(self._stream, "close", None)
                if callable(close_fn):
                    close_fn()
        except Exception as exc:
            self._emit_failed(exc)
            raise
        self._emit_completed()

    def _emit_completed(self) -> None:
        if self._closed:
            return
        self._closed = True
        emit_llm_completed(self._model, {"usage": self._usage})

    def _emit_failed(self, exc: Exception) -> None:
        if self._closed:
            return
        self._closed = True
        emit_llm_failed(self._model, exc)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


def _wrap_stream_with_events(model: str, response: Any, *, prefer_async: bool) -> Any:
    if prefer_async and hasattr(response, "__aiter__"):
        return _AsyncStreamWithEvents(model, cast(AsyncIterator[Any], response))
    if hasattr(response, "__iter__"):
        return _SyncStreamWithEvents(model, cast(Iterator[Any], response))
    if hasattr(response, "__aiter__"):
        return _AsyncStreamWithEvents(model, cast(AsyncIterator[Any], response))
    return response


class LiteLLMAdapter(BaseModelAdapter):
    def __init__(self) -> None:
        warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
        _configure_litellm_runtime()

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
        litellm_params = _build_litellm_params(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            stream_options=stream_options,
            thinking=thinking,
            drop_params=drop_params,
            extra_body=extra_body,
        )

        emit_llm_started(model, len(messages), len(tools) if tools else 0)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
                response = litellm.completion(**litellm_params)
            if stream:
                return cast(
                    list[dict[str, Any]],
                    _wrap_stream_with_events(model, response, prefer_async=False),
                )
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
        litellm_params = _build_litellm_params(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            stream_options=stream_options,
            thinking=thinking,
            drop_params=drop_params,
            extra_body=extra_body,
        )

        emit_llm_started(model, len(messages), len(tools) if tools else 0)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
                response = await litellm.acompletion(**litellm_params)
            if stream:
                return _wrap_stream_with_events(model, response, prefer_async=True)
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
        json_schema_failed = False

        if response_format is not None:
            try:
                response = litellm.completion(
                    model=model,
                    messages=payload,
                    response_format=response_format,
                )
                content = _response_content(response)
                if content:
                    try:
                        return _parse_structured_json(response_schema, content)
                    except Exception:
                        json_schema_failed = True
            except Exception as e:
                if _is_json_schema_unavailable_error(e):
                    json_schema_failed = True

        if json_schema_failed:
            result = _try_json_object_fallback(model, payload, response_schema)
            if result is not None:
                return result

        retries = max(retries, 3)
        if not _is_base_model_schema(response_schema):
            return _structured_json_retry(
                model=model,
                payload=payload,
                response_schema=response_schema,
                retries=retries,
                response_format=None,
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
        json_schema_failed = False

        if response_format is not None:
            try:
                response = await litellm.acompletion(
                    model=model,
                    messages=payload,
                    response_format=response_format,
                )
                content = _response_content(response)
                if content:
                    try:
                        return _parse_structured_json(response_schema, content)
                    except Exception:
                        json_schema_failed = True
            except Exception as e:
                if _is_json_schema_unavailable_error(e):
                    json_schema_failed = True

        if json_schema_failed:
            result = await _atry_json_object_fallback(model, payload, response_schema)
            if result is not None:
                return result

        retries = max(retries, 3)
        if not _is_base_model_schema(response_schema):
            return await _astructured_json_retry(
                model=model,
                payload=payload,
                response_schema=response_schema,
                retries=retries,
                response_format=None,
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
