import asyncio
import atexit
import json
import warnings
from collections.abc import AsyncIterator, Callable, Generator, Iterator
from dataclasses import dataclass, field
from typing import Any, cast

import litellm
from pydantic import BaseModel, TypeAdapter

from blackgeorge.adapters.base import BaseModelAdapter, ModelResponse, StructuredResponse
from blackgeorge.adapters.instructor_client import instructor_clients
from blackgeorge.adapters.litellm_callbacks import (
    callback_context,
    emit_llm_completed,
    emit_llm_failed,
    emit_llm_started,
)
from blackgeorge.core.tool_arguments import parse_tool_arguments
from blackgeorge.core.tool_call import ToolCall
from blackgeorge.core.usage import add_token_usage
from blackgeorge.utils import new_id

_litellm_runtime_configured = False


def _close_coroutine(coroutine: Any) -> None:
    close = getattr(coroutine, "close", None)
    if callable(close):
        close()


def _drain_logging_queue(worker: Any) -> None:
    queue = getattr(worker, "_queue", None)
    if queue is None:
        return
    while not queue.empty():
        try:
            task = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        if isinstance(task, dict):
            _close_coroutine(task.get("coroutine"))


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
        try:
            current_loop = asyncio.get_running_loop()
        except RuntimeError:
            _close_coroutine(async_coroutine)
            return
        bound_loop = getattr(worker, "_bound_loop", None)
        if getattr(worker, "_queue", None) is not None and bound_loop is not current_loop:
            _drain_logging_queue(worker)
        original_ensure_initialized_and_enqueue(async_coroutine)
        if getattr(worker, "_queue", None) is None:
            _close_coroutine(async_coroutine)

    worker.ensure_initialized_and_enqueue = ensure_initialized_and_enqueue
    worker._blackgeorge_enqueue_patch = True


def _configure_litellm_runtime() -> None:
    global _litellm_runtime_configured
    if _litellm_runtime_configured:
        return
    litellm.suppress_debug_info = True
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


def _response_usage(response: Any) -> dict[str, Any]:
    usage = _get(response, "usage", {}) or {}
    if isinstance(usage, BaseModel):
        usage = usage.model_dump(mode="json", warnings=False)
    return usage if isinstance(usage, dict) else {}


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
    return any(
        marker in error_str
        for marker in (
            "response_format type is unavailable",
            "json_schema is not supported",
            "json schema is not supported",
            "does not support json_schema",
        )
    )


def _is_response_format_unsupported_error(error: Exception) -> bool:
    error_str = str(error).lower()
    return _is_json_schema_unavailable_error(error) or any(
        marker in error_str
        for marker in (
            "response_format unsupported",
            "unsupported response_format",
            "does not support response_format",
        )
    )


def _build_json_object_prompt(response_schema: Any) -> str:
    schema = _build_json_schema(response_schema)
    if schema is None:
        return ""
    return f"Respond with valid JSON matching this schema: {json.dumps(schema, indent=2)}"


type SchemaAttempt = tuple[bool, Any | None]


def _schema_attempt_response(response: Any, response_schema: Any) -> SchemaAttempt:
    content = _response_content(response)
    if not content:
        return False, None
    try:
        return False, _parse_structured_json(response_schema, content)
    except Exception:
        return True, None


def _schema_attempt_error(exc: Exception) -> SchemaAttempt | None:
    if _is_json_schema_unavailable_error(exc):
        return True, None
    if _is_response_format_unsupported_error(exc):
        return False, None
    return None


@dataclass(frozen=True)
class _StructuredCall:
    messages: list[dict[str, Any]]
    response_format: dict[str, Any] | None = None
    instructor_model: type[BaseModel] | None = None


type _StructuredPipeline = Generator[_StructuredCall, Any, Any]


@dataclass
class _StructuredRequest:
    model: str
    options: dict[str, Any]
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def instructor_options(self) -> dict[str, Any]:
        return {key: value for key, value in self.options.items() if key != "thinking"}

    def _params(self, call: _StructuredCall) -> dict[str, Any]:
        params: dict[str, Any] = {"model": self.model, "messages": call.messages, **self.options}
        if call.response_format is not None:
            params["response_format"] = call.response_format
        return params

    def _record(self, response: Any) -> None:
        add_token_usage(self.usage, _response_usage(response))
        emit_llm_completed(self.model, response)

    def complete(self, call: _StructuredCall) -> Any:
        emit_llm_started(self.model, len(call.messages), 0)
        try:
            if call.instructor_model is None:
                result = response = litellm.completion(**self._params(call))
            else:
                client = instructor_clients.get(self.model, async_client=False)
                result, response = client.chat.completions.create_with_completion(
                    model=self.model,
                    messages=call.messages,
                    response_model=call.instructor_model,
                    **self.instructor_options,
                )
        except Exception as exc:
            emit_llm_failed(self.model, exc)
            raise
        self._record(response)
        return result

    async def acomplete(self, call: _StructuredCall) -> Any:
        emit_llm_started(self.model, len(call.messages), 0)
        try:
            if call.instructor_model is None:
                result = response = await litellm.acompletion(**self._params(call))
            else:
                client = instructor_clients.get(self.model, async_client=True)
                result, response = await client.chat.completions.create_with_completion(
                    model=self.model,
                    messages=call.messages,
                    response_model=call.instructor_model,
                    **self.instructor_options,
                )
        except Exception as exc:
            emit_llm_failed(self.model, exc)
            raise
        self._record(response)
        return result

    def run(self, pipeline: _StructuredPipeline) -> Any:
        outcome: Any = None
        try:
            while True:
                call = pipeline.send(outcome)
                try:
                    outcome = self.complete(call)
                except Exception as exc:
                    outcome = exc
        except StopIteration as stop:
            return stop.value

    async def arun(self, pipeline: _StructuredPipeline) -> Any:
        outcome: Any = None
        try:
            while True:
                call = pipeline.send(outcome)
                try:
                    outcome = await self.acomplete(call)
                except Exception as exc:
                    outcome = exc
        except StopIteration as stop:
            return stop.value


def _structured_request(
    model: str,
    *,
    temperature: float | None,
    max_tokens: int | None,
    thinking: dict[str, Any] | None,
    drop_params: bool | None,
    extra_body: dict[str, Any] | None,
    num_retries: int | None,
) -> _StructuredRequest:
    candidates = {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "thinking": thinking,
        "drop_params": drop_params,
        "extra_body": extra_body,
        "num_retries": num_retries,
    }
    return _StructuredRequest(
        model, {key: value for key, value in candidates.items() if value is not None}
    )


def _parse_completion_response(response: Any, response_schema: Any) -> Any:
    content = _response_content(response)
    if not content:
        raise ValueError("Empty response content")
    return _parse_structured_json(response_schema, content)


def _json_schema_attempt(outcome: Any, response_schema: Any) -> SchemaAttempt:
    if not isinstance(outcome, Exception):
        return _schema_attempt_response(outcome, response_schema)
    attempt = _schema_attempt_error(outcome)
    if attempt is None:
        raise outcome
    return attempt


def _json_object_attempt(outcome: Any, response_schema: Any) -> Any | None:
    if isinstance(outcome, Exception):
        if _is_response_format_unsupported_error(outcome):
            return None
        raise outcome
    content = _response_content(outcome)
    if not content:
        return None
    try:
        return _parse_structured_json(response_schema, content)
    except Exception:
        return None


def _structured_pipeline(
    payload: list[dict[str, Any]],
    response_schema: Any,
    retries: int,
) -> _StructuredPipeline:
    response_format = _response_format(response_schema)
    json_schema_failed = False
    if response_format is not None:
        outcome = yield _StructuredCall(payload, response_format)
        json_schema_failed, result = _json_schema_attempt(outcome, response_schema)
        if result is not None:
            return result

    schema_prompt = _build_json_object_prompt(response_schema) if json_schema_failed else ""
    if schema_prompt:
        prompted = [*payload, {"role": "user", "content": schema_prompt}]
        outcome = yield _StructuredCall(prompted, {"type": "json_object"})
        result = _json_object_attempt(outcome, response_schema)
        if result is not None:
            return result

    instructor_model = response_schema if _is_base_model_schema(response_schema) else None
    attempts = 0
    while True:
        outcome = yield _StructuredCall(payload, instructor_model=instructor_model)
        if not isinstance(outcome, Exception):
            if instructor_model is not None:
                return outcome
            try:
                return _parse_completion_response(outcome, response_schema)
            except Exception as exc:
                outcome = exc
        if attempts >= retries:
            raise outcome
        payload.append({"role": "user", "content": _structured_retry_prompt(outcome)})
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
        arguments, error = parse_tool_arguments(arguments_raw)
        if not name:
            error = "Missing tool name" if error is None else f"Missing tool name; {error}"

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
    return ModelResponse(
        content=content,
        reasoning_content=reasoning_content,
        thinking_blocks=thinking_blocks,
        tool_calls=tool_calls,
        usage=_response_usage(response),
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
    num_retries: int | None,
    response_schema: Any = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": stream,
    }
    response_format = _response_format(response_schema) if response_schema is not None else None
    if response_format is not None:
        params["response_format"] = response_format
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
    if num_retries is not None:
        params["num_retries"] = num_retries
    if tools and _supports_parallel_function_calling(model):
        params["parallel_tool_calls"] = True
    return params


def _response_format_fallback_params(
    params: dict[str, Any], response_schema: Any, exc: Exception
) -> dict[str, Any] | None:
    if response_schema is None or "response_format" not in params:
        return None
    if _is_json_schema_unavailable_error(exc):
        schema_prompt = _build_json_object_prompt(response_schema)
        if schema_prompt:
            return {
                **params,
                "messages": [*params["messages"], {"role": "user", "content": schema_prompt}],
                "response_format": {"type": "json_object"},
            }
    if _is_response_format_unsupported_error(exc):
        return {key: value for key, value in params.items() if key != "response_format"}
    return None


def _stream_usage(chunk: Any) -> dict[str, Any] | None:
    usage = _get(chunk, "usage")
    if isinstance(usage, dict):
        return usage
    if isinstance(usage, BaseModel):
        return usage.model_dump(mode="json", warnings=False)
    return None


class _StreamEvents:
    def __init__(self, model: str) -> None:
        self._model = model
        self._usage: dict[str, Any] = {}
        self._finished = False

    def _emit_completed(self) -> None:
        if self._finished:
            return
        self._finished = True
        emit_llm_completed(self._model, {"usage": self._usage})

    def _emit_failed(self, exc: Exception) -> None:
        if self._finished:
            return
        self._finished = True
        emit_llm_failed(self._model, exc)


class _SyncStreamWithEvents(_StreamEvents):
    def __init__(self, model: str, stream: Iterator[Any]) -> None:
        super().__init__(model)
        self._stream = stream
        self._resource_closed = False

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
        if self._resource_closed:
            return
        self._resource_closed = True
        try:
            close_fn = getattr(self._stream, "close", None)
            if callable(close_fn):
                close_fn()
        except Exception as exc:
            self._emit_failed(exc)
            raise
        self._emit_completed()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._stream, name)


class _AsyncStreamWithEvents(_StreamEvents):
    def __init__(self, model: str, stream: AsyncIterator[Any]) -> None:
        super().__init__(model)
        self._stream = stream
        self._iterator = stream.__aiter__()
        self._resource_closed = False

    def __aiter__(self) -> "_AsyncStreamWithEvents":
        return self

    async def __anext__(self) -> Any:
        if self._finished:
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
        if self._resource_closed:
            return
        self._resource_closed = True
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
        _configure_litellm_runtime()

    def set_callback_context(
        self, run_id: str, emit: Callable[[str, str, dict[str, Any]], None]
    ) -> None:
        callback_context.set({"run_id": run_id, "emit": emit})

    def clear_callback_context(self) -> None:
        callback_context.set(None)

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
        num_retries: int | None = None,
        response_schema: Any = None,
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
            num_retries=num_retries,
            response_schema=response_schema,
        )
        emit_llm_started(model, len(messages), len(tools) if tools else 0)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
                try:
                    response = litellm.completion(**litellm_params)
                except Exception as exc:
                    fallback = _response_format_fallback_params(
                        litellm_params, response_schema, exc
                    )
                    if fallback is None:
                        raise
                    response = litellm.completion(**fallback)
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
        num_retries: int | None = None,
        response_schema: Any = None,
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
            num_retries=num_retries,
            response_schema=response_schema,
        )
        emit_llm_started(model, len(messages), len(tools) if tools else 0)
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Pydantic serializer warnings")
                try:
                    response = await litellm.acompletion(**litellm_params)
                except Exception as exc:
                    fallback = _response_format_fallback_params(
                        litellm_params, response_schema, exc
                    )
                    if fallback is None:
                        raise
                    response = await litellm.acompletion(**fallback)
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
        temperature: float | None = None,
        max_tokens: int | None = None,
        thinking: dict[str, Any] | None = None,
        drop_params: bool | None = None,
        extra_body: dict[str, Any] | None = None,
        num_retries: int | None = None,
    ) -> StructuredResponse:
        request = _structured_request(
            model,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=thinking,
            drop_params=drop_params,
            extra_body=extra_body,
            num_retries=num_retries,
        )
        data = request.run(_structured_pipeline(list(messages), response_schema, retries))
        return StructuredResponse(data, request.usage)

    async def astructured_complete(
        self,
        *,
        model: str,
        messages: list[dict[str, Any]],
        response_schema: Any,
        retries: int,
        temperature: float | None = None,
        max_tokens: int | None = None,
        thinking: dict[str, Any] | None = None,
        drop_params: bool | None = None,
        extra_body: dict[str, Any] | None = None,
        num_retries: int | None = None,
    ) -> StructuredResponse:
        request = _structured_request(
            model,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking=thinking,
            drop_params=drop_params,
            extra_body=extra_body,
            num_retries=num_retries,
        )
        data = await request.arun(_structured_pipeline(list(messages), response_schema, retries))
        return StructuredResponse(data, request.usage)
