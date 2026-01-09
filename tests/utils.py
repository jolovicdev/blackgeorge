from typing import Any

from blackgeorge.adapters.base import BaseModelAdapter, ModelResponse


class FakeAdapter(BaseModelAdapter):
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)

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
    ) -> ModelResponse:
        if not self._responses:
            return ModelResponse(content="", tool_calls=[], usage={}, raw={})
        return self._responses.pop(0)

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
    ) -> ModelResponse:
        if not self._responses:
            return ModelResponse(content="", tool_calls=[], usage={}, raw={})
        return self._responses.pop(0)


class AsyncOnlyAdapter(BaseModelAdapter):
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)

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
    ) -> ModelResponse:
        raise RuntimeError("Sync completion not allowed")

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
    ) -> ModelResponse:
        if not self._responses:
            return ModelResponse(content="", tool_calls=[], usage={}, raw={})
        return self._responses.pop(0)


class StreamingAdapter(BaseModelAdapter):
    def __init__(self, streams: list[list[dict[str, Any]]]) -> None:
        self._streams = streams
        self._call_index = 0

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
    ) -> ModelResponse | Any:
        if stream and self._call_index < len(self._streams):
            chunks = self._streams[self._call_index]
            self._call_index += 1
            return iter(chunks)
        return ModelResponse(content="", tool_calls=[], usage={}, raw={})

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
        if stream and self._call_index < len(self._streams):
            chunks = self._streams[self._call_index]
            self._call_index += 1
            return iter(chunks)
        return ModelResponse(content="", tool_calls=[], usage={}, raw={})
