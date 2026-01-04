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
