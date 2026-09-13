from typing import Any

from pydantic import BaseModel, TypeAdapter

from blackgeorge.adapters.base import BaseModelAdapter, ModelResponse, StructuredResponse


class ScriptedAdapter(BaseModelAdapter):
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[dict[str, Any]] = []

    def _next_response(self, kind: str) -> ModelResponse:
        if not self._responses:
            raise RuntimeError(
                f"ScriptedAdapter received a {kind} call but no scripted responses remain"
            )
        return self._responses.pop(0)

    @staticmethod
    def _structured_value(response: ModelResponse, response_schema: Any) -> StructuredResponse:
        content = response.content
        if content is None:
            raise RuntimeError("Scripted structured responses must define content")
        if isinstance(response_schema, TypeAdapter):
            return StructuredResponse(response_schema.validate_json(content), response.usage)
        if isinstance(response_schema, type) and issubclass(response_schema, BaseModel):
            return StructuredResponse(response_schema.model_validate_json(content), response.usage)
        raise TypeError(f"Unsupported response_schema type: {type(response_schema).__name__}")

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
    ) -> ModelResponse:
        self.calls.append(
            {
                "kind": "complete",
                "model": model,
                "messages": messages,
                "tools": tools,
                "stream": stream,
                "response_schema": response_schema,
            }
        )
        return self._next_response("completion")

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
    ) -> ModelResponse:
        self.calls.append(
            {
                "kind": "complete",
                "model": model,
                "messages": messages,
                "tools": tools,
                "stream": stream,
                "response_schema": response_schema,
            }
        )
        return self._next_response("completion")

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
        self.calls.append(
            {
                "kind": "structured",
                "model": model,
                "messages": messages,
                "response_schema": response_schema,
                "thinking": thinking,
                "extra_body": extra_body,
            }
        )
        return self._structured_value(self._next_response("structured completion"), response_schema)

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
        self.calls.append(
            {
                "kind": "structured",
                "model": model,
                "messages": messages,
                "response_schema": response_schema,
                "thinking": thinking,
                "extra_body": extra_body,
            }
        )
        return self._structured_value(self._next_response("structured completion"), response_schema)
