from typing import Any

from blackgeorge.adapters.base import BaseModelAdapter, ModelResponse
from blackgeorge.core.message import Message
from blackgeorge.core.tool_call import ToolCall
from blackgeorge.worker_context import (
    apply_context_summary,
    litellm_model_registered,
    message_summary_text,
)


class SummaryAdapter(BaseModelAdapter):
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
    ) -> ModelResponse:
        return ModelResponse(content="summary", tool_calls=[], usage={}, raw={})

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
    ) -> ModelResponse:
        return ModelResponse(content="summary", tool_calls=[], usage={}, raw={})


def test_message_summary_text_includes_input_text_blocks() -> None:
    message = Message(
        role="user",
        content=[
            {"type": "input_text", "text": "keep this instruction"},
            {"type": "input_image", "image_url": {"url": "https://example.com/image.png"}},
        ],
    )

    summary = message_summary_text(message)
    assert summary == "user: keep this instruction"


def test_message_summary_text_keeps_multimodal_marker_without_text_blocks() -> None:
    message = Message(
        role="user",
        content=[
            {"type": "input_image", "image_url": {"url": "https://example.com/image.png"}},
        ],
    )

    summary = message_summary_text(message)
    assert summary == "user: [multimodal message]"


def test_context_summary_does_not_orphan_tool_messages() -> None:
    messages = [
        Message(role="user", content="old request"),
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="call-1", name="lookup", arguments={})],
        ),
        Message(role="tool", content="result 1", tool_call_id="call-1"),
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="call-2", name="lookup", arguments={})],
        ),
        Message(role="tool", content="result 2", tool_call_id="call-2"),
        Message(role="user", content="current request"),
    ]

    summarized = apply_context_summary(
        adapter=SummaryAdapter(),
        model_name="fake",
        messages=messages,
        temperature=None,
        metrics={},
        emit=lambda *_args: None,
        worker_name="Worker",
        model_registered=True,
    )

    assert summarized is True
    seen_tool_call_ids: set[str] = set()
    for message in messages:
        for call in message.tool_calls:
            seen_tool_call_ids.add(call.id)
        if message.role == "tool":
            assert message.tool_call_id in seen_tool_call_ids


def test_litellm_model_registered_accepts_provider_prefix(monkeypatch) -> None:
    import litellm

    monkeypatch.setattr(litellm, "model_cost", {"gpt-5": {}})

    assert litellm_model_registered("openai/gpt-5") is True
