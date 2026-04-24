from typing import Any

import pytest

from blackgeorge.adapters.base import BaseModelAdapter, ModelResponse
from blackgeorge.core.tool_call import ToolCall
from blackgeorge.desk import Desk
from blackgeorge.store.in_memory import InMemoryRunStore
from blackgeorge.tools import tool
from blackgeorge.tools.execution import aexecute_tool
from blackgeorge.tools.swarm import create_subworker_tool
from tests.utils import FakeAdapter


class ContextLimitFailingAdapter(BaseModelAdapter):
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
        raise RuntimeError("context length exceeded")

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
        raise RuntimeError("context length exceeded")


class RuntimeFailingAdapter(BaseModelAdapter):
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
        raise RuntimeError("adapter exploded")

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
        raise RuntimeError("adapter exploded")


class FailThenSucceedAdapter(BaseModelAdapter):
    def __init__(self) -> None:
        self.calls = 0

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
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("adapter exploded once")
        return ModelResponse(content="child ok", tool_calls=[], usage={}, raw={})

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
        return self.complete(
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


@pytest.mark.asyncio
async def test_subworker_runs_through_desk_and_records_child_run() -> None:
    desk = Desk(
        model="fake",
        adapter=FakeAdapter([ModelResponse(content="child ok", tool_calls=[], usage={}, raw={})]),
        run_store=InMemoryRunStore(),
    )
    spawn_tool = create_subworker_tool(desk=desk, default_model="fake")
    call = ToolCall(
        id="spawn-1",
        name=spawn_tool.name,
        arguments={
            "name": "ChildWorker",
            "instructions": "Return a short answer.",
            "task": "say ok",
        },
    )

    result = await aexecute_tool(spawn_tool, call)

    assert result.error is None
    assert result.content == "child ok"
    assert isinstance(result.data, dict)
    assert result.data.get("status") == "completed"
    run_id_value = result.data.get("run_id")
    assert isinstance(run_id_value, str)
    record = desk.run_store.get_run(run_id_value)
    assert record is not None
    assert record.status == "completed"


@pytest.mark.asyncio
async def test_subworker_paused_returns_tool_error_with_pending_metadata() -> None:
    @tool(requires_confirmation=True)
    def risky(action: str) -> str:
        return f"ok:{action}"

    desk = Desk(
        model="fake",
        adapter=FakeAdapter(
            [
                ModelResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(id="child-tool-1", name="risky", arguments={"action": "go"})
                    ],
                    usage={},
                    raw={},
                )
            ]
        ),
        run_store=InMemoryRunStore(),
    )
    spawn_tool = create_subworker_tool(
        desk=desk,
        available_tools=[risky],
        default_model="fake",
    )
    call = ToolCall(
        id="spawn-2",
        name=spawn_tool.name,
        arguments={
            "name": "ChildWorker",
            "instructions": "Use the risky tool.",
            "task": "run risky",
            "tools": ["risky"],
        },
    )

    result = await aexecute_tool(spawn_tool, call)

    assert result.error is not None
    assert "paused" in result.error.lower()
    assert isinstance(result.data, dict)
    assert result.data.get("status") == "paused"
    assert result.data.get("pending_action_type") == "confirmation"
    run_id_value = result.data.get("run_id")
    assert isinstance(run_id_value, str)
    record = desk.run_store.get_run(run_id_value)
    assert record is not None
    assert record.status == "paused"


@pytest.mark.asyncio
async def test_subworker_failure_report_is_returned_as_tool_error() -> None:
    desk = Desk(
        model="fake",
        adapter=ContextLimitFailingAdapter(),
        run_store=InMemoryRunStore(),
        respect_context_window=False,
    )
    spawn_tool = create_subworker_tool(desk=desk, default_model="fake")
    call = ToolCall(
        id="spawn-3",
        name=spawn_tool.name,
        arguments={
            "name": "ChildWorker",
            "instructions": "Return output.",
            "task": "run",
        },
    )

    result = await aexecute_tool(spawn_tool, call)

    assert result.error is not None
    assert "context length exceeded" in result.error.lower()
    assert isinstance(result.data, dict)
    assert result.data.get("status") == "failed"


@pytest.mark.asyncio
async def test_subworker_runtime_exception_marks_child_run_failed() -> None:
    desk = Desk(
        model="fake",
        adapter=RuntimeFailingAdapter(),
        run_store=InMemoryRunStore(),
    )
    spawn_tool = create_subworker_tool(desk=desk, default_model="fake")
    call = ToolCall(
        id="spawn-runtime-fail",
        name=spawn_tool.name,
        arguments={
            "name": "ChildWorker",
            "instructions": "Return output.",
            "task": "run",
        },
    )

    result = await aexecute_tool(spawn_tool, call)

    assert result.error is not None
    assert "adapter exploded" in result.error
    assert isinstance(result.data, dict)
    run_id_value = result.data.get("run_id")
    assert isinstance(run_id_value, str)
    assert result.data.get("status") == "failed"
    record = desk.run_store.get_run(run_id_value)
    assert record is not None
    assert record.status == "failed"
    events = desk.run_store.get_events(run_id_value)
    event_types = [event.type for event in events]
    assert event_types[0] == "run.started"
    assert event_types.count("run.failed") == 1
    failed_event = next(event for event in events if event.type == "run.failed")
    assert failed_event.payload.get("errors") == ["adapter exploded"]


@pytest.mark.asyncio
async def test_subworker_rejects_disallowed_model() -> None:
    desk = Desk(
        model="fake",
        adapter=FakeAdapter([ModelResponse(content="child ok", tool_calls=[], usage={}, raw={})]),
        run_store=InMemoryRunStore(),
    )
    spawn_tool = create_subworker_tool(
        desk=desk,
        default_model="fake",
        allowed_models={"fake"},
    )
    call = ToolCall(
        id="spawn-4",
        name=spawn_tool.name,
        arguments={
            "name": "ChildWorker",
            "instructions": "Return output.",
            "task": "run",
            "model": "other-model",
        },
    )

    result = await aexecute_tool(spawn_tool, call)

    assert result.error is not None
    assert "not allowed" in result.error


@pytest.mark.asyncio
async def test_subworker_budget_limit_blocks_additional_spawns() -> None:
    desk = Desk(
        model="fake",
        adapter=FakeAdapter([ModelResponse(content="first", tool_calls=[], usage={}, raw={})]),
        run_store=InMemoryRunStore(),
    )
    spawn_tool = create_subworker_tool(
        desk=desk,
        default_model="fake",
        max_subworkers=1,
    )
    first_call = ToolCall(
        id="spawn-5",
        name=spawn_tool.name,
        arguments={
            "name": "ChildWorkerOne",
            "instructions": "Return output.",
            "task": "run",
        },
    )
    second_call = ToolCall(
        id="spawn-6",
        name=spawn_tool.name,
        arguments={
            "name": "ChildWorkerTwo",
            "instructions": "Return output.",
            "task": "run",
        },
    )

    first_result = await aexecute_tool(spawn_tool, first_call)
    second_result = await aexecute_tool(spawn_tool, second_call)

    assert first_result.error is None
    assert second_result.error is not None
    assert "budget exceeded" in second_result.error


@pytest.mark.asyncio
async def test_subworker_budget_counts_only_completed_runs() -> None:
    adapter = FailThenSucceedAdapter()
    desk = Desk(
        model="fake",
        adapter=adapter,
        run_store=InMemoryRunStore(),
    )
    spawn_tool = create_subworker_tool(
        desk=desk,
        default_model="fake",
        max_subworkers=1,
    )
    first_call = ToolCall(
        id="spawn-7",
        name=spawn_tool.name,
        arguments={
            "name": "ChildWorkerOne",
            "instructions": "Return output.",
            "task": "run",
        },
    )
    second_call = ToolCall(
        id="spawn-8",
        name=spawn_tool.name,
        arguments={
            "name": "ChildWorkerTwo",
            "instructions": "Return output.",
            "task": "run",
        },
    )
    third_call = ToolCall(
        id="spawn-9",
        name=spawn_tool.name,
        arguments={
            "name": "ChildWorkerThree",
            "instructions": "Return output.",
            "task": "run",
        },
    )

    first_result = await aexecute_tool(spawn_tool, first_call)
    second_result = await aexecute_tool(spawn_tool, second_call)
    third_result = await aexecute_tool(spawn_tool, third_call)

    assert first_result.error is not None
    assert "adapter exploded once" in first_result.error
    assert second_result.error is None
    assert second_result.content == "child ok"
    assert third_result.error is not None
    assert "budget exceeded" in third_result.error


@pytest.mark.asyncio
async def test_subworker_paused_runs_do_not_consume_budget() -> None:
    @tool(requires_confirmation=True)
    def risky(action: str) -> str:
        return f"ok:{action}"

    desk = Desk(
        model="fake",
        adapter=FakeAdapter(
            [
                ModelResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(id="child-tool-2", name="risky", arguments={"action": "pause"})
                    ],
                    usage={},
                    raw={},
                ),
                ModelResponse(content="after pause", tool_calls=[], usage={}, raw={}),
            ]
        ),
        run_store=InMemoryRunStore(),
    )
    spawn_tool = create_subworker_tool(
        desk=desk,
        available_tools=[risky],
        default_model="fake",
        max_subworkers=1,
    )
    paused_call = ToolCall(
        id="spawn-10",
        name=spawn_tool.name,
        arguments={
            "name": "ChildWorkerOne",
            "instructions": "Use the risky tool.",
            "task": "run risky",
            "tools": ["risky"],
        },
    )
    completed_call = ToolCall(
        id="spawn-11",
        name=spawn_tool.name,
        arguments={
            "name": "ChildWorkerTwo",
            "instructions": "Return output.",
            "task": "run",
        },
    )
    blocked_call = ToolCall(
        id="spawn-12",
        name=spawn_tool.name,
        arguments={
            "name": "ChildWorkerThree",
            "instructions": "Return output.",
            "task": "run",
        },
    )

    paused_result = await aexecute_tool(spawn_tool, paused_call)
    completed_result = await aexecute_tool(spawn_tool, completed_call)
    blocked_result = await aexecute_tool(spawn_tool, blocked_call)

    assert paused_result.error is not None
    assert "paused" in paused_result.error.lower()
    assert completed_result.error is None
    assert completed_result.content == "after pause"
    assert blocked_result.error is not None
    assert "budget exceeded" in blocked_result.error
