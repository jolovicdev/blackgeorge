import json
from collections.abc import Iterable
from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

from blackgeorge.core.job import Job
from blackgeorge.tools.base import Tool, ToolResult
from blackgeorge.tools.registry import Toolbelt
from blackgeorge.utils import new_id
from blackgeorge.worker import Worker

if TYPE_CHECKING:
    from blackgeorge.desk import Desk


class SpawnSubworkerInput(BaseModel):
    name: str = Field(..., description="Name of the subworker.")
    instructions: str = Field(..., description="Instructions for the subworker.")
    task: str = Field(..., description="Task input for the subworker.")
    tools: list[str] = Field(
        default_factory=list,
        description="Names of tools the subworker is allowed to use.",
    )
    model: str | None = Field(
        default=None,
        description="Model override for the subworker. Defaults to configured default.",
    )


def _join_errors(errors: list[str], fallback: str) -> str:
    if not errors:
        return fallback
    text = "; ".join(error for error in errors if error)
    return text or fallback


def _child_desk(
    desk: "Desk",
    model: str,
    *,
    structured_output_retries: int,
    max_iterations: int,
    max_tool_calls: int,
) -> "Desk":
    from blackgeorge.desk import Desk

    return Desk(
        model=model,
        temperature=desk.temperature,
        max_tokens=desk.max_tokens,
        stream=False,
        structured_stream_mode=desk.structured_stream_mode,
        structured_output_retries=structured_output_retries,
        max_iterations=max_iterations,
        max_tool_calls=max_tool_calls,
        respect_context_window=desk.respect_context_window,
        event_bus=desk.event_bus,
        run_store=desk.run_store,
        memory_store=desk.memory_store,
        adapter=desk.adapter,
        storage_dir=desk.storage_dir,
    )


def _mark_child_run_failed(
    desk: "Desk",
    run_id: str,
    job: Job,
    error: str,
) -> None:
    record = desk.run_store.get_run(run_id)
    if record is None:
        desk.run_store.create_run(run_id, job.model_dump(mode="json"))
        record = desk.run_store.get_run(run_id)
    if record is not None and record.status != "running":
        return
    desk.run_store.update_run(
        run_id,
        "failed",
        None,
        {"error": error},
        None,
    )


def create_subworker_tool(
    *,
    desk: "Desk",
    available_tools: list[Tool] | None = None,
    default_model: str | None = None,
    allowed_models: Iterable[str] | None = None,
    max_subworkers: int = 5,
    max_tools_per_subworker: int = 8,
    max_iterations: int = 10,
    max_tool_calls: int = 10,
    structured_output_retries: int = 1,
    tool_name: str = "spawn_subworker",
) -> Tool:
    if max_subworkers < 1:
        raise ValueError("max_subworkers must be >= 1")
    if max_tools_per_subworker < 0:
        raise ValueError("max_tools_per_subworker must be >= 0")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    if max_tool_calls < 1:
        raise ValueError("max_tool_calls must be >= 1")

    allowed_model_set = set(allowed_models) if allowed_models is not None else None
    toolbelt = Toolbelt(available_tools or [])
    spawn_count = 0

    async def spawn_subworker(
        name: str,
        instructions: str,
        task: str,
        tools: list[str] | None = None,
        model: str | None = None,
    ) -> ToolResult:
        nonlocal spawn_count

        if spawn_count >= max_subworkers:
            return ToolResult(
                error=f"Subworker budget exceeded: max_subworkers={max_subworkers}",
            )

        resolved_model = model or default_model or desk.model
        if not resolved_model:
            return ToolResult(error="Subworker model is not configured")

        if allowed_model_set is not None and resolved_model not in allowed_model_set:
            return ToolResult(error=f"Model '{resolved_model}' is not allowed for subworkers")

        requested_tools = tools or []
        if len(requested_tools) > max_tools_per_subworker:
            return ToolResult(
                error=(
                    "Too many tools requested for subworker: "
                    f"{len(requested_tools)} > {max_tools_per_subworker}"
                )
            )

        worker_tools: list[Tool] = []
        missing_tools: list[str] = []
        for tool_name_value in requested_tools:
            resolved_tool = toolbelt.resolve(tool_name_value)
            if resolved_tool is None:
                missing_tools.append(tool_name_value)
            else:
                worker_tools.append(resolved_tool)
        if missing_tools:
            missing = ", ".join(missing_tools)
            return ToolResult(error=f"Requested tools are not available: {missing}")

        spawn_count += 1
        subworker = Worker(
            name=name,
            instructions=instructions,
            model=resolved_model,
            tools=worker_tools,
        )
        child_runner = _child_desk(
            desk,
            resolved_model,
            structured_output_retries=structured_output_retries,
            max_iterations=max_iterations,
            max_tool_calls=max_tool_calls,
        )
        child_job = Job(input=task)
        child_run_id = new_id()

        try:
            report = await child_runner.arun(
                subworker,
                child_job,
                stream=False,
                run_id=child_run_id,
            )
        except Exception as exc:
            error_message = str(exc)
            _mark_child_run_failed(
                desk=child_runner,
                run_id=child_run_id,
                job=child_job,
                error=error_message,
            )
            return ToolResult(
                error=f"Subworker execution failed: {error_message}",
                data={
                    "run_id": child_run_id,
                    "status": "failed",
                    "worker": name,
                    "errors": [error_message],
                },
            )

        result_data: dict[str, object] = {
            "run_id": report.run_id,
            "status": report.status,
            "worker": name,
        }
        if report.errors:
            result_data["errors"] = list(report.errors)
        if report.pending_action is not None:
            result_data["pending_action_type"] = report.pending_action.type

        if report.status == "paused":
            if report.pending_action is None:
                pending_type = "unknown"
            else:
                pending_type = report.pending_action.type
            return ToolResult(
                error=f"Subworker paused and requires {pending_type} follow-up",
                data=result_data,
            )

        if report.status == "failed":
            return ToolResult(
                error=_join_errors(report.errors, "Subworker failed"),
                data=result_data,
            )

        content = report.content
        if content is None and report.data is not None:
            content = json.dumps(report.data, ensure_ascii=True, default=str)
        if content is None:
            content = ""

        return ToolResult(content=content, data=result_data)

    return Tool(
        name=tool_name,
        description="Spawn a bounded autonomous subworker for delegated work.",
        schema=SpawnSubworkerInput.model_json_schema(),
        callable=spawn_subworker,
        input_model=SpawnSubworkerInput,
    )


def create_swarm_tool(
    *,
    desk: "Desk",
    available_tools: list[Tool] | None = None,
    default_model: str | None = None,
    allowed_models: Iterable[str] | None = None,
    max_subworkers: int = 5,
    max_tools_per_subworker: int = 8,
    max_iterations: int = 10,
    max_tool_calls: int = 10,
    structured_output_retries: int = 1,
) -> Tool:
    return create_subworker_tool(
        desk=desk,
        available_tools=available_tools,
        default_model=default_model,
        allowed_models=allowed_models,
        max_subworkers=max_subworkers,
        max_tools_per_subworker=max_tools_per_subworker,
        max_iterations=max_iterations,
        max_tool_calls=max_tool_calls,
        structured_output_retries=structured_output_retries,
        tool_name="spawn_agent",
    )
