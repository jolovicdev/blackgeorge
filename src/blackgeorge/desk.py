import asyncio
import json
import os
import re
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from blackgeorge.adapters.base import BaseModelAdapter
from blackgeorge.adapters.litellm import LiteLLMAdapter
from blackgeorge.async_utils import ensure_not_running_loop
from blackgeorge.config import RunConfig, validate_execution_limits
from blackgeorge.core.event import Event
from blackgeorge.core.job import Job
from blackgeorge.core.message import Message
from blackgeorge.core.report import Report
from blackgeorge.core.serialization import to_json_value
from blackgeorge.event_bus import EventBus
from blackgeorge.memory.base import MemoryStore
from blackgeorge.memory.in_memory import InMemoryMemoryStore
from blackgeorge.store.base import RunStore
from blackgeorge.store.sqlite import SQLiteRunStore
from blackgeorge.store.state import RunState
from blackgeorge.tools.base import Tool
from blackgeorge.utils import new_id, utc_now
from blackgeorge.worker import Worker
from blackgeorge.workflow.flow import Flow
from blackgeorge.workforce import Workforce

if TYPE_CHECKING:
    from blackgeorge.session import WorkerSession


UNEXPECTED_FAILURE_MESSAGE = "An unexpected error occurred"
SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]+"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/-]+=*"),
    re.compile(r"(?i)((?:api[_-]?key|token|secret|password|authorization)\s*[=:]\s*)\S+"),
)
ABSOLUTE_PATH_PATTERN = re.compile(r"(?<!\w)/(?:[^\s:/]+/)+[^\s:]+")


def _sanitize_exception_message(exc: BaseException) -> str:
    if isinstance(exc, asyncio.CancelledError):
        return "Run cancelled"
    message = str(exc).strip()
    if not message:
        return UNEXPECTED_FAILURE_MESSAGE
    for pattern in SECRET_PATTERNS:
        message = pattern.sub(
            lambda match: f"{match.group(1)}[redacted]" if match.lastindex else "[redacted]",
            message,
        )
    message = ABSOLUTE_PATH_PATTERN.sub("[path]", message)
    return message or UNEXPECTED_FAILURE_MESSAGE


class Desk:
    def __init__(
        self,
        *,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        structured_stream_mode: Literal["off", "preview"] = "off",
        structured_output_retries: int = 3,
        max_iterations: int = 10,
        max_tool_calls: int = 20,
        num_retries: int = 0,
        respect_context_window: bool = True,
        max_context_messages: int | None = None,
        event_bus: EventBus | None = None,
        run_store: RunStore | None = None,
        memory_store: MemoryStore | None = None,
        adapter: BaseModelAdapter | None = None,
        storage_dir: str | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be empty")
        if temperature is not None and temperature < 0:
            raise ValueError("temperature must be non-negative")
        validate_execution_limits(
            max_tokens=max_tokens,
            structured_output_retries=structured_output_retries,
            max_iterations=max_iterations,
            max_tool_calls=max_tool_calls,
            num_retries=num_retries,
            max_context_messages=max_context_messages,
        )
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.stream = stream
        if structured_stream_mode not in ("off", "preview"):
            raise ValueError("structured_stream_mode must be 'off' or 'preview'")
        self.structured_stream_mode = structured_stream_mode
        self.structured_output_retries = structured_output_retries
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.num_retries = num_retries
        self.respect_context_window = respect_context_window
        self.max_context_messages = max_context_messages
        self.event_bus = event_bus or EventBus()
        self.adapter = adapter or LiteLLMAdapter()
        self.storage_dir = storage_dir or ".blackgeorge"
        self.db_path = os.path.join(self.storage_dir, "blackgeorge.db")
        self.run_store: RunStore
        self._owns_run_store = run_store is None
        if run_store is None:
            os.makedirs(self.storage_dir, exist_ok=True)
            self.run_store = SQLiteRunStore(self.db_path)
        else:
            self.run_store = run_store
        self._owns_memory_store = memory_store is None
        self.memory_store = memory_store if memory_store is not None else InMemoryMemoryStore()
        self._workers: dict[str, Worker] = {}
        self._workforces: dict[str, Workforce] = {}
        self._flow_runs: dict[str, Flow] = {}
        self._runtime_tools_overrides: dict[str, list[Tool]] = {}
        self._closed = False

    def register_worker(self, worker: Worker) -> None:
        self._workers[worker.name] = worker

    def register_workforce(self, workforce: Workforce) -> None:
        self._workforces[workforce.name] = workforce

    def unregister_worker(self, worker: Worker | str) -> None:
        name = worker if isinstance(worker, str) else worker.name
        self._workers.pop(name, None)

    def unregister_workforce(self, workforce: Workforce | str) -> None:
        name = workforce if isinstance(workforce, str) else workforce.name
        self._workforces.pop(name, None)

    def register_flow_run(self, run_id: str, flow: Flow) -> None:
        self._flow_runs[run_id] = flow

    def unregister_flow_run(self, run_id: str) -> None:
        self._flow_runs.pop(run_id, None)

    def flow(self, steps: list[Any], name: str | None = None) -> Flow:
        self._ensure_open()
        return Flow(self, steps, name=name)

    def session(
        self,
        worker: Worker,
        *,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        create: bool = True,
    ) -> "WorkerSession | None":
        from blackgeorge.session import WorkerSession

        self._ensure_open()
        os.makedirs(self.storage_dir, exist_ok=True)
        if session_id is not None:
            return WorkerSession.open(
                session_id=session_id,
                worker=worker,
                desk=self,
                metadata=metadata,
                create=create,
            )
        if not create:
            raise ValueError("session_id is required when create is False")
        return WorkerSession.start(
            worker=worker, desk=self, session_id=session_id, metadata=metadata
        )

    @property
    def closed(self) -> bool:
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("Desk is closed")

    def emit(
        self,
        events: list[Event],
        run_id: str,
        event_type: str,
        source: str,
        payload: dict[str, Any],
    ) -> None:
        event = Event(
            event_id=new_id(),
            type=event_type,
            timestamp=utc_now(),
            run_id=run_id,
            source=source,
            payload=payload,
        )
        events.append(event)
        self.event_bus.emit(event)
        self.run_store.add_event(event)

    def _apply_memory(self, worker: Worker, job: Job) -> Job:
        if self.memory_store is None:
            return job
        memory_value = self.memory_store.read("context", worker.memory_scope)
        if memory_value is None:
            return job
        if isinstance(memory_value, str):
            content = memory_value
        else:
            content = json.dumps(memory_value, ensure_ascii=True, default=str)
        memory_message = Message(role="system", content=f"Memory:\n{content}")
        if job.initial_messages:
            messages = list(job.initial_messages)
            insert_index = 0
            while insert_index < len(messages) and messages[insert_index].role == "system":
                insert_index += 1
            messages.insert(insert_index, memory_message)
        else:
            messages = [memory_message]
        return job.model_copy(update={"initial_messages": messages})

    def _resolve_structured_stream_mode(self, job: Job) -> Job:
        if job.structured_stream_mode is not None:
            return job
        return job.model_copy(update={"structured_stream_mode": self.structured_stream_mode})

    def _write_memory(self, worker: Worker, report: Report) -> None:
        if self.memory_store is None:
            return
        if report.status != "completed":
            return
        value: Any | None = report.data if report.data is not None else report.content
        if value is None:
            return
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json", warnings=False)
        self.memory_store.write("last_output", value, worker.memory_scope)

    def _make_run_config(self, run_id: str, events: list[Event], stream_enabled: bool) -> RunConfig:
        stream_options = {"include_usage": True} if stream_enabled else None

        def emit(event_type: str, source: str, payload: dict[str, Any]) -> None:
            self.emit(events, run_id, event_type, source, payload)

        return RunConfig(
            adapter=self.adapter,
            emit=emit,
            run_id=run_id,
            events=events,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            stream=stream_enabled,
            stream_options=stream_options,
            structured_output_retries=self.structured_output_retries,
            max_iterations=self.max_iterations,
            max_tool_calls=self.max_tool_calls,
            num_retries=self.num_retries,
            respect_context_window=self.respect_context_window,
            max_context_messages=self.max_context_messages,
            default_model=self.model,
        )

    def _remember_runtime_tools_override(self, run_id: str, state: RunState) -> None:
        if state.job.tools_override is None:
            self._runtime_tools_overrides.pop(run_id, None)
            return
        runtime_tools = [item for item in state.job.tools_override if isinstance(item, Tool)]
        if runtime_tools:
            self._runtime_tools_overrides[run_id] = runtime_tools

    def _restore_runtime_tools_override(self, run_id: str, state: RunState) -> RunState:
        runtime_tools = self._runtime_tools_overrides.get(run_id)
        if not runtime_tools or state.job.tools_override is None:
            return state
        runtime_by_name = {tool.name: tool for tool in runtime_tools}
        restored_override: list[Any] = []
        for item in state.job.tools_override:
            if isinstance(item, str) and item in runtime_by_name:
                restored_override.append(runtime_by_name.pop(item))
            elif isinstance(item, Tool) and item.name in runtime_by_name:
                runtime_by_name.pop(item.name)
                restored_override.append(item)
            else:
                restored_override.append(item)
        restored_override.extend(runtime_by_name.values())
        job = state.job.model_copy(update={"tools_override": restored_override})
        return state.model_copy(update={"job": job})

    def _finalize_run(
        self,
        runner: Worker | Workforce | None,
        report: Report,
        state: RunState | None,
        run_id: str,
        events: list[Event],
        stream_enabled: bool,
    ) -> Report:
        if state is not None and report.status == "paused":
            state.payload["stream"] = stream_enabled
            self._remember_runtime_tools_override(run_id, state)
            self.emit(events, run_id, "run.paused", "desk", {})
            self.run_store.update_run(run_id, "paused", report.content, None, state)
        elif report.status == "completed":
            self._runtime_tools_overrides.pop(run_id, None)
            self.emit(events, run_id, "run.completed", "desk", {})
            self.run_store.update_run(
                run_id,
                "completed",
                report.content,
                to_json_value(report.data),
                None,
            )
            if isinstance(runner, Worker):
                self._write_memory(runner, report)
        else:
            self._runtime_tools_overrides.pop(run_id, None)
            self.emit(events, run_id, "run.failed", "desk", {"errors": report.errors})
            self.run_store.update_run(
                run_id,
                "failed",
                report.content,
                to_json_value(report.data),
                None,
            )
        return report

    def _fail_resume(self, report: Report, events: list[Event], message: str) -> Report:
        failed = Report(
            run_id=report.run_id,
            status="failed",
            content=None,
            data=None,
            messages=report.messages,
            tool_calls=report.tool_calls,
            metrics=report.metrics,
            events=events,
            pending_action=None,
            errors=[message],
        )
        self.emit(
            events,
            report.run_id,
            "run.failed",
            "desk",
            {"errors": failed.errors},
        )
        self.run_store.update_run(
            report.run_id,
            "failed",
            failed.content,
            to_json_value(failed.data),
            None,
        )
        self._runtime_tools_overrides.pop(report.run_id, None)
        return failed

    def _record_unexpected_failure(
        self,
        run_id: str,
        events: list[Event],
        exc: BaseException,
    ) -> None:
        error_message = _sanitize_exception_message(exc)
        errors = [error_message]
        error_type = type(exc).__name__
        try:
            self.emit(
                events,
                run_id,
                "run.failed",
                "desk",
                {"errors": errors, "error_type": error_type},
            )
        finally:
            self._runtime_tools_overrides.pop(run_id, None)
            self.run_store.update_run(
                run_id,
                "failed",
                None,
                {"error": error_message, "error_type": error_type},
                None,
            )

    def run(
        self,
        runner: Worker | Workforce,
        job: Job,
        *,
        stream: bool | None = None,
        run_id: str | None = None,
    ) -> Report:
        ensure_not_running_loop("Desk.run", "Desk.arun")
        return asyncio.run(self.arun(runner, job, stream=stream, run_id=run_id))

    async def arun(
        self,
        runner: Worker | Workforce,
        job: Job,
        *,
        stream: bool | None = None,
        run_id: str | None = None,
    ) -> Report:
        self._ensure_open()
        run_id = run_id or new_id()
        events: list[Event] = []
        stream_enabled = self.stream if stream is None else stream
        job = self._resolve_structured_stream_mode(job)
        if isinstance(runner, Worker):
            job = self._apply_memory(runner, job)
        self.run_store.create_run(run_id, job.model_dump(mode="json"))
        self.emit(events, run_id, "run.started", "desk", {"job_id": job.id})

        async def drain_handlers() -> None:
            await self.event_bus.await_pending()

        config = self._make_run_config(run_id, events, stream_enabled)

        try:
            if isinstance(runner, Worker):
                self.register_worker(runner)
                report, state = await runner.arun(config, job)
            elif isinstance(runner, Workforce):
                self.register_workforce(runner)
                report, state = await runner.arun(config, job, drain_async_handlers=drain_handlers)
            else:
                raise TypeError("Runner must be Worker or Workforce")
        except asyncio.CancelledError as exc:
            self._record_unexpected_failure(run_id, events, exc)
            raise
        except Exception as exc:
            self._record_unexpected_failure(run_id, events, exc)
            raise

        return self._finalize_run(runner, report, state, run_id, events, stream_enabled)

    def resume(
        self,
        report: Report,
        decision_or_input: Any,
        *,
        stream: bool | None = None,
    ) -> Report:
        ensure_not_running_loop("Desk.resume", "Desk.aresume")
        return asyncio.run(self.aresume(report, decision_or_input, stream=stream))

    async def aresume(
        self,
        report: Report,
        decision_or_input: Any,
        *,
        stream: bool | None = None,
    ) -> Report:
        self._ensure_open()
        record = self.run_store.get_run(report.run_id)
        if record is None:
            failed = Report(
                run_id=report.run_id,
                status="failed",
                content=None,
                reasoning_content=None,
                data=None,
                messages=report.messages,
                tool_calls=report.tool_calls,
                metrics=report.metrics,
                events=report.events,
                pending_action=None,
                errors=["No stored state"],
            )
            return failed
        events = self.run_store.get_events(report.run_id)

        if record.state is None:
            return self._fail_resume(report, events, "No stored state")

        state = self._restore_runtime_tools_override(report.run_id, record.state)
        if state.runner_type == "flow":
            flow = self._flow_runs.get(report.run_id)
            if flow is None:
                return self._fail_resume(report, events, "Flow not registered")
            return await flow.aresume(report, decision_or_input, stream=stream)
        stream_enabled = stream if stream is not None else state.payload.get("stream", self.stream)

        self.emit(events, report.run_id, "run.resumed", "desk", {})

        config = self._make_run_config(report.run_id, events, stream_enabled)

        worker: Worker | None = None
        if state.runner_type == "worker":
            worker = self._workers.get(state.runner_name)
            if worker is None:
                return self._fail_resume(report, events, "Worker not registered")
            updated_report, updated_state = await worker.aresume(config, state, decision_or_input)
        elif state.runner_type == "workforce":
            workforce = self._workforces.get(state.runner_name)
            if workforce is None:
                return self._fail_resume(report, events, "Workforce not registered")
            updated_report, updated_state = await workforce.aresume(
                config, state, decision_or_input
            )
        else:
            return self._fail_resume(report, events, "Unknown runner type")

        return self._finalize_run(
            worker, updated_report, updated_state, report.run_id, events, stream_enabled
        )

    def close(self) -> None:
        if self._closed:
            return
        try:
            if self._owns_run_store:
                self.run_store.close()
        finally:
            if self._owns_memory_store:
                self.memory_store.close()
            self._closed = True

    def __enter__(self) -> "Desk":
        return self

    def __exit__(self, *_args: Any) -> None:
        self.close()
