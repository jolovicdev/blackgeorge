import json
import os
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from blackgeorge.adapters.base import BaseModelAdapter
from blackgeorge.adapters.litellm import LiteLLMAdapter
from blackgeorge.config import RunConfig
from blackgeorge.core.event import Event
from blackgeorge.core.job import Job
from blackgeorge.core.message import Message
from blackgeorge.core.report import Report
from blackgeorge.event_bus import EventBus
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
        memory_store: Any | None = None,
        adapter: BaseModelAdapter | None = None,
        storage_dir: str | None = None,
    ) -> None:
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
        os.makedirs(self.storage_dir, exist_ok=True)
        self.run_store: RunStore
        if run_store is None:
            self.run_store = SQLiteRunStore(self.db_path)
        else:
            self.run_store = run_store
        self.memory_store = memory_store or InMemoryMemoryStore()
        self._workers: dict[str, Worker] = {}
        self._workforces: dict[str, Workforce] = {}
        self._flow_runs: dict[str, Flow] = {}
        self._runtime_tools_overrides: dict[str, list[Tool]] = {}

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
        return Flow(self, steps, name=name)

    def session(
        self,
        worker: Worker,
        *,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "WorkerSession | None":
        from blackgeorge.session import WorkerSession

        if session_id:
            session = WorkerSession.resume(session_id=session_id, worker=worker, desk=self)
            if session is not None:
                return session
            from blackgeorge.store.sqlite_session_store import SQLiteSessionStore

            store = SQLiteSessionStore(self.db_path)
            try:
                if store.get_session(session_id) is not None:
                    return None
            finally:
                store.close()
        return WorkerSession.start(
            worker=worker, desk=self, session_id=session_id, metadata=metadata
        )

    def _emit(
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

    def _output_json(self, report: Report) -> Any | None:
        if isinstance(report.data, BaseModel):
            return report.data.model_dump(mode="json", warnings=False)
        return report.data

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
            self._emit(events, run_id, event_type, source, payload)

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
            self._emit(events, run_id, "run.paused", "desk", {})
            self.run_store.update_run(run_id, "paused", report.content, None, state)
        elif report.status == "completed":
            self._runtime_tools_overrides.pop(run_id, None)
            self._emit(events, run_id, "run.completed", "desk", {})
            self.run_store.update_run(
                run_id,
                "completed",
                report.content,
                self._output_json(report),
                None,
            )
            if isinstance(runner, Worker):
                self._write_memory(runner, report)
        else:
            self._runtime_tools_overrides.pop(run_id, None)
            self._emit(events, run_id, "run.failed", "desk", {"errors": report.errors})
            self.run_store.update_run(
                run_id,
                "failed",
                report.content,
                self._output_json(report),
                None,
            )
        return report

    def _record_unexpected_failure(
        self,
        run_id: str,
        events: list[Event],
        exc: Exception,
    ) -> None:
        errors = [UNEXPECTED_FAILURE_MESSAGE]
        error_type = type(exc).__name__
        try:
            self._emit(
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
                {"error": UNEXPECTED_FAILURE_MESSAGE, "error_type": error_type},
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
        run_id = run_id or new_id()
        events: list[Event] = []
        stream_enabled = self.stream if stream is None else stream
        job = self._resolve_structured_stream_mode(job)
        if isinstance(runner, Worker):
            job = self._apply_memory(runner, job)
        self.run_store.create_run(run_id, job.model_dump(mode="json"))
        self._emit(events, run_id, "run.started", "desk", {"job_id": job.id})

        async def drain_handlers() -> None:
            await self.event_bus.await_pending()

        config = self._make_run_config(run_id, events, stream_enabled)

        try:
            if isinstance(runner, Worker):
                self.register_worker(runner)
                report, state = runner.run(config, job)
            elif isinstance(runner, Workforce):
                self.register_workforce(runner)
                report, state = runner.run(config, job, drain_async_handlers=drain_handlers)
            else:
                raise TypeError("Runner must be Worker or Workforce")
        except Exception as exc:
            self._record_unexpected_failure(run_id, events, exc)
            raise

        return self._finalize_run(runner, report, state, run_id, events, stream_enabled)

    async def arun(
        self,
        runner: Worker | Workforce,
        job: Job,
        *,
        stream: bool | None = None,
        run_id: str | None = None,
    ) -> Report:
        run_id = run_id or new_id()
        events: list[Event] = []
        stream_enabled = self.stream if stream is None else stream
        job = self._resolve_structured_stream_mode(job)
        if isinstance(runner, Worker):
            job = self._apply_memory(runner, job)
        self.run_store.create_run(run_id, job.model_dump(mode="json"))
        self._emit(events, run_id, "run.started", "desk", {"job_id": job.id})

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

        def resume_failed(message: str) -> Report:
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
            self._emit(
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
                self._output_json(failed),
                None,
            )
            self._runtime_tools_overrides.pop(report.run_id, None)
            return failed

        if record.state is None:
            return resume_failed("No stored state")

        state = self._restore_runtime_tools_override(report.run_id, record.state)
        if state.runner_type == "flow":
            flow = self._flow_runs.get(report.run_id)
            if flow is None:
                return resume_failed("Flow not registered")
            return flow.resume(report, decision_or_input, stream=stream)
        stream_enabled = stream if stream is not None else state.payload.get("stream", self.stream)

        self._emit(events, report.run_id, "run.resumed", "desk", {})

        config = self._make_run_config(report.run_id, events, stream_enabled)

        worker: Worker | None = None
        if state.runner_type == "worker":
            worker = self._workers.get(state.runner_name)
            if worker is None:
                return resume_failed("Worker not registered")
            updated_report, updated_state = worker.resume(config, state, decision_or_input)
        elif state.runner_type == "workforce":
            workforce = self._workforces.get(state.runner_name)
            if workforce is None:
                return resume_failed("Workforce not registered")
            updated_report, updated_state = workforce.resume(config, state, decision_or_input)
        else:
            return resume_failed("Unknown runner type")

        return self._finalize_run(
            worker, updated_report, updated_state, report.run_id, events, stream_enabled
        )

    async def aresume(
        self,
        report: Report,
        decision_or_input: Any,
        *,
        stream: bool | None = None,
    ) -> Report:
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

        def resume_failed(message: str) -> Report:
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
            self._emit(
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
                self._output_json(failed),
                None,
            )
            self._runtime_tools_overrides.pop(report.run_id, None)
            return failed

        if record.state is None:
            return resume_failed("No stored state")

        state = self._restore_runtime_tools_override(report.run_id, record.state)
        if state.runner_type == "flow":
            flow = self._flow_runs.get(report.run_id)
            if flow is None:
                return resume_failed("Flow not registered")
            return await flow.aresume(report, decision_or_input, stream=stream)
        stream_enabled = stream if stream is not None else state.payload.get("stream", self.stream)

        self._emit(events, report.run_id, "run.resumed", "desk", {})

        config = self._make_run_config(report.run_id, events, stream_enabled)

        worker: Worker | None = None
        if state.runner_type == "worker":
            worker = self._workers.get(state.runner_name)
            if worker is None:
                return resume_failed("Worker not registered")
            updated_report, updated_state = await worker.aresume(config, state, decision_or_input)
        elif state.runner_type == "workforce":
            workforce = self._workforces.get(state.runner_name)
            if workforce is None:
                return resume_failed("Workforce not registered")
            updated_report, updated_state = await workforce.aresume(
                config, state, decision_or_input
            )
        else:
            return resume_failed("Unknown runner type")

        return self._finalize_run(
            worker, updated_report, updated_state, report.run_id, events, stream_enabled
        )
