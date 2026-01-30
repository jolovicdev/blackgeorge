import json
import os
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from blackgeorge.adapters.base import BaseModelAdapter
from blackgeorge.adapters.litellm import LiteLLMAdapter
from blackgeorge.core.event import Event
from blackgeorge.core.job import Job
from blackgeorge.core.message import Message
from blackgeorge.core.report import Report
from blackgeorge.event_bus import EventBus
from blackgeorge.memory.in_memory import InMemoryMemoryStore
from blackgeorge.store.base import RunStore
from blackgeorge.store.sqlite import SQLiteRunStore
from blackgeorge.utils import new_id, utc_now
from blackgeorge.worker import Worker
from blackgeorge.workflow.flow import Flow
from blackgeorge.workforce import Workforce

if TYPE_CHECKING:
    from blackgeorge.session import WorkerSession


class Desk:
    def __init__(
        self,
        *,
        model: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
        structured_output_retries: int = 1,
        max_iterations: int = 10,
        max_tool_calls: int = 20,
        respect_context_window: bool = True,
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
        self.structured_output_retries = structured_output_retries
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.respect_context_window = respect_context_window
        self.event_bus = event_bus or EventBus()
        self.adapter = adapter or LiteLLMAdapter()
        self.storage_dir = storage_dir or ".blackgeorge"
        os.makedirs(self.storage_dir, exist_ok=True)
        self.db_path = os.path.join(self.storage_dir, "blackgeorge.db")
        self.run_store = run_store or SQLiteRunStore(self.db_path)
        self.memory_store = memory_store or InMemoryMemoryStore()
        self._workers: dict[str, Worker] = {}
        self._workforces: dict[str, Workforce] = {}
        self._flow_runs: dict[str, Flow] = {}

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
            return WorkerSession.resume(session_id=session_id, worker=worker, desk=self)
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
            messages = [memory_message, *job.initial_messages]
        else:
            messages = [memory_message]
        return job.model_copy(update={"initial_messages": messages})

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

    def run(self, runner: Worker | Workforce, job: Job, *, stream: bool | None = None) -> Report:
        run_id = new_id()
        events: list[Event] = []
        stream_enabled = self.stream if stream is None else stream
        stream_options = {"include_usage": True} if stream_enabled else None
        if isinstance(runner, Worker):
            job = self._apply_memory(runner, job)
        self.run_store.create_run(run_id, job.model_dump(mode="json"))
        self._emit(events, run_id, "run.started", "desk", {"job_id": job.id})

        def emit(event_type: str, source: str, payload: dict[str, Any]) -> None:
            self._emit(events, run_id, event_type, source, payload)

        if isinstance(runner, Worker):
            self.register_worker(runner)
            report, state = runner.run(
                adapter=self.adapter,
                job=job,
                run_id=run_id,
                events=events,
                emit=emit,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=stream_enabled,
                stream_options=stream_options,
                structured_output_retries=self.structured_output_retries,
                max_iterations=self.max_iterations,
                max_tool_calls=self.max_tool_calls,
                model_name=runner.model or self.model,
                respect_context_window=self.respect_context_window,
            )
        elif isinstance(runner, Workforce):
            self.register_workforce(runner)
            report, state = runner.run(
                adapter=self.adapter,
                job=job,
                run_id=run_id,
                events=events,
                emit=emit,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=stream_enabled,
                stream_options=stream_options,
                structured_output_retries=self.structured_output_retries,
                max_iterations=self.max_iterations,
                max_tool_calls=self.max_tool_calls,
                default_model=self.model,
                respect_context_window=self.respect_context_window,
            )
        else:
            raise TypeError("Runner must be Worker or Workforce")

        if state is not None and report.status == "paused":
            state.payload["stream"] = stream_enabled
            self._emit(events, run_id, "run.paused", "desk", {})
            self.run_store.update_run(run_id, "paused", report.content, None, state)
        elif report.status == "completed":
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
            self._emit(events, run_id, "run.failed", "desk", {"errors": report.errors})
            self.run_store.update_run(
                run_id,
                "failed",
                report.content,
                self._output_json(report),
                None,
            )

        return report

    async def arun(
        self,
        runner: Worker | Workforce,
        job: Job,
        *,
        stream: bool | None = None,
    ) -> Report:
        run_id = new_id()
        events: list[Event] = []
        stream_enabled = self.stream if stream is None else stream
        stream_options = {"include_usage": True} if stream_enabled else None
        if isinstance(runner, Worker):
            job = self._apply_memory(runner, job)
        self.run_store.create_run(run_id, job.model_dump(mode="json"))
        self._emit(events, run_id, "run.started", "desk", {"job_id": job.id})

        def emit(event_type: str, source: str, payload: dict[str, Any]) -> None:
            self._emit(events, run_id, event_type, source, payload)

        if isinstance(runner, Worker):
            self.register_worker(runner)
            report, state = await runner.arun(
                adapter=self.adapter,
                job=job,
                run_id=run_id,
                events=events,
                emit=emit,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=stream_enabled,
                stream_options=stream_options,
                structured_output_retries=self.structured_output_retries,
                max_iterations=self.max_iterations,
                max_tool_calls=self.max_tool_calls,
                model_name=runner.model or self.model,
                respect_context_window=self.respect_context_window,
            )
        elif isinstance(runner, Workforce):
            self.register_workforce(runner)
            report, state = await runner.arun(
                adapter=self.adapter,
                job=job,
                run_id=run_id,
                events=events,
                emit=emit,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=stream_enabled,
                stream_options=stream_options,
                structured_output_retries=self.structured_output_retries,
                max_iterations=self.max_iterations,
                max_tool_calls=self.max_tool_calls,
                default_model=self.model,
                respect_context_window=self.respect_context_window,
            )
        else:
            raise TypeError("Runner must be Worker or Workforce")

        if state is not None and report.status == "paused":
            state.payload["stream"] = stream_enabled
            self._emit(events, run_id, "run.paused", "desk", {})
            self.run_store.update_run(run_id, "paused", report.content, None, state)
        elif report.status == "completed":
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
            self._emit(events, run_id, "run.failed", "desk", {"errors": report.errors})
            self.run_store.update_run(
                run_id,
                "failed",
                report.content,
                self._output_json(report),
                None,
            )

        return report

    def resume(
        self,
        report: Report,
        decision_or_input: Any,
        *,
        stream: bool | None = None,
    ) -> Report:
        record = self.run_store.get_run(report.run_id)
        if record is None or record.state is None:
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

        state = record.state
        if state.runner_type == "flow":
            flow = self._flow_runs.get(report.run_id)
            if flow is None:
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
                    errors=["Flow not registered"],
                )
                return failed
            return flow.resume(report, decision_or_input, stream=stream)
        stream_enabled = stream if stream is not None else state.payload.get("stream", self.stream)
        stream_options = {"include_usage": True} if stream_enabled else None
        events = self.run_store.get_events(report.run_id)

        def emit(event_type: str, source: str, payload: dict[str, Any]) -> None:
            self._emit(events, report.run_id, event_type, source, payload)

        self._emit(events, report.run_id, "run.resumed", "desk", {})

        worker: Worker | None = None
        if state.runner_type == "worker":
            worker = self._workers.get(state.runner_name)
            if worker is None:
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
                    errors=["Worker not registered"],
                )
                return failed
            updated_report, updated_state = worker.resume(
                adapter=self.adapter,
                state=state,
                decision_or_input=decision_or_input,
                events=events,
                emit=emit,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=stream_enabled,
                stream_options=stream_options,
                structured_output_retries=self.structured_output_retries,
                max_iterations=self.max_iterations,
                max_tool_calls=self.max_tool_calls,
                model_name=worker.model or self.model,
                respect_context_window=self.respect_context_window,
            )
        elif state.runner_type == "workforce":
            workforce = self._workforces.get(state.runner_name)
            if workforce is None:
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
                    errors=["Workforce not registered"],
                )
                return failed
            updated_report, updated_state = workforce.resume(
                adapter=self.adapter,
                state=state,
                decision_or_input=decision_or_input,
                events=events,
                emit=emit,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=stream_enabled,
                stream_options=stream_options,
                structured_output_retries=self.structured_output_retries,
                max_iterations=self.max_iterations,
                max_tool_calls=self.max_tool_calls,
                default_model=self.model,
                respect_context_window=self.respect_context_window,
            )
        else:
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
                errors=["Unknown runner type"],
            )
            return failed

        if updated_state is not None and updated_report.status == "paused":
            self._emit(events, report.run_id, "run.paused", "desk", {})
            self.run_store.update_run(
                report.run_id,
                "paused",
                updated_report.content,
                None,
                updated_state,
            )
        elif updated_report.status == "completed":
            self._emit(events, report.run_id, "run.completed", "desk", {})
            self.run_store.update_run(
                report.run_id,
                "completed",
                updated_report.content,
                self._output_json(updated_report),
                None,
            )
            if worker is not None:
                self._write_memory(worker, updated_report)
        else:
            self._emit(
                events,
                report.run_id,
                "run.failed",
                "desk",
                {"errors": updated_report.errors},
            )
            self.run_store.update_run(
                report.run_id,
                "failed",
                updated_report.content,
                self._output_json(updated_report),
                None,
            )

        return updated_report
