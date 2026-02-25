import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from blackgeorge.collaboration.blackboard import Blackboard
from blackgeorge.collaboration.channel import Channel
from blackgeorge.core.event import Event
from blackgeorge.core.event_types import EventType
from blackgeorge.core.job import Job
from blackgeorge.core.report import Report
from blackgeorge.core.types import WorkforceMode
from blackgeorge.store.state import RunState
from blackgeorge.worker import Worker

if TYPE_CHECKING:
    from blackgeorge.config import RunConfig
from blackgeorge.workforce_helpers import (
    aggregate_reports as _aggregate_reports,
)
from blackgeorge.workforce_helpers import (
    build_workforce_state as _build_workforce_state,
)
from blackgeorge.workforce_helpers import (
    default_reducer as _default_reducer,
)
from blackgeorge.workforce_helpers import (
    find_worker as _find_worker,
)
from blackgeorge.workforce_helpers import (
    root_job as _root_job,
)
from blackgeorge.workforce_helpers import (
    select_worker_name as _select_worker_name,
)

Reducer = Callable[[list[Report]], Report]


def _ensure_not_running_loop(action: str, async_action: str) -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    raise RuntimeError(
        f"{action} cannot be called from a running event loop. Use {async_action} instead."
    )


class WorkerDecision(BaseModel):
    worker: str
    reason: str | None = None


class Workforce:
    def __init__(
        self,
        workers: list[Worker],
        mode: WorkforceMode = "managed",
        name: str | None = None,
        manager: Worker | None = None,
        reducer: Reducer | None = None,
        channel: Channel | None = None,
        blackboard: Blackboard | None = None,
    ) -> None:
        if not workers:
            raise ValueError("Workforce requires at least one worker")
        self.workers = workers
        self.mode = mode
        self.name = name or "workforce"
        self.manager = manager
        self.reducer = reducer
        self.channel = channel or Channel()
        self.blackboard = blackboard or Blackboard()

    def _run_worker(
        self,
        *,
        worker: Worker,
        adapter: Any,
        job: Job,
        run_id: str,
        events: list[Event],
        emit: Callable[[str, str, dict[str, Any]], None],
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
        stream_options: dict[str, Any] | None,
        structured_output_retries: int,
        max_iterations: int,
        max_tool_calls: int,
        default_model: str,
        respect_context_window: bool,
    ) -> tuple[Report, RunState | None]:
        model_name = worker.model or default_model
        return worker.run(
            adapter=adapter,
            job=job,
            run_id=run_id,
            events=events,
            emit=emit,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            stream_options=stream_options,
            structured_output_retries=structured_output_retries,
            max_iterations=max_iterations,
            max_tool_calls=max_tool_calls,
            model_name=model_name,
            respect_context_window=respect_context_window,
        )

    def _resume_worker(
        self,
        *,
        worker: Worker,
        adapter: Any,
        state: RunState,
        decision_or_input: Any,
        events: list[Event],
        emit: Callable[[str, str, dict[str, Any]], None],
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
        stream_options: dict[str, Any] | None,
        structured_output_retries: int,
        max_iterations: int,
        max_tool_calls: int,
        default_model: str,
        respect_context_window: bool,
    ) -> tuple[Report, RunState | None]:
        model_name = worker.model or default_model
        return worker.resume(
            adapter=adapter,
            state=state,
            decision_or_input=decision_or_input,
            events=events,
            emit=emit,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            stream_options=stream_options,
            structured_output_retries=structured_output_retries,
            max_iterations=max_iterations,
            max_tool_calls=max_tool_calls,
            model_name=model_name,
            respect_context_window=respect_context_window,
        )

    async def _arun_worker(
        self,
        *,
        worker: Worker,
        adapter: Any,
        job: Job,
        run_id: str,
        events: list[Event],
        emit: Callable[[str, str, dict[str, Any]], None],
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
        stream_options: dict[str, Any] | None,
        structured_output_retries: int,
        max_iterations: int,
        max_tool_calls: int,
        default_model: str,
        respect_context_window: bool,
    ) -> tuple[Report, RunState | None]:
        model_name = worker.model or default_model
        return await worker.arun(
            adapter=adapter,
            job=job,
            run_id=run_id,
            events=events,
            emit=emit,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            stream_options=stream_options,
            structured_output_retries=structured_output_retries,
            max_iterations=max_iterations,
            max_tool_calls=max_tool_calls,
            model_name=model_name,
            respect_context_window=respect_context_window,
        )

    async def _aresume_worker(
        self,
        *,
        worker: Worker,
        adapter: Any,
        state: RunState,
        decision_or_input: Any,
        events: list[Event],
        emit: Callable[[str, str, dict[str, Any]], None],
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
        stream_options: dict[str, Any] | None,
        structured_output_retries: int,
        max_iterations: int,
        max_tool_calls: int,
        default_model: str,
        respect_context_window: bool,
    ) -> tuple[Report, RunState | None]:
        model_name = worker.model or default_model
        return await worker.aresume(
            adapter=adapter,
            state=state,
            decision_or_input=decision_or_input,
            events=events,
            emit=emit,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
            stream_options=stream_options,
            structured_output_retries=structured_output_retries,
            max_iterations=max_iterations,
            max_tool_calls=max_tool_calls,
            model_name=model_name,
            respect_context_window=respect_context_window,
        )

    def _can_parallelize_collaborate(self, job: Job) -> bool:
        if job.tools_override is not None:
            return len(job.tools_override) == 0
        return all(not worker.tools() for worker in self.workers)

    async def _arun_collaborate_parallel(
        self,
        *,
        adapter: Any,
        job: Job,
        run_id: str,
        events: list[Event],
        emit: Callable[[str, str, dict[str, Any]], None],
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
        stream_options: dict[str, Any] | None,
        structured_output_retries: int,
        max_iterations: int,
        max_tool_calls: int,
        default_model: str,
        respect_context_window: bool,
        drain_async_handlers: Callable[[], Awaitable[None]] | None = None,
    ) -> tuple[Report, RunState | None]:
        tasks: list[asyncio.Task[tuple[Report, RunState | None]]] = []
        for worker in self.workers:
            tasks.append(
                asyncio.create_task(
                    self._arun_worker(
                        worker=worker,
                        adapter=adapter,
                        job=job,
                        run_id=run_id,
                        events=events,
                        emit=emit,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stream=stream,
                        stream_options=stream_options,
                        structured_output_retries=structured_output_retries,
                        max_iterations=max_iterations,
                        max_tool_calls=max_tool_calls,
                        default_model=default_model,
                        respect_context_window=respect_context_window,
                    )
                )
            )
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            for task in pending:
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            results: list[tuple[Report, RunState | None]] = []
            for task in tasks:
                try:
                    results.append(task.result())
                except asyncio.CancelledError:
                    results.append(
                        (
                            Report(
                                run_id=run_id,
                                status="failed",
                                content=None,
                                data=None,
                                messages=[],
                                tool_calls=[],
                                metrics={},
                                events=events,
                                pending_action=None,
                                errors=["Worker cancelled due to sibling failure"],
                            ),
                            None,
                        )
                    )
                except Exception as exc:
                    results.append(
                        (
                            Report(
                                run_id=run_id,
                                status="failed",
                                content=None,
                                data=None,
                                messages=[],
                                tool_calls=[],
                                metrics={},
                                events=events,
                                pending_action=None,
                                errors=[str(exc)],
                            ),
                            None,
                        )
                    )
            reports: list[tuple[Worker, Report]] = []
            any_failed = False
            for worker, (report, worker_state) in zip(self.workers, results, strict=False):
                if worker_state is not None:
                    state = _build_workforce_state(
                        run_id,
                        "paused",
                        self.name,
                        job,
                        worker_state,
                        "collaborate",
                        payload={
                            "root_job": job.model_dump(mode="json"),
                            "completed_reports": [
                                rep.model_dump(mode="json") for _, rep in reports
                            ],
                            "pending_worker_index": len(reports),
                        },
                    )
                    return report, state
                reports.append((worker, report))
                if report.status == "failed":
                    any_failed = True
            if any_failed:
                return _aggregate_reports(reports, run_id, events, "failed"), None
            if self.reducer:
                return self.reducer([report for _, report in reports]), None
            return _default_reducer(reports, run_id, events), None
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            if drain_async_handlers is not None:
                await drain_async_handlers()

    def run(
        self,
        *,
        adapter: Any,
        job: Job,
        run_id: str,
        events: list[Event],
        emit: Callable[[str, str, dict[str, Any]], None],
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
        stream_options: dict[str, Any] | None,
        structured_output_retries: int,
        max_iterations: int,
        max_tool_calls: int,
        default_model: str,
        respect_context_window: bool = True,
        drain_async_handlers: Callable[[], Awaitable[None]] | None = None,
    ) -> tuple[Report, RunState | None]:
        emit(EventType.WORKFORCE_STARTED, self.name, {})

        if self.mode == "managed":
            manager = self.manager or self.workers[0]
            manager_job = Job(
                input={
                    "task": job.input,
                    "workers": [worker.name for worker in self.workers],
                },
                response_schema=WorkerDecision,
                tools_override=[],
            )
            manager_report, manager_state = self._run_worker(
                worker=manager,
                adapter=adapter,
                job=manager_job,
                run_id=run_id,
                events=events,
                emit=emit,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                stream_options=stream_options,
                structured_output_retries=structured_output_retries,
                max_iterations=max_iterations,
                max_tool_calls=max_tool_calls,
                default_model=default_model,
                respect_context_window=respect_context_window,
            )
            if manager_state is not None:
                state = _build_workforce_state(
                    run_id,
                    "paused",
                    self.name,
                    job,
                    manager_state,
                    "manager",
                    payload={"root_job": job.model_dump(mode="json")},
                )
                emit(EventType.WORKFORCE_COMPLETED, self.name, {})
                return manager_report, state
            if manager_report.status == "failed":
                emit(EventType.WORKFORCE_COMPLETED, self.name, {})
                return manager_report, None

            selected = _select_worker_name(manager_report, self.workers)
            worker = _find_worker(self.workers, selected)
            report, worker_state = self._run_worker(
                worker=worker,
                adapter=adapter,
                job=job,
                run_id=run_id,
                events=events,
                emit=emit,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                stream_options=stream_options,
                structured_output_retries=structured_output_retries,
                max_iterations=max_iterations,
                max_tool_calls=max_tool_calls,
                default_model=default_model,
                respect_context_window=respect_context_window,
            )
            if worker_state is not None:
                state = _build_workforce_state(
                    run_id,
                    "paused",
                    self.name,
                    job,
                    worker_state,
                    "worker",
                    payload={
                        "root_job": job.model_dump(mode="json"),
                        "selected_worker": worker.name,
                    },
                )
                emit(EventType.WORKFORCE_COMPLETED, self.name, {})
                return report, state
            if report.status == "failed":
                emit(EventType.WORKFORCE_COMPLETED, self.name, {})
                return report, None
            emit(EventType.WORKFORCE_COMPLETED, self.name, {})
            return report, None

        if self._can_parallelize_collaborate(job):
            _ensure_not_running_loop("run", "arun")
            report, parallel_state = asyncio.run(
                self._arun_collaborate_parallel(
                    adapter=adapter,
                    job=job,
                    run_id=run_id,
                    events=events,
                    emit=emit,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream,
                    stream_options=stream_options,
                    structured_output_retries=structured_output_retries,
                    max_iterations=max_iterations,
                    max_tool_calls=max_tool_calls,
                    default_model=default_model,
                    respect_context_window=respect_context_window,
                    drain_async_handlers=drain_async_handlers,
                )
            )
            emit(EventType.WORKFORCE_COMPLETED, self.name, {})
            return report, parallel_state

        reports: list[tuple[Worker, Report]] = []
        for worker in self.workers:
            report, worker_state = self._run_worker(
                worker=worker,
                adapter=adapter,
                job=job,
                run_id=run_id,
                events=events,
                emit=emit,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                stream_options=stream_options,
                structured_output_retries=structured_output_retries,
                max_iterations=max_iterations,
                max_tool_calls=max_tool_calls,
                default_model=default_model,
                respect_context_window=respect_context_window,
            )
            if worker_state is not None:
                state = _build_workforce_state(
                    run_id,
                    "paused",
                    self.name,
                    job,
                    worker_state,
                    "collaborate",
                    payload={
                        "root_job": job.model_dump(mode="json"),
                        "completed_reports": [rep.model_dump(mode="json") for _, rep in reports],
                        "pending_worker_index": len(reports),
                    },
                )
                emit(EventType.WORKFORCE_COMPLETED, self.name, {})
                return report, state
            if report.status == "failed":
                emit(EventType.WORKFORCE_COMPLETED, self.name, {})
                return _aggregate_reports(
                    reports + [(worker, report)],
                    run_id,
                    events,
                    "failed",
                ), None
            reports.append((worker, report))

        emit(EventType.WORKFORCE_COMPLETED, self.name, {})
        if self.reducer:
            return self.reducer([report for _, report in reports]), None
        return _default_reducer(reports, run_id, events), None

    async def arun(
        self,
        *,
        adapter: Any,
        job: Job,
        run_id: str,
        events: list[Event],
        emit: Callable[[str, str, dict[str, Any]], None],
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
        stream_options: dict[str, Any] | None,
        structured_output_retries: int,
        max_iterations: int,
        max_tool_calls: int,
        default_model: str,
        respect_context_window: bool = True,
        drain_async_handlers: Callable[[], Awaitable[None]] | None = None,
    ) -> tuple[Report, RunState | None]:
        emit(EventType.WORKFORCE_STARTED, self.name, {})

        if self.mode == "managed":
            manager = self.manager or self.workers[0]
            manager_job = Job(
                input={
                    "task": job.input,
                    "workers": [worker.name for worker in self.workers],
                },
                response_schema=WorkerDecision,
                tools_override=[],
            )
            manager_report, manager_state = await self._arun_worker(
                worker=manager,
                adapter=adapter,
                job=manager_job,
                run_id=run_id,
                events=events,
                emit=emit,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                stream_options=stream_options,
                structured_output_retries=structured_output_retries,
                max_iterations=max_iterations,
                max_tool_calls=max_tool_calls,
                default_model=default_model,
                respect_context_window=respect_context_window,
            )
            if manager_state is not None:
                state = _build_workforce_state(
                    run_id,
                    "paused",
                    self.name,
                    job,
                    manager_state,
                    "manager",
                    payload={"root_job": job.model_dump(mode="json")},
                )
                emit(EventType.WORKFORCE_COMPLETED, self.name, {})
                return manager_report, state
            if manager_report.status == "failed":
                emit(EventType.WORKFORCE_COMPLETED, self.name, {})
                return manager_report, None

            selected = _select_worker_name(manager_report, self.workers)
            worker = _find_worker(self.workers, selected)
            report, worker_state = await self._arun_worker(
                worker=worker,
                adapter=adapter,
                job=job,
                run_id=run_id,
                events=events,
                emit=emit,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                stream_options=stream_options,
                structured_output_retries=structured_output_retries,
                max_iterations=max_iterations,
                max_tool_calls=max_tool_calls,
                default_model=default_model,
                respect_context_window=respect_context_window,
            )
            if worker_state is not None:
                state = _build_workforce_state(
                    run_id,
                    "paused",
                    self.name,
                    job,
                    worker_state,
                    "worker",
                    payload={
                        "root_job": job.model_dump(mode="json"),
                        "selected_worker": worker.name,
                    },
                )
                emit(EventType.WORKFORCE_COMPLETED, self.name, {})
                return report, state
            if report.status == "failed":
                emit(EventType.WORKFORCE_COMPLETED, self.name, {})
                return report, None
            emit(EventType.WORKFORCE_COMPLETED, self.name, {})
            return report, None

        if self._can_parallelize_collaborate(job):
            report, parallel_state = await self._arun_collaborate_parallel(
                adapter=adapter,
                job=job,
                run_id=run_id,
                events=events,
                emit=emit,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                stream_options=stream_options,
                structured_output_retries=structured_output_retries,
                max_iterations=max_iterations,
                max_tool_calls=max_tool_calls,
                default_model=default_model,
                respect_context_window=respect_context_window,
                drain_async_handlers=drain_async_handlers,
            )
            emit(EventType.WORKFORCE_COMPLETED, self.name, {})
            return report, parallel_state

        reports: list[tuple[Worker, Report]] = []
        for worker in self.workers:
            report, worker_state = await self._arun_worker(
                worker=worker,
                adapter=adapter,
                job=job,
                run_id=run_id,
                events=events,
                emit=emit,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                stream_options=stream_options,
                structured_output_retries=structured_output_retries,
                max_iterations=max_iterations,
                max_tool_calls=max_tool_calls,
                default_model=default_model,
                respect_context_window=respect_context_window,
            )
            if worker_state is not None:
                state = _build_workforce_state(
                    run_id,
                    "paused",
                    self.name,
                    job,
                    worker_state,
                    "collaborate",
                    payload={
                        "root_job": job.model_dump(mode="json"),
                        "completed_reports": [rep.model_dump(mode="json") for _, rep in reports],
                        "pending_worker_index": len(reports),
                    },
                )
                emit(EventType.WORKFORCE_COMPLETED, self.name, {})
                return report, state
            if report.status == "failed":
                emit(EventType.WORKFORCE_COMPLETED, self.name, {})
                return _aggregate_reports(
                    reports + [(worker, report)],
                    run_id,
                    events,
                    "failed",
                ), None
            reports.append((worker, report))

        emit(EventType.WORKFORCE_COMPLETED, self.name, {})
        if self.reducer:
            return self.reducer([report for _, report in reports]), None
        return _default_reducer(reports, run_id, events), None

    def resume(
        self,
        *,
        adapter: Any,
        state: RunState,
        decision_or_input: Any,
        events: list[Event],
        emit: Callable[[str, str, dict[str, Any]], None],
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
        stream_options: dict[str, Any] | None,
        structured_output_retries: int,
        max_iterations: int,
        max_tool_calls: int,
        default_model: str,
        respect_context_window: bool = True,
    ) -> tuple[Report, RunState | None]:
        payload = state.payload
        stage = payload.get("stage")
        worker_state_payload = payload.get("worker_state")
        if worker_state_payload is None:
            report = Report(
                run_id=state.run_id,
                status="failed",
                content=None,
                data=None,
                messages=state.messages,
                tool_calls=state.tool_calls,
                metrics=state.metrics,
                events=events,
                pending_action=None,
                errors=["Missing worker state"],
            )
            return report, None
        stored_worker_state = RunState.model_validate(worker_state_payload)

        if stage == "manager":
            manager = self.manager or self.workers[0]
            manager_report, manager_state = self._resume_worker(
                worker=manager,
                adapter=adapter,
                state=stored_worker_state,
                decision_or_input=decision_or_input,
                events=events,
                emit=emit,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                stream_options=stream_options,
                structured_output_retries=structured_output_retries,
                max_iterations=max_iterations,
                max_tool_calls=max_tool_calls,
                default_model=default_model,
                respect_context_window=respect_context_window,
            )
            if manager_state is not None:
                root_job = _root_job(payload, state.job)
                state = _build_workforce_state(
                    state.run_id,
                    "paused",
                    self.name,
                    root_job,
                    manager_state,
                    "manager",
                    payload={"root_job": root_job.model_dump(mode="json")},
                )
                return manager_report, state
            if manager_report.status == "failed":
                return manager_report, None

            root_job = _root_job(payload, state.job)
            selected = _select_worker_name(manager_report, self.workers)
            worker = _find_worker(self.workers, selected)
            report, next_state = self._run_worker(
                worker=worker,
                adapter=adapter,
                job=root_job,
                run_id=state.run_id,
                events=events,
                emit=emit,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                stream_options=stream_options,
                structured_output_retries=structured_output_retries,
                max_iterations=max_iterations,
                max_tool_calls=max_tool_calls,
                default_model=default_model,
                respect_context_window=respect_context_window,
            )
            if next_state is not None:
                state = _build_workforce_state(
                    state.run_id,
                    "paused",
                    self.name,
                    root_job,
                    next_state,
                    "worker",
                    payload={
                        "root_job": root_job.model_dump(mode="json"),
                        "selected_worker": worker.name,
                    },
                )
                return report, state
            if report.status == "failed":
                return report, None
            return report, None

        if stage == "worker":
            root_job = _root_job(payload, state.job)
            worker_name = payload.get("selected_worker")
            worker = _find_worker(self.workers, worker_name)
            report, next_state = self._resume_worker(
                worker=worker,
                adapter=adapter,
                state=stored_worker_state,
                decision_or_input=decision_or_input,
                events=events,
                emit=emit,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                stream_options=stream_options,
                structured_output_retries=structured_output_retries,
                max_iterations=max_iterations,
                max_tool_calls=max_tool_calls,
                default_model=default_model,
                respect_context_window=respect_context_window,
            )
            if next_state is not None:
                state = _build_workforce_state(
                    state.run_id,
                    "paused",
                    self.name,
                    root_job,
                    next_state,
                    "worker",
                    payload={
                        "root_job": root_job.model_dump(mode="json"),
                        "selected_worker": worker.name,
                    },
                )
                return report, state
            if report.status == "failed":
                return report, None
            return report, None

        if stage == "collaborate":
            root_job = _root_job(payload, state.job)
            completed_reports_payload = payload.get("completed_reports", [])
            completed_reports = [Report.model_validate(rep) for rep in completed_reports_payload]
            pending_index = payload.get("pending_worker_index", 0)
            if not isinstance(pending_index, int) or pending_index < 0:
                report = Report(
                    run_id=state.run_id,
                    status="failed",
                    content=None,
                    data=None,
                    messages=state.messages,
                    tool_calls=state.tool_calls,
                    metrics=state.metrics,
                    events=events,
                    pending_action=None,
                    errors=["Invalid pending worker index"],
                )
                return report, None
            if pending_index >= len(self.workers):
                report = Report(
                    run_id=state.run_id,
                    status="failed",
                    content=None,
                    data=None,
                    messages=state.messages,
                    tool_calls=state.tool_calls,
                    metrics=state.metrics,
                    events=events,
                    pending_action=None,
                    errors=["Invalid pending worker index"],
                )
                return report, None
            pending_worker = self.workers[pending_index]
            report, next_state = self._resume_worker(
                worker=pending_worker,
                adapter=adapter,
                state=stored_worker_state,
                decision_or_input=decision_or_input,
                events=events,
                emit=emit,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                stream_options=stream_options,
                structured_output_retries=structured_output_retries,
                max_iterations=max_iterations,
                max_tool_calls=max_tool_calls,
                default_model=default_model,
                respect_context_window=respect_context_window,
            )
            if next_state is not None:
                state = _build_workforce_state(
                    state.run_id,
                    "paused",
                    self.name,
                    root_job,
                    next_state,
                    "collaborate",
                    payload={
                        "root_job": root_job.model_dump(mode="json"),
                        "completed_reports": completed_reports_payload,
                        "pending_worker_index": pending_index,
                    },
                )
                return report, state
            if report.status == "failed":
                return _aggregate_reports(
                    list(zip(self.workers[: pending_index + 1], completed_reports, strict=False))
                    + [(pending_worker, report)],
                    state.run_id,
                    events,
                    "failed",
                ), None
            completed_reports.append(report)
            reports: list[tuple[Worker, Report]] = []
            for worker, rep in zip(
                self.workers[: pending_index + 1],
                completed_reports,
                strict=False,
            ):
                reports.append((worker, rep))
            for worker in self.workers[pending_index + 1 :]:
                rep, next_state = self._run_worker(
                    worker=worker,
                    adapter=adapter,
                    job=root_job,
                    run_id=state.run_id,
                    events=events,
                    emit=emit,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream,
                    stream_options=stream_options,
                    structured_output_retries=structured_output_retries,
                    max_iterations=max_iterations,
                    max_tool_calls=max_tool_calls,
                    default_model=default_model,
                    respect_context_window=respect_context_window,
                )
                if next_state is not None:
                    new_payload = {
                        "root_job": root_job.model_dump(mode="json"),
                        "completed_reports": [r.model_dump(mode="json") for _, r in reports],
                        "pending_worker_index": len(reports),
                    }
                    state = _build_workforce_state(
                        state.run_id,
                        "paused",
                        self.name,
                        root_job,
                        next_state,
                        "collaborate",
                        payload=new_payload,
                    )
                    return rep, state
                if rep.status == "failed":
                    return _aggregate_reports(
                        reports + [(worker, rep)],
                        state.run_id,
                        events,
                        "failed",
                    ), None
                reports.append((worker, rep))
            if self.reducer:
                return self.reducer([rep for _, rep in reports]), None
            return _default_reducer(reports, state.run_id, events), None

        report = Report(
            run_id=state.run_id,
            status="failed",
            content=None,
            data=None,
            messages=state.messages,
            tool_calls=state.tool_calls,
            metrics=state.metrics,
            events=events,
            pending_action=None,
            errors=["Unknown workflow stage"],
        )
        return report, None

    async def aresume(
        self,
        *,
        adapter: Any,
        state: RunState,
        decision_or_input: Any,
        events: list[Event],
        emit: Callable[[str, str, dict[str, Any]], None],
        temperature: float | None,
        max_tokens: int | None,
        stream: bool,
        stream_options: dict[str, Any] | None,
        structured_output_retries: int,
        max_iterations: int,
        max_tool_calls: int,
        default_model: str,
        respect_context_window: bool = True,
    ) -> tuple[Report, RunState | None]:
        payload = state.payload
        stage = payload.get("stage")
        worker_state_payload = payload.get("worker_state")
        if worker_state_payload is None:
            report = Report(
                run_id=state.run_id,
                status="failed",
                content=None,
                data=None,
                messages=state.messages,
                tool_calls=state.tool_calls,
                metrics=state.metrics,
                events=events,
                pending_action=None,
                errors=["Missing worker state"],
            )
            return report, None
        stored_worker_state = RunState.model_validate(worker_state_payload)

        if stage == "manager":
            manager = self.manager or self.workers[0]
            manager_report, manager_state = await self._aresume_worker(
                worker=manager,
                adapter=adapter,
                state=stored_worker_state,
                decision_or_input=decision_or_input,
                events=events,
                emit=emit,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                stream_options=stream_options,
                structured_output_retries=structured_output_retries,
                max_iterations=max_iterations,
                max_tool_calls=max_tool_calls,
                default_model=default_model,
                respect_context_window=respect_context_window,
            )
            if manager_state is not None:
                root_job = _root_job(payload, state.job)
                state = _build_workforce_state(
                    state.run_id,
                    "paused",
                    self.name,
                    root_job,
                    manager_state,
                    "manager",
                    payload={"root_job": root_job.model_dump(mode="json")},
                )
                return manager_report, state
            if manager_report.status == "failed":
                return manager_report, None

            root_job = _root_job(payload, state.job)
            selected = _select_worker_name(manager_report, self.workers)
            worker = _find_worker(self.workers, selected)
            report, next_state = await self._arun_worker(
                worker=worker,
                adapter=adapter,
                job=root_job,
                run_id=state.run_id,
                events=events,
                emit=emit,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                stream_options=stream_options,
                structured_output_retries=structured_output_retries,
                max_iterations=max_iterations,
                max_tool_calls=max_tool_calls,
                default_model=default_model,
                respect_context_window=respect_context_window,
            )
            if next_state is not None:
                state = _build_workforce_state(
                    state.run_id,
                    "paused",
                    self.name,
                    root_job,
                    next_state,
                    "worker",
                    payload={
                        "root_job": root_job.model_dump(mode="json"),
                        "selected_worker": worker.name,
                    },
                )
                return report, state
            if report.status == "failed":
                return report, None
            return report, None

        if stage == "worker":
            root_job = _root_job(payload, state.job)
            worker_name = payload.get("selected_worker")
            worker = _find_worker(self.workers, worker_name)
            report, next_state = await self._aresume_worker(
                worker=worker,
                adapter=adapter,
                state=stored_worker_state,
                decision_or_input=decision_or_input,
                events=events,
                emit=emit,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                stream_options=stream_options,
                structured_output_retries=structured_output_retries,
                max_iterations=max_iterations,
                max_tool_calls=max_tool_calls,
                default_model=default_model,
                respect_context_window=respect_context_window,
            )
            if next_state is not None:
                state = _build_workforce_state(
                    state.run_id,
                    "paused",
                    self.name,
                    root_job,
                    next_state,
                    "worker",
                    payload={
                        "root_job": root_job.model_dump(mode="json"),
                        "selected_worker": worker.name,
                    },
                )
                return report, state
            if report.status == "failed":
                return report, None
            return report, None

        if stage == "collaborate":
            root_job = _root_job(payload, state.job)
            completed_reports_payload = payload.get("completed_reports", [])
            completed_reports = [Report.model_validate(rep) for rep in completed_reports_payload]
            pending_index = payload.get("pending_worker_index", 0)
            if not isinstance(pending_index, int) or pending_index < 0:
                report = Report(
                    run_id=state.run_id,
                    status="failed",
                    content=None,
                    data=None,
                    messages=state.messages,
                    tool_calls=state.tool_calls,
                    metrics=state.metrics,
                    events=events,
                    pending_action=None,
                    errors=["Invalid pending worker index"],
                )
                return report, None
            if pending_index >= len(self.workers):
                report = Report(
                    run_id=state.run_id,
                    status="failed",
                    content=None,
                    data=None,
                    messages=state.messages,
                    tool_calls=state.tool_calls,
                    metrics=state.metrics,
                    events=events,
                    pending_action=None,
                    errors=["Invalid pending worker index"],
                )
                return report, None
            pending_worker = self.workers[pending_index]
            report, next_state = await self._aresume_worker(
                worker=pending_worker,
                adapter=adapter,
                state=stored_worker_state,
                decision_or_input=decision_or_input,
                events=events,
                emit=emit,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                stream_options=stream_options,
                structured_output_retries=structured_output_retries,
                max_iterations=max_iterations,
                max_tool_calls=max_tool_calls,
                default_model=default_model,
                respect_context_window=respect_context_window,
            )
            if next_state is not None:
                state = _build_workforce_state(
                    state.run_id,
                    "paused",
                    self.name,
                    root_job,
                    next_state,
                    "collaborate",
                    payload={
                        "root_job": root_job.model_dump(mode="json"),
                        "completed_reports": completed_reports_payload,
                        "pending_worker_index": pending_index,
                    },
                )
                return report, state
            if report.status == "failed":
                return _aggregate_reports(
                    list(zip(self.workers[: pending_index + 1], completed_reports, strict=False))
                    + [(pending_worker, report)],
                    state.run_id,
                    events,
                    "failed",
                ), None
            completed_reports.append(report)
            reports: list[tuple[Worker, Report]] = []
            for worker, rep in zip(
                self.workers[: pending_index + 1],
                completed_reports,
                strict=False,
            ):
                reports.append((worker, rep))
            for worker in self.workers[pending_index + 1 :]:
                rep, next_state = await self._arun_worker(
                    worker=worker,
                    adapter=adapter,
                    job=root_job,
                    run_id=state.run_id,
                    events=events,
                    emit=emit,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream,
                    stream_options=stream_options,
                    structured_output_retries=structured_output_retries,
                    max_iterations=max_iterations,
                    max_tool_calls=max_tool_calls,
                    default_model=default_model,
                    respect_context_window=respect_context_window,
                )
                if next_state is not None:
                    new_payload = {
                        "root_job": root_job.model_dump(mode="json"),
                        "completed_reports": [r.model_dump(mode="json") for _, r in reports],
                        "pending_worker_index": len(reports),
                    }
                    state = _build_workforce_state(
                        state.run_id,
                        "paused",
                        self.name,
                        root_job,
                        next_state,
                        "collaborate",
                        payload=new_payload,
                    )
                    return rep, state
                if rep.status == "failed":
                    return _aggregate_reports(
                        reports + [(worker, rep)],
                        state.run_id,
                        events,
                        "failed",
                    ), None
                reports.append((worker, rep))
            if self.reducer:
                return self.reducer([rep for _, rep in reports]), None
            return _default_reducer(reports, state.run_id, events), None

        report = Report(
            run_id=state.run_id,
            status="failed",
            content=None,
            data=None,
            messages=state.messages,
            tool_calls=state.tool_calls,
            metrics=state.metrics,
            events=events,
            pending_action=None,
            errors=["Unknown workflow stage"],
        )
        return report, None

    def run_with_config(
        self,
        config: "RunConfig",
        job: Job,
        drain_async_handlers: Callable[[], Awaitable[None]] | None = None,
    ) -> tuple[Report, RunState | None]:
        _ensure_not_running_loop("run", "arun")
        return asyncio.run(self.arun_with_config(config, job, drain_async_handlers))

    async def arun_with_config(
        self,
        config: "RunConfig",
        job: Job,
        drain_async_handlers: Callable[[], Awaitable[None]] | None = None,
    ) -> tuple[Report, RunState | None]:
        return await self.arun(
            adapter=config.adapter,
            job=job,
            run_id=config.run_id,
            events=config.events,
            emit=config.emit,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            stream=config.stream,
            stream_options=config.stream_options,
            structured_output_retries=config.structured_output_retries,
            max_iterations=config.max_iterations,
            max_tool_calls=config.max_tool_calls,
            default_model=config.default_model or "",
            respect_context_window=config.respect_context_window,
            drain_async_handlers=drain_async_handlers,
        )

    def resume_with_config(
        self,
        config: "RunConfig",
        state: RunState,
        decision_or_input: Any,
    ) -> tuple[Report, RunState | None]:
        _ensure_not_running_loop("resume", "aresume")
        return asyncio.run(self.aresume_with_config(config, state, decision_or_input))

    async def aresume_with_config(
        self,
        config: "RunConfig",
        state: RunState,
        decision_or_input: Any,
    ) -> tuple[Report, RunState | None]:
        return await self.aresume(
            adapter=config.adapter,
            state=state,
            decision_or_input=decision_or_input,
            events=config.events,
            emit=config.emit,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            stream=config.stream,
            stream_options=config.stream_options,
            structured_output_retries=config.structured_output_retries,
            max_iterations=config.max_iterations,
            max_tool_calls=config.max_tool_calls,
            default_model=config.default_model or "",
            respect_context_window=config.respect_context_window,
        )
