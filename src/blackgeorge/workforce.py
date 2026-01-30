from collections.abc import Callable
from typing import Any

from pydantic import BaseModel

from blackgeorge.collaboration.blackboard import Blackboard
from blackgeorge.collaboration.channel import Channel
from blackgeorge.core.event import Event
from blackgeorge.core.job import Job
from blackgeorge.core.report import Report
from blackgeorge.core.types import RunStatus, WorkforceMode
from blackgeorge.store.state import RunState
from blackgeorge.worker import Worker

Reducer = Callable[[list[Report]], Report]


class WorkerDecision(BaseModel):
    worker: str
    reason: str | None = None


def _build_workforce_state(
    run_id: str,
    status: RunStatus,
    workforce_name: str,
    job: Job,
    worker_state: RunState,
    stage: str,
    payload: dict[str, Any] | None = None,
) -> RunState:
    return RunState(
        run_id=run_id,
        status=status,
        runner_type="workforce",
        runner_name=workforce_name,
        job=job,
        messages=worker_state.messages,
        tool_calls=worker_state.tool_calls,
        pending_action=worker_state.pending_action,
        metrics=worker_state.metrics,
        iteration=worker_state.iteration,
        payload={
            "stage": stage,
            "worker_state": worker_state.model_dump(mode="json"),
            **(payload or {}),
        },
    )


def _select_worker_name(report: Report, workers: list[Worker]) -> str:
    worker_names = {w.name for w in workers}
    if isinstance(report.data, BaseModel) and hasattr(report.data, "worker"):
        candidate = report.data.worker
        if isinstance(candidate, str) and candidate in worker_names:
            return candidate
    if report.content:
        for worker in workers:
            if worker.name in report.content:
                return worker.name
    return workers[0].name


def _find_worker(workers: list[Worker], name: str | None) -> Worker:
    if name is None:
        return workers[0]
    for worker in workers:
        if worker.name == name:
            return worker
    return workers[0]


def _root_job(payload: dict[str, Any], fallback: Job) -> Job:
    raw = payload.get("root_job")
    if raw is None:
        return fallback
    return Job.model_validate(raw)


def _default_reducer(
    reports: list[tuple[Worker, Report]],
    run_id: str,
    events: list[Event],
) -> Report:
    status: RunStatus = "completed"
    if any(report.status == "failed" for _, report in reports):
        status = "failed"
    return _aggregate_reports(reports, run_id, events, status)


def _aggregate_reports(
    reports: list[tuple[Worker, Report]],
    run_id: str,
    events: list[Event],
    status: RunStatus,
) -> Report:
    has_data = any(report.data is not None for _, report in reports)
    content_parts: list[str] = []
    data: Any | None = None
    if has_data:
        data = []
        for worker, report in reports:
            data.append({"worker": worker.name, "data": report.data, "content": report.content})
    for worker, report in reports:
        content_parts.append(f"[{worker.name}] {report.content or ''}")
    return Report(
        run_id=run_id,
        status=status,
        content="\n\n".join(content_parts),
        data=data,
        messages=[message for _, report in reports for message in report.messages],
        tool_calls=[call for _, report in reports for call in report.tool_calls],
        metrics={},
        events=events,
        pending_action=None,
        errors=[error for _, report in reports for error in report.errors],
    )


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
    ) -> tuple[Report, RunState | None]:
        emit("workforce.started", self.name, {})

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
                emit("workforce.completed", self.name, {})
                return manager_report, state
            if manager_report.status == "failed":
                emit("workforce.completed", self.name, {})
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
                emit("workforce.completed", self.name, {})
                return report, state
            if report.status == "failed":
                emit("workforce.completed", self.name, {})
                return report, None

            emit("workforce.completed", self.name, {})
            return report, None

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
                emit("workforce.completed", self.name, {})
                return report, state
            if report.status == "failed":
                emit("workforce.completed", self.name, {})
                return _aggregate_reports(
                    reports + [(worker, report)],
                    run_id,
                    events,
                    "failed",
                ), None
            reports.append((worker, report))

        emit("workforce.completed", self.name, {})
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
    ) -> tuple[Report, RunState | None]:
        emit("workforce.started", self.name, {})

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
                emit("workforce.completed", self.name, {})
                return manager_report, state
            if manager_report.status == "failed":
                emit("workforce.completed", self.name, {})
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
                emit("workforce.completed", self.name, {})
                return report, state
            if report.status == "failed":
                emit("workforce.completed", self.name, {})
                return report, None

            emit("workforce.completed", self.name, {})
            return report, None

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
                emit("workforce.completed", self.name, {})
                return report, state
            if report.status == "failed":
                emit("workforce.completed", self.name, {})
                return _aggregate_reports(
                    reports + [(worker, report)],
                    run_id,
                    events,
                    "failed",
                ), None
            reports.append((worker, report))

        emit("workforce.completed", self.name, {})
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
