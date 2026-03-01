import asyncio
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from blackgeorge.collaboration.blackboard import Blackboard
from blackgeorge.collaboration.channel import Channel
from blackgeorge.core.event_types import EventType
from blackgeorge.core.job import Job
from blackgeorge.core.message import Message
from blackgeorge.core.report import Report
from blackgeorge.core.types import WorkforceMode
from blackgeorge.store.state import RunState
from blackgeorge.tools.base import Tool
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
    merge_swarm_reports as _merge_swarm_reports,
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
        self._worker_by_name: dict[str, Worker] = {w.name: w for w in workers}
        if manager is not None:
            self._worker_by_name[manager.name] = manager

    async def _arun_worker(
        self, config: "RunConfig", worker: Worker, job: Job
    ) -> tuple[Report, RunState | None]:
        return await worker.arun(config, job)

    async def _aresume_worker(
        self, config: "RunConfig", worker: Worker, state: RunState, decision_or_input: Any
    ) -> tuple[Report, RunState | None]:
        return await worker.aresume(config, state, decision_or_input)

    def _can_parallelize_collaborate(self, job: Job) -> bool:
        if job.tools_override is not None:
            return len(job.tools_override) == 0
        return all(not worker.tools() for worker in self.workers)

    async def _arun_collaborate_parallel(
        self,
        config: "RunConfig",
        job: Job,
        drain_async_handlers: Callable[[], Awaitable[None]] | None = None,
    ) -> tuple[Report, RunState | None]:
        tasks = [asyncio.create_task(self._arun_worker(config, w, job)) for w in self.workers]
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
                                run_id=config.run_id,
                                status="failed",
                                messages=[],
                                tool_calls=[],
                                metrics={},
                                events=config.events,
                                errors=["Worker cancelled due to sibling failure"],
                            ),
                            None,
                        )
                    )
                except Exception as exc:
                    results.append(
                        (
                            Report(
                                run_id=config.run_id,
                                status="failed",
                                messages=[],
                                tool_calls=[],
                                metrics={},
                                events=config.events,
                                errors=[str(exc)],
                            ),
                            None,
                        )
                    )
            reports: list[tuple[Worker, Report]] = []
            any_failed = False
            for worker, (report, worker_state) in zip(self.workers, results, strict=False):
                if worker_state:
                    state = _build_workforce_state(
                        config.run_id,
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
                return _aggregate_reports(reports, config.run_id, config.events, "failed"), None
            return (
                self.reducer([r for _, r in reports])
                if self.reducer
                else _default_reducer(reports, config.run_id, config.events)
            ), None
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        finally:
            if drain_async_handlers:
                await drain_async_handlers()

    def _reduce_reports(
        self, reports: list[tuple[Worker, Report]], run_id: str, events: list[Any]
    ) -> Report:
        if self.reducer:
            return self.reducer([report for _, report in reports])
        return _default_reducer(reports, run_id, events)

    def _build_paused_state(
        self,
        run_id: str,
        job: Job,
        worker_state: RunState,
        stage: str,
        payload: dict[str, Any],
    ) -> RunState:
        return _build_workforce_state(
            run_id,
            "paused",
            self.name,
            job,
            worker_state,
            stage,
            payload=payload,
        )

    def _swarm_worker(self, name: str | None) -> Worker:
        if not isinstance(name, str) or not name:
            raise ValueError("Swarm current worker is missing.")
        worker = self._worker_by_name.get(name)
        if worker is None:
            raise ValueError(f"Swarm current worker '{name}' is not registered.")
        return worker

    def _resume_failed_report(self, state: RunState, config: "RunConfig", message: str) -> Report:
        return Report(
            run_id=state.run_id,
            status="failed",
            messages=state.messages,
            tool_calls=state.tool_calls,
            metrics=state.metrics,
            events=config.events,
            errors=[message],
        )

    def _swarm_resume_failed_report(
        self, state: RunState, config: "RunConfig", message: str
    ) -> Report:
        return self._resume_failed_report(state, config, message)

    def _swarm_handoff_budget(self, config: "RunConfig") -> int:
        return max(1, min(config.max_iterations, config.max_tool_calls))

    def _handoff_tool(self, worker: Worker, job: Job, tool_name: str) -> Tool | None:
        if job.tools_override is not None:
            resolved: Tool | None = None
            for item in job.tools_override:
                if isinstance(item, Tool):
                    if item.name == tool_name:
                        resolved = item
                    continue
                if isinstance(item, str) and item == tool_name:
                    candidate = worker.toolbelt.resolve(item)
                    if candidate is not None and candidate.name == tool_name:
                        resolved = candidate
            return resolved
        return worker.toolbelt.resolve(tool_name)

    def _handoff_allowed_agents(self, worker: Worker, job: Job, tool_name: str) -> set[str] | None:
        tool = self._handoff_tool(worker, job, tool_name)
        if tool is None or not tool.requires_handoff:
            return None
        properties = tool.schema.get("properties")
        if not isinstance(properties, dict):
            return None
        agent_name_schema = properties.get("agent_name")
        if not isinstance(agent_name_schema, dict):
            return None
        raw_enum = agent_name_schema.get("enum")
        if not isinstance(raw_enum, list):
            return None
        allowed = {item for item in raw_enum if isinstance(item, str) and item}
        return allowed

    def _handoff_messages(
        self, report: Report, target_name: str, tool_call_id: str
    ) -> list[Message]:
        messages = [message for message in report.messages if message.role != "system"]
        messages.append(
            Message(
                role="tool",
                content=f"Transferred to {target_name}.",
                tool_call_id=tool_call_id,
            )
        )
        return messages

    def _handoff_transition(
        self, report: Report, current_job: Job, current_worker: Worker
    ) -> tuple[Worker, Job] | None:
        pending = report.pending_action
        if pending is None or pending.type != "handoff":
            return None
        target_name = pending.tool_call.arguments.get("agent_name")
        if not isinstance(target_name, str) or not target_name:
            raise ValueError("Handoff target is missing or invalid.")
        handoff_context = pending.tool_call.arguments.get("context", "")
        allowed_agents = self._handoff_allowed_agents(
            current_worker, current_job, pending.tool_call.name
        )
        if allowed_agents is not None and target_name not in allowed_agents:
            raise ValueError(
                f"Handoff target '{target_name}' is not allowed for worker '{current_worker.name}'."
            )
        try:
            next_worker = self._swarm_worker(target_name)
        except ValueError:
            raise ValueError(f"Handoff target '{target_name}' is not in the workforce.") from None
        messages = self._handoff_messages(report, target_name, pending.tool_call.id)
        next_job = current_job.model_copy(
            update={
                "initial_messages": messages,
                "input": handoff_context or current_job.input,
            }
        )
        return next_worker, next_job

    def _handoff_failed_report(self, report: Report, message: str) -> Report:
        errors = list(report.errors)
        errors.append(message)
        return report.model_copy(
            update={"status": "failed", "pending_action": None, "errors": errors}
        )

    async def _arun_swarm_mode(
        self, config: "RunConfig", job: Job
    ) -> tuple[Report, RunState | None]:
        current_worker = self.manager or self.workers[0]
        current_job = job
        handoff_count = 0
        handoff_budget = self._swarm_handoff_budget(config)
        swarm_history: list[Report] = []
        while True:
            report, worker_state = await self._arun_worker(config, current_worker, current_job)
            if worker_state and report.status == "paused":
                try:
                    handoff = self._handoff_transition(report, current_job, current_worker)
                except ValueError as exc:
                    failed = self._handoff_failed_report(report, str(exc))
                    return _merge_swarm_reports(swarm_history, failed), None
                if handoff is not None:
                    swarm_history.append(report.model_copy(update={"pending_action": None}))
                    handoff_count += 1
                    if handoff_count > handoff_budget:
                        failed = self._handoff_failed_report(
                            report, "Max handoff transitions exceeded."
                        )
                        return _merge_swarm_reports(swarm_history, failed), None
                    current_worker, current_job = handoff
                    continue
            if worker_state:
                return report, self._build_paused_state(
                    config.run_id,
                    current_job,
                    worker_state,
                    "swarm",
                    payload={
                        "root_job": current_job.model_dump(mode="json"),
                        "current_worker": current_worker.name,
                        "handoff_count": handoff_count,
                        "swarm_history": [r.model_dump(mode="json") for r in swarm_history],
                    },
                )
            return _merge_swarm_reports(swarm_history, report), None

    async def _arun_managed_mode(
        self, config: "RunConfig", job: Job
    ) -> tuple[Report, RunState | None]:
        manager = self.manager or self.workers[0]
        manager_job = Job(
            input={"task": job.input, "workers": [worker.name for worker in self.workers]},
            response_schema=WorkerDecision,
            tools_override=[],
        )
        manager_report, manager_state = await self._arun_worker(config, manager, manager_job)
        if manager_state:
            return manager_report, self._build_paused_state(
                config.run_id,
                job,
                manager_state,
                "manager",
                payload={"root_job": job.model_dump(mode="json")},
            )
        if manager_report.status == "failed":
            return manager_report, None
        selected_worker = _find_worker(
            self.workers, _select_worker_name(manager_report, self.workers)
        )
        worker_report, worker_state = await self._arun_worker(config, selected_worker, job)
        if worker_state:
            return worker_report, self._build_paused_state(
                config.run_id,
                job,
                worker_state,
                "worker",
                payload={
                    "root_job": job.model_dump(mode="json"),
                    "selected_worker": selected_worker.name,
                },
            )
        return worker_report, None

    async def _arun_collaborate_mode(
        self,
        config: "RunConfig",
        job: Job,
        drain_async_handlers: Callable[[], Awaitable[None]] | None = None,
    ) -> tuple[Report, RunState | None]:
        if self._can_parallelize_collaborate(job):
            return await self._arun_collaborate_parallel(config, job, drain_async_handlers)
        reports: list[tuple[Worker, Report]] = []
        for worker in self.workers:
            report, worker_state = await self._arun_worker(config, worker, job)
            if worker_state:
                return report, self._build_paused_state(
                    config.run_id,
                    job,
                    worker_state,
                    "collaborate",
                    payload={
                        "root_job": job.model_dump(mode="json"),
                        "completed_reports": [rep.model_dump(mode="json") for _, rep in reports],
                        "pending_worker_index": len(reports),
                    },
                )
            if report.status == "failed":
                return _aggregate_reports(
                    reports + [(worker, report)], config.run_id, config.events, "failed"
                ), None
            reports.append((worker, report))
        return self._reduce_reports(reports, config.run_id, config.events), None

    def _missing_worker_state_report(self, state: RunState, config: "RunConfig") -> Report:
        return self._resume_failed_report(state, config, "Missing worker state")

    def _invalid_pending_worker_index_report(self, state: RunState, config: "RunConfig") -> Report:
        return self._resume_failed_report(state, config, "Invalid pending worker index")

    def _unknown_stage_report(self, state: RunState, config: "RunConfig") -> Report:
        return self._resume_failed_report(state, config, "Unknown workflow stage")

    async def _aresume_swarm_stage(
        self,
        config: "RunConfig",
        state: RunState,
        payload: dict[str, Any],
        stored_worker_state: RunState,
        decision_or_input: Any,
    ) -> tuple[Report, RunState | None]:
        current_worker_name = payload.get("current_worker")
        if not isinstance(current_worker_name, str) or not current_worker_name:
            current_worker_name = stored_worker_state.runner_name
        try:
            current_worker = self._swarm_worker(current_worker_name)
        except ValueError as exc:
            return self._swarm_resume_failed_report(state, config, str(exc)), None
        current_job = state.job
        handoff_count = payload.get("handoff_count", 0)
        if not isinstance(handoff_count, int) or handoff_count < 0:
            handoff_count = 0
        handoff_budget = self._swarm_handoff_budget(config)
        swarm_history_payload = payload.get("swarm_history", [])
        swarm_history: list[Report] = [Report.model_validate(r) for r in swarm_history_payload]
        report, worker_state = await self._aresume_worker(
            config, current_worker, stored_worker_state, decision_or_input
        )
        while True:
            if worker_state and report.status == "paused":
                try:
                    handoff = self._handoff_transition(report, current_job, current_worker)
                except ValueError as exc:
                    failed = self._handoff_failed_report(report, str(exc))
                    return _merge_swarm_reports(swarm_history, failed), None
                if handoff is not None:
                    swarm_history.append(report.model_copy(update={"pending_action": None}))
                    handoff_count += 1
                    if handoff_count > handoff_budget:
                        failed = self._handoff_failed_report(
                            report, "Max handoff transitions exceeded."
                        )
                        return _merge_swarm_reports(swarm_history, failed), None
                    current_worker, current_job = handoff
                    report, worker_state = await self._arun_worker(
                        config, current_worker, current_job
                    )
                    continue
            if worker_state:
                return report, self._build_paused_state(
                    state.run_id,
                    current_job,
                    worker_state,
                    "swarm",
                    payload={
                        "root_job": current_job.model_dump(mode="json"),
                        "current_worker": current_worker.name,
                        "handoff_count": handoff_count,
                        "swarm_history": [r.model_dump(mode="json") for r in swarm_history],
                    },
                )
            return _merge_swarm_reports(swarm_history, report), None

    async def _aresume_manager_stage(
        self,
        config: "RunConfig",
        state: RunState,
        payload: dict[str, Any],
        stored_worker_state: RunState,
        decision_or_input: Any,
    ) -> tuple[Report, RunState | None]:
        manager = self.manager or self.workers[0]
        root_job = _root_job(payload, state.job)
        manager_report, manager_state = await self._aresume_worker(
            config, manager, stored_worker_state, decision_or_input
        )
        if manager_state:
            return manager_report, self._build_paused_state(
                state.run_id,
                root_job,
                manager_state,
                "manager",
                payload={"root_job": root_job.model_dump(mode="json")},
            )
        if manager_report.status == "failed":
            return manager_report, None
        worker = _find_worker(self.workers, _select_worker_name(manager_report, self.workers))
        report, next_state = await self._arun_worker(config, worker, root_job)
        if next_state:
            return report, self._build_paused_state(
                state.run_id,
                root_job,
                next_state,
                "worker",
                payload={
                    "root_job": root_job.model_dump(mode="json"),
                    "selected_worker": worker.name,
                },
            )
        return report, None

    async def _aresume_worker_stage(
        self,
        config: "RunConfig",
        state: RunState,
        payload: dict[str, Any],
        stored_worker_state: RunState,
        decision_or_input: Any,
    ) -> tuple[Report, RunState | None]:
        root_job = _root_job(payload, state.job)
        worker = _find_worker(self.workers, payload.get("selected_worker"))
        report, next_state = await self._aresume_worker(
            config, worker, stored_worker_state, decision_or_input
        )
        if next_state:
            return report, self._build_paused_state(
                state.run_id,
                root_job,
                next_state,
                "worker",
                payload={
                    "root_job": root_job.model_dump(mode="json"),
                    "selected_worker": worker.name,
                },
            )
        return report, None

    async def _aresume_collaborate_stage(
        self,
        config: "RunConfig",
        state: RunState,
        payload: dict[str, Any],
        stored_worker_state: RunState,
        decision_or_input: Any,
    ) -> tuple[Report, RunState | None]:
        root_job = _root_job(payload, state.job)
        completed_reports_payload = payload.get("completed_reports", [])
        pending_index = payload.get("pending_worker_index", 0)
        if (
            not isinstance(pending_index, int)
            or pending_index < 0
            or pending_index >= len(self.workers)
        ):
            return self._invalid_pending_worker_index_report(state, config), None
        pending_worker = self.workers[pending_index]
        report, next_state = await self._aresume_worker(
            config, pending_worker, stored_worker_state, decision_or_input
        )
        if next_state:
            return report, self._build_paused_state(
                state.run_id,
                root_job,
                next_state,
                "collaborate",
                payload={
                    "root_job": root_job.model_dump(mode="json"),
                    "completed_reports": completed_reports_payload,
                    "pending_worker_index": pending_index,
                },
            )
        completed_reports = [Report.model_validate(rep) for rep in completed_reports_payload]
        if report.status == "failed":
            return _aggregate_reports(
                list(zip(self.workers[: pending_index + 1], completed_reports, strict=False))
                + [(pending_worker, report)],
                state.run_id,
                config.events,
                "failed",
            ), None
        completed_reports.append(report)
        reports: list[tuple[Worker, Report]] = [
            (worker, worker_report)
            for worker, worker_report in zip(
                self.workers[: pending_index + 1], completed_reports, strict=False
            )
        ]
        for worker in self.workers[pending_index + 1 :]:
            rep, next_state = await self._arun_worker(config, worker, root_job)
            if next_state:
                return rep, self._build_paused_state(
                    state.run_id,
                    root_job,
                    next_state,
                    "collaborate",
                    payload={
                        "root_job": root_job.model_dump(mode="json"),
                        "completed_reports": [r.model_dump(mode="json") for _, r in reports],
                        "pending_worker_index": len(reports),
                    },
                )
            if rep.status == "failed":
                return _aggregate_reports(
                    reports + [(worker, rep)], state.run_id, config.events, "failed"
                ), None
            reports.append((worker, rep))
        return self._reduce_reports(reports, state.run_id, config.events), None

    def run(
        self,
        config: "RunConfig",
        job: Job,
        drain_async_handlers: Callable[[], Awaitable[None]] | None = None,
    ) -> tuple[Report, RunState | None]:
        _ensure_not_running_loop("run", "arun")
        return asyncio.run(self.arun(config, job, drain_async_handlers))

    async def arun(
        self,
        config: "RunConfig",
        job: Job,
        drain_async_handlers: Callable[[], Awaitable[None]] | None = None,
    ) -> tuple[Report, RunState | None]:
        config.emit(EventType.WORKFORCE_STARTED, self.name, {})
        if self.mode == "swarm":
            result = await self._arun_swarm_mode(config, job)
        elif self.mode == "managed":
            result = await self._arun_managed_mode(config, job)
        else:
            result = await self._arun_collaborate_mode(config, job, drain_async_handlers)
        config.emit(EventType.WORKFORCE_COMPLETED, self.name, {})
        return result

    def resume(
        self, config: "RunConfig", state: RunState, decision_or_input: Any
    ) -> tuple[Report, RunState | None]:
        _ensure_not_running_loop("resume", "aresume")
        return asyncio.run(self.aresume(config, state, decision_or_input))

    async def aresume(
        self, config: "RunConfig", state: RunState, decision_or_input: Any
    ) -> tuple[Report, RunState | None]:
        if config.run_id != state.run_id:
            config = config.with_overrides(run_id=state.run_id)
        payload, stage = state.payload, state.payload.get("stage")
        worker_state_payload = payload.get("worker_state")
        if worker_state_payload is None:
            return self._missing_worker_state_report(state, config), None
        stored_worker_state = RunState.model_validate(worker_state_payload)
        if stage == "swarm":
            return await self._aresume_swarm_stage(
                config, state, payload, stored_worker_state, decision_or_input
            )
        if stage == "manager":
            return await self._aresume_manager_stage(
                config, state, payload, stored_worker_state, decision_or_input
            )
        if stage == "worker":
            return await self._aresume_worker_stage(
                config, state, payload, stored_worker_state, decision_or_input
            )
        if stage == "collaborate":
            return await self._aresume_collaborate_stage(
                config, state, payload, stored_worker_state, decision_or_input
            )
        return self._unknown_stage_report(state, config), None
