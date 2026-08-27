import asyncio
from typing import Any, cast

from blackgeorge.async_utils import ensure_not_running_loop
from blackgeorge.config import RunConfig
from blackgeorge.core.event import Event
from blackgeorge.core.job import Job
from blackgeorge.core.report import Report
from blackgeorge.core.serialization import to_json_value
from blackgeorge.store.state import RunState
from blackgeorge.utils import new_id
from blackgeorge.worker import Worker
from blackgeorge.workflow.context import WorkflowContext
from blackgeorge.workflow.nodes import (
    OutputContinuation,
    Step,
    WorkflowGraph,
    restore_continuations,
    serialize_continuations,
)
from blackgeorge.workflow.result import (
    StepOutput,
    StepResult,
    WorkflowContinuation,
)
from blackgeorge.workforce import Workforce


class Flow:
    def __init__(self, desk: Any, steps: list[Any], name: str | None = None) -> None:
        self.desk = desk
        self.steps = list(steps)
        self.name = name or "flow"
        self._run_id = ""
        self._events: list[Event] = []
        self._stream: bool = False

    def emit(self, event_type: str, source: str, payload: dict[str, Any]) -> None:
        if not self._run_id:
            return
        self.desk.emit(self._events, self._run_id, event_type, source, payload)

    def _make_run_config(self, stream: bool) -> RunConfig:
        return RunConfig(
            adapter=self.desk.adapter,
            emit=self.emit,
            run_id=self._run_id,
            events=self._events,
            temperature=self.desk.temperature,
            max_tokens=self.desk.max_tokens,
            stream=stream,
            stream_options={"include_usage": True} if stream else None,
            structured_output_retries=self.desk.structured_output_retries,
            max_iterations=self.desk.max_iterations,
            max_tool_calls=self.desk.max_tool_calls,
            num_retries=self.desk.num_retries,
            respect_context_window=self.desk.respect_context_window,
            max_context_messages=self.desk.max_context_messages,
            max_cost_usd=self.desk.max_cost_usd,
            default_model=self.desk.model,
        )

    async def run_runner(
        self,
        runner: Worker | Workforce,
        job: Job,
    ) -> tuple[Report, RunState | None]:
        config = self._make_run_config(self._stream)
        if isinstance(runner, Worker):
            self.desk.register_worker(runner)
            return await runner.arun(config, job)
        if isinstance(runner, Workforce):
            self.desk.register_workforce(runner)
            return await runner.arun(config, job)
        raise TypeError("Runner must be Worker or Workforce")

    async def _resume_runner(
        self,
        state: RunState,
        decision_or_input: Any,
        stream: bool,
    ) -> tuple[Report, RunState | None]:
        config = self._make_run_config(stream)
        if state.runner_type == "worker":
            worker = self.desk._workers.get(state.runner_name)
            if worker is None:
                report = Report(
                    run_id=state.run_id,
                    status="failed",
                    content=None,
                    data=None,
                    messages=state.messages,
                    tool_calls=state.tool_calls,
                    metrics=state.metrics,
                    events=self._events,
                    pending_action=None,
                    errors=["Worker not registered"],
                )
                return report, None
            worker = cast(Worker, worker)
            return await worker.aresume(config, state, decision_or_input)
        if state.runner_type == "workforce":
            workforce = self.desk._workforces.get(state.runner_name)
            if workforce is None:
                report = Report(
                    run_id=state.run_id,
                    status="failed",
                    content=None,
                    data=None,
                    messages=state.messages,
                    tool_calls=state.tool_calls,
                    metrics=state.metrics,
                    events=self._events,
                    pending_action=None,
                    errors=["Workforce not registered"],
                )
                return report, None
            workforce = cast(Workforce, workforce)
            return await workforce.aresume(config, state, decision_or_input)
        report = Report(
            run_id=state.run_id,
            status="failed",
            content=None,
            data=None,
            messages=state.messages,
            tool_calls=state.tool_calls,
            metrics=state.metrics,
            events=self._events,
            pending_action=None,
            errors=["Unknown runner type"],
        )
        return report, None

    def _build_flow_state(
        self,
        job: Job,
        step_index: int,
        reports: list[Report],
        step_state: RunState,
        stream: bool,
        context: WorkflowContext,
        continuations: list[WorkflowContinuation],
    ) -> RunState:
        graph = WorkflowGraph(tuple(self.steps))
        payload = {
            "step_index": step_index,
            "outputs": [report.model_dump(mode="json") for report in reports],
            "step_state": step_state.model_dump(mode="json"),
            "stream": stream,
            "flow_signature": graph.signature,
            "context": context.snapshot(),
            "continuations": serialize_continuations(graph, continuations),
        }
        return RunState(
            run_id=self._run_id,
            status="paused",
            runner_type="flow",
            runner_name=self.name,
            job=job,
            messages=step_state.messages,
            tool_calls=step_state.tool_calls,
            pending_action=step_state.pending_action,
            metrics=step_state.metrics,
            iteration=step_state.iteration,
            payload=payload,
        )

    def _restore_outputs(self, payload: Any) -> list[Report]:
        if payload is None:
            return []
        if not isinstance(payload, list):
            raise ValueError("Invalid flow outputs")
        return [Report.model_validate(item) for item in payload]

    def _combine_reports(self, reports: list[Report]) -> Report:
        content_parts: list[str] = []
        for idx, report in enumerate(reports, start=1):
            content_parts.append(f"[step {idx}] {report.content or ''}")
        data = [
            {"index": idx, "content": report.content, "data": report.data}
            for idx, report in enumerate(reports, start=1)
        ]
        return Report(
            run_id=self._run_id,
            status="completed",
            content="\n\n".join(content_parts),
            data=data,
            messages=[message for report in reports for message in report.messages],
            tool_calls=[call for report in reports for call in report.tool_calls],
            metrics={},
            events=self._events,
            pending_action=None,
            errors=[error for report in reports for error in report.errors],
        )

    def _normalize_results(self, results: list[StepOutput]) -> list[StepResult]:
        normalized: list[StepResult] = []
        for item in results:
            if isinstance(item, StepResult):
                normalized.append(item)
            elif isinstance(item, Report):
                normalized.append(StepResult(item, None))
            else:
                raise TypeError("Step must return Report or StepResult")
        return normalized

    def _emit_run_failed(self, report: Report) -> None:
        self.desk.emit(self._events, self._run_id, "run.failed", self.name, {})
        self.desk.run_store.update_run(
            self._run_id,
            "failed",
            report.content,
            to_json_value(report.data),
            None,
        )
        self.desk.unregister_flow_run(self._run_id)

    def _emit_run_completed(self, report: Report) -> None:
        self.desk.emit(self._events, self._run_id, "run.completed", self.name, {})
        self.desk.run_store.update_run(
            self._run_id,
            "completed",
            report.content,
            to_json_value(report.data),
            None,
        )
        self.desk.unregister_flow_run(self._run_id)

    def _pause_flow(
        self,
        report: Report,
        step_state: RunState,
        job: Job,
        step_index: int,
        all_reports: list[Report],
        stream_enabled: bool,
        context: WorkflowContext,
        continuations: list[WorkflowContinuation],
    ) -> Report:
        step_state.payload["stream"] = stream_enabled
        try:
            state = self._build_flow_state(
                job,
                step_index,
                all_reports,
                step_state,
                stream_enabled,
                context,
                continuations,
            )
        except (TypeError, ValueError) as exc:
            failed = report.model_copy(
                update={
                    "status": "failed",
                    "pending_action": None,
                    "errors": [*report.errors, f"Could not persist flow state: {exc}"],
                }
            )
            self._emit_run_failed(failed)
            return failed
        self.desk.emit(self._events, self._run_id, "run.paused", self.name, {})
        self.desk.run_store.update_run(
            self._run_id,
            "paused",
            report.content,
            None,
            state,
        )
        return report

    async def _execute_step_results(
        self,
        results: list[StepOutput],
        context: WorkflowContext,
        all_reports: list[Report],
        job: Job,
        stream_enabled: bool,
        step_index: int,
        remaining_continuations: list[WorkflowContinuation] | None = None,
    ) -> Report | None:
        normalized = self._normalize_results(results)
        for result_index, result in enumerate(normalized):
            report = result.report
            if report.status == "paused":
                if result.state is None:
                    failed = Report(
                        run_id=self._run_id,
                        status="failed",
                        content=None,
                        data=None,
                        messages=report.messages,
                        tool_calls=report.tool_calls,
                        metrics=report.metrics,
                        events=self._events,
                        pending_action=None,
                        errors=["Missing runner state"],
                    )
                    self.desk.emit(self._events, self._run_id, "run.failed", self.name, {})
                    self.desk.run_store.update_run(self._run_id, "failed", None, None, None)
                    self.desk.unregister_flow_run(self._run_id)
                    return failed
                continuations = list(result.continuations)
                tail = normalized[result_index + 1 :]
                if tail:
                    continuations.append(OutputContinuation(tuple(tail)))
                if remaining_continuations:
                    continuations.extend(remaining_continuations)
                return self._pause_flow(
                    report,
                    result.state,
                    job,
                    step_index,
                    all_reports,
                    stream_enabled,
                    context,
                    continuations,
                )
            if report.status == "failed":
                self._emit_run_failed(report)
                return report
            all_reports.append(report)
            context.outputs.append(report)
        return None

    async def _run_steps(
        self,
        steps: list[Any],
        context: WorkflowContext,
        all_reports: list[Report],
        job: Job,
        stream_enabled: bool,
        start_index: int = 0,
    ) -> Report | None:
        for step_index, step in enumerate(steps, start=start_index):
            results = await step.execute(self, context)
            report = await self._execute_step_results(
                results, context, all_reports, job, stream_enabled, step_index
            )
            if report is not None:
                return report
        return None

    def _finalize_flow_run(self, all_reports: list[Report]) -> Report:
        if not all_reports:
            empty_report = Report(
                run_id=self._run_id,
                status="failed",
                content=None,
                data=None,
                messages=[],
                tool_calls=[],
                metrics={},
                events=self._events,
                pending_action=None,
                errors=["No steps executed"],
            )
            self.desk.emit(self._events, self._run_id, "run.failed", self.name, {})
            self.desk.run_store.update_run(self._run_id, "failed", None, None, None)
            self.desk.unregister_flow_run(self._run_id)
            return empty_report

        if len(all_reports) > 1:
            final_report = self._combine_reports(all_reports)
        else:
            final_report = all_reports[0]
        self._emit_run_completed(final_report)
        return final_report

    async def _run(self, job: Job) -> Report:
        if self.desk.closed:
            raise RuntimeError("Desk is closed")
        self._run_id = new_id()
        self._events = []
        self._stream = self.desk.stream
        self.desk.run_store.create_run(self._run_id, job.model_dump(mode="json"))
        self.desk.register_flow_run(self._run_id, self)
        self.desk.emit(self._events, self._run_id, "run.started", self.name, {"job_id": job.id})
        context = WorkflowContext(job=job)
        all_reports: list[Report] = []

        report = await self._run_steps(self.steps, context, all_reports, job, self._stream)
        if report is not None:
            return report

        return self._finalize_flow_run(all_reports)

    def _resume_failure(self, state: RunState, error: str) -> Report:
        report = Report(
            run_id=state.run_id,
            status="failed",
            content=None,
            data=None,
            messages=state.messages,
            tool_calls=state.tool_calls,
            metrics=state.metrics,
            events=self._events,
            pending_action=None,
            errors=[error],
        )
        self._emit_run_failed(report)
        return report

    async def _resume(
        self,
        state: RunState,
        decision_or_input: Any,
        stream: bool | None = None,
    ) -> Report:
        self._run_id = state.run_id
        self._events = self.desk.run_store.get_events(state.run_id)
        payload = state.payload
        step_index = payload.get("step_index")
        step_state_payload = payload.get("step_state")
        if isinstance(step_index, bool) or not isinstance(step_index, int):
            return self._resume_failure(state, "Invalid flow state")
        if step_state_payload is None:
            return self._resume_failure(state, "Invalid flow state")
        if step_index < 0 or step_index >= len(self.steps):
            return self._resume_failure(state, "Invalid step index")

        graph = WorkflowGraph(tuple(self.steps))
        stored_signature = payload.get("flow_signature")
        if stored_signature is not None and stored_signature != graph.signature:
            return self._resume_failure(state, "Flow definition does not match stored state")
        if "continuations" not in payload and not isinstance(self.steps[step_index], Step):
            return self._resume_failure(
                state,
                "Stored flow state does not include composite continuation data",
            )
        try:
            all_reports = self._restore_outputs(payload.get("outputs"))
            context = WorkflowContext.restore(state.job, all_reports, payload.get("context"))
            continuations = restore_continuations(graph, payload.get("continuations", []))
            step_state = RunState.model_validate(step_state_payload)
        except (TypeError, ValueError) as exc:
            return self._resume_failure(state, f"Invalid flow state: {exc}")

        self.desk.register_flow_run(self._run_id, self)
        self.desk.emit(self._events, self._run_id, "run.resumed", self.name, {})

        payload_stream = payload.get("stream")
        if stream is None:
            stream_enabled = (
                payload_stream if isinstance(payload_stream, bool) else self.desk.stream
            )
        else:
            stream_enabled = stream
        self._stream = stream_enabled

        step_state.payload["stream"] = stream_enabled
        report, updated_state = await self._resume_runner(
            step_state,
            decision_or_input,
            stream_enabled,
        )
        if report.status == "paused":
            if updated_state is None:
                failed = Report(
                    run_id=self._run_id,
                    status="failed",
                    content=None,
                    data=None,
                    messages=report.messages,
                    tool_calls=report.tool_calls,
                    metrics=report.metrics,
                    events=self._events,
                    pending_action=None,
                    errors=["Missing runner state"],
                )
                self.desk.emit(self._events, self._run_id, "run.failed", self.name, {})
                self.desk.run_store.update_run(self._run_id, "failed", None, None, None)
                self.desk.unregister_flow_run(self._run_id)
                return failed
            return self._pause_flow(
                report,
                updated_state,
                state.job,
                step_index,
                all_reports,
                stream_enabled,
                context,
                continuations,
            )
        if report.status == "failed":
            self._emit_run_failed(report)
            return report

        all_reports.append(report)
        context.outputs.append(report)

        while continuations:
            continuation = continuations.pop(0)
            continuation_results = await continuation(self, context)
            continuation_report = await self._execute_step_results(
                continuation_results,
                context,
                all_reports,
                state.job,
                stream_enabled,
                step_index,
                continuations,
            )
            if continuation_report is not None:
                return continuation_report

        remaining_report = await self._run_steps(
            self.steps[step_index + 1 :],
            context,
            all_reports,
            state.job,
            stream_enabled,
            start_index=step_index + 1,
        )
        if remaining_report is not None:
            return remaining_report

        return self._finalize_flow_run(all_reports)

    async def arun(self, job: Job) -> Report:
        return await self._run(job)

    async def aresume(
        self,
        report: Report,
        decision_or_input: Any,
        *,
        stream: bool | None = None,
    ) -> Report:
        if self.desk.closed:
            raise RuntimeError("Desk is closed")
        record = self.desk.run_store.get_run(report.run_id)
        if record is None or record.state is None:
            failed = Report(
                run_id=report.run_id,
                status="failed",
                content=None,
                data=None,
                messages=report.messages,
                tool_calls=report.tool_calls,
                metrics=report.metrics,
                events=report.events,
                pending_action=None,
                errors=["No stored state"],
            )
            return failed
        if record.state.runner_type != "flow":
            failed = Report(
                run_id=report.run_id,
                status="failed",
                content=None,
                data=None,
                messages=report.messages,
                tool_calls=report.tool_calls,
                metrics=report.metrics,
                events=report.events,
                pending_action=None,
                errors=["Run is not a flow"],
            )
            return failed
        return await self._resume(record.state, decision_or_input, stream=stream)

    def run(self, job: Job) -> Report:
        ensure_not_running_loop("Flow.run", "Flow.arun")
        return asyncio.run(self._run(job))

    def resume(
        self,
        report: Report,
        decision_or_input: Any,
        *,
        stream: bool | None = None,
    ) -> Report:
        ensure_not_running_loop("Flow.resume", "Flow.aresume")
        return asyncio.run(self.aresume(report, decision_or_input, stream=stream))
