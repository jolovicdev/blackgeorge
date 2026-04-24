import asyncio
from typing import Any, cast

from pydantic import BaseModel

from blackgeorge.async_utils import ensure_not_running_loop
from blackgeorge.config import RunConfig
from blackgeorge.core.event import Event
from blackgeorge.core.job import Job
from blackgeorge.core.report import Report
from blackgeorge.store.state import RunState
from blackgeorge.utils import new_id
from blackgeorge.worker import Worker
from blackgeorge.workflow.context import WorkflowContext
from blackgeorge.workflow.result import StepOutput, StepResult
from blackgeorge.workforce import Workforce


class Flow:
    def __init__(self, desk: Any, steps: list[Any], name: str | None = None) -> None:
        self.desk = desk
        self.steps = steps
        self.name = name or "flow"
        self._run_id = ""
        self._events: list[Event] = []
        self._stream: bool = False

    def emit(self, event_type: str, source: str, payload: dict[str, Any]) -> None:
        if not self._run_id:
            return
        self.desk._emit(self._events, self._run_id, event_type, source, payload)

    def _output_json(self, report: Report) -> Any | None:
        if isinstance(report.data, BaseModel):
            return report.data.model_dump(mode="json", warnings=False)
        return report.data

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
            default_model=self.desk.model,
        )

    async def _run_runner(
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
    ) -> RunState:
        payload = {
            "step_index": step_index,
            "outputs": [report.model_dump(mode="json") for report in reports],
            "step_state": step_state.model_dump(mode="json"),
            "stream": stream,
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
        if not isinstance(payload, list):
            return []
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
        self.desk._emit(self._events, self._run_id, "run.failed", self.name, {})
        self.desk.run_store.update_run(
            self._run_id,
            "failed",
            report.content,
            self._output_json(report),
            None,
        )
        self.desk.unregister_flow_run(self._run_id)

    def _emit_run_completed(self, report: Report) -> None:
        self.desk._emit(self._events, self._run_id, "run.completed", self.name, {})
        self.desk.run_store.update_run(
            self._run_id,
            "completed",
            report.content,
            self._output_json(report),
            None,
        )
        self.desk.unregister_flow_run(self._run_id)

    async def _execute_step_results(
        self,
        results: list[StepOutput],
        context: WorkflowContext,
        all_reports: list[Report],
        job: Job,
        stream_enabled: bool,
        step_index: int,
    ) -> Report | None:
        for result in self._normalize_results(results):
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
                    self.desk._emit(self._events, self._run_id, "run.failed", self.name, {})
                    self.desk.run_store.update_run(self._run_id, "failed", None, None, None)
                    self.desk.unregister_flow_run(self._run_id)
                    return failed
                self.desk._emit(self._events, self._run_id, "run.paused", self.name, {})
                result.state.payload["stream"] = stream_enabled
                state = self._build_flow_state(
                    job,
                    step_index,
                    all_reports,
                    result.state,
                    stream_enabled,
                )
                self.desk.run_store.update_run(
                    self._run_id,
                    "paused",
                    report.content,
                    None,
                    state,
                )
                return report
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
            self.desk._emit(self._events, self._run_id, "run.failed", self.name, {})
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
        self._run_id = new_id()
        self._events = []
        self._stream = self.desk.stream
        self.desk.run_store.create_run(self._run_id, job.model_dump(mode="json"))
        self.desk.register_flow_run(self._run_id, self)
        self.desk._emit(self._events, self._run_id, "run.started", self.name, {"job_id": job.id})
        context = WorkflowContext(job=job)
        all_reports: list[Report] = []

        report = await self._run_steps(self.steps, context, all_reports, job, self._stream)
        if report is not None:
            return report

        return self._finalize_flow_run(all_reports)

    async def _resume(
        self,
        state: RunState,
        decision_or_input: Any,
        stream: bool | None = None,
    ) -> Report:
        payload = state.payload
        step_index = payload.get("step_index")
        step_state_payload = payload.get("step_state")
        if not isinstance(step_index, int) or step_state_payload is None:
            events = self.desk.run_store.get_events(state.run_id)
            failed = Report(
                run_id=state.run_id,
                status="failed",
                content=None,
                data=None,
                messages=state.messages,
                tool_calls=state.tool_calls,
                metrics=state.metrics,
                events=events,
                pending_action=None,
                errors=["Invalid flow state"],
            )
            return failed
        if step_index < 0 or step_index >= len(self.steps):
            events = self.desk.run_store.get_events(state.run_id)
            failed = Report(
                run_id=state.run_id,
                status="failed",
                content=None,
                data=None,
                messages=state.messages,
                tool_calls=state.tool_calls,
                metrics=state.metrics,
                events=events,
                pending_action=None,
                errors=["Invalid step index"],
            )
            return failed

        self._run_id = state.run_id
        self._events = self.desk.run_store.get_events(state.run_id)
        self.desk.register_flow_run(self._run_id, self)
        self.desk._emit(self._events, self._run_id, "run.resumed", self.name, {})

        payload_stream = payload.get("stream")
        if stream is None:
            stream_enabled = (
                payload_stream if isinstance(payload_stream, bool) else self.desk.stream
            )
        else:
            stream_enabled = stream
        self._stream = stream_enabled

        context = WorkflowContext(job=state.job)
        all_reports = self._restore_outputs(payload.get("outputs"))
        context.outputs.extend(all_reports)

        step_state = RunState.model_validate(step_state_payload)
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
                self.desk._emit(self._events, self._run_id, "run.failed", self.name, {})
                self.desk.run_store.update_run(self._run_id, "failed", None, None, None)
                self.desk.unregister_flow_run(self._run_id)
                return failed
            self.desk._emit(self._events, self._run_id, "run.paused", self.name, {})
            updated_state.payload["stream"] = stream_enabled
            flow_state = self._build_flow_state(
                state.job,
                step_index,
                all_reports,
                updated_state,
                stream_enabled,
            )
            self.desk.run_store.update_run(
                self._run_id,
                "paused",
                report.content,
                None,
                flow_state,
            )
            return report
        if report.status == "failed":
            self._emit_run_failed(report)
            return report

        all_reports.append(report)
        context.outputs.append(report)

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
