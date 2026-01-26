import asyncio
from collections.abc import Callable
from typing import Any, Protocol

from blackgeorge.core.job import Job
from blackgeorge.workflow.context import WorkflowContext
from blackgeorge.workflow.result import StepOutput, StepResult


class Executable(Protocol):
    async def execute(self, flow: Any, context: WorkflowContext) -> list[StepOutput]: ...


class Step:
    def __init__(
        self,
        runner: Any,
        name: str | None = None,
        job_builder: Callable[[WorkflowContext], Job] | None = None,
    ) -> None:
        self.runner = runner
        self.name = name or getattr(runner, "name", "step")
        self.job_builder = job_builder

    async def execute(self, flow: Any, context: WorkflowContext) -> list[StepOutput]:
        job = self.job_builder(context) if self.job_builder else context.job
        flow.emit("step.started", self.name, {})
        report, state = await flow._run_runner(self.runner, job)
        if state is not None and report.status == "paused":
            flow.emit("step.paused", self.name, {"status": report.status})
            return [StepResult(report, state)]
        flow.emit("step.completed", self.name, {"status": report.status})
        return [StepResult(report, state)]


def _should_stop(outputs: list[StepOutput]) -> bool:
    for item in outputs:
        report = item.report if isinstance(item, StepResult) else item
        if report.status in ("paused", "failed"):
            return True
    return False


class Parallel:
    def __init__(self, *steps: Executable) -> None:
        self.steps = list(steps)

    async def execute(self, flow: Any, context: WorkflowContext) -> list[StepOutput]:
        async def run_step(step: Executable) -> list[StepOutput]:
            return await step.execute(flow, context)

        results = await asyncio.gather(*(run_step(step) for step in self.steps))
        return [report for group in results for report in group]


class Condition:
    def __init__(
        self,
        predicate: Callable[[WorkflowContext], bool],
        if_true: list[Executable],
        if_false: list[Executable] | None = None,
    ) -> None:
        self.predicate = predicate
        self.if_true = if_true
        self.if_false = if_false or []

    async def execute(self, flow: Any, context: WorkflowContext) -> list[StepOutput]:
        steps = self.if_true if self.predicate(context) else self.if_false
        reports: list[StepOutput] = []
        for step in steps:
            step_outputs = await step.execute(flow, context)
            reports.extend(step_outputs)
            if _should_stop(step_outputs):
                return reports
        return reports


class Router:
    def __init__(
        self,
        selector: Callable[[WorkflowContext], str],
        routes: dict[str, list[Executable]],
    ) -> None:
        self.selector = selector
        self.routes = routes

    async def execute(self, flow: Any, context: WorkflowContext) -> list[StepOutput]:
        key = self.selector(context)
        steps = self.routes.get(key, [])
        reports: list[StepOutput] = []
        for step in steps:
            step_outputs = await step.execute(flow, context)
            reports.extend(step_outputs)
            if _should_stop(step_outputs):
                return reports
        return reports


class Loop:
    def __init__(
        self,
        steps: list[Executable],
        stop: Callable[[WorkflowContext], bool],
        max_iterations: int = 10,
    ) -> None:
        self.steps = steps
        self.stop = stop
        self.max_iterations = max_iterations

    async def execute(self, flow: Any, context: WorkflowContext) -> list[StepOutput]:
        reports: list[StepOutput] = []
        iteration = 0
        while iteration < self.max_iterations:
            iteration += 1
            for step in self.steps:
                step_outputs = await step.execute(flow, context)
                reports.extend(step_outputs)
                if _should_stop(step_outputs):
                    return reports
            if self.stop(context):
                break
        return reports
