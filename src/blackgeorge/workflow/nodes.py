import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from inspect import isawaitable
from itertools import count
from typing import Any, Protocol, cast

from blackgeorge.core.event_types import EventType
from blackgeorge.core.job import Job
from blackgeorge.core.report import Report
from blackgeorge.store.state import RunState
from blackgeorge.workflow.context import WorkflowContext
from blackgeorge.workflow.result import (
    StepOutput,
    StepResult,
    WorkflowContinuation,
)


class Executable(Protocol):
    async def execute(self, flow: Any, context: WorkflowContext) -> list[StepOutput]: ...


def report_for(output: StepOutput) -> Report:
    return output.report if isinstance(output, StepResult) else output


def should_stop(outputs: list[StepOutput]) -> bool:
    return any(report_for(output).status in ("paused", "failed") for output in outputs)


@dataclass(frozen=True)
class SequenceContinuation:
    steps: tuple[Executable, ...]

    async def __call__(self, flow: Any, context: WorkflowContext) -> list[StepOutput]:
        return await execute_sequence(flow, context, self.steps)


@dataclass(frozen=True)
class OutputContinuation:
    outputs: tuple[StepOutput, ...]

    async def __call__(self, flow: Any, context: Any) -> list[StepOutput]:
        return list(self.outputs)


def defer_after_pause(
    outputs: list[StepOutput],
    *continuations: WorkflowContinuation,
) -> list[StepOutput]:
    for index, output in enumerate(outputs):
        report = report_for(output)
        if report.status != "paused":
            continue
        result = output if isinstance(output, StepResult) else StepResult(report, None)
        deferred = list(result.continuations)
        if tail := outputs[index + 1 :]:
            deferred.append(OutputContinuation(tuple(tail)))
        deferred.extend(continuations)
        return [*outputs[:index], replace(result, continuations=tuple(deferred))]
    return outputs


async def execute_sequence(
    flow: Any,
    context: WorkflowContext,
    steps: tuple[Executable, ...] | list[Executable],
) -> list[StepOutput]:
    outputs: list[StepOutput] = []
    for index, step in enumerate(steps):
        step_outputs = await step.execute(flow, context)
        outputs.extend(step_outputs)
        if not should_stop(step_outputs):
            continue
        if any(report_for(output).status == "paused" for output in step_outputs):
            remaining = tuple(steps[index + 1 :])
            if remaining:
                return defer_after_pause(outputs, SequenceContinuation(remaining))
            return defer_after_pause(outputs)
        return outputs
    return outputs


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
        flow.emit(EventType.STEP_STARTED, self.name, {})
        report, state = await flow.run_runner(self.runner, job)
        event = (
            EventType.STEP_PAUSED
            if state is not None and report.status == "paused"
            else EventType.STEP_COMPLETED
        )
        flow.emit(event, self.name, {"status": report.status})
        return [StepResult(report, state)]


class Parallel:
    def __init__(self, *steps: Executable) -> None:
        if not steps:
            raise ValueError("Parallel requires at least one step")
        self.steps = tuple(steps)

    async def execute(self, flow: Any, context: WorkflowContext) -> list[StepOutput]:
        groups = await asyncio.gather(*(step.execute(flow, context) for step in self.steps))
        return defer_after_pause([output for group in groups for output in group])


class Condition:
    def __init__(
        self,
        predicate: Callable[[WorkflowContext], bool | Awaitable[bool]],
        if_true: list[Executable],
        if_false: list[Executable] | None = None,
    ) -> None:
        self.predicate = predicate
        self.if_true = tuple(if_true)
        self.if_false = tuple(if_false or [])

    async def execute(self, flow: Any, context: WorkflowContext) -> list[StepOutput]:
        selected = self.predicate(context)
        if isawaitable(selected):
            selected = await cast(Awaitable[bool], selected)
        steps = self.if_true if selected else self.if_false
        return await execute_sequence(flow, context, steps)


class AsyncCondition(Condition):
    pass


class Router:
    def __init__(
        self,
        selector: Callable[[WorkflowContext], str],
        routes: dict[str, list[Executable]],
    ) -> None:
        self.selector = selector
        self.routes = {key: tuple(steps) for key, steps in routes.items()}

    async def execute(self, flow: Any, context: WorkflowContext) -> list[StepOutput]:
        return await execute_sequence(flow, context, self.routes.get(self.selector(context), ()))


class Loop:
    counter = count(1)
    name_prefix = "loop"

    def __init__(
        self,
        steps: list[Executable],
        stop: Callable[[WorkflowContext], bool | Awaitable[bool]],
        max_iterations: int = 10,
        name: str | None = None,
    ) -> None:
        if not steps:
            raise ValueError("Loop requires at least one step")
        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if name is not None and not name.strip():
            raise ValueError("Loop name must not be empty")
        self.steps = tuple(steps)
        self.stop = stop
        self.max_iterations = max_iterations
        self._explicit_name = name is not None
        self.name = name if name is not None else f"{self.name_prefix}_{next(self.counter)}"

    async def execute(self, flow: Any, context: WorkflowContext) -> list[StepOutput]:
        context.set_loop_iteration(self.name, 0)
        return await self.run_iterations(flow, context, 0, False)

    async def run_iterations(
        self,
        flow: Any,
        context: WorkflowContext,
        start_index: int,
        iteration_started: bool,
    ) -> list[StepOutput]:
        outputs: list[StepOutput] = []
        while iteration_started or context.loop_iteration(self.name) < self.max_iterations:
            if not iteration_started:
                context.increment_loop_iteration(self.name)
            iteration_started = True
            for index in range(start_index, len(self.steps)):
                step_outputs = await self.steps[index].execute(flow, context)
                outputs.extend(step_outputs)
                if not should_stop(step_outputs):
                    continue
                if any(report_for(output).status == "paused" for output in step_outputs):
                    continuation = LoopContinuation(self, index + 1)
                    return defer_after_pause(outputs, continuation)
                return outputs
            stopped = self.stop(context)
            if isawaitable(stopped):
                stopped = await cast(Awaitable[bool], stopped)
            if stopped:
                break
            start_index = 0
            iteration_started = False
        return outputs


@dataclass(frozen=True)
class LoopContinuation:
    loop: Loop
    start_index: int
    previous_name: str | None = None

    async def __call__(self, flow: Any, context: WorkflowContext) -> list[StepOutput]:
        if self.previous_name is not None and self.previous_name != self.loop.name:
            iteration = context.loop_iteration(self.previous_name)
            context.set_loop_iteration(self.loop.name, iteration)
        return await self.loop.run_iterations(flow, context, self.start_index, True)


class AsyncLoop(Loop):
    counter = count(1)
    name_prefix = "async_loop"


type StepPath = tuple[str | int, ...]


def executable_entries(
    steps: Sequence[Executable],
    prefix: StepPath = (),
) -> Iterator[tuple[StepPath, Executable]]:
    for index, step in enumerate(steps):
        path = (*prefix, index)
        yield path, step
        if isinstance(step, Parallel):
            yield from executable_entries(step.steps, (*path, "parallel"))
        elif isinstance(step, Condition):
            yield from executable_entries(step.if_true, (*path, "true"))
            yield from executable_entries(step.if_false, (*path, "false"))
        elif isinstance(step, Router):
            for route, route_steps in step.routes.items():
                yield from executable_entries(route_steps, (*path, "route", route))
        elif isinstance(step, Loop):
            yield from executable_entries(step.steps, (*path, "loop"))


def executable_shape(step: Executable) -> dict[str, Any]:
    qualified_type = f"{type(step).__module__}.{type(step).__qualname__}"
    if isinstance(step, Step):
        runner_type = f"{type(step.runner).__module__}.{type(step.runner).__qualname__}"
        return {
            "type": qualified_type,
            "name": step.name,
            "runner_type": runner_type,
            "runner_name": getattr(step.runner, "name", None),
        }
    if isinstance(step, Parallel):
        return {
            "type": qualified_type,
            "steps": [executable_shape(child) for child in step.steps],
        }
    if isinstance(step, Condition):
        return {
            "type": qualified_type,
            "true": [executable_shape(child) for child in step.if_true],
            "false": [executable_shape(child) for child in step.if_false],
        }
    if isinstance(step, Router):
        return {
            "type": qualified_type,
            "routes": {
                route: [executable_shape(child) for child in route_steps]
                for route, route_steps in step.routes.items()
            },
        }
    if isinstance(step, Loop):
        return {
            "type": qualified_type,
            "name": step.name if step._explicit_name else None,
            "max_iterations": step.max_iterations,
            "steps": [executable_shape(child) for child in step.steps],
        }
    return {"type": qualified_type}


@dataclass(frozen=True)
class WorkflowGraph:
    steps: tuple[Executable, ...]

    @property
    def signature(self) -> str:
        shape = [executable_shape(step) for step in self.steps]
        encoded = json.dumps(shape, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()

    def path_for(self, target: Executable) -> StepPath:
        paths = [path for path, step in executable_entries(self.steps) if step is target]
        if not paths:
            raise ValueError("Workflow continuation references a step outside the flow")
        if len(paths) > 1:
            raise ValueError("Workflow step instances cannot be reused across continuation paths")
        return paths[0]

    def resolve(self, path: StepPath) -> Executable:
        for candidate, step in executable_entries(self.steps):
            if candidate == path:
                return step
        raise ValueError("Workflow continuation path does not exist in this flow")


def serialize_step_output(graph: WorkflowGraph, output: StepOutput) -> dict[str, Any]:
    result = output if isinstance(output, StepResult) else StepResult(output, None)
    return {
        "report": result.report.model_dump(mode="json", warnings=False),
        "state": (
            result.state.model_dump(mode="json", warnings=False)
            if result.state is not None
            else None
        ),
        "continuations": serialize_continuations(graph, result.continuations),
    }


def restore_step_output(graph: WorkflowGraph, payload: Any) -> StepResult:
    if not isinstance(payload, dict):
        raise ValueError("Invalid workflow continuation output")
    report = Report.model_validate(payload.get("report"))
    state_payload = payload.get("state")
    state = RunState.model_validate(state_payload) if state_payload is not None else None
    continuations = restore_continuations(graph, payload.get("continuations", []))
    return StepResult(report, state, tuple(continuations))


def serialize_continuation(
    graph: WorkflowGraph,
    continuation: WorkflowContinuation,
) -> dict[str, Any]:
    if isinstance(continuation, SequenceContinuation):
        return {
            "kind": "sequence",
            "paths": [list(graph.path_for(step)) for step in continuation.steps],
        }
    if isinstance(continuation, LoopContinuation):
        return {
            "kind": "loop",
            "path": list(graph.path_for(continuation.loop)),
            "start_index": continuation.start_index,
            "loop_name": continuation.loop.name,
        }
    if isinstance(continuation, OutputContinuation):
        return {
            "kind": "outputs",
            "outputs": [serialize_step_output(graph, output) for output in continuation.outputs],
        }
    raise TypeError(f"Unsupported workflow continuation: {type(continuation).__name__}")


def serialize_continuations(
    graph: WorkflowGraph,
    continuations: Sequence[WorkflowContinuation],
) -> list[dict[str, Any]]:
    return [serialize_continuation(graph, continuation) for continuation in continuations]


def parse_step_path(payload: Any) -> StepPath:
    if not isinstance(payload, list):
        raise ValueError("Invalid workflow continuation path")
    if any(isinstance(part, bool) or not isinstance(part, (str, int)) for part in payload):
        raise ValueError("Invalid workflow continuation path")
    return tuple(payload)


def restore_continuation(
    graph: WorkflowGraph,
    payload: Any,
) -> WorkflowContinuation:
    if not isinstance(payload, dict):
        raise ValueError("Invalid workflow continuation")
    kind = payload.get("kind")
    if kind == "sequence":
        paths = payload.get("paths")
        if not isinstance(paths, list):
            raise ValueError("Invalid sequence continuation")
        return SequenceContinuation(tuple(graph.resolve(parse_step_path(path)) for path in paths))
    if kind == "loop":
        loop = graph.resolve(parse_step_path(payload.get("path")))
        start_index = payload.get("start_index")
        loop_name = payload.get("loop_name")
        if (
            not isinstance(loop, Loop)
            or isinstance(start_index, bool)
            or not isinstance(start_index, int)
            or not isinstance(loop_name, str)
        ):
            raise ValueError("Invalid loop continuation")
        if start_index < 0 or start_index > len(loop.steps):
            raise ValueError("Invalid loop continuation index")
        return LoopContinuation(loop, start_index, loop_name)
    if kind == "outputs":
        outputs = payload.get("outputs")
        if not isinstance(outputs, list):
            raise ValueError("Invalid output continuation")
        return OutputContinuation(tuple(restore_step_output(graph, output) for output in outputs))
    raise ValueError("Unknown workflow continuation type")


def restore_continuations(
    graph: WorkflowGraph,
    payload: Any,
) -> list[WorkflowContinuation]:
    if not isinstance(payload, list):
        raise ValueError("Invalid workflow continuations")
    return [restore_continuation(graph, continuation) for continuation in payload]
