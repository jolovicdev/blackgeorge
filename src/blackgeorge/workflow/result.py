from dataclasses import dataclass
from typing import Any, Protocol

from blackgeorge.core.report import Report
from blackgeorge.store.state import RunState


@dataclass(frozen=True)
class StepResult:
    report: Report
    state: RunState | None
    continuations: tuple["WorkflowContinuation", ...] = ()


type StepOutput = Report | StepResult


class WorkflowContinuation(Protocol):
    async def __call__(self, flow: Any, context: Any) -> list[StepOutput]: ...
