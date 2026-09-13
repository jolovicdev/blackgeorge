from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from blackgeorge.core.report import Report
from blackgeorge.store.state import RunState

if TYPE_CHECKING:
    from blackgeorge.workflow.context import WorkflowContext
    from blackgeorge.workflow.flow import Flow


@dataclass(frozen=True)
class StepResult:
    report: Report
    state: RunState | None
    continuations: tuple["WorkflowContinuation", ...] = ()


type StepOutput = Report | StepResult


class WorkflowContinuation(Protocol):
    async def __call__(self, flow: "Flow", context: "WorkflowContext") -> list[StepOutput]: ...
