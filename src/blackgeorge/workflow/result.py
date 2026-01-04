from dataclasses import dataclass

from blackgeorge.core.report import Report
from blackgeorge.store.state import RunState


@dataclass(frozen=True)
class StepResult:
    report: Report
    state: RunState | None


type StepOutput = Report | StepResult
