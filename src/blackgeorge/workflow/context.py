from dataclasses import dataclass, field
from typing import Any

from blackgeorge.core.job import Job
from blackgeorge.core.report import Report


@dataclass
class WorkflowContext:
    job: Job
    outputs: list[Report] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
    _loop_iterations: dict[str, int] = field(default_factory=dict)

    def loop_iteration(self, loop_name: str) -> int:
        return self._loop_iterations.get(loop_name, 0)

    def set_loop_iteration(self, loop_name: str, iteration: int) -> None:
        self._loop_iterations[loop_name] = iteration

    def increment_loop_iteration(self, loop_name: str) -> int:
        current = self._loop_iterations.get(loop_name, 0)
        current += 1
        self._loop_iterations[loop_name] = current
        return current
