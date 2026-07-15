import json
from dataclasses import dataclass, field
from typing import Any

from blackgeorge.core.job import Job
from blackgeorge.core.report import Report
from blackgeorge.core.serialization import to_json_value


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

    def snapshot(self) -> dict[str, Any]:
        artifacts = to_json_value(self.artifacts)
        try:
            json.dumps(artifacts, ensure_ascii=True)
        except (TypeError, ValueError) as exc:
            raise TypeError("Workflow context artifacts must be JSON-serializable") from exc
        return {
            "artifacts": artifacts,
            "loop_iterations": dict(self._loop_iterations),
        }

    @classmethod
    def restore(
        cls,
        job: Job,
        outputs: list[Report],
        payload: Any,
    ) -> "WorkflowContext":
        if payload is None:
            return cls(job=job, outputs=list(outputs))
        if not isinstance(payload, dict):
            raise ValueError("Invalid workflow context")
        artifacts = payload.get("artifacts", {})
        loop_iterations = payload.get("loop_iterations", {})
        if not isinstance(artifacts, dict):
            raise ValueError("Invalid workflow context artifacts")
        if not isinstance(loop_iterations, dict) or any(
            not isinstance(name, str)
            or isinstance(iteration, bool)
            or not isinstance(iteration, int)
            or iteration < 0
            for name, iteration in loop_iterations.items()
        ):
            raise ValueError("Invalid workflow loop iterations")
        return cls(
            job=job,
            outputs=list(outputs),
            artifacts=dict(artifacts),
            _loop_iterations=dict(loop_iterations),
        )
