from dataclasses import dataclass, field
from typing import Any

from blackgeorge.core.job import Job
from blackgeorge.core.report import Report


@dataclass
class WorkflowContext:
    job: Job
    outputs: list[Report] = field(default_factory=list)
    artifacts: dict[str, Any] = field(default_factory=dict)
