from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RunStartedPayload:
    job_id: str


@dataclass(frozen=True)
class RunFailedPayload:
    errors: list[str]


@dataclass(frozen=True)
class WorkerPausedPayload:
    pending_action_type: str


@dataclass(frozen=True)
class WorkerFailedPayload:
    error: str


@dataclass(frozen=True)
class WorkerContextSummarizedPayload:
    model: str
    summarized_messages: int
    kept_messages: int
    unregistered_model: bool = False
    registration_hint: str | None = None


@dataclass(frozen=True)
class ToolStartedPayload:
    tool_call_id: str


@dataclass(frozen=True)
class ToolCompletedPayload:
    tool_call_id: str
    result_preview: str | None = None
    result_truncated: bool = False
    timed_out: bool = False
    cancelled: bool = False


@dataclass(frozen=True)
class ToolFailedPayload:
    tool_call_id: str
    error: str


@dataclass(frozen=True)
class StreamTokenPayload:
    token: str


@dataclass(frozen=True)
class AssistantMessagePayload:
    content: str
    tool_calls: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class LLMCompletedPayload:
    model: str
    latency_ms: int
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float | None = None


@dataclass(frozen=True)
class LLMFailedPayload:
    model: str
    latency_ms: int
    error_type: str
    error_message: str


@dataclass(frozen=True)
class StepCompletedPayload:
    status: str


@dataclass(frozen=True)
class StepPausedPayload:
    status: str


@dataclass(frozen=True)
class WorkforcePausedPayload:
    root_job: dict[str, Any]
    completed_reports: list[dict[str, Any]]
    pending_worker_index: int
