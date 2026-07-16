from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from blackgeorge.adapters.base import BaseModelAdapter
from blackgeorge.core.event import Event

EventEmitter = Callable[[str, str, dict[str, Any]], None]


def validate_execution_limits(
    *,
    max_tokens: int | None,
    structured_output_retries: int,
    max_iterations: int,
    max_tool_calls: int,
    num_retries: int,
    max_context_messages: int | None,
    max_cost_usd: float | None,
) -> None:
    if max_tokens is not None and max_tokens < 1:
        raise ValueError("max_tokens must be >= 1")
    if structured_output_retries < 0:
        raise ValueError("structured_output_retries must be >= 0")
    if max_iterations < 1:
        raise ValueError("max_iterations must be >= 1")
    if max_tool_calls < 0:
        raise ValueError("max_tool_calls must be >= 0")
    if num_retries < 0:
        raise ValueError("num_retries must be >= 0")
    if max_context_messages is not None and max_context_messages < 1:
        raise ValueError("max_context_messages must be >= 1")
    if max_cost_usd is not None and max_cost_usd < 0:
        raise ValueError("max_cost_usd must be >= 0")


@dataclass(frozen=True)
class RunConfig:
    adapter: BaseModelAdapter
    emit: EventEmitter
    run_id: str
    events: list[Event] = field(default_factory=list)
    temperature: float | None = None
    max_tokens: int | None = None
    stream: bool = False
    stream_options: dict[str, Any] | None = None
    structured_output_retries: int = 3
    max_iterations: int = 10
    max_tool_calls: int = 20
    num_retries: int = 0
    respect_context_window: bool = True
    max_context_messages: int | None = None
    max_cost_usd: float | None = None
    default_model: str | None = None
    usage_totals: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_execution_limits(
            max_tokens=self.max_tokens,
            structured_output_retries=self.structured_output_retries,
            max_iterations=self.max_iterations,
            max_tool_calls=self.max_tool_calls,
            num_retries=self.num_retries,
            max_context_messages=self.max_context_messages,
            max_cost_usd=self.max_cost_usd,
        )

    def with_overrides(self, **kwargs: Any) -> "RunConfig":
        current = dict(self.__dict__)
        if "events" not in kwargs and current.get("events") is not None:
            current["events"] = list(current["events"])
        if "stream_options" not in kwargs and current.get("stream_options") is not None:
            current["stream_options"] = dict(current["stream_options"])
        current.update(kwargs)
        return RunConfig(**current)

    def model_name(self, worker_model: str | None) -> str:
        return worker_model or self.default_model or ""
