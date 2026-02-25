import copy
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from blackgeorge.adapters.base import BaseModelAdapter
from blackgeorge.core.event import Event

EventEmitter = Callable[[str, str, dict[str, Any]], None]


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
    respect_context_window: bool = True
    default_model: str | None = None

    def with_overrides(self, **kwargs: Any) -> "RunConfig":
        current = copy.deepcopy(self.__dict__)
        current.update(kwargs)
        return RunConfig(**current)

    def model_name(self, worker_model: str | None) -> str:
        return worker_model or self.default_model or ""
