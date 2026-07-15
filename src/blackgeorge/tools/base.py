from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel

from blackgeorge.core.tool_call import ToolCall

ToolPreHook = Callable[[ToolCall], Any]
ToolPostHook = Callable[[ToolCall, "ToolResult"], Any]
ProgressCallback = Callable[[str], None]


@dataclass(frozen=True)
class ToolResult:
    content: str | None = None
    data: Any | None = None
    error: str | None = None
    timed_out: bool = False
    cancelled: bool = False
    exception_type: str | None = None


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    schema: dict[str, Any]
    callable: Callable[..., Any]
    input_model: type[BaseModel]
    requires_confirmation: bool = False
    requires_user_input: bool = False
    requires_handoff: bool = False
    external_execution: bool = False
    pre: tuple[ToolPreHook, ...] = ()
    post: tuple[ToolPostHook, ...] = ()
    confirmation_prompt: str | None = None
    user_input_prompt: str | None = None
    input_key: str | None = None
    timeout: float | None = None
    retries: int = 0
    retry_delay: float = 1.0
    output_type: type[BaseModel] | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Tool name must not be empty")
        if sum((self.requires_confirmation, self.requires_user_input, self.requires_handoff)) > 1:
            raise ValueError("A tool can require only one interactive action")
        if self.timeout is not None and self.timeout <= 0:
            raise ValueError("Tool timeout must be greater than zero")
        if self.retries < 0:
            raise ValueError("Tool retries must be non-negative")
        if self.retry_delay < 0:
            raise ValueError("Tool retry_delay must be non-negative")
        object.__setattr__(self, "pre", tuple(self.pre))
        object.__setattr__(self, "post", tuple(self.post))
