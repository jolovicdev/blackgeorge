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


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    schema: dict[str, Any]
    callable: Callable[..., Any]
    input_model: type[BaseModel]
    requires_confirmation: bool = False
    requires_user_input: bool = False
    external_execution: bool = False
    pre: tuple[ToolPreHook, ...] = ()
    post: tuple[ToolPostHook, ...] = ()
    confirmation_prompt: str | None = None
    user_input_prompt: str | None = None
    input_key: str | None = None
    timeout: float | None = None
    retries: int = 0
    retry_delay: float = 1.0
