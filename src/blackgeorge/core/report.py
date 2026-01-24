from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from blackgeorge.core.event import Event
from blackgeorge.core.message import Message
from blackgeorge.core.pending_action import PendingAction
from blackgeorge.core.tool_call import ToolCall
from blackgeorge.core.types import RunStatus


class Report(BaseModel):
    run_id: str
    status: RunStatus
    pending_action: PendingAction | None = None
    content: str | None = None
    reasoning_content: str | None = None
    data: Any | None = None
    messages: list[Message] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    events: list[Event] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)
