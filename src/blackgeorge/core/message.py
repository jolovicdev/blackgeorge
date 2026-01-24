from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from blackgeorge.core.tool_call import ToolCall
from blackgeorge.core.types import MessageRole


class Message(BaseModel):
    role: MessageRole
    content: str
    reasoning_content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)
