from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from blackgeorge.core.tool_call import ToolCall
from blackgeorge.core.types import PendingActionType


class PendingAction(BaseModel):
    action_id: str
    type: PendingActionType
    tool_call: ToolCall
    prompt: str
    options: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(frozen=True)
