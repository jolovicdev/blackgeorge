from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from blackgeorge.core.job import Job
from blackgeorge.core.message import Message
from blackgeorge.core.pending_action import PendingAction
from blackgeorge.core.tool_call import ToolCall
from blackgeorge.core.types import RunStatus


class RunState(BaseModel):
    run_id: str
    status: RunStatus
    runner_type: str
    runner_name: str
    job: Job
    messages: list[Message] = Field(default_factory=list)
    tool_calls: list[ToolCall] = Field(default_factory=list)
    pending_action: PendingAction | None = None
    metrics: dict[str, Any] = Field(default_factory=dict)
    iteration: int = 0
    payload: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)
