from typing import Literal

MessageRole = Literal["system", "user", "assistant", "tool"]
PendingActionType = Literal["confirmation", "user_input"]
RunStatus = Literal["completed", "paused", "failed", "running"]
WorkforceMode = Literal["managed", "collaborate"]
