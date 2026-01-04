from typing import Any

from blackgeorge.core.pending_action import PendingAction
from blackgeorge.core.tool_call import ToolCall
from blackgeorge.core.types import PendingActionType
from blackgeorge.tools.base import Tool


def tool_prompt(tool: Tool, action_type: str, tool_call: ToolCall) -> str:
    if action_type == "user_input":
        question = tool_call.arguments.get("question")
        if isinstance(question, str) and question.strip():
            return question
        return tool.user_input_prompt or tool.description or f"Provide input for {tool.name}"
    base = tool.confirmation_prompt or tool.description or f"Confirm {tool.name}"
    path = tool_call.arguments.get("path")
    if isinstance(path, str) and path.strip():
        return f"{base} ({path})"
    return base


def tool_action_type(tool: Tool) -> PendingActionType | None:
    if tool.requires_confirmation:
        return "confirmation"
    if tool.requires_user_input:
        return "user_input"
    return None


def pending_options(action_type: PendingActionType) -> list[str]:
    if action_type == "confirmation":
        return ["yes", "no"]
    return []


def resume_argument_key(pending_action: PendingAction) -> str:
    value = pending_action.metadata.get("input_key")
    return value if isinstance(value, str) else "user_input"


def update_arguments(call: ToolCall, key: str, value: Any) -> ToolCall:
    updated = dict(call.arguments)
    updated[key] = value
    return call.model_copy(update={"arguments": updated})
