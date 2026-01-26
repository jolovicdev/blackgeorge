from collections.abc import Callable
from typing import Any

from blackgeorge.tools.base import Tool
from blackgeorge.tools.schema import build_input_model, build_schema


def tool(
    name: str | None = None,
    description: str | None = None,
    requires_confirmation: bool = False,
    requires_user_input: bool = False,
    external_execution: bool = False,
    confirmation_prompt: str | None = None,
    user_input_prompt: str | None = None,
    input_key: str | None = None,
    timeout: float | None = None,
    retries: int = 0,
    retry_delay: float = 1.0,
) -> Callable[[Callable[..., Any]], Tool]:
    def wrapper(fn: Callable[..., Any]) -> Tool:
        input_model = build_input_model(fn)
        schema = build_schema(input_model)
        tool_name = name or fn.__name__
        tool_description = description or (fn.__doc__ or "")
        return Tool(
            name=tool_name,
            description=tool_description,
            schema=schema,
            callable=fn,
            input_model=input_model,
            requires_confirmation=requires_confirmation,
            requires_user_input=requires_user_input,
            external_execution=external_execution,
            confirmation_prompt=confirmation_prompt,
            user_input_prompt=user_input_prompt,
            input_key=input_key,
            timeout=timeout,
            retries=retries,
            retry_delay=retry_delay,
        )

    return wrapper
