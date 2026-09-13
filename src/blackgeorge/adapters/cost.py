from typing import Any

import litellm
from litellm.cost_calculator import completion_cost


def calculate_cost(response: Any) -> float | None:
    try:
        return completion_cost(completion_response=response)
    except Exception:
        return None


def get_model_pricing(model: str) -> dict[str, float] | None:
    try:
        return litellm.model_cost.get(model)
    except Exception:
        return None


def get_prompt_cost(model: str, tokens: int) -> float | None:
    pricing = get_model_pricing(model)
    if not pricing:
        return None
    if "input_cost_per_million" in pricing:
        return pricing["input_cost_per_million"] * tokens / 1_000_000
    if "input_cost_per_token" in pricing:
        return pricing["input_cost_per_token"] * tokens
    return None


def get_completion_cost(model: str, tokens: int) -> float | None:
    pricing = get_model_pricing(model)
    if not pricing:
        return None
    if "output_cost_per_million" in pricing:
        return pricing["output_cost_per_million"] * tokens / 1_000_000
    if "output_cost_per_token" in pricing:
        return pricing["output_cost_per_token"] * tokens
    return None


def usage_cost(model: str, usage: dict[str, Any]) -> float:
    cost = 0.0
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    if isinstance(prompt_tokens, (int, float)):
        cost += get_prompt_cost(model, int(prompt_tokens)) or 0.0
    if isinstance(completion_tokens, (int, float)):
        cost += get_completion_cost(model, int(completion_tokens)) or 0.0
    return cost
