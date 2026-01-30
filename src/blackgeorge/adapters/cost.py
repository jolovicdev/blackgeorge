from typing import Any

import litellm


def calculate_cost(response: Any) -> float | None:
    try:
        return litellm.completion_cost(completion_response=response)
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
