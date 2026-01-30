import json
from collections.abc import Callable
from typing import Any

from blackgeorge.adapters.base import BaseModelAdapter, ModelResponse
from blackgeorge.core.message import Message

CONTEXT_LIMIT_ERRORS = (
    "expected a string with maximum length",
    "maximum context length",
    "context length exceeded",
    "context_length_exceeded",
    "context window full",
    "too many tokens",
    "input is too long",
    "exceeds token limit",
)

SUMMARY_SYSTEM_PROMPT = (
    "You are a summarization assistant. Preserve key facts, decisions, tool results, "
    "names, numbers, and user constraints. Be concise and avoid adding information."
)
SUMMARY_INSTRUCTION = "Summarize the following conversation history:\n\n{content}"
SUMMARY_CHUNK_TOKENS = 2000
SUMMARY_TAIL_MESSAGES = 4
SUMMARY_ATTEMPT_LIMIT = 2
SUMMARY_MAX_OUTPUT_TOKENS = 800
MODEL_REGISTRATION_HINT = (
    "Register the model in LiteLLM with context window and token limits "
    "(register_model or proxy model_info) for accurate context handling."
)


def is_context_limit_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(phrase in message for phrase in CONTEXT_LIMIT_ERRORS)


def litellm_model_registered(model_name: str) -> bool:
    try:
        import litellm

        return model_name in getattr(litellm, "model_cost", {})
    except Exception:
        return False


def token_count(model_name: str, text: str) -> int:
    if not text:
        return 0
    try:
        import litellm

        count = litellm.token_counter(
            model=model_name,
            messages=[{"role": "user", "content": text}],
        )
        if isinstance(count, int):
            return count
    except Exception:
        pass
    return max(1, len(text) // 4)


def chunk_text(model_name: str, text: str, target_tokens: int) -> list[str]:
    if not text:
        return [""]
    total_tokens = token_count(model_name, text)
    if total_tokens <= target_tokens:
        return [text]
    chunk_size = max(200, target_tokens * 4)
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def message_summary_text(message: Message) -> str:
    parts: list[str] = []
    if message.content:
        parts.append(message.content)
    if message.tool_calls:
        calls: list[str] = []
        for call in message.tool_calls:
            args = json.dumps(call.arguments, ensure_ascii=True, sort_keys=True)
            calls.append(f"{call.name}({args})")
        parts.append(f"tool_calls={'; '.join(calls)}")
    if message.role == "tool" and message.tool_call_id:
        parts.append(f"tool_call_id={message.tool_call_id}")
    if not parts:
        return ""
    return f"{message.role}: " + " ".join(parts)


def summarize_chunk(
    adapter: BaseModelAdapter,
    model_name: str,
    chunk: str,
    temperature: float | None,
) -> str:
    response = adapter.complete(
        model=model_name,
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": SUMMARY_INSTRUCTION.format(content=chunk)},
        ],
        tools=None,
        tool_choice=None,
        temperature=temperature,
        max_tokens=SUMMARY_MAX_OUTPUT_TOKENS,
        stream=False,
        stream_options=None,
    )
    if isinstance(response, ModelResponse):
        return response.content or ""
    return ""


async def asummarize_chunk(
    adapter: BaseModelAdapter,
    model_name: str,
    chunk: str,
    temperature: float | None,
) -> str:
    response = await adapter.acomplete(
        model=model_name,
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": SUMMARY_INSTRUCTION.format(content=chunk)},
        ],
        tools=None,
        tool_choice=None,
        temperature=temperature,
        max_tokens=SUMMARY_MAX_OUTPUT_TOKENS,
        stream=False,
        stream_options=None,
    )
    if isinstance(response, ModelResponse):
        return response.content or ""
    return ""


def summarize_messages(
    adapter: BaseModelAdapter,
    model_name: str,
    messages: list[Message],
    temperature: float | None,
) -> str | None:
    lines = [message_summary_text(message) for message in messages]
    lines = [line for line in lines if line]
    if not lines:
        return ""
    text = "\n".join(lines)
    target_tokens = SUMMARY_CHUNK_TOKENS
    attempts = 0
    while attempts < SUMMARY_ATTEMPT_LIMIT:
        try:
            chunks = chunk_text(model_name, text, target_tokens)
            summaries: list[str] = []
            for chunk in chunks:
                summary = summarize_chunk(adapter, model_name, chunk, temperature)
                if summary:
                    summaries.append(summary)
            if not summaries:
                return None
            merged = "\n".join(summaries)
            if len(summaries) > 1:
                merged = summarize_chunk(adapter, model_name, merged, temperature) or merged
            return merged
        except Exception as exc:
            if not is_context_limit_error(exc):
                raise
            attempts += 1
            target_tokens = max(200, target_tokens // 2)
    return None


async def asummarize_messages(
    adapter: BaseModelAdapter,
    model_name: str,
    messages: list[Message],
    temperature: float | None,
) -> str | None:
    lines = [message_summary_text(message) for message in messages]
    lines = [line for line in lines if line]
    if not lines:
        return ""
    text = "\n".join(lines)
    target_tokens = SUMMARY_CHUNK_TOKENS
    attempts = 0
    while attempts < SUMMARY_ATTEMPT_LIMIT:
        try:
            chunks = chunk_text(model_name, text, target_tokens)
            summaries: list[str] = []
            for chunk in chunks:
                summary = await asummarize_chunk(adapter, model_name, chunk, temperature)
                if summary:
                    summaries.append(summary)
            if not summaries:
                return None
            merged = "\n".join(summaries)
            if len(summaries) > 1:
                merged = await asummarize_chunk(adapter, model_name, merged, temperature) or merged
            return merged
        except Exception as exc:
            if not is_context_limit_error(exc):
                raise
            attempts += 1
            target_tokens = max(200, target_tokens // 2)
    return None


def apply_context_summary(
    *,
    adapter: BaseModelAdapter,
    model_name: str,
    messages: list[Message],
    temperature: float | None,
    metrics: dict[str, Any],
    emit: Callable[[str, str, dict[str, Any]], None],
    worker_name: str,
    model_registered: bool,
) -> bool:
    system_messages = [message for message in messages if message.role == "system"]
    non_system = [message for message in messages if message.role != "system"]
    if not non_system:
        return False
    tail_count = SUMMARY_TAIL_MESSAGES
    if len(non_system) <= tail_count:
        tail_count = 0
    head = non_system if tail_count == 0 else non_system[:-tail_count]
    tail = [] if tail_count == 0 else non_system[-tail_count:]
    try:
        summary = summarize_messages(adapter, model_name, head, temperature)
    except Exception:
        return False
    if summary is None:
        return False
    summary_message = Message(
        role="user",
        content=f"Summary of previous context:\n{summary}",
        metadata={"summary": True},
    )
    messages[:] = system_messages + [summary_message] + tail
    info = {
        "model": model_name,
        "summarized_messages": len(head),
        "kept_messages": len(tail),
    }
    if not model_registered:
        info["unregistered_model"] = True
        info["registration_hint"] = MODEL_REGISTRATION_HINT
        warnings = metrics.setdefault("warnings", [])
        if isinstance(warnings, list) and MODEL_REGISTRATION_HINT not in warnings:
            warnings.append(MODEL_REGISTRATION_HINT)
    summaries = metrics.setdefault("context_summaries", [])
    if isinstance(summaries, list):
        summaries.append(info)
    emit("worker.context_summarized", worker_name, info)
    return True


async def aapply_context_summary(
    *,
    adapter: BaseModelAdapter,
    model_name: str,
    messages: list[Message],
    temperature: float | None,
    metrics: dict[str, Any],
    emit: Callable[[str, str, dict[str, Any]], None],
    worker_name: str,
    model_registered: bool,
) -> bool:
    system_messages = [message for message in messages if message.role == "system"]
    non_system = [message for message in messages if message.role != "system"]
    if not non_system:
        return False
    tail_count = SUMMARY_TAIL_MESSAGES
    if len(non_system) <= tail_count:
        tail_count = 0
    head = non_system if tail_count == 0 else non_system[:-tail_count]
    tail = [] if tail_count == 0 else non_system[-tail_count:]
    try:
        summary = await asummarize_messages(adapter, model_name, head, temperature)
    except Exception:
        return False
    if summary is None:
        return False
    summary_message = Message(
        role="user",
        content=f"Summary of previous context:\n{summary}",
        metadata={"summary": True},
    )
    messages[:] = system_messages + [summary_message] + tail
    info = {
        "model": model_name,
        "summarized_messages": len(head),
        "kept_messages": len(tail),
    }
    if not model_registered:
        info["unregistered_model"] = True
        info["registration_hint"] = MODEL_REGISTRATION_HINT
        warnings = metrics.setdefault("warnings", [])
        if isinstance(warnings, list) and MODEL_REGISTRATION_HINT not in warnings:
            warnings.append(MODEL_REGISTRATION_HINT)
    summaries = metrics.setdefault("context_summaries", [])
    if isinstance(summaries, list):
        summaries.append(info)
    emit("worker.context_summarized", worker_name, info)
    return True


def context_error_message(model_registered: bool, allow_summary: bool) -> str:
    if allow_summary:
        message = "Context length exceeded after summarization attempts."
    else:
        message = (
            "Context length exceeded. Set Desk(respect_context_window=True) to allow summarization."
        )
    if not model_registered:
        message = f"{message} {MODEL_REGISTRATION_HINT}"
    return message
