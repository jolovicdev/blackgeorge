import pytest
from pydantic import BaseModel

from blackgeorge import Desk, Job, ScriptedAdapter, Worker
from blackgeorge.adapters.base import ModelResponse
from blackgeorge.core.tool_call import ToolCall
from blackgeorge.store.in_memory import InMemoryRunStore
from blackgeorge.tools import tool


class Answer(BaseModel):
    answer: str


def _response(
    content: str | None = "ok", tool_calls: list[ToolCall] | None = None
) -> ModelResponse:
    return ModelResponse(content=content, tool_calls=tool_calls or [], usage={}, raw={})


def _desk(adapter: ScriptedAdapter) -> Desk:
    return Desk(model="fake", adapter=adapter, run_store=InMemoryRunStore())


def test_scripted_adapter_completes_run() -> None:
    adapter = ScriptedAdapter([_response("hello")])
    report = _desk(adapter).run(Worker(name="W"), Job(input="hi"))
    assert report.status == "completed"
    assert report.content == "hello"
    assert len(adapter.calls) == 1
    assert adapter.calls[0]["kind"] == "complete"
    assert adapter.calls[0]["model"] == "fake"
    assert adapter.calls[0]["messages"][-1]["content"] == "hi"


async def test_scripted_adapter_supports_arun() -> None:
    adapter = ScriptedAdapter([_response("async")])
    report = await _desk(adapter).arun(Worker(name="W"), Job(input="hi"))
    assert report.status == "completed"
    assert report.content == "async"


def test_scripted_adapter_executes_tools() -> None:
    @tool()
    def echo(text: str) -> str:
        return text.upper()

    adapter = ScriptedAdapter(
        [
            _response(None, [ToolCall(id="1", name="echo", arguments={"text": "hi"})]),
            _response("done"),
        ]
    )
    report = _desk(adapter).run(Worker(name="W", tools=[echo]), Job(input="go"))
    assert report.status == "completed"
    assert report.content == "done"
    assert report.tool_calls[0].result.content == "HI"
    assert len(adapter.calls) == 2


def test_scripted_adapter_structured_output() -> None:
    adapter = ScriptedAdapter([_response('{"answer": "ok"}')])
    report = _desk(adapter).run(Worker(name="W"), Job(input="hi", response_schema=Answer))
    assert report.status == "completed"
    assert report.data is not None
    assert report.data.answer == "ok"
    assert adapter.calls[0]["kind"] == "structured"


def test_scripted_adapter_structured_output_rejects_invalid_json() -> None:
    adapter = ScriptedAdapter([_response("not json")])
    report = _desk(adapter).run(Worker(name="W"), Job(input="hi", response_schema=Answer))
    assert report.status == "failed"
    assert report.errors


def test_scripted_adapter_raises_when_exhausted() -> None:
    adapter = ScriptedAdapter([])
    with pytest.raises(RuntimeError, match="no scripted responses remain"):
        _desk(adapter).run(Worker(name="W"), Job(input="hi"))


def test_scripted_adapter_stream_flag_degrades_to_non_streaming() -> None:
    adapter = ScriptedAdapter([_response("streamed")])
    report = _desk(adapter).run(Worker(name="W"), Job(input="hi"), stream=True)
    assert report.status == "completed"
    assert report.content == "streamed"
    assert adapter.calls[0]["stream"] is True
