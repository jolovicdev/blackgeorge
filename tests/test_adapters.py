from blackgeorge.adapters.litellm import _parse_tool_calls


def test_parse_tool_calls_with_valid_json() -> None:
    message = {
        "tool_calls": [
            {
                "id": "call_123",
                "function": {
                    "name": "test_tool",
                    "arguments": '{"param1": "value1", "param2": 42}',
                },
            }
        ]
    }

    calls = _parse_tool_calls(message)

    assert len(calls) == 1
    assert calls[0].id == "call_123"
    assert calls[0].name == "test_tool"
    assert calls[0].arguments == {"param1": "value1", "param2": 42}
    assert calls[0].error is None


def test_parse_tool_calls_with_invalid_json() -> None:
    message = {
        "tool_calls": [
            {
                "id": "call_456",
                "function": {
                    "name": "test_tool",
                    "arguments": '{"param1": invalid json here}',
                },
            }
        ]
    }

    calls = _parse_tool_calls(message)

    assert len(calls) == 1
    assert calls[0].id == "call_456"
    assert calls[0].name == "test_tool"
    assert calls[0].arguments == {}
    assert calls[0].error is not None
    assert "Invalid JSON" in calls[0].error
    assert "invalid json" in calls[0].error


def test_parse_tool_calls_with_empty_arguments() -> None:
    message = {
        "tool_calls": [
            {
                "id": "call_789",
                "function": {
                    "name": "test_tool",
                    "arguments": "",
                },
            }
        ]
    }

    calls = _parse_tool_calls(message)

    assert len(calls) == 1
    assert calls[0].arguments == {}
    assert calls[0].error is None


def test_parse_tool_calls_with_dict_arguments() -> None:
    message = {
        "tool_calls": [
            {
                "id": "call_abc",
                "function": {
                    "name": "test_tool",
                    "arguments": {"param1": "value1"},
                },
            }
        ]
    }

    calls = _parse_tool_calls(message)

    assert len(calls) == 1
    assert calls[0].arguments == {}
    assert calls[0].error is None


def test_parse_tool_calls_generates_id_if_missing() -> None:
    message = {
        "tool_calls": [
            {
                "function": {
                    "name": "test_tool",
                    "arguments": "{}",
                },
            }
        ]
    }

    calls = _parse_tool_calls(message)

    assert len(calls) == 1
    assert calls[0].id is not None
    assert len(calls[0].id) > 0
