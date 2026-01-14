from blackgeorge.collaboration.blackboard import Blackboard
from blackgeorge.collaboration.channel import Channel
from blackgeorge.collaboration.tools import (
    blackboard_read_tool,
    blackboard_write_tool,
    channel_broadcast_tool,
    channel_receive_tool,
    channel_send_tool,
)
from blackgeorge.core.tool_call import ToolCall
from blackgeorge.tools.execution import execute_tool


def test_channel_send_receive() -> None:
    channel = Channel()
    msg = channel.send("worker_a", "worker_b", {"task": "analyze"})
    assert msg.sender == "worker_a"
    assert msg.recipient == "worker_b"
    messages = channel.receive("worker_b")
    assert len(messages) == 1
    assert messages[0].content == {"task": "analyze"}


def test_channel_receive_clears() -> None:
    channel = Channel()
    channel.send("a", "b", "hello")
    messages1 = channel.receive("b")
    assert len(messages1) == 1
    messages2 = channel.receive("b")
    assert len(messages2) == 0


def test_channel_peek_does_not_clear() -> None:
    channel = Channel()
    channel.send("a", "b", "hello")
    peeked = channel.peek("b")
    assert len(peeked) == 1
    received = channel.receive("b")
    assert len(received) == 1


def test_channel_broadcast() -> None:
    channel = Channel()
    channel.broadcast("manager", {"status": "start"})
    msgs_a = channel.receive("worker_a")
    msgs_b = channel.receive("worker_b")
    assert len(msgs_a) >= 1
    assert len(msgs_b) >= 1
    assert msgs_a[0].content == {"status": "start"}
    msgs_a_again = channel.receive("worker_a")
    assert len(msgs_a_again) == 0


def test_channel_broadcast_all_mode() -> None:
    channel = Channel()
    channel.broadcast("manager", {"status": "start"})
    msgs_a = channel.receive("worker_a", broadcast_mode="all")
    msgs_a_again = channel.receive("worker_a", broadcast_mode="all")
    assert len(msgs_a) >= 1
    assert len(msgs_a_again) >= 1


def test_channel_isolation() -> None:
    channel = Channel()
    channel.send("a", "b", "for_b")
    channel.send("a", "c", "for_c")
    msgs_b = channel.receive("b")
    msgs_c = channel.receive("c")
    assert len(msgs_b) == 1
    assert len(msgs_c) == 1
    assert msgs_b[0].content == "for_b"
    assert msgs_c[0].content == "for_c"


def test_blackboard_write_read() -> None:
    bb = Blackboard()
    bb.write("result", {"score": 95}, "analyst")
    value = bb.read("result")
    assert value == {"score": 95}


def test_blackboard_read_missing() -> None:
    bb = Blackboard()
    value = bb.read("nonexistent")
    assert value is None


def test_blackboard_overwrite() -> None:
    bb = Blackboard()
    bb.write("key", "v1", "a")
    bb.write("key", "v2", "b")
    assert bb.read("key") == "v2"
    entry = bb.read_entry("key")
    assert entry is not None
    assert entry.author == "b"


def test_blackboard_subscription() -> None:
    bb = Blackboard()
    notifications: list[tuple[str, object, str]] = []

    def callback(key: str, value: object, author: str) -> None:
        notifications.append((key, value, author))

    bb.subscribe("result", callback)
    bb.write("result", 42, "worker")
    assert len(notifications) == 1
    assert notifications[0] == ("result", 42, "worker")


def test_blackboard_global_subscription() -> None:
    bb = Blackboard()
    notifications: list[str] = []

    def callback(key: str, value: object, author: str) -> None:
        notifications.append(key)

    bb.subscribe_all(callback)
    bb.write("a", 1, "x")
    bb.write("b", 2, "y")
    assert "a" in notifications
    assert "b" in notifications


def test_blackboard_delete() -> None:
    bb = Blackboard()
    bb.write("key", "value", "author")
    assert bb.exists("key")
    bb.delete("key")
    assert not bb.exists("key")


def test_blackboard_keys() -> None:
    bb = Blackboard()
    bb.write("a", 1, "x")
    bb.write("b", 2, "y")
    keys = bb.keys()
    assert "a" in keys
    assert "b" in keys


def test_blackboard_clear() -> None:
    bb = Blackboard()
    bb.write("a", 1, "x")
    bb.write("b", 2, "y")
    bb.clear()
    assert bb.read("a") is None
    assert bb.read("b") is None


def test_channel_tools_round_trip() -> None:
    channel = Channel()
    send_tool = channel_send_tool(channel, sender="worker_a")
    receive_tool = channel_receive_tool(channel, recipient="worker_b")
    send_call = ToolCall(
        id="1",
        name=send_tool.name,
        arguments={"recipient": "worker_b", "content": "hello"},
    )
    send_result = execute_tool(send_tool, send_call)
    assert send_result.error is None
    receive_call = ToolCall(id="2", name=receive_tool.name, arguments={})
    receive_result = execute_tool(receive_tool, receive_call)
    assert receive_result.error is None
    assert receive_result.data
    assert receive_result.data[0]["content"] == "hello"


def test_channel_broadcast_tool() -> None:
    channel = Channel()
    broadcast_tool = channel_broadcast_tool(channel, sender="manager")
    receive_tool = channel_receive_tool(channel, recipient="worker_a")
    broadcast_call = ToolCall(id="1", name=broadcast_tool.name, arguments={"content": "start"})
    broadcast_result = execute_tool(broadcast_tool, broadcast_call)
    assert broadcast_result.error is None
    receive_call = ToolCall(id="2", name=receive_tool.name, arguments={})
    receive_result = execute_tool(receive_tool, receive_call)
    assert receive_result.error is None
    assert receive_result.data
    assert receive_result.data[0]["content"] == "start"


def test_blackboard_tools_round_trip() -> None:
    bb = Blackboard()
    write_tool = blackboard_write_tool(bb, author="worker_a")
    read_tool = blackboard_read_tool(bb)
    write_call = ToolCall(id="1", name=write_tool.name, arguments={"key": "result", "value": 42})
    write_result = execute_tool(write_tool, write_call)
    assert write_result.error is None
    read_call = ToolCall(id="2", name=read_tool.name, arguments={"key": "result"})
    read_result = execute_tool(read_tool, read_call)
    assert read_result.error is None
    assert read_result.data == 42
