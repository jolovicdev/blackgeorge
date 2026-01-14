from blackgeorge.collaboration.blackboard import Blackboard
from blackgeorge.collaboration.channel import BroadcastMode, Channel
from blackgeorge.tools import Tool, tool

type JsonValue = str | int | float | bool | None | list[object] | dict[str, object]


def channel_send_tool(channel: Channel, sender: str, name: str = "channel_send") -> Tool:
    @tool(name=name)
    def channel_send(
        recipient: str,
        content: JsonValue,
        metadata: dict[str, JsonValue] | None = None,
    ) -> str:
        message = channel.send(sender, recipient, content, metadata)
        return message.id

    return channel_send


def channel_broadcast_tool(channel: Channel, sender: str, name: str = "channel_broadcast") -> Tool:
    @tool(name=name)
    def channel_broadcast(
        content: JsonValue,
        metadata: dict[str, JsonValue] | None = None,
    ) -> str:
        message = channel.broadcast(sender, content, metadata)
        return message.id

    return channel_broadcast


def channel_receive_tool(
    channel: Channel,
    recipient: str,
    name: str = "channel_receive",
) -> Tool:
    @tool(name=name)
    def channel_receive(
        broadcast_mode: BroadcastMode = "one_shot",
        clear: bool = True,
    ) -> list[dict[str, JsonValue]]:
        messages = channel.receive(recipient, clear=clear, broadcast_mode=broadcast_mode)
        output: list[dict[str, JsonValue]] = []
        for msg in messages:
            output.append(
                {
                    "id": msg.id,
                    "sender": msg.sender,
                    "recipient": msg.recipient,
                    "content": msg.content,
                    "timestamp": msg.timestamp.isoformat(),
                    "metadata": msg.metadata,
                }
            )
        return output

    return channel_receive


def blackboard_write_tool(
    blackboard: Blackboard,
    author: str,
    name: str = "blackboard_write",
) -> Tool:
    @tool(name=name)
    def blackboard_write(key: str, value: JsonValue) -> str:
        blackboard.write(key, value, author)
        return key

    return blackboard_write


def blackboard_read_tool(blackboard: Blackboard, name: str = "blackboard_read") -> Tool:
    @tool(name=name)
    def blackboard_read(key: str) -> JsonValue | None:
        return blackboard.read(key)

    return blackboard_read
