from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Any, Literal

from blackgeorge.utils import new_id, utc_now


@dataclass
class ChannelMessage:
    id: str
    sender: str
    recipient: str | None
    content: Any
    timestamp: datetime
    metadata: dict[str, Any] = field(default_factory=dict)


BroadcastMode = Literal["all", "one_shot"]


class Channel:
    def __init__(self) -> None:
        self._messages: dict[str, list[ChannelMessage]] = defaultdict(list)
        self._broadcast: list[ChannelMessage] = []
        self._broadcast_positions: dict[str, int] = defaultdict(int)
        self._lock = Lock()

    def send(
        self,
        sender: str,
        recipient: str,
        content: Any,
        metadata: dict[str, Any] | None = None,
    ) -> ChannelMessage:
        message = ChannelMessage(
            id=new_id(),
            sender=sender,
            recipient=recipient,
            content=content,
            timestamp=utc_now(),
            metadata=metadata or {},
        )
        with self._lock:
            self._messages[recipient].append(message)
        return message

    def broadcast(
        self,
        sender: str,
        content: Any,
        metadata: dict[str, Any] | None = None,
    ) -> ChannelMessage:
        message = ChannelMessage(
            id=new_id(),
            sender=sender,
            recipient=None,
            content=content,
            timestamp=utc_now(),
            metadata=metadata or {},
        )
        with self._lock:
            self._broadcast.append(message)
        return message

    def receive(
        self,
        recipient: str,
        clear: bool = True,
        broadcast_mode: BroadcastMode = "one_shot",
    ) -> list[ChannelMessage]:
        with self._lock:
            direct = list(self._messages.get(recipient, []))
            broadcasts = list(self._broadcast)
            if clear:
                self._messages[recipient] = []
            if broadcast_mode == "one_shot":
                start = self._broadcast_positions.get(recipient, 0)
                broadcasts = self._broadcast[start:]
                if clear:
                    self._broadcast_positions[recipient] = len(self._broadcast)
            elif broadcast_mode == "all":
                broadcasts = list(self._broadcast)
            else:
                broadcasts = []
        return direct + list(broadcasts)

    def peek(self, recipient: str) -> list[ChannelMessage]:
        return self.receive(recipient, clear=False)

    def clear(self, recipient: str | None = None) -> None:
        with self._lock:
            if recipient is None:
                self._messages.clear()
                self._broadcast.clear()
                self._broadcast_positions.clear()
            else:
                self._messages[recipient] = []

    def all_messages(self) -> list[ChannelMessage]:
        with self._lock:
            all_msgs: list[ChannelMessage] = []
            for msgs in self._messages.values():
                all_msgs.extend(msgs)
            all_msgs.extend(self._broadcast)
        return sorted(all_msgs, key=lambda m: m.timestamp)
