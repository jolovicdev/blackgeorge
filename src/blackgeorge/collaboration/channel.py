import asyncio
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Any, Literal

from blackgeorge.utils import new_id, utc_now


@dataclass(frozen=True)
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
            content=deepcopy(content),
            timestamp=utc_now(),
            metadata=deepcopy(metadata) if metadata is not None else {},
        )
        with self._lock:
            self._messages[recipient].append(message)
        return deepcopy(message)

    async def asend(
        self,
        sender: str,
        recipient: str,
        content: Any,
        metadata: dict[str, Any] | None = None,
    ) -> ChannelMessage:
        return await asyncio.to_thread(self.send, sender, recipient, content, metadata)

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
            content=deepcopy(content),
            timestamp=utc_now(),
            metadata=deepcopy(metadata) if metadata is not None else {},
        )
        with self._lock:
            self._broadcast.append(message)
        return deepcopy(message)

    async def abroadcast(
        self,
        sender: str,
        content: Any,
        metadata: dict[str, Any] | None = None,
    ) -> ChannelMessage:
        return await asyncio.to_thread(self.broadcast, sender, content, metadata)

    def receive(
        self,
        recipient: str,
        clear: bool = True,
        broadcast_mode: BroadcastMode = "one_shot",
    ) -> list[ChannelMessage]:
        if broadcast_mode not in ("one_shot", "all"):
            raise ValueError("broadcast_mode must be 'one_shot' or 'all'")
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
            else:
                broadcasts = list(self._broadcast)
        messages = sorted([*direct, *broadcasts], key=lambda message: message.timestamp)
        return deepcopy(messages)

    async def areceive(
        self,
        recipient: str,
        clear: bool = True,
        broadcast_mode: BroadcastMode = "one_shot",
    ) -> list[ChannelMessage]:
        return await asyncio.to_thread(self.receive, recipient, clear, broadcast_mode)

    def peek(self, recipient: str) -> list[ChannelMessage]:
        return self.receive(recipient, clear=False)

    async def apeek(self, recipient: str) -> list[ChannelMessage]:
        return await self.areceive(recipient, clear=False)

    def clear(self, recipient: str | None = None) -> None:
        with self._lock:
            if recipient is None:
                self._messages.clear()
                self._broadcast.clear()
                self._broadcast_positions.clear()
            else:
                self._messages[recipient] = []
                self._broadcast_positions[recipient] = len(self._broadcast)

    async def aclear(self, recipient: str | None = None) -> None:
        await asyncio.to_thread(self.clear, recipient)

    def all_messages(self) -> list[ChannelMessage]:
        with self._lock:
            all_msgs: list[ChannelMessage] = []
            for msgs in self._messages.values():
                all_msgs.extend(msgs)
            all_msgs.extend(self._broadcast)
        return deepcopy(sorted(all_msgs, key=lambda message: message.timestamp))

    async def aall_messages(self) -> list[ChannelMessage]:
        return await asyncio.to_thread(self.all_messages)
