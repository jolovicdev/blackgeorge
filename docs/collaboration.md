# Collaboration

Blackgeorge provides built-in collaboration primitives for multi-agent coordination. Workers can communicate through direct messaging and share state through a common data store.

## Blackboard

A `Blackboard` provides shared state across workers. Multiple workers can read from and write to the same blackboard, enabling coordination through shared memory.

### Creating a blackboard

```python
from blackgeorge.collaboration import Blackboard

blackboard = Blackboard()
```

### Writing to the blackboard

```python
blackboard.write("analysis_result", {"score": 95}, author="analyst")
```

The `write` method takes:
- `key`: A string identifier for the data
- `value`: Any JSON-serializable value
- `author`: The name of the worker writing the data

### Reading from the blackboard

```python
result = blackboard.read("analysis_result")
print(result)  # {"score": 95}

# Check if a key exists
if blackboard.exists("analysis_result"):
    value = blackboard.read("analysis_result")
```

### Reading entries with metadata

```python
entry = blackboard.read_entry("analysis_result")
print(entry.key)         # "analysis_result"
print(entry.value)       # {"score": 95}
print(entry.author)      # "analyst"
print(entry.created_at)  # datetime of first write
print(entry.updated_at)  # datetime of last update
```

### Listing all entries

```python
entries = blackboard.all_entries()
for key, entry in entries.items():
    print(f"{key} by {entry.author}: {entry.value}")
```

### Subscribing to changes

```python
def on_write(key: str, value: Any, author: str) -> None:
    print(f"{author} wrote {key}: {value}")

# Subscribe to a specific key
blackboard.subscribe("analysis_result", on_write)

# Subscribe to all changes
blackboard.subscribe_all(on_write)
```

### Other operations

```python
# Delete a key
blackboard.delete("analysis_result")

# List all keys
keys = blackboard.keys()

# Clear all entries
blackboard.clear()
```

## Channel

A `Channel` enables direct messaging between workers. Messages can be sent to specific recipients or broadcast to all workers.

### Creating a channel

```python
from blackgeorge.collaboration import Channel

channel = Channel()
```

### Sending direct messages

```python
message = channel.send(
    sender="worker_a",
    recipient="worker_b",
    content={"task": "analyze data"},
    metadata={"priority": "high"}
)
print(message.id)  # Unique message ID
```

### Broadcasting messages

```python
message = channel.broadcast(
    sender="manager",
    content={"status": "starting"},
    metadata={"phase": "1"}
)
```

### Receiving messages

```python
# Receive all messages for this worker (clears after reading)
messages = channel.receive("worker_b")
for msg in messages:
    print(f"From {msg.sender}: {msg.content}")

# Peek at messages without clearing
messages = channel.peek("worker_b")

# Receive with broadcast mode options
messages = channel.receive(
    "worker_b",
    clear=False,                # Don't clear after reading
    broadcast_mode="one_shot"   # Each broadcast seen once (default)
)

# Receive all broadcasts (including previously seen)
messages = channel.receive(
    "worker_b",
    broadcast_mode="all"
)
```

### Broadcast modes

- `one_shot`: Each broadcast is delivered only once per recipient. The channel tracks which broadcasts each recipient has seen.
- `all`: All broadcasts are returned every time. Useful for treating broadcasts as a log.

### Message structure

Each message has:

```python
@dataclass
class ChannelMessage:
    id: str                        # Unique message ID
    sender: str                    # Sender's name
    recipient: str | None          # None for broadcasts
    content: Any                   # Message content
    timestamp: datetime            # When sent
    metadata: dict[str, Any]       # Additional data
```

### Channel utilities

```python
# Get all messages across all recipients
all_messages = channel.all_messages()

# Clear messages for a specific recipient
channel.clear("worker_b")

# Clear all messages
channel.clear()
```

## Using with Workforce

Pass channel and blackboard when creating a workforce:

```python
from blackgeorge import Worker, Workforce
from blackgeorge.collaboration import Channel, Blackboard

channel = Channel()
blackboard = Blackboard()

worker_a = Worker(name="worker_a")
worker_b = Worker(name="worker_b")

workforce = Workforce(
    [worker_a, worker_b],
    mode="collaborate",
    channel=channel,
    blackboard=blackboard,
)
```

Workers can access the collaboration primitives:

```python
# In worker tool implementations
workforce.channel.send("worker_a", "worker_b", {"data": "value"})
workforce.blackboard.write("result", {"status": "done"}, author="worker_a")
```

## Collaboration Tools

Blackgeorge provides tool wrappers around collaboration primitives so workers can use them through tool calls.

### Channel tools

```python
from blackgeorge.collaboration import channel_send_tool, channel_receive_tool, channel_broadcast_tool

send_tool = channel_send_tool(channel, sender="worker_a")
receive_tool = channel_receive_tool(channel, recipient="worker_b")
broadcast_tool = channel_broadcast_tool(channel, sender="manager")

worker = Worker(name="worker_a", tools=[send_tool, receive_tool, broadcast_tool])
```

### Blackboard tools

```python
from blackgeorge.collaboration import blackboard_read_tool, blackboard_write_tool

read_tool = blackboard_read_tool(blackboard)
write_tool = blackboard_write_tool(blackboard, author="worker_a")

worker = Worker(name="worker_a", tools=[read_tool, write_tool])
```

### Tool parameters

**channel_send:**
- `recipient` (str): The recipient worker name
- `content` (JsonValue): Message content
- `metadata` (dict, optional): Additional message metadata
- Returns: Message ID

**channel_broadcast:**
- `content` (JsonValue): Broadcast content
- `metadata` (dict, optional): Additional metadata
- Returns: Message ID

**channel_receive:**
- `broadcast_mode` ("all" or "one_shot"): How to handle broadcasts
- `clear` (bool): Whether to clear messages after reading
- Returns: List of message dictionaries

**blackboard_write:**
- `key` (str): The key to write
- `value` (JsonValue): The value to store
- Returns: The key that was written

**blackboard_read:**
- `key` (str): The key to read
- Returns: The stored value or None

## Example: Multi-agent analysis

```python
from blackgeorge import Desk, Job, Worker, Workforce
from blackgeorge.collaboration import Channel, Blackboard
from blackgeorge.collaboration.tools import (
    channel_send_tool,
    channel_receive_tool,
    blackboard_read_tool,
    blackboard_write_tool,
)

# Create collaboration primitives
channel = Channel()
blackboard = Blackboard()

# Create workers with collaboration tools
analyst = Worker(
    name="analyst",
    tools=[
        blackboard_write_tool(blackboard, "analyst"),
        channel_send_tool(channel, "analyst"),
    ],
)

reviewer = Worker(
    name="reviewer",
    tools=[
        blackboard_read_tool(blackboard),
        blackboard_write_tool(blackboard, "reviewer"),
        channel_send_tool(channel, "reviewer"),
        channel_receive_tool(channel, "reviewer"),
    ],
)

# Create workforce
workforce = Workforce(
    [analyst, reviewer],
    mode="collaborate",
    channel=channel,
    blackboard=blackboard,
)

# Run
desk = Desk(model="openai/gpt-5-nano")
job = Job(input="Analyze the data and have it reviewed")
report = desk.run(workforce, job)
```

## Thread safety

Both `Blackboard` and `Channel` are thread-safe. They use internal locks to ensure safe concurrent access from multiple workers.

## Async methods

For use in async contexts (like `Workforce` with `mode="collaborate"`), both `Blackboard` and `Channel` provide async method variants:

### Async Blackboard

```python
# Async variants of all methods
await blackboard.awrite("key", value, "author")
value = await blackboard.aread("key")
entry = await blackboard.aread_entry("key")
exists = await blackboard.aexists("key")
deleted = await blackboard.adelete("key")
keys = await blackboard.akeys()
entries = await blackboard.aall_entries()
await blackboard.asubscribe("key", callback)
await blackgeorge.asubscribe_all(callback)
await blackboard.aunsubscribe("key", callback)
await blackboard.aclear()
```

### Async Channel

```python
# Async variants of all methods
message = await channel.asend("sender", "recipient", content, metadata)
message = await channel.abroadcast("sender", content, metadata)
messages = await channel.areceive("recipient", clear=True, broadcast_mode="one_shot")
messages = await channel.apeek("recipient")
await channel.aclear("recipient")
messages = await channel.aall_messages()
```

### Why async methods?

When using `Workforce` with `mode="collaborate"`, workers run in parallel using `asyncio.gather`. The async variants use `asyncio.to_thread()` to delegate to sync methods, ensuring thread-safe access through a single `threading.Lock`. This prevents blocking the event loop while maintaining correct synchronization.
