# WorkerSession

A `WorkerSession` manages multi-turn conversations with automatic message persistence and context compaction integration. Use sessions for chatbots, conversational agents, or any scenario where you need to maintain conversation state across multiple user interactions.

## What a WorkerSession does

- Accumulates conversation history across `run()` calls
- Persists messages to SQLite for recovery across program restarts
- Integrates with context compaction (handles long conversations automatically)
- Provides both sync (`run()`) and async (`arun()`) APIs
- Supports streaming responses with `stream_run()` and `astream_run()`
- Supports thinking models (DeepSeek Reasoner, Claude 3.7 Sonnet, etc.) via Job parameters

## Creating a session

Sessions are created through the `Desk`:

```python
from blackgeorge import Desk, Worker

desk = Desk(model="openai/gpt-4o")
worker = Worker(name="Assistant", instructions="You are helpful")

session = desk.session(worker)
```

The session ID is auto-generated, or you can provide your own:

```python
session = desk.session(
    worker=worker,
    session_id="user-123",
    metadata={"user_id": "123", "topic": "support"},
)
```

## Sending messages

Use `run()` for sync or `arun()` for async:

```python
report1 = session.run("Hello, I'm Alice")
print(report1.content)

report2 = session.run("What's my name?")
print(report2.content)
```

The session automatically:
1. Loads previous messages from storage
2. Sends them with the new user message
3. Saves the updated conversation after each response

## Loading an existing session

```python
restored = desk.session(worker, session_id="user-123")
if restored:
    report = restored.run("Continue our conversation")
```

Returns `None` if the session doesn't exist or belongs to a different worker.

## Session history

Get the full conversation history:

```python
messages = session.history()
for message in messages:
    print(f"{message.role}: {message.content}")
```

## Closing a session

```python
session.close()
```

This removes the session and all associated messages from storage.

## Context compaction

Sessions integrate with the existing context compaction mechanism. When a conversation grows too large:

1. The worker detects the context limit
2. Old messages are summarized automatically
3. The compacted messages are saved back to the session
4. Next turn loads the already-compacted messages

You don't need to handle this manually—it just works.

## Session storage

Sessions use the `SessionStore` interface:

- `SQLiteSessionStore`: Default, persists to `.blackgeorge/blackgeorge.db`
- `InMemorySessionStore`: For testing or ephemeral sessions

Storage schema:

```sql
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    worker_name TEXT NOT NULL,
    metadata TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE session_messages (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    message_json TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);
```

## Use cases

**Chatbot**: Maintain conversation state for each user

```python
session = desk.session(chatbot_worker, session_id=f"user:{user_id}")
while user_input := get_user_input():
    response = session.run(user_input)
    send_response(response.content)
```

**Support agent**: Resume conversations across sessions

```python
session = desk.session(support_worker, session_id=f"ticket:{ticket_id}")
if not session:
    session = desk.session(support_worker, session_id=f"ticket:{ticket_id}")
```

**Multi-step task**: Track progress across multiple interactions

```python
session = desk.session(
    worker=planner_worker,
    session_id=f"plan:{plan_id}",
    metadata={"step": 1, "total_steps": 5},
)
```

## Sync vs async

```python
report = session.run("Hello")              # sync
report = await session.arun("Hello")       # async
```

Both use the same underlying storage and produce identical results.

## Streaming responses

The `stream_run()` and `astream_run()` methods yield events during execution, useful for real-time token display or progress tracking:

```python
# Sync streaming
for event in session.stream_run("Tell me a story"):
    if event.type == "stream.token":
        print(event.payload["token"], end="", flush=True)
    elif event.type == "worker.completed":
        print(f"\nDone! Metrics: {event.payload}")

# Async streaming
async for event in session.astream_run("Count to 10"):
    if event.type == "stream.token":
        print(event.payload["token"], end="", flush=True)
```

Streaming also works with the `stream` parameter on `run()`/`arun()`:

```python
report = session.run("Hello", stream=True)
for event in report.events:
    if event.type == "stream.token":
        print(event.payload["token"], end="")
```

Events are only collected when `stream=True`. Without streaming, `report.events` contains lifecycle events only (started, completed, etc.).

## Thinking models

For models that output separate reasoning content (DeepSeek Reasoner, Claude 3.7 Sonnet, o1, etc.), use Job parameters:

```python
from blackgeorge import Job

# DeepSeek Reasoner (no budget_tokens support)
report = session.run(
    "Which is larger: 9.11 or 9.8?",
    thinking={"type": "enabled"},
)

# Anthropic Claude 3.7+ (supports budget_tokens)
report = session.run(
    "Which is larger: 9.11 or 9.8?",
    thinking={"type": "enabled", "budget_tokens": 1024},
)

print(f"Reasoning: {report.reasoning_content}")
print(f"Answer: {report.content}")
```

### drop_params

Some thinking models reject standard parameters like `temperature` or `top_p`. Use `drop_params=True` to automatically omit unsupported parameters:

```python
report = session.run(
    "Explain quantum computing",
    thinking={"type": "enabled"},
    drop_params=True,  # Auto-drops unsupported params
)
```

### extra_body

Pass provider-specific parameters via `extra_body`:

```python
report = session.run(
    "Analyze this data",
    extra_body={
        "guardrails": ["pii-detection"],
        "custom_setting": 42,
    },
)
```

### Multi-turn behavior

Sessions automatically handle `reasoning_content` correctly across turns:
- **Within a turn**: Reasoning content is preserved during tool-call loops
- **Between turns**: Reasoning content is cleared from stored messages (it will be regenerated if needed)

```python
# First turn generates reasoning_content
report1 = session.run("What is 9.11 vs 9.8?", thinking={"type": "enabled"})
print(report1.reasoning_content)  # Has reasoning

# Second turn - reasoning_content is cleared from history
report2 = session.run("How many Rs in strawberry?")
print(report2.reasoning_content)  # New reasoning generated

# History never contains stale reasoning_content
for msg in session.history():
    if msg.role == "assistant":
        assert msg.reasoning_content is None  # Always None in storage
```

## Worker binding

Sessions are bound to a specific worker by name. Loading a session with a different worker returns `None`:

```python
session = desk.session(worker1)
loaded = desk.session(worker2, session_id=session.session_id)
assert loaded is None  # worker1 != worker2
```

## Integration with Job

Sessions use the `initial_messages` field on `Job` to inject conversation history:

```python
from blackgeorge import Job

messages = session.history()
job = Job(input="new message", initial_messages=messages)
```

This is handled automatically by `session.run()`, but you can use it directly if needed.
