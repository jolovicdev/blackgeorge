# Workforce

A `Workforce` coordinates multiple workers. It supports two modes: managed and collaborate.

## Create a workforce

```python
from blackgeorge import Worker, Workforce

w1 = Worker(name="Researcher")
w2 = Worker(name="Writer")
workforce = Workforce([w1, w2], mode="managed", name="team")
```

## Managed mode

In managed mode, a manager chooses which worker should handle the job.

- If you pass `manager`, that worker is used.
- If you do not pass `manager`, the first worker in the list is used.
- The manager receives a job with a response schema that contains a single field, `worker`.

The selection rules are:

- If the manager returns a structured response with a `worker` field, that worker is used.
- Otherwise, the system scans the manager output for a worker name.
- If nothing matches, the first worker is used.

If the manager or selected worker pauses, the workforce returns a paused report with enough state to resume later.

## Collaborate mode

In collaborate mode, workers run sequentially and their reports are combined.

- If a worker pauses, the workforce returns a paused report and stores state.
- When resumed, the workforce continues from the paused worker and then completes the remaining workers.

You can pass a reducer to combine worker reports. If you do not, the default reducer concatenates content and collects data per worker.

## Worker collaboration

Workforces include built-in collaboration primitives for worker-to-worker communication.

### Channel

A channel enables direct messaging between workers.
Broadcasts are delivered once per recipient by default; use `broadcast_mode="all"` to treat broadcasts as a log.

```python
from blackgeorge.collaboration import Channel

channel = Channel()

channel.send("worker_a", "worker_b", {"task": "analyze data"})

channel.broadcast("manager", {"status": "starting"})

messages = channel.receive("worker_b")
for msg in messages:
    print(f"From {msg.sender}: {msg.content}")

all_messages = channel.receive("worker_b", broadcast_mode="all")
```

### Blackboard

A blackboard provides shared state across workers.

```python
from blackgeorge.collaboration import Blackboard

bb = Blackboard()

bb.write("analysis_result", {"score": 95}, author="analyst")

result = bb.read("analysis_result")

bb.subscribe("analysis_result", lambda k, v, a: print(f"Updated by {a}"))
```

### Using with Workforce

Pass channel and blackboard when creating a workforce:

```python
from blackgeorge import Worker, Workforce
from blackgeorge.collaboration import Channel, Blackboard

channel = Channel()
blackboard = Blackboard()

workforce = Workforce(
    [w1, w2],
    mode="collaborate",
    channel=channel,
    blackboard=blackboard,
)
```

Workers can access `workforce.channel` and `workforce.blackboard` for communication.

## Resume behavior

Resuming a workforce uses the stored stage in run state:

- manager: resume the manager step
- worker: resume the selected worker
- collaborate: resume the paused worker in sequence

The desk handles these transitions when you call `desk.resume(report, decision_or_input)`.
