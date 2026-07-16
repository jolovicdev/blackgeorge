# Workforce

A `Workforce` coordinates multiple workers. It supports three modes: managed, collaborate, and swarm.

Workforce reports carry run-level `usage` and `cost_usd` totals in `Report.metrics`, summed across
every metered worker execution in the run. See [Configuration](configuration.md#cost-budgets) for
how totals and budgets are computed.

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
- Manager selection runs with tools disabled, even if the manager has tools, to force a structured selection response.

The selection rules are:

- If the manager returns a structured response with a `worker` field, that worker is used.
- Otherwise, the system scans the manager output for a worker name.
- If nothing matches, the first worker is used.

If the manager or selected worker pauses, the workforce returns a paused report with enough state to resume later.

## Swarm mode

In swarm mode, workers can dynamically hand off execution and context to another agent in the workforce at any time using the `transfer_to_agent_tool`. This enables complex, dynamic multi-agent orchestrations without a hardcoded manager.

```python
from blackgeorge.tools import transfer_to_agent_tool

handoff_tool = transfer_to_agent_tool(["coder", "reviewer"])

coder = Worker(name="coder", tools=[handoff_tool])
reviewer = Worker(name="reviewer")

swarm = Workforce([coder, reviewer], mode="swarm")
```

When a worker calls `transfer_to_agent`, the orchestrator intercepts the pending tool action and
switches the active worker inside the same run.

Swarm handoff safety rules:

- The target must exist in the current workforce.
- If the handoff tool has an `agent_name` allowlist (from `transfer_to_agent_tool([...])`), the target must be in that allowlist.
- Allowlist enforcement works for both worker-level tools and `Job.tools_override` tools.
- System-role messages from the prior worker are not carried into the next worker handoff context, preventing instruction leakage across roles.
- Handoff transitions are bounded per run. Exceeding the cap returns a failed report.

## Collaborate mode

In collaborate mode, reports are combined across all workers.

- If all workers have no tools, workers run concurrently for lower latency.
- If any worker has tools, workers run sequentially to preserve deterministic pause/resume behavior.
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

- swarm: resume the exact paused worker identity, including handoff continuation
- manager: resume the manager step
- worker: resume the selected worker
- collaborate: resume the paused worker in sequence

The desk handles these transitions when you call `desk.resume(report, decision_or_input)`.

If the stored pending worker index no longer matches the current worker list, resume fails with a `failed` report rather than raising.

For swarm runs, resume also fails explicitly if the paused worker identity is missing or no longer registered.
