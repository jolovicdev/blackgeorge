# Workflow

The workflow layer lets you compose multiple steps into a flow. Use it when you need a multi-step pipeline or parallel branches.

## Flow

Create a flow from steps using the desk.

```python
from blackgeorge import Desk, Job, Worker
from blackgeorge.workflow import Step

desk = Desk(model="openai/gpt-5-nano")
analyst = Worker(name="Analyst")
writer = Worker(name="Writer")

flow = desk.flow([Step(analyst), Step(writer)])
report = flow.run(Job(input="Analyze feedback"))
```

A flow produces a report. If there are multiple steps, the content is combined with step headers.

## Steps

- Step: runs a worker or workforce once
- Parallel: runs steps concurrently and returns all results
- Condition: chooses a branch based on a predicate
- Router: selects a route by string key
- Loop: repeats steps until a stop predicate or max iterations
- AsyncCondition: chooses a branch based on an async predicate
- AsyncLoop: repeats steps until an async stop predicate or max iterations

### Step

`Step` wraps a runner and optionally provides a `job_builder` to create a job per step.

```python
from blackgeorge import Job, Worker
from blackgeorge.workflow import Step

worker = Worker(name="Analyst")

def build_job(context):
    return Job(input={"task": "Analyze", "seed": context.job.input})

step = Step(worker, job_builder=build_job)
```

### Parallel

```python
from blackgeorge.workflow import Parallel, Step

parallel = Parallel(Step(worker_a), Step(worker_b))
```

### Condition

```python
from blackgeorge.workflow import Condition, Step

condition = Condition(
    predicate=lambda ctx: bool(ctx.outputs),
    if_true=[Step(worker_a)],
    if_false=[Step(worker_b)],
)
```

### Router

```python
from blackgeorge.workflow import Router, Step

router = Router(
    selector=lambda ctx: "fast",
    routes={"fast": [Step(worker_a)], "deep": [Step(worker_b)]},
)
```

### Loop

```python
from blackgeorge.workflow import Loop, Step

loop = Loop(
    steps=[Step(worker_a)],
    stop=lambda ctx: len(ctx.outputs) >= 3,
    max_iterations=5,
    name="analysis_loop",  # Optional: for tracking iterations
)
```

Loops track iterations in the `WorkflowContext`:

```python
def stop_when_done(ctx):
    iteration = ctx.loop_iteration("analysis_loop")
    return iteration >= 3 or len(ctx.outputs) > 0
```

### AsyncCondition

Use async predicates when the condition requires async operations:

```python
from blackgeorge.workflow import AsyncCondition, Step

async def check_external_service(ctx):
    result = await fetch_status()
    return result.ready

condition = AsyncCondition(
    predicate=check_external_service,
    if_true=[Step(worker_a)],
    if_false=[Step(worker_b)],
)
```

### AsyncLoop

Use async predicates for loop termination:

```python
from blackgeorge.workflow import AsyncLoop, Step

async def should_stop(ctx):
    result = await check_completion()
    return result.done

loop = AsyncLoop(
    steps=[Step(worker_a)],
    stop=should_stop,
    max_iterations=10,
    name="polling_loop",
)
```

## Pause and resume

If a step pauses, the flow stores the current step state, outputs, workflow context, and nested continuation path. Composite nodes (`Condition`, `Router`, `Loop`, and `Parallel`) short-circuit on paused or failed results so work is not skipped or repeated.

```python
report = flow.run(job)
if report.status == "paused":
    report = flow.resume(report, True)
```

Continuation state is persisted through the configured `RunStore`. An equivalent recreated flow can resume after a process restart by calling `recreated_flow.resume(report, decision)`. The recreated flow must have the same node structure, runner names, explicit loop names, and limits. Context artifacts must be JSON-serializable when a flow can pause.

## Async usage

Use the async APIs when you already have an event loop.

```python
report = await flow.arun(job)
if report.status == "paused":
    report = await flow.aresume(report, True)
```

Async flows use non-blocking model calls, and `Parallel` runs steps concurrently. The sync wrappers `flow.run` and `flow.resume` raise if called from a running event loop.
