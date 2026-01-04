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

## Resume behavior

Resuming a workforce uses the stored stage in run state:

- manager: resume the manager step
- worker: resume the selected worker
- collaborate: resume the paused worker in sequence

The desk handles these transitions when you call `desk.resume(report, decision_or_input)`.
