# Tutorials

Tested examples demonstrating blackgeorge capabilities.

## 1. Basic Worker

```python
from blackgeorge import Desk, Job, Worker

desk = Desk(model="openai/gpt-5-mini")
worker = Worker(name="Assistant")
job = Job(input="Say hello")

report = desk.run(worker, job)
print(report.content)
```

## 2. Tools

```python
from blackgeorge import Desk, Job, Worker
from blackgeorge.tools import tool

@tool()
def search(query: str) -> str:
    """Search for information"""
    return f"Results for: {query}"

worker = Worker(name="Agent", tools=[search])
job = Job(input="Search for Python tutorials")

report = desk.run(worker, job)
```

## 3. Structured Output

```python
from pydantic import BaseModel
from blackgeorge import Desk, Job, Worker

class Analysis(BaseModel):
    sentiment: str
    confidence: float

worker = Worker(name="Analyst")
job = Job(
    input="Analyze: 'I love this product!'",
    response_schema=Analysis
)

report = desk.run(worker, job)
print(report.data)  # Analysis(sentiment='positive', confidence=0.95)
```

## 4. Pause/Resume

```python
from blackgeorge import Desk, Job, Worker
from blackgeorge.tools import tool

@tool(requires_confirmation=True)
def delete_file(path: str) -> str:
    return f"Deleted: {path}"

worker = Worker(name="Agent", tools=[delete_file])
job = Job(input="Delete config.yaml")

report = desk.run(worker, job)
if report.status == "paused":
    print(report.pending_action.prompt)  # "Confirm delete_file (config.yaml)"
    report = desk.resume(report, True)   # Confirm
```

## 5. Events

```python
from blackgeorge import Desk, EventType, Job, Worker

def on_event(event):
    if event.type == EventType.TOOL_COMPLETED:
        print(f"Tool finished: {event.source}")

desk = Desk(model="openai/gpt-5-mini")
desk.event_bus.subscribe("*", on_event)

report = desk.run(worker, job)
```

## 6. Multi-Agent (Workforce)

```python
from blackgeorge import Desk, Job, Worker, Workforce

researcher = Worker(name="Researcher", tools=[search])
writer = Worker(name="Writer")

workforce = Workforce([researcher, writer], mode="managed")
job = Job(input="Research and write about AI")

report = desk.run(workforce, job)
```

## 7. Swarm Mode

```python
from blackgeorge import Workforce
from blackgeorge.tools import transfer_to_agent_tool

handoff = transfer_to_agent_tool(["Writer", "Editor"])

researcher = Worker(name="Researcher", tools=[search, handoff])
writer = Worker(name="Writer", tools=[handoff])
editor = Worker(name="Editor")

workforce = Workforce([researcher, writer, editor], mode="swarm")
```

## Patterns

### Tool Safety

```python
@tool(requires_confirmation=True)  # Ask before executing
def delete(path: str) -> str: ...

@tool(requires_user_input=True)    # Collect user input
def ask(question: str) -> str: ...

@tool(timeout=30.0, retries=3)     # Resilient execution
def fetch(url: str) -> str: ...
```

### RunConfig

```python
from blackgeorge import RunConfig

config = RunConfig(
    model="openai/gpt-5-mini",
    max_iterations=20,
    max_context_messages=15,
)

report = worker.run(config, job)
```

### Async

```python
report = await desk.arun(worker, job)
report = await desk.aresume(report, decision)
```
