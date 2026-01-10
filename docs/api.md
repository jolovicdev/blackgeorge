# REST API

**The REST API is experimental and subject to change.**

The Blackgeorge REST API exposes all framework functionality via HTTP endpoints.

## Quick Start

```bash
# Install dependencies
uv pip install -e .[dev]

# Start the server
uv run uvicorn blackgeorge.api:create_app --factory --reload

# Or with host/port
uv run uvicorn blackgeorge.api:create_app --factory --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

Interactive documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Authentication

Currently no authentication. For production use, add API keys or OAuth2.

## Core Concepts

### Workers
Single AI agents with tools, instructions, and models.

### Workforces
Multi-agent coordination in managed or collaborate mode.

### Runs
Individual job executions with status tracking and pause/resume.

## Endpoints

### Health

#### GET /health
Check API health status.

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "version": "0.1.0"
}
```

### Workers

#### POST /api/v1/workers
Register a new worker.

```bash
curl -X POST http://localhost:8000/api/v1/workers \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Researcher",
    "model": "openai/gpt-4",
    "instructions": "You are a researcher."
  }'
```

#### GET /api/v1/workers
List all registered workers.

```bash
curl http://localhost:8000/api/v1/workers
```

#### GET /api/v1/workers/{name}
Get specific worker details.

```bash
curl http://localhost:8000/api/v1/workers/Researcher
```

#### DELETE /api/v1/workers/{name}
Unregister a worker.

```bash
curl -X DELETE http://localhost:8000/api/v1/workers/Researcher
```

### Workforces

#### POST /api/v1/workforces
Create a workforce.

```bash
curl -X POST http://localhost:8000/api/v1/workforces \
  -H "Content-Type: application/json" \
  -d '{
    "name": "MyTeam",
    "workers": ["Researcher", "Writer"],
    "mode": "managed"
  }'
```

#### GET /api/v1/workforces
List all workforces.

```bash
curl http://localhost:8000/api/v1/workforces
```

#### GET /api/v1/workforces/{name}
Get specific workforce details.

```bash
curl http://localhost:8000/api/v1/workforces/MyTeam
```

#### DELETE /api/v1/workforces/{name}
Unregister a workforce.

```bash
curl -X DELETE http://localhost:8000/api/v1/workforces/MyTeam
```

### Runs

#### POST /api/v1/runs/worker/{name}
Execute a job with a worker.

```bash
curl -X POST http://localhost:8000/api/v1/runs/worker/Researcher \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Summarize AI trends in 2024",
    "expected_output": "A concise summary"
  }'
```

Response:
```json
{
  "run_id": "abc123",
  "status": "completed",
  "content": "AI trends in 2024 include...",
  "data": null,
  "pending_action": null,
  "metrics": {},
  "errors": [],
  "messages": [],
  "tool_calls": []
}
```

#### POST /api/v1/runs/workforce/{name}
Execute a job with a workforce.

```bash
curl -X POST http://localhost:8000/api/v1/runs/workforce/MyTeam \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Write a research report on quantum computing"
  }'
```

#### POST /api/v1/runs/{run_id}/resume
Resume a paused run.

```bash
curl -X POST http://localhost:8000/api/v1/runs/{run_id}/resume \
  -H "Content-Type: application/json" \
  -d '{
    "decision": true
  }'
```

### Status

#### GET /api/v1/runs/{run_id}
Get run status and results.

```bash
curl http://localhost:8000/api/v1/runs/{run_id}
```

#### GET /api/v1/runs
List all runs with optional filtering.

```bash
# List all runs
curl http://localhost:8000/api/v1/runs

# Filter by status
curl "http://localhost:8000/api/v1/runs?status=completed"

# With pagination
curl "http://localhost:8000/api/v1/runs?limit=10&offset=20"
```

#### GET /api/v1/runs/{run_id}/events
Get events for a run.

```bash
curl http://localhost:8000/api/v1/runs/{run_id}/events
```

## Configuration

Configure the API via environment variables with the `BLACKGEORGE_API_` prefix:

```bash
export BLACKGEORGE_API_DEFAULT_MODEL="openai/gpt-4"
export BLACKGEORGE_API_STORAGE_DIR="/data/blackgeorge"
export BLACKGEORGE_API_CORS_ORIGINS='["http://localhost:3000"]'
```

## Development

### Running Tests

```bash
# Run all tests
uv run pytest

# Run API tests only
uv run pytest tests/api/

# Run with coverage
uv run pytest tests/api/ --cov=blackgeorge.api
```

### Type Checking

```bash
uv run mypy src/blackgeorge/api/
```

### Linting

```bash
uv run ruff check src/blackgeorge/api/
uv run ruff format src/blackgeorge/api/
```
