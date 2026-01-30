# AGENTS.md

## Project Overview

Blackgeorge is a code-first Python framework for building AI agents with explicit APIs, tool calling, structured outputs, and multi-agent orchestration.

Core primitives: **Desk** (runtime), **Worker** (single agent), **Workforce** (multi-agent coordination), **Workflow** (step-based flows).

## Tech Stack

- Python 3.12+
- `uv` for package management
- Pydantic v2 for data models
- LiteLLM + Instructor for LLM calls
- pytest + ruff + mypy for dev tooling

## Project Structure

- `src/blackgeorge/` — Library source
- `src/blackgeorge/core/` — Pydantic models (Event, Job, Report, etc.)
- `src/blackgeorge/adapters/` — LLM provider adapters
- `src/blackgeorge/tools/` — Tool system (@tool decorator, registry, execution)
- `src/blackgeorge/workflow/` — Flow orchestration (Step, Parallel)
- `src/blackgeorge/store/` — Run persistence backends
- `src/blackgeorge/memory/` — Memory stores
- `docs/` — MkDocs documentation
- `tests/` — pytest suite

## Commands

Install dev deps:
```bash
uv pip install -e .[dev]
```

Run tests:
```bash
uv run pytest
```

Lint and format:
```bash
uv run ruff check .
uv run ruff format .
```

Type check:
```bash
uv run mypy src
```

Serve docs:
```bash
uv run mkdocs serve
```

## Code Conventions

### Style
- No comments or docstrings in code; rely on clear naming
- All functions must be type annotated
- Avoid `Any`; use concrete types, Protocols, or TypeVars
- Prefer absolute imports from `blackgeorge`
- No unused imports or variables
- Line length: 100

### Naming
- `PascalCase` for classes
- `snake_case` for functions, methods, variables
- `_prefix` for private methods/attributes

### Types
- Use `str | None` not `Optional[str]`
- Use `collections.abc` for abstract base classes
- Use `TYPE_CHECKING` to avoid circular imports
- Pydantic models for complex data with validation

### Patterns
- Use `@dataclass(frozen=True)` for immutable value objects
- Use `ConfigDict(frozen=True, arbitrary_types_allowed=True)` for immutable Pydantic models
- Provide both sync and async variants: `run()` / `arun()`
- Avoid module import side effects

## Dev Environment Tips

- Use `uv add <package>` for deps, `uv add --dev <package>` for dev deps
- Never use pip directly
- Flow.run/Flow.resume cannot be called from a running event loop (use arun/aresume)

## Testing Instructions

- Run the full suite: `uv run pytest`
- Run specific test: `uv run pytest tests/test_worker.py::test_name -v`
- Tests use pytest-asyncio with `asyncio_mode = auto`
- Add tests for behavior changes, even if not asked

## PR / Commit Guidelines

Commit message format (Conventional Commits):
```
type: subject
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`
Subject: imperative, lowercase, no trailing period
Example: `feat: add vector memory chunking`

Pre-commit checklist:
```bash
uv run ruff check .
uv run ruff format .
uv run mypy src
uv run pytest
```

## Critical Rules

- **DO NOT** commit `.blackgeorge/` run state or database files
- **DO NOT** commit `.env` files or API keys
- **DO NOT** bump versions in `pyproject.toml` unless asked
- **ALWAYS** update docs in `docs/` when changing public API or behavior (if module does not exists, make new .md file and update mkdocs.yml)
- **ALWAYS** add tests when changing behavior
