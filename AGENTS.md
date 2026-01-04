# AGENTS.md

## Project overview
- Name: Blackgeorge
- Description: Code-first agentic framework built around Desk, Worker, and Workforce.
- Language: Python 3.12 (see `.python-version`)
- Tooling: uv, ruff, mypy, pytest, mkdocs
- Paradigm: Object-oriented core with Pydantic models and small dataclasses

## Commands
- Install dev deps: `uv pip install -e .[dev]`
- Lint: `uv run ruff check .`
- Format: `uv run ruff format .`
- Type check: `uv run mypy src`
- Tests: `uv run pytest`
- Docs: `uv run mkdocs serve` or `uv run mkdocs build`

## Dependency management
- Use `uv add <package>` and `uv add --dev <package>` only.
- Do not use pip or venv directly.

## Code style and constraints
- No comments or docstrings in code; rely on clear naming and structure.
- All functions must be type annotated; keep mypy strict config passing.
- Avoid `Any` when a concrete type, Protocol, or TypeVar is available.
- Prefer absolute imports from `blackgeorge`.
- No unused imports or variables.
- Avoid excessive try/except; handle errors only where needed.
- Avoid module import side effects and network calls.
- Keep modules focused; no large refactors unless explicitly requested.
- Follow established codebase rules and architecture.

## Project structure
- `src/blackgeorge/`: library source
- `src/blackgeorge/core/`: core Pydantic models and types
- `src/blackgeorge/adapters/`: model adapters and Instructor wiring
- `src/blackgeorge/tools/`: tool schema, registry, and execution
- `src/blackgeorge/workflow/`: flow orchestration
- `src/blackgeorge/store/` and `src/blackgeorge/memory/`: persistence backends
- `tests/`: pytest suite
- `docs/`: mkdocs content
- `examples/`: example projects

## Repo hygiene
- Do not commit `.blackgeorge/` run state or database files.
- Do not bump versions in `pyproject.toml` unless asked.
- If behavior changes, update or add tests and docs as needed.

## Commit messages
- Use Conventional Commits: `type: subject`
- Types: feat, fix, refactor, docs, test, chore
- Subject is imperative, lowercase, and has no trailing period
- Example: `refactor: split worker runtime`

## Pre-flight checklist
- `uv run ruff check .`
- `uv run ruff format .`
- `uv run mypy src`
- `uv run pytest`
