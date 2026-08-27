# Development

This project uses pytest for tests, ruff for linting, and mypy for type checking.

## Development workflow

Typical local workflow:

1. Install dev dependencies.
1. Make code and docs changes.
1. Run ruff to catch lint and style issues early.
1. Run mypy to ensure type correctness under the strict config.
1. Run pytest to validate behavior.
1. Run MkDocs serve/build to verify documentation changes.

This order keeps fast feedback loops first (ruff and mypy), then tests, then docs.

MkDocs uses `docs/index.md` as the site home. The `docs/README.md` file is excluded to avoid a name conflict with the home page.

## Install dev dependencies

```text
uv pip install -e .[dev]
```

## Run tests

```text
uv run pytest
```

## Lint

```text
uv run ruff check .
```

```text
uv run ruff format .
```

## Docs site

```text
uv run mkdocs serve
```

```text
uv run mkdocs build
```

## Type check

```text
uv run mypy src
```
